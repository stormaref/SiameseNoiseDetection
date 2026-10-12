"""Co-teaching as a two-network noise detector, scored against our ensemble on the same noisy labels.

Training (Han et al., NeurIPS 2018): two PreAct-ResNet18 networks on the full noisy training set (labels
taken from the committed predictions, so both methods see exactly the same labels). In every
mini-batch each network ranks its per-sample cross-entropy and keeps the R(T) smallest; each network
is updated on the other's selection. R(T) = 1 - min(T / Tk, 1) * tau with tau the actual noise rate
and Tk = 10; 200 epochs, batch 128, Adam (lr 1e-3); from epoch 80 the learning rate decays linearly
to 0 and beta1 drops from 0.9 to 0.1. Training images get random crop (padding 4) and horizontal flip.

Detection: a sample is flagged when neither network selected it during the final epoch ("final
small-loss selection"), or when it is among the tau-fraction of largest losses of both final networks
on the un-augmented image ("loss ranking, both networks"). Flagged samples are removed. Our ensemble is
scored from the committed predictions at the paper's thresholds (relabel or remove).

Usage:
    uv run python scripts/coteaching.py --dataset cifar10 --noise_ratio 20            # train + score (GPU)
    uv run python scripts/coteaching.py --dataset cifar10 --noise_ratio 20 --analyze results/coteaching_cifar10_20.npz
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import snd.config as config
from snd.models.preact import PreActResNet18

PARAMS = {('cifar10', '20'): 'CIFAR10_20_PARAMS', ('cifar10', '30'): 'CIFAR10_30_PARAMS',
          ('cifar10', '40'): 'CIFAR10_40_PARAMS', ('fashionmnist', '20'): 'FashionMNIST_20_PARAMS',
          ('fashionmnist', '30'): 'FashionMNIST_30_PARAMS', ('fashionmnist', '40'): 'FashionMNIST_40_PARAMS'}
DATA = {'cifar10': ('CIFAR10_TRAIN_DATASET', 'CIFAR10_TEST_DATASET',
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'fashionmnist': ('FashionMNIST_TRAIN_DATASET', 'FashionMNIST_TEST_DATASET',
                         (0.2860,) * 3, (0.3530,) * 3)}
SCHEDULE = dict(epochs=200, batch_size=128, lr=1e-3, decay_start=80, tk=10)


def label_table(dataset, noise_ratio):
    """Per-sample noisy/true labels and our ensemble's votes, indexed by dataset position."""
    params = getattr(config, PARAMS[(dataset, noise_ratio)])
    df = pd.concat([pd.read_csv(params['prediction_path'].format(k))
                    for k in range(1, params['outer_folds_num'] + 1)])
    df = df.set_index('index').sort_index()
    df['is_noisy'] = df['noisy_label'] != df['real_label']
    return df


def images(dataset, train):
    ds = getattr(config, DATA[dataset][0 if train else 1])
    x = ds.data.numpy() if torch.is_tensor(ds.data) else np.asarray(ds.data)
    if x.ndim == 3:                                    # grayscale -> 3 identical channels
        x = np.repeat(x[..., None], 3, axis=3)
    return np.ascontiguousarray(x), np.asarray(ds.targets, dtype=np.int64)


def augment(x, pad=4):
    """RandomCrop(size, padding=pad) + RandomHorizontalFlip on a float NCHW batch (zero padding)."""
    b, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (i[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    out = xp[torch.arange(b, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None],
             rows, cols]
    flip = torch.rand(b, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


@torch.no_grad()
def predict(net, x_u8, mean, std, amp, batch=1024):
    net.eval()
    out = []
    for s in range(0, len(x_u8), batch):
        x = (x_u8[s:s + batch].float() / 255 - mean) / std
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = net(x)
        out.append(torch.softmax(logits.float(), 1))
    return torch.cat(out)


def train(dataset, table, seed, out_path, epochs=None, subsample=None):
    """Train the two networks and save the final selections and losses to `out_path` (.npz)."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = dev.type == 'cuda'
    torch.backends.cudnn.benchmark = True
    x_train, _ = images(dataset, train=True)
    x_test, y_test = images(dataset, train=False)
    index = table.index.to_numpy()
    if subsample:                                      # quick checks only
        index = index[:subsample]
    noisy = table.loc[index, 'noisy_label'].to_numpy().astype(np.int64)
    is_noisy_np = table.loc[index, 'is_noisy'].to_numpy(bool)

    X = torch.from_numpy(x_train[index]).permute(0, 3, 1, 2).contiguous().to(dev)
    Y = torch.from_numpy(noisy).to(dev)
    Xte = torch.from_numpy(x_test).permute(0, 3, 1, 2).contiguous().to(dev)
    Yte = torch.from_numpy(y_test).to(dev)
    is_noisy = torch.from_numpy(is_noisy_np).to(dev)
    mean, std = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in DATA[dataset][2:])
    n, bs = len(Y), SCHEDULE['batch_size']
    epochs = epochs or SCHEDULE['epochs']

    tau = float(is_noisy_np.mean())                   # forget rate = actual noise rate
    rate = np.full(epochs, tau)
    k = min(SCHEDULE['tk'], epochs)
    rate[:k] = np.linspace(0, tau, SCHEDULE['tk'])[:k]
    lr_plan, beta_plan = np.full(epochs, SCHEDULE['lr']), np.full(epochs, 0.9)
    for e in range(SCHEDULE['decay_start'], epochs):
        lr_plan[e] = (epochs - e) / (epochs - SCHEDULE['decay_start']) * SCHEDULE['lr']
        beta_plan[e] = 0.1

    torch.manual_seed(seed)
    net1, net2 = PreActResNet18(10).to(dev), PreActResNet18(10).to(dev)      # different inits
    opt1 = torch.optim.Adam(net1.parameters(), lr=SCHEDULE['lr'])
    opt2 = torch.optim.Adam(net2.parameters(), lr=SCHEDULE['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    for epoch in range(epochs):
        for opt in (opt1, opt2):
            for g in opt.param_groups:
                g['lr'], g['betas'] = float(lr_plan[epoch]), (float(beta_plan[epoch]), 0.999)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100003 + epoch)).to(dev)
        sel1 = torch.zeros(n, dtype=torch.bool, device=dev)
        sel2 = torch.zeros(n, dtype=torch.bool, device=dev)
        net1.train()
        net2.train()
        for idx in torch.tensor_split(perm, math.ceil(n / bs)):
            x = (augment(X[idx].float() / 255) - mean) / std
            y = Y[idx]
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out1, out2 = net1(x), net2(x)
            l1 = F.cross_entropy(out1.float(), y, reduction='none')
            l2 = F.cross_entropy(out2.float(), y, reduction='none')
            keep = int((1 - float(rate[epoch])) * len(idx))
            i1 = torch.argsort(l1.detach())[:keep]                # net1's small-loss samples
            i2 = torch.argsort(l2.detach())[:keep]
            sel1[idx[i1]] = True
            sel2[idx[i2]] = True
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            scaler.scale(l1[i2].mean() + l2[i1].mean()).backward()      # cross update
            scaler.step(opt1)
            scaler.step(opt2)
            scaler.update()
        flag = ~sel1 & ~sel2
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            p = predict(net1, Xte, mean, std, amp) + predict(net2, Xte, mean, std, amp)
            test_acc = float((p.argmax(1) == Yte).float().mean() * 100)
            print(f'epoch {epoch + 1}/{epochs}: forget rate {rate[epoch]:.3f}, flagged {int(flag.sum())} '
                  f'(precision {float(is_noisy[flag].float().mean()) if flag.any() else float("nan"):.3f}), '
                  f'test accuracy {test_acc:.2f}%', flush=True)

    s1, s2 = predict(net1, X, mean, std, amp), predict(net2, X, mean, std, amp)
    eval_loss1 = F.nll_loss(torch.log(s1.clamp_min(1e-12)), Y, reduction='none')
    eval_loss2 = F.nll_loss(torch.log(s2.clamp_min(1e-12)), Y, reduction='none')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, index=index, sel1=sel1.cpu().numpy(), sel2=sel2.cpu().numpy(),
                        eval_loss1=eval_loss1.cpu().numpy(), eval_loss2=eval_loss2.cpu().numpy(),
                        forget_rate=tau)
    print(f'wrote {out_path}')


def metrics(flag, score, new_label, rows):
    """Detection and correction metrics; new_label >= 0 relabels a flagged sample, -1 removes it."""
    is_noisy = rows['is_noisy'].to_numpy(bool)
    noisy, true = rows['noisy_label'].to_numpy(), rows['real_label'].to_numpy()
    tp, fp = int((flag & is_noisy).sum()), int((flag & ~is_noisy).sum())
    prec, rec = (tp / (tp + fp) if tp + fp else 0.0), tp / is_noisy.sum()
    final = np.where(flag, new_label, noisy)
    retained = ~(flag & (new_label < 0))
    return {'Precision': prec, 'Recall': rec, 'F1': 2 * prec * rec / (prec + rec) if tp else 0.0,
            'FPR': fp / (~is_noisy).sum(), 'ROC-AUC': roc_auc_score(is_noisy, score),
            'Retained': int(retained.sum()),
            'Residual noise': float((retained & (final != true)).sum() / retained.sum())}


def analyze(dataset, noise_ratio, table, npz_path):
    with np.load(npz_path) as z:
        rows = table.loc[z['index']]
        n = len(rows)
        score = (z['eval_loss1'] + z['eval_loss2']) / 2
        remove = np.full(n, -1)
        k = int(round(float(z['forget_rate']) * n))
        # stable sort: ties at the cutoff (saturated or equal losses) are broken by sample order
        top = lambda loss: np.isin(np.arange(n), np.argsort(-loss, kind='stable')[:k])  # noqa: E731
        results = {
            'Co-teaching (final small-loss selection)': metrics(~z['sel1'] & ~z['sel2'], score, remove, rows),
            'Co-teaching (loss ranking, both networks)': metrics(top(z['eval_loss1']) & top(z['eval_loss2']),
                                                                 score, remove, rows),
        }
    td, tr = config.PAPER_THRESHOLDS[(dataset, noise_ratio)]
    votes = np.stack(rows['preds'].astype(str).str.split('|').map(lambda x: np.array(x, dtype=int)).values)
    hist = (votes[:, :, None] == np.arange(10)).sum(axis=1)
    meets = hist >= tr
    flag = rows['mistakes'].to_numpy() >= td
    results[f'Ours (T_D={td}, T_R={tr})'] = metrics(flag, rows['mistakes'].to_numpy() / votes.shape[1],
                                                    np.where(meets.any(1), meets.argmax(1), -1), rows)
    out = pd.DataFrame(results).T
    pd.set_option('display.width', 200)
    print(out.to_string(formatters={c: (lambda v: f'{v:,.0f}') if c == 'Retained' else
                                    (lambda v: f'{100 * v:.1f}%') if c == 'Residual noise' else
                                    (lambda v: f'{v:.3f}') for c in out.columns}))
    return out


def main():
    p = argparse.ArgumentParser(description='Co-teaching vs. our ensemble on the same noisy labels')
    p.add_argument('--dataset', required=True, choices=['cifar10', 'fashionmnist'])
    p.add_argument('--noise_ratio', required=True, choices=['20', '30', '40'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--analyze', metavar='NPZ', help='score a saved run instead of training')
    p.add_argument('--output_dir', default='results')
    p.add_argument('--epochs', type=int, help='override the 200-epoch schedule (quick checks only)')
    p.add_argument('--subsample', type=int, help='train on the first N samples (quick checks only)')
    args = p.parse_args()

    table = label_table(args.dataset, args.noise_ratio)
    npz = args.analyze
    if npz is None:
        npz = os.path.join(args.output_dir, f'coteaching_{args.dataset}_{args.noise_ratio}.npz')
        train(args.dataset, table, args.seed, npz, args.epochs, args.subsample)
    analyze(args.dataset, args.noise_ratio, table, npz)


if __name__ == '__main__':
    main()
