"""R5.2 worker: two-network ensemble baselines on our exact noisy labels.

Jobs are (method, dataset, noise) triples from the config; each trains on the full noisy
training set (50k CIFAR-10 / 60k Fashion-MNIST, labels rebuilt from the committed preds CSVs by
snd_kaggle) and writes <out_dir>/<job>/final.npz, which r52_analysis.py scores.

coteaching  Han et al., NeurIPS 2018, reimplemented here: two PreAct-ResNet18 (snd.models.preact),
            each mini-batch every net ranks its per-sample CE loss and keeps the R(T) smallest;
            net1 is updated on net2's small-loss samples and vice versa. R(T) = 1 - min(T/Tk, 1)*tau,
            tau = actual noise rate, Tk = 10; 200 epochs, batch 128, Adam lr 1e-3, from epoch 80 the
            lr decays linearly to 0 and beta1 drops 0.9 -> 0.1 (the official schedule).
            Saved: which samples each net selected during the final epoch (detection), per-sample
            losses, and both nets' softmax on the un-augmented training images (correction).
dividemix   Li et al., ICLR 2020, the OFFICIAL code (github.com/LiJunnan1992/DivideMix, MIT licence,
            pinned commit) cloned at runtime and patched by `patch_dividemix` below: runs on current
            PyTorch/CPU/MPS, AMP, reads our images + our noisy labels (its own noise_file JSON
            mechanism), dumps both nets' GMM clean probabilities every epoch, checkpoints every
            `ckpt_every` epochs and before the session deadline (exit code 3 = paused, resume by
            rerunning), and exports the final GMM division + both nets' softmax.

Usage:  python r52_worker.py config.json             # worker loop (one per GPU via launch_workers)
        python r52_worker.py config.json --prepare   # one-off: data, preds, DivideMix clone/patch
"""
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snd_kaggle as K  # noqa: E402
from r52_filters import detection_metrics  # noqa: E402

DM_URL = 'https://github.com/LiJunnan1992/DivideMix.git'
DM_COMMIT = 'd9d3058fa69a952463b896f84730378cdee6ec39'      # master, 2020-09-14

COTEACHING = dict(epochs=200, batch_size=128, lr=1e-3, decay_start=80, tk=10, exponent=1,
                  forget_rate=None,          # None -> actual noise rate of the training labels
                  augment=True,              # random crop (pad 4) + horizontal flip, done on the GPU
                  ckpt_every=10, eval_every=5, eval_batch=1024, max_epochs_this_run=0)
# Paper Sec. 5.1 / App. B + the official Train_cifar.py defaults. noise_mode 'asym' turns on the
# warm-up confidence penalty the paper prescribes for non-uniform (class-conditional) noise, and
# lambda_u = 0 is Table 7's value for every CIFAR-10 setting with <= 40% noise (sym 20%, asym 40%).
DIVIDEMIX = dict(commit=DM_COMMIT, epochs=300, lr_drop=150, batch_size=64, lr=0.02, noise_mode='asym',
                 lambda_u=0, p_threshold=0.5, T=0.5, alpha=4, ckpt_every=5, num_workers=3,
                 warm_up=0, max_epochs_this_run=0, progress_every=120)    # warm_up 0 = official (10)
NORM = {'cifar10': (K.CIFAR_MEAN, K.CIFAR_STD),
        'fashionmnist': ((0.2860,) * 3, (0.3530,) * 3)}      # FMNIST train-set mean/std, 3 channels


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if os.environ.get('R52_MPS') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# --------------------------------------------------------------------------- data
_DATA = {}


def job_data(cfg, dataset, noise):
    """Our noisy training set as uint8 NHWC arrays (+ the label table rows, same order)."""
    key = (dataset, noise)
    if key in _DATA:
        return _DATA[key]
    proto = K.PROTOCOL[(dataset, noise)]
    repo = K.setup_repo(cfg['code_ref'])
    table = K.load_label_table(K.extract_preds(repo, cfg['preds_ref'], proto['preds']))
    ds = K.noisy_train_set(dataset, table)            # checks CSVs against torchvision targets
    x = ds.data.numpy() if torch.is_tensor(ds.data) else np.asarray(ds.data)
    test = K.base_dataset(dataset, train=False)
    xt = test.data.numpy() if torch.is_tensor(test.data) else np.asarray(test.data)
    yt = np.asarray(test.targets, dtype=np.int64)
    if x.ndim == 3:                                   # grayscale -> 3 identical channels
        x, xt = np.repeat(x[..., None], 3, axis=3), np.repeat(xt[..., None], 3, axis=3)
    table, keep = K.smoke_subsample(table, cfg)       # smoke tests only; same draw as r51_worker
    if keep is not None:
        x = x[keep]
    if cfg.get('test_subsample'):
        xt, yt = xt[:cfg['test_subsample']], yt[:cfg['test_subsample']]
    if (table['is_noisy'].to_numpy() != (table['noisy_label'] != table['real_label']).to_numpy()).any():
        raise ValueError('is_noisy disagrees with noisy_label != real_label')
    data = dict(table=table, index=table.index.to_numpy(), x_train=np.ascontiguousarray(x),
                noisy=table['noisy_label'].to_numpy().astype(np.int64),
                true=table['real_label'].to_numpy().astype(np.int64),
                is_noisy=table['is_noisy'].to_numpy(bool), x_test=np.ascontiguousarray(xt), y_test=yt)
    _DATA.clear()                                     # keep one dataset in memory
    _DATA[key] = data
    return data


def job_dir(cfg, job):
    path = os.path.join(cfg['out_dir'], job['id'])
    os.makedirs(path, exist_ok=True)
    return path


def add_minutes(path, minutes):
    """Accumulate wall-clock minutes of a job across sessions (timing.json)."""
    f = os.path.join(path, 'timing.json')
    t = json.load(open(f)) if os.path.exists(f) else {'sessions': []}
    t['sessions'].append(round(minutes, 2))
    json.dump(t, open(f, 'w'))
    return sum(t['sessions'])


# --------------------------------------------------------------------------- Co-teaching
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
def predict(net, x_u8, mean, std, amp, batch):
    net.eval()
    out = []
    for s in range(0, len(x_u8), batch):
        x = (x_u8[s:s + batch].float() / 255 - mean) / std
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = net(x)
        out.append(torch.softmax(logits.float(), 1))
    return torch.cat(out)


def coteaching_job(cfg, job, deadline_end):
    from snd.models.preact import PreActResNet18
    ct = dict(COTEACHING, **cfg.get('coteaching', {}))
    data = job_data(cfg, job['dataset'], job['noise'])
    out_dir = job_dir(cfg, job)
    dev = device()
    amp = bool(cfg.get('amp', True)) and dev.type == 'cuda'
    torch.backends.cudnn.benchmark = True

    X = torch.from_numpy(data['x_train']).permute(0, 3, 1, 2).contiguous().to(dev)       # uint8
    Y = torch.from_numpy(data['noisy']).to(dev)
    Xte = torch.from_numpy(data['x_test']).permute(0, 3, 1, 2).contiguous().to(dev)
    Yte = torch.from_numpy(data['y_test']).to(dev)
    is_noisy = torch.from_numpy(data['is_noisy']).to(dev)
    mean, std = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in NORM[job['dataset']])
    n, epochs, bs = len(Y), ct['epochs'], ct['batch_size']

    tau = float(data['is_noisy'].mean()) if ct['forget_rate'] is None else float(ct['forget_rate'])
    rate = np.full(epochs, tau)
    k = min(ct['tk'], ct['epochs'])                 # short (smoke) runs end inside the ramp
    rate[:k] = np.linspace(0, tau ** ct['exponent'], ct['tk'])[:k]
    lr_plan = np.full(epochs, ct['lr'])
    beta_plan = np.full(epochs, 0.9)
    for e in range(ct['decay_start'], epochs):
        lr_plan[e] = (epochs - e) / (epochs - ct['decay_start']) * ct['lr']
        beta_plan[e] = 0.1

    torch.manual_seed(cfg['seed'])
    net1, net2 = PreActResNet18(10).to(dev), PreActResNet18(10).to(dev)      # different inits
    opt1 = torch.optim.Adam(net1.parameters(), lr=ct['lr'])
    opt2 = torch.optim.Adam(net2.parameters(), lr=ct['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ckpt = os.path.join(out_dir, 'ckpt.pt')
    history, start = [], 0
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)   # RNG states must stay on CPU
        net1.load_state_dict(ck['net1'])
        net2.load_state_dict(ck['net2'])
        opt1.load_state_dict(ck['opt1'])
        opt2.load_state_dict(ck['opt2'])
        if ck['scaler']:
            scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['rng_cpu'])
        if dev.type == 'cuda' and ck['rng_cuda'] is not None:
            torch.cuda.set_rng_state(ck['rng_cuda'])
        history, start = ck['history'], ck['epoch'] + 1
        print(f'{job["id"]}: resumed after epoch {ck["epoch"]}', flush=True)
        del ck

    def save(epoch):
        torch.save(dict(epoch=epoch, net1=net1.state_dict(), net2=net2.state_dict(), opt1=opt1.state_dict(),
                        opt2=opt2.state_dict(), scaler=scaler.state_dict(), history=history,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state() if dev.type == 'cuda' else None), ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)

    t_session = time.time()
    sel1 = sel2 = loss1 = loss2 = None
    for epoch in range(start, epochs):
        t0 = time.time()
        for opt in (opt1, opt2):
            for g in opt.param_groups:
                g['lr'], g['betas'] = float(lr_plan[epoch]), (float(beta_plan[epoch]), 0.999)
        fr = float(rate[epoch])
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg['seed'] * 100003 + epoch)).to(dev)
        sel1 = torch.zeros(n, dtype=torch.bool, device=dev)
        sel2 = torch.zeros(n, dtype=torch.bool, device=dev)
        loss1 = torch.zeros(n, device=dev)
        loss2 = torch.zeros(n, device=dev)
        upd = torch.zeros(2, device=dev)
        net1.train()
        net2.train()
        batches = torch.tensor_split(perm, math.ceil(n / bs))      # near-equal sizes: no 1-sample tail
        for idx in batches:
            x = X[idx].float() / 255
            if ct['augment']:
                x = augment(x)
            x = (x - mean) / std
            y = Y[idx]
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out1, out2 = net1(x), net2(x)
            l1 = F.cross_entropy(out1.float(), y, reduction='none')
            l2 = F.cross_entropy(out2.float(), y, reduction='none')
            loss1[idx], loss2[idx] = l1.detach(), l2.detach()
            keep = int((1 - fr) * len(idx))                       # official: int(remember_rate * batch)
            i1 = torch.argsort(l1.detach())[:keep]                # net1's small-loss samples
            i2 = torch.argsort(l2.detach())[:keep]
            sel1[idx[i1]] = True
            sel2[idx[i2]] = True
            u1, u2 = l1[i2].mean(), l2[i1].mean()                 # cross update (exchange)
            opt1.zero_grad(set_to_none=True)
            opt2.zero_grad(set_to_none=True)
            scaler.scale(u1 + u2).backward()
            scaler.step(opt1)
            scaler.step(opt2)
            scaler.update()
            upd += torch.stack([u1.detach(), u2.detach()])
        flag = ~sel1 & ~sel2
        row = dict(epoch=epoch, lr=float(lr_plan[epoch]), beta1=float(beta_plan[epoch]), forget_rate=fr,
                   loss1=float(upd[0]) / len(batches), loss2=float(upd[1]) / len(batches),
                   label_prec1=float(1 - is_noisy[sel1].float().mean()),
                   label_prec2=float(1 - is_noisy[sel2].float().mean()),
                   flagged=int(flag.sum()), flag_prec=float(is_noisy[flag].float().mean()) if flag.any() else np.nan,
                   test_acc1=np.nan, test_acc2=np.nan, test_acc=np.nan)
        last = epoch == epochs - 1
        if last or (epoch + 1) % ct['eval_every'] == 0:
            p1 = predict(net1, Xte, mean, std, amp, ct['eval_batch'])
            p2 = predict(net2, Xte, mean, std, amp, ct['eval_batch'])
            row.update(test_acc1=float((p1.argmax(1) == Yte).float().mean() * 100),
                       test_acc2=float((p2.argmax(1) == Yte).float().mean() * 100),
                       test_acc=float(((p1 + p2).argmax(1) == Yte).float().mean() * 100))
        dt = time.time() - t0
        row['seconds'] = dt
        history.append(row)
        print(f'{job["id"]} ep{epoch} lr {row["lr"]:.2e} fr {fr:.3f} loss {row["loss1"]:.3f}/{row["loss2"]:.3f} '
              f'label-prec {row["label_prec1"]:.3f}/{row["label_prec2"]:.3f} flagged {row["flagged"]} '
              f'(prec {row["flag_prec"]:.3f}) test {row["test_acc"]:.2f} [{dt:.0f}s/ep, '
              f'{(time.time() - t_session) / 60:.0f} min]', flush=True)
        ran = epoch + 1 - start
        stop = not last and (time.time() + 1.5 * dt + 600 > deadline_end or
                             (ct['max_epochs_this_run'] and ran >= ct['max_epochs_this_run']))
        if last or stop or (epoch + 1) % ct['ckpt_every'] == 0:
            save(epoch)
        if stop:
            add_minutes(out_dir, (time.time() - t_session) / 60)
            print(f'{job["id"]}: paused after epoch {epoch}; rerun to resume', flush=True)
            return 'paused', None

    if sel1 is None:
        raise RuntimeError(f'{job["id"]}: checkpoint already at the last epoch but final.npz missing; '
                           'delete ckpt.pt to retrain')
    s1 = predict(net1, X, mean, std, amp, ct['eval_batch'])
    s2 = predict(net2, X, mean, std, amp, ct['eval_batch'])
    ce1 = F.nll_loss(torch.log(s1.clamp_min(1e-12)), Y, reduction='none')
    ce2 = F.nll_loss(torch.log(s2.clamp_min(1e-12)), Y, reduction='none')
    cols = list(history[0].keys())
    np.savez_compressed(os.path.join(out_dir, 'final.npz'), index=data['index'],
                        sel1=sel1.cpu().numpy(), sel2=sel2.cpu().numpy(),
                        loss1=loss1.cpu().numpy(), loss2=loss2.cpu().numpy(),
                        eval_loss1=ce1.cpu().numpy(), eval_loss2=ce2.cpu().numpy(),
                        soft1=s1.cpu().numpy().astype(np.float16), soft2=s2.cpu().numpy().astype(np.float16),
                        forget_rate=tau, history=np.array([[r[c] for c in cols] for r in history], float),
                        history_cols=np.array(cols))
    minutes = add_minutes(out_dir, (time.time() - t_session) / 60)
    flag = (~sel1 & ~sel2).cpu().numpy()
    m = detection_metrics(flag, ((ce1 + ce2) / 2).cpu().numpy(), np.full(n, -1), data['table'])
    return 'done', dict(job=job['id'], method='coteaching', dataset=job['dataset'], noise=job['noise'],
                        n=n, epochs=epochs, minutes=round(minutes, 1), forget_rate=tau,
                        test_acc=history[-1]['test_acc'], flagged=m['flagged'], precision=m['precision'],
                        recall=m['recall'], f1=m['f1'], auc=m['auc'], residual_noise=m['residual_noise'])


# --------------------------------------------------------------------------- DivideMix
def _edit(text, edits, name):
    for old, new, count in edits:
        if isinstance(old, re.Pattern):
            text, k = old.subn(new, text)
        else:
            k = text.count(old)
            text = text.replace(old, new)
        if k == 0 or (count is not None and k != count):
            raise RuntimeError(f'{name}: patch anchor matched {k}x, expected {count}: {str(old)[:70]!r}')
    return text


TRAIN_ARGS = """parser.add_argument('--noise_file', default='', type=str)  # r52: our noisy labels (JSON list)
parser.add_argument('--out_dir', default='./checkpoint', type=str)  # r52
parser.add_argument('--num_workers', default=5, type=int)  # r52 (was hard-coded 5)
parser.add_argument('--lr_drop', default=150, type=int)  # r52 (was hard-coded 150)
parser.add_argument('--ckpt_every', default=5, type=int)  # r52: checkpoint period in epochs
parser.add_argument('--deadline', default=0.0, type=float)  # r52: unix time; checkpoint + exit(3) before it
parser.add_argument('--max_epochs_this_run', default=0, type=int)  # r52: smoke tests of resume
parser.add_argument('--amp', default=1, type=int)  # r52: fp16 autocast on CUDA
parser.add_argument('--warm_up', default=0, type=int)  # r52: 0 = official per-dataset value
args = parser.parse_args()"""

TRAIN_HELPERS = """import time  # r52 ---------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if os.environ.get('R52_MPS') and torch.backends.mps.is_available() else 'cpu')
AMP = bool(args.amp) and DEVICE.type == 'cuda'
SCALERS = {}
R52_TEST_ACC = []
if DEVICE.type == 'cuda':
    torch.cuda.set_device(args.gpuid)
if args.noise_file and not os.path.exists(args.noise_file):
    raise FileNotFoundError('r52: noise file %s missing' % args.noise_file)
os.makedirs(args.out_dir, exist_ok=True)


def fwd(net, x):
    # r52: forward under fp16 autocast on CUDA, logits back in fp32
    with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
        out = net(x)
    return out.float()


def step(loss, optimizer):
    # r52: AMP-aware backward + SGD step, one GradScaler per optimiser
    if id(optimizer) not in SCALERS:
        SCALERS[id(optimizer)] = torch.amp.GradScaler('cuda', enabled=AMP)
    sc = SCALERS[id(optimizer)]
    sc.scale(loss).backward()
    sc.step(optimizer)
    sc.update()
# r52 end ------------------------------------------------------------------------"""

TRAIN_RESUME = """# r52: resume, per-epoch co-divide dump, checkpoints, final export -----------------
R52_CKPT = os.path.join(args.out_dir, 'ckpt.pt')
R52_HIST = os.path.join(args.out_dir, 'gmm_history.npz')
r52_hist = {}
start_epoch = 0
for _opt in (optimizer1, optimizer2):
    SCALERS[id(_opt)] = torch.amp.GradScaler('cuda', enabled=AMP)
if os.path.exists(R52_CKPT):
    _ck = torch.load(R52_CKPT, map_location='cpu', weights_only=False)
    net1.load_state_dict(_ck['net1'])
    net2.load_state_dict(_ck['net2'])
    optimizer1.load_state_dict(_ck['opt1'])
    optimizer2.load_state_dict(_ck['opt2'])
    if _ck['scaler1']:
        SCALERS[id(optimizer1)].load_state_dict(_ck['scaler1'])
    if _ck['scaler2']:
        SCALERS[id(optimizer2)].load_state_dict(_ck['scaler2'])
    all_loss = _ck['all_loss']
    R52_TEST_ACC.extend(_ck['test_acc'])
    random.setstate(_ck['rng_py'])
    np.random.set_state(_ck['rng_np'])
    torch.set_rng_state(_ck['rng_torch'])
    if DEVICE.type == 'cuda' and _ck['rng_cuda'] is not None:
        torch.cuda.set_rng_state(_ck['rng_cuda'])
    if os.path.exists(R52_HIST):
        with np.load(R52_HIST) as _h:
            r52_hist = {int(e): p for e, p in zip(_h['epochs'], _h['probs'])}
    start_epoch = _ck['epoch'] + 1
    print('| r52: resumed after epoch %d' % _ck['epoch'], flush=True)
    del _ck


def r52_save(epoch):
    state = dict(epoch=epoch, net1=net1.state_dict(), net2=net2.state_dict(),
                 opt1=optimizer1.state_dict(), opt2=optimizer2.state_dict(),
                 scaler1=SCALERS[id(optimizer1)].state_dict(), scaler2=SCALERS[id(optimizer2)].state_dict(),
                 all_loss=[l[-5:] for l in all_loss], test_acc=list(R52_TEST_ACC),
                 rng_py=random.getstate(), rng_np=np.random.get_state(), rng_torch=torch.get_rng_state(),
                 rng_cuda=torch.cuda.get_rng_state() if DEVICE.type == 'cuda' else None)
    torch.save(state, R52_CKPT + '.tmp')
    os.replace(R52_CKPT + '.tmp', R52_CKPT)
    if r52_hist:
        ep = sorted(r52_hist)
        np.savez(R52_HIST[:-4] + '.tmp.npz', epochs=np.array(ep), probs=np.stack([r52_hist[e] for e in ep]))
        os.replace(R52_HIST[:-4] + '.tmp.npz', R52_HIST)


def r52_epoch_end(epoch, t0):
    dt = time.time() - t0
    last = epoch == args.num_epochs
    stop = not last and ((args.deadline > 0 and time.time() + 1.5 * dt + 600 > args.deadline) or
                         (args.max_epochs_this_run > 0 and epoch + 1 - start_epoch >= args.max_epochs_this_run))
    save = last or stop or (epoch + 1) % args.ckpt_every == 0
    if save:
        r52_save(epoch)
    print('\\n| r52: epoch %d took %.1f s%s' % (epoch, dt, ' (checkpoint saved)' if save else ''), flush=True)
    if stop:
        print('| r52: stopping before the session deadline; rerun to resume from %s' % R52_CKPT, flush=True)
        sys.exit(3)


def r52_finish():
    # final division (same GMM as eval_train) + both nets' softmax on un-augmented training images
    global eval_loader
    eval_loader = loader.run('eval_train')
    p1, _ = eval_train(net1, list(all_loss[0]))
    p2, _ = eval_train(net2, list(all_loss[1]))
    soft = []
    for net in (net1, net2):
        net.eval()
        out = []
        with torch.no_grad():
            for inputs, _, index in eval_loader:
                out.append(torch.softmax(fwd(net, inputs.to(DEVICE)), 1).cpu())
        soft.append(torch.cat(out).numpy().astype(np.float16))
    extra = {'last_division': r52_hist[args.num_epochs]} if args.num_epochs in r52_hist else {}
    np.savez_compressed(os.path.join(args.out_dir, 'final.npz'), prob1=p1, prob2=p2, soft1=soft[0],
                        soft2=soft[1], test_acc=np.array(R52_TEST_ACC), **extra)
    print('| r52: wrote final.npz', flush=True)
# r52 end ------------------------------------------------------------------------

for epoch in range(start_epoch, args.num_epochs+1):
    t_epoch = time.time()  # r52"""

LOADER_AUC = """class AUCMeter(object):  # r52: torchnet is unmaintained; same AUC via scikit-learn
    def reset(self):
        self.scores, self.targets = None, None

    def add(self, scores, targets):
        self.scores, self.targets = np.asarray(scores), np.asarray(targets)

    def value(self):
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(self.targets, self.scores), None, None
        except ValueError:
            return float('nan'), None, None"""

LOADER_FMNIST = """        elif self.dataset=='fashionmnist':  # r52 adaptation: 28x28 grayscale replicated to 3 channels
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
                ])
    def run(self,mode,pred=[],prob=[]):"""


def patch_dividemix(train_src, loader_src):
    """Return patched (Train_cifar_r52.py, dataloader_cifar_r52.py) sources. Trailing whitespace
    is stripped first so anchors are exact; every anchor must match the expected number of times."""
    train = '\n'.join(line.rstrip() for line in train_src.split('\n'))
    loader = '\n'.join(line.rstrip() for line in loader_src.split('\n'))
    train = _edit(train, [
        # modern PyTorch / any device
        (re.compile(r'\.cuda\(\)'), '.to(DEVICE)', None),
        ('unlabeled_train_iter.next()', 'next(unlabeled_train_iter)', 2),
        (re.compile(r'= (net|net1|net2|model)\((inputs\w*|mixed_input)\)'), r'= fwd(\1, \2)', 11),
        ('        loss.backward()\n        optimizer.step()', '        step(loss, optimizer)  # r52', 1),
        ('        L.backward()\n        optimizer.step()', '        step(L, optimizer)  # r52', 1),
        ('args = parser.parse_args()', TRAIN_ARGS, 1),
        ('torch.cuda.set_device(args.gpuid)', TRAIN_HELPERS, 1),
        ('import dataloader_cifar as dataloader', 'import dataloader_cifar_r52 as dataloader  # r52', 1),
        # any training-set size; vectorised loss gather (same values as the per-sample loop)
        ('    losses = torch.zeros(50000)', '    losses = torch.zeros(len(eval_loader.dataset))  # r52', 1),
        ('            for b in range(inputs.size(0)):\n                losses[index[b]]=loss[b]',
         '            losses[index] = loss.detach().float().cpu()  # r52', 1),
        ('    acc = 100.*correct/total', '    acc = 100.*correct/total\n    R52_TEST_ACC.append(acc)  # r52', 1),
        # logs into the job directory, appended across resumed sessions
        ("open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w')",
         "open(os.path.join(args.out_dir,'%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt'),'a')", 1),
        ("open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')",
         "open(os.path.join(args.out_dir,'%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt'),'a')", 1),
        ("if args.dataset=='cifar10':\n    warm_up = 10",
         "if args.dataset in ('cifar10', 'fashionmnist'):  # r52: FMNIST reuses the CIFAR-10 warm-up\n"
         "    warm_up = 10", 1),
        ("elif args.dataset=='cifar100':\n    warm_up = 30",
         "elif args.dataset=='cifar100':\n    warm_up = 30\nif args.warm_up:  # r52 (smoke tests only)\n"
         "    warm_up = args.warm_up", 1),
        ('num_workers=5,\\', 'num_workers=args.num_workers,\\', 1),
        ("noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))",
         "noise_file=args.noise_file or '%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))", 1),
        ('    if epoch >= 150:', '    if epoch >= args.lr_drop:  # r52', 1),
        # resume + per-epoch GMM dump + checkpoint/deadline + final export
        ('for epoch in range(args.num_epochs+1):', TRAIN_RESUME, 1),
        ('        prob2,all_loss[1]=eval_train(net2,all_loss[1])',
         '        prob2,all_loss[1]=eval_train(net2,all_loss[1])\n'
         '        r52_hist[epoch] = np.stack([prob1, prob2]).astype(np.float16)  # r52', 1),
        ('\n    test(epoch,net1,net2)', '\n    test(epoch,net1,net2)\n    r52_epoch_end(epoch, t_epoch)  # r52', 1),
    ], 'Train_cifar.py').rstrip() + '\n\nr52_finish()  # r52\n'
    loader = _edit(loader, [
        ('from torchnet.meter import AUCMeter', LOADER_AUC, 1),
        ("        if self.mode=='test':\n            if dataset=='cifar10':",
         "        if self.mode=='test':\n"
         "            if os.path.exists('%s/r52_test.npz'%root_dir):  # r52: uint8 NHWC arrays\n"
         "                _d = np.load('%s/r52_test.npz'%root_dir)\n"
         "                self.test_data, self.test_label = _d['x'], _d['y'].tolist()\n"
         "            elif dataset=='cifar10':", 1),
        ("            train_data=[]\n            train_label=[]\n            if dataset=='cifar10':",
         "            train_data=[]\n            train_label=[]\n"
         "            if os.path.exists('%s/r52_train.npz'%root_dir):  # r52: uint8 NHWC arrays, true labels\n"
         "                _d = np.load('%s/r52_train.npz'%root_dir)\n"
         "                train_data, train_label = _d['x'], _d['y'].tolist()\n"
         "            elif dataset=='cifar10':", 1),
        ('            train_data = train_data.reshape((50000, 3, 32, 32))\n'
         '            train_data = train_data.transpose((0, 2, 3, 1))',
         '            if train_data.ndim == 2:  # r52: only the CIFAR pickles need reshaping\n'
         '                train_data = train_data.reshape((50000, 3, 32, 32))\n'
         '                train_data = train_data.transpose((0, 2, 3, 1))', 1),
        ('            else:    #inject noise',
         "            else:    #inject noise\n"
         "                raise FileNotFoundError('r52: noise file %s missing; refusing to draw new noise' % noise_file)",
         1),
        ('    def run(self,mode,pred=[],prob=[]):', LOADER_FMNIST, 1),
    ], 'dataloader_cifar.py')
    return train, loader


def _write_atomic(path, text):
    if os.path.exists(path) and open(path).read() == text:
        return
    with open(path + '.tmp', 'w') as f:
        f.write(text)
    os.replace(path + '.tmp', path)


def prepare_dividemix(commit=DM_COMMIT):
    """Clone the official repo at `commit` (once) and write the patched copies + a diff."""
    repo = os.path.join(K.SCRATCH, 'DivideMix')      # re-creatable: kept out of the saved output
    if not os.path.isdir(os.path.join(repo, '.git')):
        tmp = f'{repo}.tmp{os.getpid()}'
        subprocess.run(['git', 'clone', '--quiet', DM_URL, tmp], check=True)
        subprocess.run(['git', '-C', tmp, 'checkout', '--quiet', commit], check=True)
        try:
            os.rename(tmp, repo)
        except OSError:                        # another worker won the race
            shutil.rmtree(tmp, ignore_errors=True)
    head = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != commit:
        raise RuntimeError(f'{repo} is at {head}, expected {commit}')
    srcs = [open(os.path.join(repo, f)).read() for f in ('Train_cifar.py', 'dataloader_cifar.py')]
    patched = patch_dividemix(*srcs)
    diff = []
    for name, old, new in zip(('Train_cifar.py', 'dataloader_cifar.py'), srcs, patched):
        new_name = name.replace('.py', '_r52.py')
        _write_atomic(os.path.join(repo, new_name), new)
        diff += difflib.unified_diff([l.rstrip() + '\n' for l in old.split('\n')], new.splitlines(True),
                                     name, new_name)
    _write_atomic(os.path.join(repo, 'r52_patch.diff'), ''.join(diff))
    return repo


def dividemix_arrays(cfg, dataset, data):
    """Images for the patched loader: <dir>/r52_{train,test}.npz with x (uint8 NHWC), y (TRUE labels)."""
    tag = dataset + (f'-sub{cfg["subsample"]}' if cfg.get('subsample') else '') + \
        (f'-test{cfg["test_subsample"]}' if cfg.get('test_subsample') else '')
    out = os.path.join(K.DATA_ROOT, 'r52', tag)
    os.makedirs(out, exist_ok=True)
    for split, x, y in (('train', data['x_train'], data['true']), ('test', data['x_test'], data['y_test'])):
        path = os.path.join(out, f'r52_{split}.npz')
        if not os.path.exists(path):
            tmp = path[:-4] + f'.tmp{os.getpid()}.npz'
            np.savez(tmp, x=x, y=y)
            os.replace(tmp, path)
    return out


def stream(cmd, cwd, every):
    """Run cmd, echo its output; DivideMix's per-iteration '\\r' progress is thinned to one line/`every` s."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf, last = b'', 0.0
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            *parts, buf = re.split(rb'[\r\n]', buf + chunk)
            for part in parts:
                line = part.decode(errors='replace').rstrip()
                if not line:
                    continue
                if 'Iter[' in line:
                    if time.time() - last < every:
                        continue
                    last = time.time()
                print(line, flush=True)
    except BaseException:
        proc.kill()
        raise
    if buf.strip():
        print(buf.decode(errors='replace'), flush=True)
    return proc.wait()


def dividemix_job(cfg, job, deadline_end):
    dm = dict(DIVIDEMIX, **cfg.get('dividemix', {}))
    data = job_data(cfg, job['dataset'], job['noise'])
    repo = prepare_dividemix(dm['commit'])
    data_dir = dividemix_arrays(cfg, job['dataset'], data)
    out_dir = job_dir(cfg, job)
    np.save(os.path.join(out_dir, 'index.npy'), data['index'])
    shutil.copy(os.path.join(repo, 'r52_patch.diff'), out_dir)
    noise_file = os.path.join(out_dir, 'noise_labels.json')
    labels = [int(v) for v in data['noisy']]
    if os.path.exists(noise_file):
        if json.load(open(noise_file)) != labels:
            raise RuntimeError(f'{noise_file} does not match the label table')
    else:
        _write_atomic(noise_file, json.dumps(labels))
    cmd = [sys.executable, '-u', 'Train_cifar_r52.py', '--dataset', job['dataset'], '--data_path', data_dir,
           '--noise_file', noise_file, '--out_dir', out_dir, '--r', str(job['noise'] / 100),
           '--noise_mode', dm['noise_mode'], '--lambda_u', str(dm['lambda_u']),
           '--p_threshold', str(dm['p_threshold']), '--T', str(dm['T']), '--alpha', str(dm['alpha']),
           '--num_epochs', str(dm['epochs']), '--lr_drop', str(dm['lr_drop']), '--batch_size', str(dm['batch_size']),
           '--lr', str(dm['lr']), '--num_class', '10', '--seed', str(cfg['seed']),
           '--num_workers', str(dm['num_workers']), '--ckpt_every', str(dm['ckpt_every']),
           '--deadline', str(deadline_end), '--amp', str(int(bool(cfg.get('amp', True)))),
           '--max_epochs_this_run', str(dm['max_epochs_this_run']), '--warm_up', str(dm['warm_up'])]
    print(f'{job["id"]}: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    rc = stream(cmd, repo, dm['progress_every'])
    minutes = add_minutes(out_dir, (time.time() - t0) / 60)
    if rc == 3:
        print(f'{job["id"]}: paused (checkpoint in {out_dir}); rerun to resume', flush=True)
        return 'paused', None
    if rc != 0:
        raise RuntimeError(f'{job["id"]}: DivideMix exited with {rc}')
    with np.load(os.path.join(out_dir, 'final.npz')) as z:
        clean = (z['prob1'] + z['prob2']) / 2
        pred = (z['soft1'].astype(np.float32) + z['soft2'].astype(np.float32)).argmax(1)
        test_acc = float(z['test_acc'][-1]) if len(z['test_acc']) else np.nan
    flag = clean < dm['p_threshold']
    m = detection_metrics(flag, 1 - clean, pred, data['table'])
    return 'done', dict(job=job['id'], method='dividemix', dataset=job['dataset'], noise=job['noise'],
                        n=len(labels), epochs=dm['epochs'] + 1, minutes=round(minutes, 1), test_acc=test_acc,
                        flagged=m['flagged'], precision=m['precision'], recall=m['recall'], f1=m['f1'],
                        auc=m['auc'], residual_noise=m['residual_noise'], relabel_acc=m['relabel_acc'])


RUNNERS = {'coteaching': coteaching_job, 'dividemix': dividemix_job}


# --------------------------------------------------------------------------- queue
def jobs_for(cfg, worker):
    jobs = [dict(id=f'{m}_{d}_{n}', method=m, dataset=d, noise=int(n)) for m, d, n in cfg['jobs']]
    prefs = cfg.get('worker_prefs', {'0': ['dividemix', 'coteaching'],
                                     '1': ['coteaching', 'dividemix']}).get(str(worker))
    if prefs:
        jobs.sort(key=lambda j: prefs.index(j['method']) if j['method'] in prefs else len(prefs))
    return jobs


def prepare(cfg):
    """One-off setup before launching workers (avoids download/clone races between them)."""
    for method, dataset, noise in cfg['jobs']:
        data = job_data(cfg, dataset, int(noise))
        if method == 'dividemix':
            dividemix_arrays(cfg, dataset, data)
    if any(m == 'dividemix' for m, _, _ in cfg['jobs']):
        repo = prepare_dividemix(dict(DIVIDEMIX, **cfg.get('dividemix', {}))['commit'])
        print(f'DivideMix patched in {repo} (see r52_patch.diff)')
    print('prepared', flush=True)


def main(argv):
    with open(argv[0]) as f:
        cfg = json.load(f)
    K.setup_repo(cfg['code_ref'])
    if '--prepare' in argv:
        prepare(cfg)
        return
    worker = int(os.environ.get('SND_WORKER', 0))
    queue = K.JobQueue(cfg['out_dir'])
    deadline = K.Deadline(cfg['session_hours'] - (time.time() - cfg['session_start']) / 3600)
    for job in jobs_for(cfg, worker):
        if queue.done(job['id']):
            continue
        if not deadline.allows(cfg.get('min_job_hours', 0.5)):
            print('session deadline reached; remaining jobs left for the next version', flush=True)
            break
        if not queue.claim(job['id']):
            continue
        print(f'worker {worker}: start {job["id"]}', flush=True)
        try:
            status, summary = RUNNERS[job['method']](cfg, job, deadline.end)
        except BaseException:
            queue.release(job['id'])
            raise
        if status != 'done':
            queue.release(job['id'])
            break
        queue.finish(job['id'], summary)
        print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
