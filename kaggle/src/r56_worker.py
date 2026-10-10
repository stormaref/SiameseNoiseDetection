"""R5.6 worker: train downstream classifiers on noisy / cleaned / oracle training sets.

Training sets (all built from the same published noisy labels):
  noisy            every sample, observed labels;
  ours             DetectAndRelabel output at (td, tr): flagged samples relabelled or removed;
  oracle_filtered  every truly noisy sample removed, the rest keep their (correct) labels;
  oracle_clean     every sample, ground-truth labels;
  given            a cleaned label vector shipped in the config (`given_labels`, see
                   encode_labels), e.g. from an ensemble whose predictions are not in the repo.
`oracle_filtered` separates "fewer samples" from "better labels"; `oracle_clean` is the ceiling.

Protocol = snd.evaluation.final_model_tester.FinalModelTester as called in main.ipynb:
stratified train/val split of the training set, Adam, linear warm-up for `warmup` epochs and
then a constant rate, label smoothing, early stopping on validation accuracy, best weights
reloaded, accuracy on the clean test set. Usage: python r56_worker.py <config.json>
"""
import base64
import copy
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snd_kaggle as K  # noqa: E402

TRAIN_SETS = ('noisy', 'ours', 'oracle_filtered', 'oracle_clean', 'given')


def encode_labels(labels):
    """int labels (-1 = removed) -> compact text for the `given_labels` config entry."""
    return base64.b64encode(zlib.compress(np.asarray(labels, dtype=np.int8).tobytes(), 9)).decode()


def decode_labels(text, n):
    labels = np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype=np.int8).astype(int)
    if len(labels) != n:
        raise ValueError(f'given_labels has {len(labels)} entries, the training set {n}')
    return labels


def build_arch(name, dataset=None):
    """32x32-style variants of four architecture families (also valid at 28x28)."""
    from torchvision import models
    if name == 'preact_resnet34':                    # the paper's downstream model
        from snd.models.preact import PreActResNet34
        m = PreActResNet34()
        if dataset == 'animal10n':                   # 64x64 inputs: 2x2x512 features (cnn_size=2048)
            m.linear = nn.Linear(2048, 10)
        return m
    if name == 'resnet18':
        m = models.resnet18(num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    if name == 'vgg16_bn':
        m = models.vgg16_bn(num_classes=10)
        m.features[-1] = nn.Identity()               # keep >= 1x1 maps for 28x28 inputs
        m.avgpool = nn.AdaptiveAvgPool2d(1)
        m.classifier = nn.Linear(512, 10)
        return m
    if name == 'mobilenet_v2':
        m = models.mobilenet_v2(num_classes=10)
        m.features[0][0].stride = (1, 1)
        return m
    if name == 'densenet121':
        m = models.densenet121(num_classes=10)
        m.features.conv0 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.features.pool0 = nn.Identity()
        return m
    raise ValueError(name)


def training_labels(table, kind, td, tr):
    """Label per sample for one training set; -1 = not in the set."""
    if kind == 'noisy':
        return table['noisy_label'].to_numpy().copy()
    if kind == 'ours':
        return K.cleaned_labels(table, td, tr)
    if kind == 'oracle_filtered':
        return np.where(table['is_noisy'].to_numpy(), -1, table['noisy_label'].to_numpy())
    if kind == 'oracle_clean':
        return table['real_label'].to_numpy().copy()
    if kind == 'given':
        return table['given_label'].to_numpy().copy()
    raise ValueError(kind)


class Relabelled(Dataset):
    def __init__(self, base, indices, labels, transform):
        self.base, self.indices, self.labels, self.transform = base, indices, labels, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]
        return self.transform(img), int(self.labels[i])


def evaluate(model, loader, device, amp):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def run_job(cfg, job, base_train, base_test, table):
    from sklearn.model_selection import train_test_split
    p = dict(K.DOWNSTREAM[cfg['dataset']], **cfg.get('overrides', {}))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = cfg['amp'] and device.type == 'cuda'
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])

    labels = training_labels(table, job['train_set'], cfg['td'], cfg['tr'])
    keep = np.flatnonzero(labels >= 0)
    true = table['real_label'].to_numpy()
    tr_pos, va_pos = train_test_split(np.arange(len(keep)), test_size=p['val_ratio'],
                                      stratify=labels[keep], random_state=job['seed'])
    aug, plain = K.augmentation({'cifar10': 'down_cifar', 'cifar10n': 'down_cifar',
                                 'animal10n': 'down_animal'}.get(cfg['dataset'], 'down_fmnist'))
    workers = cfg.get('loader_workers', 2)
    mk = lambda pos, t, shuffle, bs: DataLoader(  # noqa: E731
        Relabelled(base_train, keep[pos], labels[keep][pos], t), batch_size=bs, shuffle=shuffle,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = mk(tr_pos, aug, True, p['batch_size'])
    val_loader = mk(va_pos, plain, False, 512)
    test_loader = DataLoader(Relabelled(base_test, np.arange(len(base_test)),
                                        np.asarray(base_test.targets), plain),
                             batch_size=512, num_workers=workers)

    model = build_arch(job['arch'], cfg['dataset']).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=p['lr'], betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=p['wd'])
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=p['warmup'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ce = nn.CrossEntropyLoss(label_smoothing=p['smoothing'])

    best, best_state, stale, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(p['max_epochs']):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                loss = ce(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
        val_acc = evaluate(model, val_loader, device, amp)
        history.append([epoch, loss_sum / len(train_loader), val_acc])
        if val_acc > best:
            best, stale, best_state = val_acc, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f'{job["id"]} ep{epoch} loss {loss_sum / len(train_loader):.3f} val {val_acc:.4f} '
              f'best {best:.4f} stale {stale} [{(time.time() - t0) / 60:.0f} min]', flush=True)
        if stale >= p['patience']:
            break
        if epoch < p['warmup']:
            warmup.step()
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device, amp)
    kept_labels = labels[keep]
    return dict(job=job['id'], dataset=cfg['dataset'], noise=cfg['noise'], preds_ref=cfg['preds_ref'],
                td=cfg['td'], tr=cfg['tr'], train_set=job['train_set'], arch=job['arch'], seed=job['seed'],
                n_train_total=int(len(keep)), n_removed=int(len(labels) - len(keep)),
                label_noise_in_set=float((kept_labels != true[keep]).mean()),
                best_val_acc=best, test_acc=test_acc, epochs=len(history),
                minutes=round((time.time() - t0) / 60, 1), history=history)


def jobs_for(cfg):
    return [dict(id=f'{cfg["dataset"]}{cfg["noise"]}_{s}_{a}_s{seed}', train_set=s, arch=a, seed=seed)
            for seed in cfg['seeds'] for a in cfg['archs'] for s in cfg['train_sets']]


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    repo = K.setup_repo(cfg['code_ref'])
    proto = K.PROTOCOL[(cfg['dataset'], cfg['noise'])]
    # Animal-10N and F-MNIST 60% have no published-preds entry in PROTOCOL (their runs draw new
    # folds/noise); their re-run predictions are committed under preds/<preds_name>
    table = K.load_label_table(K.extract_preds(repo, cfg['preds_ref'], cfg.get('preds_name') or proto['preds']))
    base_train = K.noisy_train_set(cfg['dataset'], table)     # images only; labels come from the table
    base_test = K.base_dataset(cfg['dataset'], train=False)
    if cfg.get('given_labels'):
        table['given_label'] = decode_labels(cfg['given_labels'], len(table))
    table, keep = K.smoke_subsample(table, cfg)
    if keep is not None:
        base_train = torch.utils.data.Subset(base_train, keep)
        base_test = torch.utils.data.Subset(base_test, np.arange(cfg['subsample'] // 5))
        base_test.targets = np.asarray(base_test.dataset.targets)[base_test.indices]

    queue = K.JobQueue(cfg['out_dir'])
    deadline = K.Deadline(cfg['session_hours'] - (time.time() - cfg['session_start']) / 3600)
    for job in jobs_for(cfg):
        if queue.done(job['id']):
            continue
        if not deadline.allows(cfg['job_hours']):
            print('session deadline reached; remaining jobs left for the next version', flush=True)
            break
        if not queue.claim(job['id']):
            continue
        try:
            summary = run_job(cfg, job, base_train, base_test, table)
        except BaseException:
            queue.release(job['id'])
            raise
        queue.finish(job['id'], summary)
        print('DONE', json.dumps({k: v for k, v in summary.items() if k != 'history'}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
