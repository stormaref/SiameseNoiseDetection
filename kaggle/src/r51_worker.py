"""R5.1 worker: train one inner-ensemble member per job and save what the analysis needs.

Variants (identical data, inner split, pair sampling, seed, schedule and early stopping):
  siamese    -- our objective: CE on both branches + the contrastive loss of
                snd.training.contrastive.ContrastiveLoss (main, corrected form
                same*d^2 + (1-same)*[m-d]_+^2), as in the CIFAR-10 re-run;
  ce         -- the same network and pairs with the contrastive term switched off (ratio 0),
                i.e. an ordinary cross-entropy classifier trained on the same samples;
  ce_linear  -- (optional) torchvision backbone with a plain linear head, CE only;
  siamese_noreg -- the regularization ablation (main.ipynb cell 54): `siamese` with dropout 0
                and a fixed 40 epochs (patience 50, so never stops early; the best
                validation-accuracy weights are still restored, as Trainer did).

Per job it writes <job>.npz with, for the held-out outer fold: softmax outputs and
embeddings; for the member's own training subset: embeddings (reference set for
centroid and k-NN scores). Usage: python r51_worker.py <config.json>
"""
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snd_kaggle as K  # noqa: E402


def build_model(variant, proto, dropout):
    from snd.models.siamese import SiameseNetwork
    if variant in ('siamese', 'ce', 'siamese_noreg'):
        dropout = 0.0 if variant == 'siamese_noreg' else dropout
        return SiameseNetwork(num_classes=10, model=proto['backbone'], embedding_dimension=proto['emb'],
                              pre_trained=proto['pre_trained'], dropout_prob=dropout,
                              trainable=True, parallel=False)
    if variant == 'ce_linear':
        return LinearHead(proto)
    raise ValueError(variant)


class LinearHead(nn.Module):
    """Backbone + linear classifier; `classify` mirrors SiameseNetwork's (emb, logits) API."""

    def __init__(self, proto):
        super().__init__()
        from snd.models.siamese import BACKBONES
        build, width = BACKBONES[proto['backbone']]
        base = build(10, proto['pre_trained'])
        base.fc = nn.Flatten()
        self.feature_extractor = base
        self.fc = nn.Linear(width, 10)

    def classify(self, x):
        feat = self.feature_extractor(x)
        return feat, self.fc(feat)

    def forward(self, x1, x2):
        e1, c1 = self.classify(x1)
        e2, c2 = self.classify(x2)
        return e1, e2, c1, c2


def contrastive(emb1, emb2, same, margin):
    """snd.training.contrastive.ContrastiveLoss (Euclidean), evaluated in float32 under AMP."""
    d = nn.functional.pairwise_distance(emb1.float(), emb2.float())
    return torch.mean(same * d.pow(2) + (1 - same) * torch.clamp(margin - d, min=0).pow(2))


def run_epoch(model, loader, device, ratio, margin, ce, opt=None, scaler=None, amp=False):
    """One pass over pair batches; mirrors snd.training.trainer.Trainer.calc_loss."""
    train = opt is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for img1, img2, y1, y2, _, _ in loader:
            img1, img2 = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True)
            y1, y2 = y1.to(device), y2.to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                e1, e2, c1, c2 = model(img1, img2)
                loss = ce(c1.float(), y1) + ce(c2.float(), y2)
            if ratio:
                loss = loss + ratio * contrastive(e1, e2, (y1 == y2).float(), margin)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            loss_sum += loss.item()
            correct += (c1.argmax(1) == y1).sum().item() + (c2.argmax(1) == y2).sum().item()
            total += 2 * len(y1)
    return loss_sum / len(loader), 100.0 * correct / total


@torch.no_grad()
def embed(model, ds, indices, transform, device, amp, batch=1024, workers=2):
    from snd.data.dataset import DatasetSingle
    loader = DataLoader(DatasetSingle(Subset(ds, indices), transform), batch_size=batch,
                        shuffle=False, num_workers=workers)
    model.eval()
    embs, probs = [], []
    for img, _, _ in loader:
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            e, logits = model.classify(img.to(device))
        embs.append(e.float().cpu())
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(embs).numpy(), torch.cat(probs).numpy()


def cifar_affine_gpu(x):
    """'cifar_affine' (training) on a float batch [B, 3, 32, 32] in [0, 1], drawn per sample:
    RandomRotation(15) -> RandomHorizontalFlip -> RandomAffine(translate=0.1) (nearest, zero
    fill, as torchvision) -> RandomResizedCrop(32, scale=(0.9, 1)) (bilinear)."""
    n, dev = x.shape[0], x.device
    ang = torch.deg2rad(torch.empty(n, device=dev).uniform_(-15, 15))
    cos, sin = torch.cos(ang), torch.sin(ang)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    shift = torch.round(torch.empty(n, 2, device=dev).uniform_(-3.2, 3.2)) * (2 / 32)
    # rotation + flip, then the integer shift as its own resample: shifting in from outside the
    # frame must give fill even where the rotation alone would land inside (torchvision order)
    a = torch.stack([torch.stack([cos * flip, sin], 1), torch.stack([-sin * flip, cos], 1)], 1)
    for theta in (torch.cat([a, torch.zeros(n, 2, 1, device=dev)], 2),
                  torch.cat([torch.eye(2, device=dev).expand(n, 2, 2), -shift[:, :, None]], 2)):
        grid = nn.functional.affine_grid(theta, x.shape, align_corners=False)
        x = nn.functional.grid_sample(x, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    # RandomResizedCrop: up to 10 tries of (area, log-ratio), else the whole image
    area = torch.empty(n, 10, device=dev).uniform_(0.9, 1.0) * 1024
    ratio = torch.exp(torch.empty(n, 10, device=dev).uniform_(np.log(3 / 4), np.log(4 / 3)))
    w, h = torch.round(torch.sqrt(area * ratio)), torch.round(torch.sqrt(area / ratio))
    ok = (w <= 32) & (h <= 32)
    first = ok.float().argmax(1)
    w = torch.where(ok.any(1), w.gather(1, first[:, None])[:, 0], torch.full_like(w[:, 0], 32))
    h = torch.where(ok.any(1), h.gather(1, first[:, None])[:, 0], torch.full_like(h[:, 0], 32))
    top = torch.floor(torch.rand(n, device=dev) * (32 - h + 1))
    left = torch.floor(torch.rand(n, device=dev) * (32 - w + 1))
    zero = torch.zeros(n, device=dev)
    theta = torch.stack([torch.stack([w / 32, zero, (left + w / 2) / 16 - 1], 1),
                         torch.stack([zero, h / 32, (top + h / 2) / 16 - 1], 1)], 1)
    grid = nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)


def gpu_transform(x, aug, train):
    """snd_kaggle.augmentation on a uint8 batch: [B, 28, 28] for F-MNIST, [B, 32, 32, 3] for CIFAR.

    CIFAR ('cifar_affine'): ToTensor + Normalize(CIFAR_MEAN, CIFAR_STD), with cifar_affine_gpu
    in training. F-MNIST:

    Grayscale(3) + ToTensor (+ Normalize(0.5, 0.5) for 'fmnist_norm'); for 'fmnist_aug' in
    training, RandomCrop(28, padding=2) (zero fill, offsets uniform in 0..4) and
    RandomHorizontalFlip(0.5) per sample, as torchvision draws them.
    """
    x = x.float().div_(255)
    if aug == 'cifar_affine':
        x = x.permute(0, 3, 1, 2)
        if train:
            x = cifar_affine_gpu(x)
        mean = torch.tensor(K.CIFAR_MEAN, device=x.device)[None, :, None, None]
        std = torch.tensor(K.CIFAR_STD, device=x.device)[None, :, None, None]
        return (x - mean) / std
    if train and aug == 'fmnist_aug':
        n, h, w = x.shape
        padded = nn.functional.pad(x, (2, 2, 2, 2))
        top = torch.randint(0, 5, (n,), device=x.device)
        left = torch.randint(0, 5, (n,), device=x.device)
        rows = (top[:, None] + torch.arange(h, device=x.device))[:, :, None]
        cols = (left[:, None] + torch.arange(w, device=x.device))[:, None, :]
        x = padded[torch.arange(n, device=x.device)[:, None, None], rows, cols]
        flip = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip[:, None, None], x.flip(-1), x)
    elif aug == 'fmnist_norm':
        x = (x - 0.5) / 0.5
    elif aug not in ('fmnist_plain', 'fmnist_aug'):
        raise ValueError(f'no GPU transform for {aug!r}')
    return x[:, None].repeat(1, 3, 1, 1)


class GpuPairs:
    """Fast path (F-MNIST, CIFAR 'cifar_affine'): the images stay on the GPU and augmentation runs batched there.

    Pairs are DatasetPairs' own (same `random` draws, so identical to the CPU path); only the
    per-sample PIL work, which left the T4s mostly idle, moves to the GPU. Yields the same
    tuples as a DataLoader over DatasetPairs.
    """

    def __init__(self, images, labels, subset, num_pairs, aug, train, batch_size, shuffle):
        from types import SimpleNamespace
        from snd.data.dataset import DatasetPairs
        sub_labels = labels[torch.as_tensor(subset, device=labels.device)].tolist()
        local = DatasetPairs.faster_generate_pairs_indices(
            SimpleNamespace(dataset=[(None, y) for y in sub_labels], num_pairs_per_epoch=num_pairs))
        subset = torch.as_tensor(subset, device=images.device)
        self.pairs = subset[torch.tensor(local, device=images.device)]     # original indices
        self.images, self.labels = images, labels
        self.aug, self.train, self.batch_size, self.shuffle = aug, train, batch_size, shuffle

    def __len__(self):
        return -(-len(self.pairs) // self.batch_size)

    def __iter__(self):
        n = len(self.pairs)
        order = torch.randperm(n, device=self.pairs.device) if self.shuffle else torch.arange(n, device=self.pairs.device)
        for start in range(0, n, self.batch_size):
            a, b = self.pairs[order[start:start + self.batch_size]].unbind(1)
            yield (gpu_transform(self.images[a], self.aug, self.train),
                   gpu_transform(self.images[b], self.aug, self.train), self.labels[a], self.labels[b], a, b)


@torch.no_grad()
def embed_gpu(model, images, indices, aug, amp, batch=1024):
    model.eval()
    indices = torch.as_tensor(indices, device=images.device)
    embs, probs = [], []
    for start in range(0, len(indices), batch):
        x = gpu_transform(images[indices[start:start + batch]], aug, train=False)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            e, logits = model.classify(x)
        embs.append(e.float().cpu())
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(embs).numpy(), torch.cat(probs).numpy()


def raw_images(ds, device):
    """uint8 images [N, 28, 28] and (noisy) targets of a torchvision F-MNIST set, on `device`."""
    base, idx = (ds.dataset, np.asarray(ds.indices)) if isinstance(ds, Subset) else (ds, np.arange(len(ds)))
    images = torch.as_tensor(np.asarray(base.data))[torch.as_tensor(idx)]
    labels = torch.as_tensor(np.asarray(base.targets)[idx])
    return images.to(device), labels.to(device)


def train_member(cfg, job, ds, table, splits):
    from snd.data.dataset import DatasetPairs
    variant, outer, member = job['variant'], job['outer'], job['member']
    # protocol_overrides: hyperparameter pilots (e.g. margin, patience) without new variants
    proto = dict(K.PROTOCOL[(cfg['dataset'], cfg['noise'])], **cfg.get('protocol_overrides', {}))
    # per-dataset settings in PROTOCOL (Animal-10N: dropout 0.1, batch 400; F-MNIST 60%: 300k
    # pairs) override the shared ones; `overrides` (smoke tests) override both
    common = {**K.SIAMESE_COMMON,
              **{k: proto[k] for k in ('dropout', 'batch_size', 'train_pairs') if k in proto},
              **cfg.get('overrides', {})}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = cfg['amp'] and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True        # fixed input sizes: pick the fastest kernels
    aug, plain = K.augmentation(proto['aug'])

    tr_idx, va_idx = splits[outer][member]
    seed = cfg['seed'] * 1000 + outer * 10 + member          # same pairs/init for every variant
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    workers = cfg.get('loader_workers', 2)
    # gpu_data (F-MNIST only): same pairs and transforms, batched on the GPU (the CPU loaders
    # were the bottleneck on Kaggle's 4 cores)
    gpu_data = cfg.get('gpu_data', proto.get('gpu_data', False))
    if gpu_data:
        images, labels = raw_images(ds, device)
        train_loader = GpuPairs(images, labels, tr_idx, common['train_pairs'], proto['aug'], True,
                                common['batch_size'], shuffle=True)
        val_loader = GpuPairs(images, labels, va_idx, common['val_pairs'], proto['aug'], False,
                              512, shuffle=False)
    else:
        train_pairs = DatasetPairs(Subset(ds, tr_idx), smart_count=False,
                                   num_pairs_per_epoch=common['train_pairs'], transform=aug)
        val_pairs = DatasetPairs(Subset(ds, va_idx), smart_count=False,
                                 num_pairs_per_epoch=common['val_pairs'], transform=plain)
        train_loader = DataLoader(train_pairs, batch_size=common['batch_size'], shuffle=True,
                                  num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
        val_loader = DataLoader(val_pairs, batch_size=512, shuffle=False, num_workers=workers,
                                pin_memory=True, persistent_workers=workers > 0)

    model = build_model(variant, proto, common['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=common['lr'], weight_decay=proto['wd'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ce = nn.CrossEntropyLoss(label_smoothing=common['label_smoothing'])
    ratio = 1.0 if variant in ('siamese', 'siamese_noreg') else 0.0
    max_epochs, patience = common['max_epochs'], proto['patience']
    if variant == 'siamese_noreg':
        max_epochs, patience = min(40, max_epochs), 50   # min(): smoke-test overrides

    # Early stopping on validation accuracy with patience, keep the best weights
    # (Trainer.train with freeze_epoch=None).
    best_acc, best_state, stale, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(max_epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, ratio, proto['margin'], ce,
                                    opt, scaler, amp)
        va_loss, va_acc = run_epoch(model, val_loader, device, ratio, proto['margin'], ce, amp=amp)
        history.append([epoch, tr_loss, tr_acc, va_loss, va_acc])
        if va_acc > best_acc:
            best_acc, stale = va_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f'{job["id"]} ep{epoch} train {tr_loss:.3f}/{tr_acc:.2f} val {va_loss:.3f}/{va_acc:.2f} '
              f'best {best_acc:.2f} stale {stale} [{(time.time() - t0) / 60:.0f} min]', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    train_minutes = (time.time() - t0) / 60

    outer_idx = np.flatnonzero(table['outer_fold'].to_numpy() == outer)
    if gpu_data:
        emb, probs = embed_gpu(model, images, outer_idx, proto['aug'], amp)
        ref_emb, _ = embed_gpu(model, images, tr_idx, proto['aug'], amp)
    else:
        emb, probs = embed(model, ds, outer_idx, plain, device, amp, workers=workers)
        ref_emb, _ = embed(model, ds, tr_idx, plain, device, amp, workers=workers)
    out = os.path.join(cfg['out_dir'], job['id'])
    np.savez_compressed(out + '.npz', outer_idx=outer_idx, probs=probs.astype(np.float16),
                        emb=emb.astype(np.float16), ref_idx=np.asarray(tr_idx),
                        ref_emb=ref_emb.astype(np.float16), history=np.asarray(history))
    if cfg.get('save_checkpoints', True):
        torch.save({k: v.half() if v.is_floating_point() else v
                    for k, v in model.state_dict().items()}, out + '.pt')

    noisy = table['noisy_label'].to_numpy()[outer_idx]
    true = table['real_label'].to_numpy()[outer_idx]
    pred = probs.argmax(1)
    return dict(job=job['id'], variant=variant, outer=outer, member=member, epochs=len(history),
                best_val_acc=best_acc, minutes=round(train_minutes, 1),
                outer_acc_noisy=float((pred == noisy).mean()), outer_acc_true=float((pred == true).mean()))


def inner_splits(table, outer_folds, seed, n_inner):
    """Seeded StratifiedKFold over each outer-training set (original indices)."""
    from sklearn.model_selection import StratifiedKFold
    splits = {}
    folds = table['outer_fold'].to_numpy()
    noisy = table['noisy_label'].to_numpy()
    for outer in outer_folds:
        pool = np.flatnonzero(folds != outer)
        skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + outer)
        splits[outer] = [(pool[a], pool[b]) for a, b in skf.split(pool, noisy[pool])]
    return splits


def jobs_for(cfg):
    jobs = []
    # member_ids: train only some members (e.g. finish an ensemble whose first members ran
    # elsewhere); a list applies to every outer fold, a dict {"outer": [ids]} per fold
    member_ids = cfg.get('member_ids') or range(cfg.get('members', 10))
    for outer in cfg['outer_folds']:
        ids = member_ids.get(str(outer), []) if isinstance(member_ids, dict) else member_ids
        for member in ids:
            for variant in cfg['variants']:
                # variant_suffix keeps pilot runs apart from the base variant when outputs are merged
                jobs.append(dict(id=f"{variant}{cfg.get('variant_suffix', '')}_o{outer}_m{member}",
                                 variant=variant, outer=outer, member=member))
    return jobs


def main(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    K.setup_repo(cfg['code_ref'])
    proto = K.PROTOCOL[(cfg['dataset'], cfg['noise'])]
    table = K.label_table(K.setup_repo(cfg['code_ref']), cfg['dataset'], cfg['noise'],
                          cfg['preds_ref'], cfg['seed'])
    ds = K.noisy_train_set(cfg['dataset'], table)
    table, keep = K.smoke_subsample(table, cfg)
    if keep is not None:
        ds = Subset(ds, keep)
    n_inner = proto.get('inner_folds', K.SIAMESE_COMMON['inner_folds'])
    cfg.setdefault('members', n_inner)         # one member per inner fold
    splits = inner_splits(table, cfg['outer_folds'], cfg['seed'], n_inner)

    queue = K.JobQueue(cfg['out_dir'])
    deadline = K.Deadline(cfg['session_hours'] - (time.time() - cfg['session_start']) / 3600)
    for job in jobs_for(cfg):
        if queue.done(job['id']):
            continue
        if not deadline.allows(cfg['member_hours']):
            print('session deadline reached; remaining jobs left for the next version', flush=True)
            break
        if not queue.claim(job['id']):
            continue
        try:
            summary = train_member(cfg, job, ds, table, splits)
        except BaseException:
            queue.release(job['id'])
            raise
        queue.finish(job['id'], summary)
        print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
