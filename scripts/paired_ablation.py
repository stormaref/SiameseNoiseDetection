"""Paired ablation of the ensemble members on one outer fold (the paper's regularization and
contrastive-objective ablations).

Three member variants are trained on identical inner splits, training pairs, initializations and
seeds, so every difference comes from the objective:
  siamese        ours: cross-entropy on both branches + the contrastive loss, dropout, early stopping;
  siamese_noreg  ours without dropout and early stopping (a fixed 40 epochs; the best
                 validation-accuracy weights are still restored);
  ce             the same network and pairs with the contrastive term switched off, i.e. an
                 ordinary cross-entropy classifier.
Each member scores the held-out outer fold. The analysis ranks the held-out samples by ensemble
disagreement and by embedding-level scores computed inside each member (centroid margin, k-nearest
neighbours), reports detection at T_D = m, a paired bootstrap of ROC-AUC differences, and how often
a member misclassifies the observed label of mislabeled and clean samples.

Usage:
    uv run python scripts/paired_ablation.py train --variants siamese ce siamese_noreg      # GPU
    uv run python scripts/paired_ablation.py analyze --members_dir results/paired_ablation
"""
import argparse
import copy
import glob
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

import snd.config as config
from snd.data.dataset import DatasetPairs, DatasetSingle
from snd.models.siamese import SiameseNetwork

CIFAR_NORM = [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
CROP = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(0.5), T.RandomRotation(15), *CIFAR_NORM])
AFFINE = T.Compose([T.RandomRotation(15), T.RandomHorizontalFlip(0.5), T.RandomAffine(0, translate=(0.1, 0.1)),
                    T.RandomResizedCrop(32, scale=(0.9, 1.0)), *CIFAR_NORM])
# Member settings of the ablation runs (ResNet50 pre-trained on ImageNet, 64-d embedding, margin 2)
SETTINGS = {'20': dict(params='CIFAR10_20_PARAMS', patience=8, transform=CROP),
            '30': dict(params='CIFAR10_30_PARAMS', patience=10, transform=CROP),
            '40': dict(params='CIFAR10_40_PARAMS', patience=15, transform=AFFINE)}
COMMON = dict(lr=5e-5, weight_decay=5e-4, batch_size=2048, train_pairs=200_000, val_pairs=20_000,
              dropout=0.5, label_smoothing=0.1, margin=2, embedding=64, max_epochs=1000, inner_folds=10)
SCORES = ['disagreement', 'confidence', 'centroid_margin', 'knn_disagreement', 'emb_instability']


# --------------------------------------------------------------------------- data
def label_table(noise_ratio):
    """Noisy/true labels and outer folds of the committed CIFAR-10 predictions, by dataset position."""
    params = getattr(config, SETTINGS[noise_ratio]['params'])
    parts = [pd.read_csv(params['prediction_path'].format(k)).assign(outer_fold=k)
             for k in range(1, params['outer_folds_num'] + 1)]
    table = pd.concat(parts).set_index('index').sort_index()
    assert (table.index == np.arange(len(table))).all()
    table['is_noisy'] = table['noisy_label'] != table['real_label']
    return table


def inner_splits(table, outer, seed, n_inner):
    """Seeded stratified split of the outer-training pool into the members' training subsets."""
    pool = np.flatnonzero(table['outer_fold'].to_numpy() != outer)
    noisy = table['noisy_label'].to_numpy()
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + outer)
    return [(pool[a], pool[b]) for a, b in skf.split(pool, noisy[pool])]


# --------------------------------------------------------------------------- training
def contrastive(emb1, emb2, same, margin):
    d = nn.functional.pairwise_distance(emb1.float(), emb2.float())
    return torch.mean(same * d.pow(2) + (1 - same) * torch.clamp(margin - d, min=0).pow(2))


def run_epoch(model, loader, device, ratio, margin, ce, opt=None, scaler=None, amp=False):
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
def embed(model, ds, indices, transform, device, amp, workers):
    loader = DataLoader(DatasetSingle(Subset(ds, indices), transform), batch_size=1024, shuffle=False,
                        num_workers=workers)
    model.eval()
    embs, probs = [], []
    for img, _, _ in loader:
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            e, logits = model.classify(img.to(device))
        embs.append(e.float().cpu())
        probs.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(embs).numpy(), torch.cat(probs).numpy()


def train_member(variant, outer, member, ds, table, split, setting, args):
    common = dict(COMMON)
    if args.train_pairs:                                     # quick checks only
        common.update(train_pairs=args.train_pairs, val_pairs=min(COMMON['val_pairs'], args.train_pairs))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    torch.backends.cudnn.benchmark = True
    plain = T.Compose(CIFAR_NORM)
    tr_idx, va_idx = split
    seed = args.seed * 1000 + outer * 10 + member           # same pairs/init for every variant
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    workers = args.workers
    train_loader = DataLoader(DatasetPairs(Subset(ds, tr_idx), smart_count=False,
                                           num_pairs_per_epoch=common['train_pairs'], transform=setting['transform']),
                              batch_size=common['batch_size'], shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0)
    val_loader = DataLoader(DatasetPairs(Subset(ds, va_idx), smart_count=False,
                                         num_pairs_per_epoch=common['val_pairs'], transform=plain),
                            batch_size=512, shuffle=False, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0)
    model = SiameseNetwork(num_classes=10, model='resnet50', embedding_dimension=common['embedding'],
                           pre_trained=True, dropout_prob=0.0 if variant == 'siamese_noreg' else common['dropout'],
                           trainable=True, parallel=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=common['lr'], weight_decay=common['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    ce = nn.CrossEntropyLoss(label_smoothing=common['label_smoothing'])
    ratio = 0.0 if variant == 'ce' else 1.0
    max_epochs, patience = (40, 50) if variant == 'siamese_noreg' else (common['max_epochs'], setting['patience'])
    if args.max_epochs:
        max_epochs = min(max_epochs, args.max_epochs)

    best_acc, best_state, stale = -1.0, None, 0
    for epoch in range(max_epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, ratio, common['margin'], ce, opt, scaler, amp)
        va_loss, va_acc = run_epoch(model, val_loader, device, ratio, common['margin'], ce, amp=amp)
        if va_acc > best_acc:
            best_acc, stale, best_state = va_acc, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f'{variant} outer {outer} member {member} epoch {epoch}: train {tr_loss:.3f}/{tr_acc:.2f} '
              f'val {va_loss:.3f}/{va_acc:.2f} best {best_acc:.2f} stale {stale}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)

    outer_idx = np.flatnonzero(table['outer_fold'].to_numpy() == outer)
    emb, probs = embed(model, ds, outer_idx, plain, device, amp, workers)
    ref_emb, _ = embed(model, ds, tr_idx, plain, device, amp, workers)
    out = os.path.join(args.members_dir, f'{variant}_o{outer}_m{member}.npz')
    np.savez_compressed(out, outer_idx=outer_idx, probs=probs.astype(np.float16), emb=emb.astype(np.float16),
                        ref_idx=np.asarray(tr_idx), ref_emb=ref_emb.astype(np.float16))
    print(f'wrote {out}', flush=True)


def train(args):
    table = label_table(args.noise_ratio)
    if args.subsample:                                       # quick checks only
        table = table.iloc[:args.subsample]
    ds = config.CIFAR10_TRAIN_DATASET
    ds.targets = table['noisy_label'].tolist() + list(ds.targets[len(table):])   # noisy labels
    os.makedirs(args.members_dir, exist_ok=True)
    splits = inner_splits(table, args.outer, args.seed, COMMON['inner_folds'])
    for variant in args.variants:
        for member in range(args.members):
            if os.path.exists(os.path.join(args.members_dir, f'{variant}_o{args.outer}_m{member}.npz')):
                continue
            train_member(variant, args.outer, member, ds, table, splits[member],
                         SETTINGS[args.noise_ratio], args)


# --------------------------------------------------------------------------- analysis
def member_scores(npz, noisy_all, k=20):
    """Per-sample scalars from one member: prediction, p(observed label) and embedding scores."""
    d = np.load(npz)
    idx, ref_idx = d['outer_idx'], d['ref_idx']
    y, ref_y = noisy_all[idx], noisy_all[ref_idx]
    probs = d['probs'].astype(np.float32)
    h = torch.from_numpy(d['emb'].astype(np.float32))
    ref = torch.from_numpy(d['ref_emb'].astype(np.float32))
    classes = np.unique(ref_y)
    mu = torch.stack([ref[torch.from_numpy(ref_y == c)].mean(0) for c in classes])
    dist_mu = torch.cdist(h, mu).numpy()
    own = dist_mu[np.arange(len(y)), np.searchsorted(classes, y)]
    other = np.where(classes[None, :] == y[:, None], np.inf, dist_mu).min(1)
    ref_own = torch.cdist(ref, mu).numpy()[np.arange(len(ref_y)), np.searchsorted(classes, ref_y)]
    knn = np.empty(len(y))
    for start in range(0, len(y), 1024):
        nearest = torch.cdist(h[start:start + 1024], ref).topk(k, largest=False).indices.numpy()
        knn[start:start + 1024] = (ref_y[nearest] != y[start:start + 1024, None]).mean(1)
    return dict(idx=idx, pred=probs.argmax(1), p_obs=probs[np.arange(len(y)), y],
                margin=own - other, knn=knn, own_scaled=own / np.median(ref_own))


def ensemble_scores(paths, noisy_all):
    per = [member_scores(p, noisy_all) for p in paths]
    idx = per[0]['idx']
    assert all((m['idx'] == idx).all() for m in per)
    y = noisy_all[idx]
    stack = lambda key: np.stack([m[key] for m in per], 1)  # noqa: E731
    pred = stack('pred')
    return idx, pred, {
        'disagreement': (pred != y[:, None]).mean(1),
        'confidence': 1 - stack('p_obs').mean(1),
        'centroid_margin': stack('margin').mean(1),
        'knn_disagreement': stack('knn').mean(1),
        'emb_instability': stack('own_scaled').std(1) if len(per) > 1 else np.zeros(len(idx)),
    }


def detection_at(flag, truth):
    tp, fp, fn = (flag & truth).sum(), (flag & ~truth).sum(), (~flag & truth).sum()
    return dict(precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn),
                f1=2 * tp / (2 * tp + fp + fn), fpr=fp / (~truth).sum())


def paired_bootstrap_auc(s1, s2, truth, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        b = rng.integers(0, len(truth), len(truth))
        if truth[b].all() or not truth[b].any():
            continue
        diffs.append(roc_auc_score(truth[b], s1[b]) - roc_auc_score(truth[b], s2[b]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def analyze(args):
    table = label_table(args.noise_ratio)
    noisy_all, noise_all = table['noisy_label'].to_numpy(), table['is_noisy'].to_numpy()
    runs = {}
    for path in glob.glob(os.path.join(args.members_dir, '*.npz')):
        variant, outer, member = re.match(r'(.+)_o(\d+)_m(\d+)\.npz', os.path.basename(path)).groups()
        if int(outer) == args.outer:
            runs.setdefault(variant, {})[int(member)] = path
    cache, rows = {}, []
    for variant in sorted(runs):
        paths = [runs[variant][m] for m in sorted(runs[variant])]
        idx, pred, scores = ensemble_scores(paths, noisy_all)
        truth = noise_all[idx]
        cache[variant] = (scores, truth)
        wrong = pred != noisy_all[idx][:, None]
        rows.append(dict(variant=variant, members=len(paths),
                         **{f'auc_{s}': roc_auc_score(truth, v) for s, v in scores.items()},
                         ap=average_precision_score(truth, scores['disagreement']),
                         **detection_at(wrong.sum(1) >= len(paths), truth),
                         miscls_noisy=wrong[truth].mean(), miscls_clean=wrong[~truth].mean()))
    res = pd.DataFrame(rows).set_index('variant')
    pd.set_option('display.width', 200)
    print(f'Outer fold {args.outer}: {len(truth):,} held-out samples, {truth.sum():,} mislabeled\n')
    print('Detection by disagreement (ROC-AUC, AP) and at T_D = m:')
    print(res[['members', 'auc_disagreement', 'ap', 'precision', 'recall', 'f1', 'fpr']].round(3).to_string())
    print('\nROC-AUC of the embedding-level scores:')
    print(res[['auc_centroid_margin', 'auc_knn_disagreement', 'auc_confidence']].round(3).to_string())
    print('\nShare of members misclassifying the observed label (mislabeled / clean samples):')
    print(res[['miscls_noisy', 'miscls_clean']].round(3).to_string())
    if 'siamese' in cache:
        for other in ('ce', 'siamese_noreg'):
            if other in cache:
                mean, lo, hi = paired_bootstrap_auc(cache['siamese'][0]['disagreement'],
                                                    cache[other][0]['disagreement'], cache['siamese'][1])
                print(f'\nPaired bootstrap, ROC-AUC siamese - {other}: {mean:+.4f}, 95% CI [{lo:.3f}, {hi:.3f}]')
        if 'ce' in cache:
            print(f"Correct labels discarded at T_D = m: {1 - res.loc['siamese', 'fpr'] / res.loc['ce', 'fpr']:.0%} "
                  'fewer with the contrastive objective than with cross-entropy only')
    res.to_csv(os.path.join(args.members_dir, f'paired_ablation_outer{args.outer}.csv'))


def main():
    p = argparse.ArgumentParser(description='Paired ablation of the ensemble members (CIFAR-10)')
    p.add_argument('mode', choices=['train', 'analyze'])
    p.add_argument('--noise_ratio', default='20', choices=sorted(SETTINGS))
    p.add_argument('--outer', type=int, default=1, help='outer fold to hold out (1-based)')
    p.add_argument('--variants', nargs='+', default=['siamese', 'ce', 'siamese_noreg'],
                   choices=['siamese', 'ce', 'siamese_noreg'])
    p.add_argument('--members', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--members_dir', default='results/paired_ablation')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max_epochs', type=int, help='cap the epochs (quick checks only)')
    p.add_argument('--train_pairs', type=int, help='override the 200k training pairs (quick checks only)')
    p.add_argument('--subsample', type=int, help='use only the first N samples (quick checks only)')
    args = p.parse_args()
    train(args) if args.mode == 'train' else analyze(args)


if __name__ == '__main__':
    main()
