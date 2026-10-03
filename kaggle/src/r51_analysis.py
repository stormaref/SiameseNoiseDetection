"""R5.1 analysis: which part of the detector carries the signal?

For every (variant, outer fold) it scores the held-out samples with
  disagreement      r(x)/m, the published detector;
  confidence        1 - mean_j p_j(y~|x), the members' mean softmax on the observed label;
  centroid_margin   mean_j [d(h_j, mu_j[y~]) - min_{k != y~} d(h_j, mu_j[k])];
  knn_disagreement  mean_j share of the k nearest training embeddings with a label != y~;
  emb_instability   std_j of d(h_j, mu_j[y~]) / s_j, s_j = member j's median own-class distance.
Embedding spaces of separately trained members are not aligned, so every embedding score is
computed inside one member (against that member's own training embeddings, labelled with
the observed labels) and only the resulting scalars are aggregated across members.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

SCORES = ['disagreement', 'confidence', 'centroid_margin', 'knn_disagreement', 'emb_instability']


def load_runs(out_dir):
    runs = {}
    for path in glob.glob(os.path.join(out_dir, '*.npz')):
        variant, outer, member = re.match(r'(.+)_o(\d+)_m(\d+)\.npz', os.path.basename(path)).groups()
        runs.setdefault((variant, int(outer)), {})[int(member)] = path
    return {key: [members[m] for m in sorted(members)] for key, members in runs.items()}


def member_scores(npz, noisy_all, k=20):
    """Per-sample scalars from one member: prediction, p(y~), and the embedding scores."""
    d = np.load(npz)
    idx, ref_idx = d['outer_idx'], d['ref_idx']
    y, ref_y = noisy_all[idx], noisy_all[ref_idx]
    probs = d['probs'].astype(np.float32)
    h = torch.from_numpy(d['emb'].astype(np.float32))
    ref = torch.from_numpy(d['ref_emb'].astype(np.float32))
    classes = np.unique(ref_y)
    mu = torch.stack([ref[torch.from_numpy(ref_y == c)].mean(0) for c in classes])
    dist_mu = torch.cdist(h, mu).numpy()                         # n x c
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


def best_f1(score, truth):
    order = np.argsort(-score, kind='stable')
    hits = np.cumsum(truth[order])
    k = np.arange(1, len(score) + 1)
    f1 = 2 * hits / (k + truth.sum())
    # only cut between distinct score values
    valid = np.r_[score[order][1:] != score[order][:-1], True]
    return f1[valid].max()


def detection_at(flag, truth):
    tp, fp = (flag & truth).sum(), (flag & ~truth).sum()
    fn = (~flag & truth).sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn)
    return dict(precision=prec, recall=rec, f1=2 * tp / (2 * tp + fp + fn), fpr=fp / (~truth).sum())


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


def analyse(out_dir, table, min_members=10):
    noisy_all = table['noisy_label'].to_numpy()
    noise_all = table['is_noisy'].to_numpy()
    rows, threshold_rows, member_rows, cache = [], [], [], {}
    for (variant, outer), paths in sorted(load_runs(out_dir).items()):
        if len(paths) < min_members:
            print(f'{variant} outer {outer}: {len(paths)} members, skipped (< {min_members})')
            continue
        idx, pred, scores = ensemble_scores(paths, noisy_all)
        truth = noise_all[idx]
        cache[(variant, outer)] = (idx, scores, truth)
        for name, s in scores.items():
            rows.append(dict(variant=variant, outer=outer, m=len(paths), score=name,
                             auc=roc_auc_score(truth, s), ap=average_precision_score(truth, s),
                             best_f1=best_f1(s, truth)))
        r = (pred != noisy_all[idx][:, None]).sum(1)
        for tau in range(1, len(paths) + 1):
            threshold_rows.append(dict(variant=variant, outer=outer, tau=tau,
                                       **detection_at(r >= tau, truth)))
        wrong = pred != noisy_all[idx][:, None]
        member_rows.append(dict(variant=variant, outer=outer,
                                miscls_rate_noisy=wrong[truth].mean(), miscls_rate_clean=wrong[~truth].mean(),
                                predicts_true_label_on_noisy=(pred[truth] == table['real_label'].to_numpy()[idx][truth, None]).mean()))

    # the published run on the same outer folds, for reference
    for outer in sorted({o for _, o in cache}):
        sub = table[table['outer_fold'] == outer]
        truth = sub['is_noisy'].to_numpy()
        s = sub['mistakes'].to_numpy() / 10
        rows.append(dict(variant='published', outer=outer, m=10, score='disagreement',
                         auc=roc_auc_score(truth, s), ap=average_precision_score(truth, s),
                         best_f1=best_f1(s, truth)))

    # paired comparisons on identical samples: variant vs variant (same score), and each
    # embedding score vs the disagreement detector (same variant)
    boot = []
    pairs = [('siamese', 'ce'), ('siamese', 'siamese_noreg'), ('siamese', 'ce_linear'), ('ce', 'ce_linear')]
    for outer in sorted({o for _, o in cache}):
        for a, b in pairs:
            if (a, outer) not in cache or (b, outer) not in cache:
                continue
            i1, s1, t1 = cache[(a, outer)]
            i2, s2, _ = cache[(b, outer)]
            assert (i1 == i2).all()
            for name in SCORES:
                mean, lo, hi = paired_bootstrap_auc(s1[name], s2[name], t1)
                boot.append(dict(outer=outer, comparison=f'{a} - {b}', score=name,
                                 auc_diff=mean, ci_lo=lo, ci_hi=hi))
        for variant in ('siamese', 'ce'):
            if (variant, outer) not in cache:
                continue
            _, s, t = cache[(variant, outer)]
            for name in SCORES[1:]:
                mean, lo, hi = paired_bootstrap_auc(s[name], s['disagreement'], t)
                boot.append(dict(outer=outer, comparison=f'{variant}: {name} - disagreement',
                                 score=name, auc_diff=mean, ci_lo=lo, ci_hi=hi))
    return (pd.DataFrame(rows), pd.DataFrame(threshold_rows), pd.DataFrame(member_rows),
            pd.DataFrame(boot))


def report(out_dir, table, results_dir, min_members=10):
    os.makedirs(results_dir, exist_ok=True)
    scores, thresholds, members, boot = analyse(out_dir, table, min_members)
    for name, df in [('scores', scores), ('thresholds', thresholds), ('members', members),
                     ('bootstrap', boot)]:
        df.to_csv(os.path.join(results_dir, f'r51_{name}.csv'), index=False)
    pd.set_option('display.width', 200)
    if len(scores):
        print('\nThreshold-free detection quality (held-out outer folds):')
        print(scores.pivot_table(index=['score'], columns='variant', values=['auc', 'best_f1'])
              .round(4).to_string())
        print('\nPer-member misclassification w.r.t. the observed label (R4.3 support):')
        print(members.round(4).to_string(index=False))
    if len(boot):
        print('\nPaired bootstrap of ROC-AUC differences (95% CI):')
        print(boot.round(4).to_string(index=False))
    return scores, thresholds, members, boot
