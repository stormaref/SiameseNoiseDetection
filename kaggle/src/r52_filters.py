"""R5.2 shared metrics + classic ensemble filters on the R5.1 out-of-fold members (no training).

Metrics (identical for every method, so the rows of r52_analysis are comparable):
  detection   flag vs table['is_noisy']: precision, recall, F1, FPR, ROC-AUC/AP of a
              continuous noise score (higher = more likely noisy);
  correction  every flagged sample is either relabelled (new label >= 0) or removed (-1);
              unflagged samples keep their noisy label.
              residual noise = wrong labels kept / size of the cleaned set (removed excluded);
              relabel acc    = relabelled to the true class / relabelled (repo definition,
                               CleanerMetricsMixin.analyze_with_mistakes_count);
              relabel score  = paper Sec. "Quantifying Label Correction Quality":
                               +2 noisy->true, 0 noisy->wrong, -2 clean->wrong, 0 clean->own label,
                               +1 noisy removed, -1 clean removed; divided by #flagged.

Filters, from the R5.1 files `<variant>_o<outer>_m<member>.npz` (outer_idx = positions in the
label table, probs = n x 10 softmax of a member that never saw that outer fold):
  majority   (Brodley & Friedl 1999): flag if more than half of the m members misclassify
             (argmax != given label), i.e. r > m/2;
  consensus  (Brodley & Friedl 1999): flag if all m members misclassify (r = m);
  confident_learning (Northcutt et al. 2021): cleanlab.filter.find_label_issues(noisy labels,
             mean of the m members' softmax) with cleanlab's defaults
             (filter_by='prune_by_noise_rate'), run once on all covered samples.
  Scores for AUC: r/m for majority/consensus (the same score, two thresholds); 1 - mean
  softmax probability of the given label (cleanlab's self-confidence) for CL.
  Cleaning: 'remove' (the filters' native action) and 'relabel' (secondary): majority and
  consensus relabel to a class predicted by a strict majority (> m/2) of the members, else
  remove; CL relabels to the argmax of the mean softmax.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

R51_FILE = re.compile(r'(?P<variant>.+)_o(?P<outer>\d+)_m(?P<member>\d+)\.npz$')


# --------------------------------------------------------------------------- metrics
def detection_metrics(flag, score, new_label, table_rows):
    """Detection + correction metrics for one method on the rows of `table_rows`.

    flag      bool array, True = flagged as noisy
    score     float array (higher = noisier) or None
    new_label int array: label after cleaning for flagged samples (>= 0 relabel, -1 remove);
              ignored where flag is False
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    flag = np.asarray(flag, bool)
    noisy_lab = table_rows['noisy_label'].to_numpy()
    true_lab = table_rows['real_label'].to_numpy()
    is_noisy = table_rows['is_noisy'].to_numpy(bool)
    n = len(flag)
    tp = int((flag & is_noisy).sum())
    fp = int((flag & ~is_noisy).sum())
    fn = int((~flag & is_noisy).sum())
    tn = int((~flag & ~is_noisy).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = dict(N=n, n_noisy=int(is_noisy.sum()), noise_rate=float(is_noisy.mean()) if n else np.nan,
               flagged=int(flag.sum()), tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec,
               f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
               fpr=fp / (fp + tn) if fp + tn else 0.0, auc=np.nan, ap=np.nan)
    if score is not None and 0 < is_noisy.sum() < n:
        score = np.asarray(score, float)
        out['auc'] = float(roc_auc_score(is_noisy, score))
        out['ap'] = float(average_precision_score(is_noisy, score))

    new_label = np.asarray(new_label, int)
    final = np.where(flag, new_label, noisy_lab)
    removed = flag & (new_label < 0)
    relabeled = flag & (new_label >= 0)
    retained = ~removed
    wrong_kept = int((retained & (final != true_lab)).sum())
    rel_ok = relabeled & (final == true_lab)
    score_sum = (2 * (relabeled & is_noisy & rel_ok).sum() - 2 * (relabeled & ~is_noisy & ~rel_ok).sum()
                 + (removed & is_noisy).sum() - (removed & ~is_noisy).sum())
    out.update(relabeled=int(relabeled.sum()), relabeled_changed=int((relabeled & (final != noisy_lab)).sum()),
               removed=int(removed.sum()), removed_noisy=int((removed & is_noisy).sum()),
               retained=int(retained.sum()), wrong_kept=wrong_kept,
               residual_noise=wrong_kept / retained.sum() if retained.sum() else np.nan,
               relabel_acc=float(rel_ok.sum() / relabeled.sum()) if relabeled.sum() else np.nan,
               relabel_score=float(score_sum / flag.sum()) if flag.sum() else np.nan)
    return out


# --------------------------------------------------------------------------- R5.1 members
def load_r51_members(r51_dir, variant, min_members=10):
    """[(outer_idx, probs)] for every `<variant>_o<outer>_m<member>.npz` in r51_dir (sorted).

    Outer folds with fewer than `min_members` member files (an unfinished R5.1 run) are skipped,
    as in r51_analysis, so every scored sample has the full ensemble behind it."""
    found = {}
    for path in glob.glob(os.path.join(r51_dir, f'{variant}_o*_m*.npz')):
        m = R51_FILE.search(os.path.basename(path))
        if m and m.group('variant') == variant:
            found.setdefault(int(m.group('outer')), []).append((int(m.group('member')), path))
    members = []
    for outer in sorted(found):
        if len(found[outer]) < min_members:
            print(f'{variant} outer {outer}: {len(found[outer])} members < {min_members}, skipped', flush=True)
            continue
        for _, path in sorted(found[outer]):
            with np.load(path) as z:
                members.append((z['outer_idx'].astype(int), z['probs'].astype(np.float32)))
    return members


def r51_variants(r51_dir):
    names = {R51_FILE.search(os.path.basename(p)).group('variant')
             for p in glob.glob(os.path.join(r51_dir, '*_o*_m*.npz')) if R51_FILE.search(os.path.basename(p))}
    return sorted(names)


def aggregate(members, n, n_classes=10):
    """Per-position member count, prob sum and class-vote counts over the given members."""
    count = np.zeros(n, int)
    prob_sum = np.zeros((n, n_classes))
    votes = np.zeros((n, n_classes), int)
    for outer_idx, probs in members:
        count[outer_idx] += 1
        prob_sum[outer_idx] += probs
        np.add.at(votes, (outer_idx, probs.argmax(1)), 1)
    return count, prob_sum, votes


def ensemble_filters(members, table, n_classes=10):
    """Run the three filters; returns ({name: (flag, score, relabel_target)}, covered positions)."""
    n = len(table)
    noisy = table['noisy_label'].to_numpy()
    count, prob_sum, votes = aggregate(members, n, n_classes)
    cov = np.flatnonzero(count > 0)
    if len(cov) == 0:
        raise ValueError('no R5.1 member covers any sample')
    m = count[cov]
    if len(np.unique(m)) > 1:
        print(f'warning: unequal member counts per sample {np.unique(m, return_counts=True)}; '
              'majority/consensus use each sample\'s own m', flush=True)
    v = votes[cov]
    mistakes = m - v[np.arange(len(cov)), noisy[cov]]
    avg = prob_sum[cov] / m[:, None]
    majority_class = np.where((v * 2 > m[:, None]).any(1), v.argmax(1), -1)    # strict majority or -1

    out = {
        'majority': (mistakes * 2 > m, mistakes / m, majority_class),
        'consensus': (mistakes == m, mistakes / m, majority_class),
    }
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        print('warning: cleanlab not installed (pip install cleanlab); Confident Learning skipped', flush=True)
    else:
        issues = find_label_issues(labels=noisy[cov], pred_probs=avg, n_jobs=1)
        out['confident_learning'] = (issues, 1.0 - avg[np.arange(len(cov)), noisy[cov]], avg.argmax(1))
    return out, cov


def filter_metrics(members, table, scopes=None, n_classes=10):
    """Main entry: members = [(outer_idx, probs)], table = label table (positions = outer_idx).

    scopes: {name: bool mask over table rows}; each filter is evaluated on covered ∩ mask.
    Default: {'covered': all samples with at least one member}.
    Returns one row per filter x cleaning mode x scope.
    """
    results, cov = ensemble_filters(members, table, n_classes)
    scopes = scopes or {'covered': np.ones(len(table), bool)}
    rows = []
    for name, (flag, score, target) in results.items():
        for mode in ('remove', 'relabel'):
            new_label = np.full(len(cov), -1) if mode == 'remove' else target
            for scope, mask in scopes.items():
                keep = np.asarray(mask, bool)[cov]
                if not keep.any():
                    continue
                row = detection_metrics(flag[keep], score[keep], new_label[keep], table.iloc[cov[keep]])
                rows.append(dict(method=name, cleaning=mode, scope=scope, **row))
    return pd.DataFrame(rows)
