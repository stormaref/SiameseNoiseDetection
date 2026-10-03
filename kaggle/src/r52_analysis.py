"""R5.2 analysis: every method scored with the same detection/correction metrics on the same labels.

Rows (per dataset/noise, per scope):
  ours                 the published Siamese ensemble from the committed CSVs at the paper's oracle
                       thresholds (flag: mistakes >= TD; relabel to a class with >= TR votes, else remove)
  coteaching           flag = not selected as small-loss by net1 AND not by net2 during the final epoch;
                       cleaning 'remove' (primary) | 'agree-relabel' (secondary: relabel to the joint
                       argmax when both nets' argmax agree on the un-augmented image, else remove);
                       AUC score = mean of the two nets' CE loss on the un-augmented image (final weights)
  coteaching-global    (secondary) flag = in the top forget-rate fraction of the un-augmented CE loss of
                       BOTH final nets (the dataset-level version of the per-batch rule); 'remove'
  dividemix            flag = mean of the two nets' GMM clean probability < p_threshold (0.5), GMM fitted
                       on the final weights' losses; AUC score = 1 - mean clean probability;
                       cleaning 'relabel' (primary: argmax of the averaged softmax of both nets on the
                       un-augmented image) | 'remove' (secondary)
  majority / consensus / confident_learning [variant]
                       r52_filters on the R5.1 out-of-fold members ('ce' = plain classifier ensemble,
                       'siamese' = ours retrained); 'remove' (primary) | 'relabel' (secondary)
Scopes: 'all' = every training sample the method scored (R5.1 filters: the samples their members
cover); 'folds' = only samples whose outer fold is in config['outer_folds'] (comparable to our per-fold
numbers). Usage: python r52_analysis.py config.json [--out results.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snd_kaggle as K  # noqa: E402
from r52_filters import detection_metrics, filter_metrics, load_r51_members, r51_variants  # noqa: E402

# Paper Table "Optimal Configurations" (oracle thresholds, TD/TR)
OURS_THRESHOLDS = {('cifar10', 20): (8, 10), ('cifar10', 30): (7, 10), ('cifar10', 40): (6, 10),
                   ('fashionmnist', 20): (9, 10), ('fashionmnist', 30): (9, 10), ('fashionmnist', 40): (7, 10)}
SHOW = [('precision', 'P'), ('recall', 'R'), ('f1', 'F1'), ('fpr', 'FPR'), ('auc', 'AUC'),
        ('flagged', 'flagged'), ('residual_noise', 'resid'), ('relabel_acc', 'relab_acc')]


def label_table(cfg, dataset, noise):
    """Label table in the row order the workers used (subsample draw identical to r51/r52 workers)."""
    proto = K.PROTOCOL[(dataset, noise)]
    repo = K.setup_repo(cfg['code_ref'])
    table = K.load_label_table(K.extract_preds(repo, cfg['preds_ref'], proto['preds']))
    return K.smoke_subsample(table, cfg)[0]


def rows_for(table, scopes, method, cleaning, flag, score, new_label, **extra):
    out = []
    for scope, mask in scopes.items():
        if mask.any():
            m = detection_metrics(flag[mask], None if score is None else score[mask], new_label[mask],
                                  table[mask])
            out.append(dict(method=method, cleaning=cleaning, scope=scope, **m, **extra))
    return out


def ours(table, scopes, td, tr):
    flag = table['mistakes'].to_numpy() >= td
    new = K.cleaned_labels(table, td, tr)
    score = table['mistakes'].to_numpy() / table['preds'].astype(str).str.count(r'\|').add(1).to_numpy()
    return rows_for(table, scopes, f'ours (TD={td},TR={tr})', 'relabel', flag, score, new)


def coteaching(z, table, scopes):
    pos = table.index.get_indexer(z['index'])
    if (pos < 0).any() or len(pos) != len(table):
        raise ValueError('Co-teaching output does not cover the label table')
    t = table.iloc[pos]
    sc = {k: v[pos] for k, v in scopes.items()}
    flag = ~z['sel1'] & ~z['sel2']
    score = (z['eval_loss1'] + z['eval_loss2']) / 2
    s1, s2 = z['soft1'].astype(np.float32), z['soft2'].astype(np.float32)
    agree = s1.argmax(1) == s2.argmax(1)
    target = np.where(agree, (s1 + s2).argmax(1), -1)
    hist = dict(zip(z['history_cols'].tolist(), z['history'][-1]))
    extra = dict(test_acc=hist.get('test_acc', np.nan), forget_rate=float(z['forget_rate']))
    n = len(t)
    rows = rows_for(t, sc, 'coteaching', 'remove', flag, score, np.full(n, -1), **extra)
    rows += rows_for(t, sc, 'coteaching', 'agree-relabel', flag, score, target, **extra)
    k = int(round(float(z['forget_rate']) * n))                       # global small-loss version
    top = lambda loss: np.isin(np.arange(n), np.argsort(-loss)[:k])  # noqa: E731
    gflag = top(z['eval_loss1']) & top(z['eval_loss2'])
    rows += rows_for(t, sc, 'coteaching-global', 'remove', gflag, score, np.full(n, -1), **extra)
    return rows


def dividemix(z, index, table, scopes, p_threshold):
    pos = table.index.get_indexer(index)
    if (pos < 0).any() or len(pos) != len(table):
        raise ValueError('DivideMix output does not cover the label table')
    t = table.iloc[pos]
    sc = {k: v[pos] for k, v in scopes.items()}
    clean = (z['prob1'] + z['prob2']) / 2
    flag = clean < p_threshold
    pred = (z['soft1'].astype(np.float32) + z['soft2'].astype(np.float32)).argmax(1)
    extra = dict(test_acc=float(z['test_acc'][-1]) if len(z['test_acc']) else np.nan)
    rows = rows_for(t, sc, 'dividemix', 'relabel', flag, 1 - clean, pred, **extra)
    rows += rows_for(t, sc, 'dividemix', 'remove', flag, 1 - clean, np.full(len(t), -1), **extra)
    return rows


def resolve(pattern):
    import glob
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    out_dir = cfg['out_dir']
    p_thr = cfg.get('dividemix', {}).get('p_threshold', 0.5)
    thresholds = dict(OURS_THRESHOLDS)
    for key, v in cfg.get('ours_thresholds', {}).items():
        d, n = key.rsplit('_', 1)
        thresholds[(d, int(n))] = tuple(v)
    r51_dirs = cfg.get('r51_dirs', {})
    settings = sorted({(d, int(n)) for _, d, n in cfg.get('jobs', [])} |
                      {(k.rsplit('_', 1)[0], int(k.rsplit('_', 1)[1])) for k in r51_dirs})

    rows = []
    for dataset, noise in settings:
        table = label_table(cfg, dataset, noise)
        folds = table['outer_fold'].isin(cfg.get('outer_folds', [])).to_numpy()
        scopes = {'all': np.ones(len(table), bool), 'folds': folds}
        new = []
        if (dataset, noise) in thresholds:
            new += ours(table, scopes, *thresholds[(dataset, noise)])
        ct = os.path.join(out_dir, f'coteaching_{dataset}_{noise}', 'final.npz')
        if os.path.exists(ct):
            with np.load(ct) as z:
                new += coteaching(z, table, scopes)
        dm_dir = os.path.join(out_dir, f'dividemix_{dataset}_{noise}')
        if os.path.exists(os.path.join(dm_dir, 'final.npz')):
            with np.load(os.path.join(dm_dir, 'final.npz')) as z:
                new += dividemix(z, np.load(os.path.join(dm_dir, 'index.npy')), table, scopes, p_thr)
        r51 = r51_dirs.get(f'{dataset}_{noise}')
        r51 = resolve(r51) if r51 else None
        if r51:
            for variant in [v for v in cfg.get('r51_variants', ['ce', 'siamese']) if v in r51_variants(r51)]:
                members = load_r51_members(r51, variant, cfg.get('r51_min_members', 10))
                if not members:
                    continue
                df = filter_metrics(members, table, scopes)
                df['method'] = df['method'] + f' [{variant}]'
                new += df.to_dict('records')
        elif f'{dataset}_{noise}' in r51_dirs:
            print(f'warning: no R5.1 outputs found for {dataset}_{noise}: {r51_dirs[f"{dataset}_{noise}"]}')
        for r in new:
            r.update(dataset=dataset, noise=noise)
        rows += new

    res = pd.DataFrame(rows)
    if res.empty:
        print('no results found')
        return res
    lead = ['dataset', 'noise', 'scope', 'method', 'cleaning']
    res = res[lead + [c for c in res.columns if c not in lead]]
    out = a.out or os.path.join(out_dir, 'r52_results.csv')
    res.to_csv(out, index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 60)
    for scope, part in res.groupby('scope', sort=False):
        view = part.set_index(['dataset', 'noise', 'method', 'cleaning'])[[c for c, _ in SHOW]]
        view.columns = [s for _, s in SHOW]
        n = part.groupby(['dataset', 'noise'])['N'].max().to_dict()
        print(f'\n=== scope: {scope}  (N per setting: {n}) ===')
        print(view.to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'\nwrote {out}')
    return res


if __name__ == '__main__':
    main(sys.argv[1:])
