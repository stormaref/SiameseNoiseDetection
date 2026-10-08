"""Cleaning outcome (paper's DetectAndRelabel rule) of new runs vs the published predictions.

Usage: python cleaning_compare.py <noise> <variant> <TD> <TR> <run_dir> [<run_dir> ...]
Uses only outer folds where the new variant has all 10 members, and the published members on
the same folds. Rule: flag if mistakes >= TD; relabel to the class with >= TR votes, else remove.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402

# F-MNIST 'published' = the round-3 paper run (513d833); main now holds the 2026-10 re-run
PUB_REF = '513d833'

noise, variant, TD, TR = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
repo = K.setup_repo('main')
pred_dir = K.extract_preds(repo, PUB_REF, K.PROTOCOL[('fashionmnist', noise)]['preds'])
table = K.load_label_table(pred_dir)

new = {}
for run in sys.argv[5:]:
    for path in glob.glob(os.path.join(run, 'r51_out', f'{variant}_o*_m*.npz')):
        o, m = map(int, re.search(r'_o(\d+)_m(\d+)\.npz$', path).groups())
        new.setdefault(o, {})[m] = path
folds = sorted(o for o, ms in new.items() if len(ms) == 10)


def published(fold):
    f = glob.glob(os.path.join(pred_dir, '**', f'fold{fold}_analysis.csv'), recursive=True)[0]
    df = pd.read_csv(f)
    P = np.stack(df['preds'].astype(str).str.split('|').map(lambda x: np.array(x, dtype=int)).values)
    return df['index'].to_numpy(), P


def new_preds(fold):
    idx, cols = None, []
    for m in range(10):
        z = np.load(new[fold][m])
        if idx is None:
            idx = z['outer_idx']
        assert np.array_equal(idx, z['outer_idx'])
        cols.append(z['probs'].astype(np.float32).argmax(1))
    return idx, np.stack(cols, 1)


def clean(idx, P):
    noisy = table['noisy_label'].to_numpy()[idx]
    true = table['real_label'].to_numpy()[idx]
    is_noisy = noisy != true
    mistakes = (P != noisy[:, None]).sum(1)
    flagged = mistakes >= TD
    votes = np.stack([(P == c).sum(1) for c in range(10)], 1)
    target = np.where(votes.max(1) >= TR, votes.argmax(1), -1)
    new_label = np.where(flagged, target, noisy)
    kept = new_label >= 0
    tp = (flagged & is_noisy).sum()
    prec, rec = tp / max(flagged.sum(), 1), tp / max(is_noisy.sum(), 1)
    return dict(n=len(idx), noisy_before=int(is_noisy.sum()), flagged=int(flagged.sum()),
                precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                relabeled=int((flagged & kept).sum()), removed=int((~kept).sum()),
                retained=int(kept.sum()), noisy_after=int((new_label[kept] != true[kept]).sum()),
                residual_noise=float((new_label[kept] != true[kept]).mean()),
                clean_retained=int((new_label[kept] == true[kept]).sum()))


rows = []
for fold in folds:
    for name, (idx, P) in [('published', published(fold)), (variant, new_preds(fold))]:
        rows.append(dict(fold=fold, model=name, **clean(idx, P)))
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(f'F-MNIST {noise}%  TD={TD} TR={TR}  folds {folds}')
print(df.round(4).to_string(index=False))
tot = df.groupby('model')[['n', 'noisy_before', 'flagged', 'retained', 'noisy_after', 'clean_retained', 'removed']].sum()
tot['residual_noise_%'] = (100 * tot.noisy_after / tot.retained).round(2)
print(tot.to_string())
