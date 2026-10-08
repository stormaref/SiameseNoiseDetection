"""New vs published ensemble on one outer fold with the same member subset (e.g. 9 of 10).

Usage: python gate_compare.py <noise> <variant> <fold> <run_dir> [<run_dir> ...]
Reports disagreement AUC, best F1, and the oracle-best residual noise over the (TD, TR) grid
TD, TR in 5..m, for both ensembles restricted to the members the new run has.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402

# F-MNIST 'published' = the round-3 paper run (513d833); main now holds the 2026-10 re-run
PUB_REF = '513d833'

noise, variant, fold = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
repo = K.setup_repo('main')
pred_dir = K.extract_preds(repo, PUB_REF, K.PROTOCOL[('fashionmnist', noise)]['preds'])
table = K.load_label_table(pred_dir)
paths = {}
for run in sys.argv[4:]:
    for p in glob.glob(os.path.join(run, 'r51_out', f'{variant}_o{fold}_m*.npz')):
        paths[int(re.search(r'_m(\d+)\.npz$', p).group(1))] = p
members = sorted(paths)
idx, cols = None, []
for m in members:
    z = np.load(paths[m])
    idx = z['outer_idx'] if idx is None else idx
    assert np.array_equal(idx, z['outer_idx'])
    cols.append(z['probs'].astype(np.float32).argmax(1))
P_new = np.stack(cols, 1)
f = glob.glob(os.path.join(pred_dir, '**', f'fold{fold}_analysis.csv'), recursive=True)[0]
df = pd.read_csv(f).set_index('index').loc[idx]
P_pub = np.stack(df['preds'].astype(str).str.split('|').map(lambda x: np.array(x, dtype=int)).values)[:, members]
noisy = table['noisy_label'].to_numpy()[idx]
true = table['real_label'].to_numpy()[idx]
is_noisy = noisy != true
m = len(members)


def evaluate(P):
    r = (P != noisy[:, None]).sum(1)
    f1s = []
    for td in range(1, m + 1):
        fl = r >= td
        tp = (fl & is_noisy).sum()
        p, q = tp / max(fl.sum(), 1), tp / is_noisy.sum()
        f1s.append(2 * p * q / max(p + q, 1e-12))
    votes = np.stack([(P == c).sum(1) for c in range(10)], 1)
    best = None
    lo = 5 if m >= 9 else m // 2 + 1                 # 5..m as before; majority..m for small m
    for td in range(lo, m + 1):
        for tr in range(lo, m + 1):
            fl = r >= td
            target = np.where(votes.max(1) >= tr, votes.argmax(1), -1)
            lab = np.where(fl, target, noisy)
            kept = lab >= 0
            res = (lab[kept] != true[kept]).mean()
            if best is None or res < best[0]:
                best = (res, td, tr, int(kept.sum()), int((lab[kept] == true[kept]).sum()))
    acc = (P == true[:, None]).mean(0)
    return dict(auc=roc_auc_score(is_noisy, r), best_f1=max(f1s), member_true_acc=acc.mean(),
                best_residual_noise=best[0], at_TD=best[1], at_TR=best[2], retained=best[3], clean_retained=best[4])


out = pd.DataFrame([dict(model='published', **evaluate(P_pub)), dict(model=variant, **evaluate(P_new))])
print(f'F-MNIST {noise}% fold {fold}, members {members} ({m}), noisy {is_noisy.sum()} of {len(idx)}')
print(out.round(4).to_string(index=False))
