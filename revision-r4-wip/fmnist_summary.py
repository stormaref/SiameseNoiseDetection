"""Paper-style F-MNIST numbers (oracle TD* = argmax F1 over 6..10, TR = 10) for the published and
the new ensemble, pooled over the outer folds where the new run has all 10 members.

Usage: python fmnist_summary.py <noise> <variant> <run_dir> [<run_dir> ...]
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

noise, variant = int(sys.argv[1]), sys.argv[2]
pred_dir = K.extract_preds(K.setup_repo('main'), PUB_REF, K.PROTOCOL[('fashionmnist', noise)]['preds'])
table = K.load_label_table(pred_dir)
paths = {}
for run in sys.argv[3:]:
    for p in glob.glob(os.path.join(run, 'r51_out', f'{variant}_o*_m*.npz')):
        o, m = map(int, re.search(r'_o(\d+)_m(\d+)\.npz$', p).groups())
        paths.setdefault(o, {})[m] = p
folds = sorted(o for o, ms in paths.items() if len(ms) == 10)
idx_all, P_new, P_pub = [], [], []
for fold in folds:
    idx = np.load(paths[fold][0])['outer_idx']
    P_new.append(np.stack([np.load(paths[fold][m])['probs'].astype(np.float32).argmax(1) for m in range(10)], 1))
    f = glob.glob(os.path.join(pred_dir, '**', f'fold{fold}_analysis.csv'), recursive=True)[0]
    df = pd.read_csv(f).set_index('index').loc[idx]
    P_pub.append(np.stack(df['preds'].astype(str).str.split('|').map(lambda x: np.array(x, dtype=int)).values))
    idx_all.append(idx)
idx = np.concatenate(idx_all)
noisy, true = table['noisy_label'].to_numpy()[idx], table['real_label'].to_numpy()[idx]
nz = noisy != true


def at(P, td, tr=10):
    r = (P != noisy[:, None]).sum(1)
    fl = r >= td
    votes = np.stack([(P == c).sum(1) for c in range(10)], 1)
    target = np.where(votes.max(1) >= tr, votes.argmax(1), -1)
    lab = np.where(fl, target, noisy)
    kept = lab >= 0
    tp = (fl & nz).sum()
    p, q = tp / max(fl.sum(), 1), tp / nz.sum()
    return dict(TD=td, acc=(fl == nz).mean(), precision=p, recall=q, f1=2 * p * q / max(p + q, 1e-12),
                noisy_after=int((lab[kept] != true[kept]).sum()), clean_after=int((lab[kept] == true[kept]).sum()),
                residual_pct=100 * (lab[kept] != true[kept]).mean())


rows = []
for name, P in [('published', np.concatenate(P_pub)), (variant, np.concatenate(P_new))]:
    best = max((at(P, td) for td in range(6, 11)), key=lambda e: e['f1'])
    rows.append(dict(model=name, **best))
print(f'F-MNIST {noise}%: folds {folds} (n={len(idx)}, noisy before {nz.sum()}, clean before {(~nz).sum()})')
print(pd.DataFrame(rows).round(4).to_string(index=False))
