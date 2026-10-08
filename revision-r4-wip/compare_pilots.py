"""Compare F-MNIST 20% fold-1 pilots at equal ensemble size (members 0..k-1).

Usage: python compare_pilots.py <k> <run_dir> [<run_dir> ...]
"""
import glob
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402

# F-MNIST 'published' = the round-3 paper run (513d833); main now holds the 2026-10 re-run
PUB_REF = '513d833'
import r51_analysis as A  # noqa: E402

k = int(sys.argv[1])
tmp = tempfile.mkdtemp()
val = {}
for run in sys.argv[2:]:
    for path in glob.glob(os.path.join(run, 'r51_out', '*.npz')):
        name = os.path.basename(path)
        m = re.match(r'(.+)_o(\d+)_m(\d+)\.npz', name)
        if int(m.group(3)) >= k:
            continue
        os.symlink(os.path.abspath(path), os.path.join(tmp, name))
        done = json.load(open(path[:-4] + '.done'))
        val.setdefault(m.group(1), []).append((done['best_val_acc'], done['outer_acc_true'], done['epochs'], done['minutes']))
repo = K.setup_repo('main')
table = K.load_label_table(K.extract_preds(repo, PUB_REF, K.PROTOCOL[('fashionmnist', 20)]['preds']))
scores, thresholds, members, boot = A.analyse(tmp, table, min_members=k)
s = scores[scores.score == 'disagreement'][['variant', 'm', 'auc', 'best_f1']]
for v, rows in sorted(val.items()):
    r = np.array(rows)
    print(f'{v:16s} n={len(r)} inner val acc {r[:, 0].mean():.2f}  member true acc {r[:, 1].mean():.3f}  '
          f'epochs {r[:, 2].mean():.0f}  min/member {r[:, 3].mean():.0f}')
print(s.sort_values('best_f1', ascending=False).to_string(index=False))
