"""Cleaned CIFAR-10N training labels (DetectAndRelabel at TD, TR) from the re-run members, as the
`given_labels` text for r56_worker (train_set 'given').

Usage: python c10n_given_labels.py <TD> <TR> <out.txt> <run_dir> [<run_dir> ...]
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402
import r56_worker as W  # noqa: E402

td, tr, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
table = K.load_label_table(K.extract_preds(K.setup_repo('main'), 'main', K.PROTOCOL[('cifar10n', 0)]['preds']))
noisy, true = table['noisy_label'].to_numpy(), table['real_label'].to_numpy()
paths = {}
for run in sys.argv[4:]:
    for p in glob.glob(os.path.join(run, 'r51_out', 'siamese_o*_m*.npz')):
        o, m = map(int, re.search(r'_o(\d+)_m(\d+)\.npz$', p).groups())
        paths.setdefault(o, {})[m] = p
assert sorted(paths) == list(range(1, 11)) and all(len(v) == 10 for v in paths.values()), paths.keys()
labels = np.full(len(table), -2)
for fold, members in paths.items():
    idx = np.load(members[0])['outer_idx']
    P = np.stack([np.load(members[m])['probs'].astype(np.float32).argmax(1) for m in range(10)], 1)
    r = (P != noisy[idx, None]).sum(1)
    votes = np.stack([(P == c).sum(1) for c in range(10)], 1)
    target = np.where(votes.max(1) >= tr, votes.argmax(1), -1)
    labels[idx] = np.where(r >= td, target, noisy[idx])
assert (labels >= -1).all(), 'some samples are in no outer fold'
kept = labels >= 0
print(f'TD {td} TR {tr}: kept {kept.sum()}, removed {(~kept).sum()}, relabelled {(kept & (labels != noisy)).sum()}, '
      f'residual {100 * (labels[kept] != true[kept]).mean():.4f}%')
text = W.encode_labels(labels)
assert (W.decode_labels(text, len(labels)) == labels).all()
open(out, 'w').write(text)
print(out, len(text), 'chars')
