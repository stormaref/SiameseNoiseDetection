"""Convert R5.1 worker outputs (one variant, all outer folds) into the published preds format.

Usage: python r51_to_preds.py <r51_out_dir> <variant> <dataset> <noise> <dest_preds_dir>
Writes dest/fold{k}_analysis.csv (index,noisy_label,is_noisy,real_label,mistakes,label_pred,preds)
and dest/fold{k}_noisy_indices.csv (mistakes >= m), exactly as NoiseCleaner.process_predictions
and save_noisy_indices did, so every existing analysis/plot script runs on the re-run.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402

src, variant, dataset, noise, dest = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
# label_table also covers runs without published preds (Animal-10N folds, the F-MNIST 60% noise file)
table = K.label_table(K.setup_repo('main'), dataset, noise, 'main')
noisy, true, flag = table['noisy_label'].to_numpy(), table['real_label'].to_numpy(), table['is_noisy'].to_numpy()
runs = {}
for path in glob.glob(os.path.join(src, f'{variant}_o*_m*.npz')):
    outer, member = map(int, re.search(r'_o(\d+)_m(\d+)\.npz$', path).groups())
    runs.setdefault(outer, {})[member] = path
os.makedirs(dest, exist_ok=True)
for outer, members in sorted(runs.items()):
    paths = [members[m] for m in sorted(members)]
    loaded = [np.load(p) for p in paths]
    idx = loaded[0]['outer_idx']
    preds = np.stack([d['probs'].astype(np.float32).argmax(1) for d in loaded], 1)
    with open(os.path.join(dest, f'fold{outer}_analysis.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['index', 'noisy_label', 'is_noisy', 'real_label', 'mistakes',
                                          'label_pred', 'preds'])
        w.writeheader()
        flagged = []
        for row, i in enumerate(idx):
            p = preds[row]
            mistakes = int((p != noisy[i]).sum())
            values, counts = np.unique(p, return_counts=True)
            top = np.sort(-counts)
            label_pred = -1 if len(top) > 1 and top[0] == top[1] else int(values[np.argmax(counts)])
            w.writerow(dict(index=int(i), noisy_label=int(noisy[i]), is_noisy=bool(flag[i]),
                            real_label=int(true[i]), mistakes=mistakes, label_pred=label_pred,
                            preds='|'.join(map(str, p))))
            if mistakes >= len(paths):
                flagged.append(int(i))
    with open(os.path.join(dest, f'fold{outer}_noisy_indices.csv'), 'w', newline='') as f:
        csv.writer(f).writerow(flagged)
    print(f'outer fold {outer}: {len(paths)} members, {len(idx)} samples')
