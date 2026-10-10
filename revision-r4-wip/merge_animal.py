"""Collect the 100 Animal-10N re-run members (10 outer x 10 inner) into one folder of symlinks.

Usage: python merge_animal.py <dest_dir>
Fold 1 members 0-1 exist twice (CPU path in snd-animal10n-f1, GPU path in snd-anm-gpucheck): the GPU
copies are used, like every other member. Fold 3 m2 and fold 7 m2 were trained twice by overlapping
sessions; the first session's copy is used.
"""
import glob
import os
import re
import sys

ORDER = ['snd-anm-gpucheck', 'snd-anm-o1o10', 'snd-anm-o2-3', 'snd-anm-o4-5', 'snd-anm-o6-7', 'snd-anm-o8-9',
         'snd-anm-tail-a', 'snd-anm-tail-b', 'snd-anm-tail-c', 'snd-anm-rest-d', 'snd-anm-rest-e',
         'snd-anm-rest-f', 'snd-anm-last']
here = os.path.dirname(os.path.abspath(__file__))
dest = sys.argv[1]
os.makedirs(dest, exist_ok=True)
chosen = {}
for run in ORDER:
    for p in sorted(glob.glob(os.path.join(here, 'kaggle-results', run, 'r51_out', 'siamese_o*_m*.npz'))):
        o, m = map(int, re.search(r'_o(\d+)_m(\d+)\.npz$', p).groups())
        chosen.setdefault((o, m), p)
missing = [(o, m) for o in range(1, 11) for m in range(10) if (o, m) not in chosen]
if missing:
    sys.exit(f'missing members: {missing}')
for (o, m), p in chosen.items():
    link = os.path.join(dest, f'siamese_o{o}_m{m}.npz')
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(p, link)
print(f'{len(chosen)} members linked into {dest}')
