"""Generate a fixed, seeded IDN noise draw plus outer folds, stored in the repo.

The published Fashion-MNIST 60% predictions mix several noise draws (the IDN generator's
add_noise() ignores its seed, and that run was resumed across sessions), so its labels
cannot be reused. This writes one reproducible draw that every Kaggle run reads.

Usage: python make_fixed_noise.py fashionmnist 60 51 15
       (dataset, noise %, seed, number of outer folds) -> kaggle/data/<dataset><noise>_idn_seed<seed>.npz
"""
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, 'src'))
import snd_kaggle as K  # noqa: E402
from snd.data.instance_dependent import InstanceDependentNoiseAdder  # noqa: E402

dataset, noise, seed, n_outer = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ds = K.base_dataset(dataset, train=True)
true = np.asarray(ds.targets).copy()
x0, _ = ds[0]
image_size = int(np.prod(np.asarray(x0).shape))          # 28*28 for F-MNIST (grayscale PIL)
adder = InstanceDependentNoiseAdder(ds, image_size=image_size, ratio=noise / 100, num_classes=10)
noisy = adder.get_noisy_labels(norm_std=0.1, seed=seed)  # the seeded path; add_noise() ignores seed
skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
fold = np.zeros(len(true), dtype=np.int8)
for k, (_, held) in enumerate(skf.split(np.zeros(len(true)), noisy), start=1):
    fold[held] = k
out = os.path.join(HERE, 'data', f'{dataset}{noise}_idn_seed{seed}.npz')
np.savez_compressed(out, noisy_labels=noisy.astype(np.int8), true_labels=true.astype(np.int8),
                    outer_fold=fold)
print(f'{out}: noise rate {np.mean(noisy != true):.4f}, folds {np.bincount(fold)[1:].tolist()}')
