"""Train the downstream classifier on a cleaned training set and report its test accuracy.

The training set is cleaned from the committed per-fold predictions with the paper's rule: a sample
is flagged when at least T_D ensemble members disagree with its label, relabeled to the first class
that at least T_R members predict, and removed otherwise. The defaults are the oracle thresholds of
the paper. A PreAct-ResNet34 is then trained once per seed with the settings in
`snd.config.DOWNSTREAM_PARAMS`, and the mean and standard deviation over seeds are printed.

Usage:
    uv run python scripts/train_downstream.py --dataset fashionmnist --noise_ratio 20 --seeds 0 1 2 3 4 5 6 7 8 9
    uv run python scripts/train_downstream.py --dataset cifar10 --noise_ratio 30 --td 8 --tr 10 --seeds 0
"""
import argparse
import os
import pickle
import statistics

import numpy as np
import pandas as pd

import snd.config as config
from snd.cli import get_dataset_config
from snd.evaluation.final_model_tester import FinalModelTester
from snd.utils import set_global_seed

# Oracle thresholds (T_D, T_R) used in the paper
PAPER_THRESHOLDS = {
    ('cifar10', '20'): (10, 10), ('cifar10', '30'): (8, 10), ('cifar10', '40'): (8, 10),
    ('cifar10n', 'n'): (10, 10),
    ('fashionmnist', '20'): (9, 10), ('fashionmnist', '30'): (8, 10),
    ('fashionmnist', '40'): (7, 10), ('fashionmnist', '60'): (9, 11),
}


def parse_args():
    p = argparse.ArgumentParser(description='Downstream accuracy after cleaning with (T_D, T_R)')
    p.add_argument('--dataset', required=True, choices=sorted(config.DOWNSTREAM_PARAMS))
    p.add_argument('--noise_ratio', required=True, help='20, 30, 40, 60 (Fashion-MNIST) or n (cifar10n)')
    p.add_argument('--td', type=int, help='detection threshold (default: the paper\'s)')
    p.add_argument('--tr', type=int, help='relabeling threshold (default: the paper\'s)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--epochs', type=int, help='maximum epochs (default: the config value)')
    p.add_argument('--output_dir', default='results/downstream')
    return p.parse_args()


def load_predictions(params):
    """Per-sample rows of all outer folds (index, noisy/real label, mistakes, member votes)."""
    folds = [pd.read_csv(params['prediction_path'].format(k)) for k in range(1, params['outer_folds_num'] + 1)]
    return pd.concat(folds, ignore_index=True)


def clean(df, td, tr, num_classes=10):
    """Return (dataset indices, labels) of the cleaned set."""
    votes = np.stack(df['preds'].astype(str).str.split('|').map(lambda x: np.array(x, dtype=int)).values)
    hist = (votes[:, :, None] == np.arange(num_classes)).sum(axis=1)
    flagged = df['mistakes'].to_numpy() >= td
    meets = hist >= tr
    keep = ~flagged | meets.any(axis=1)
    labels = np.where(flagged, meets.argmax(axis=1), df['noisy_label'].to_numpy())
    return df['index'].to_numpy()[keep], labels[keep], flagged


def main():
    args = parse_args()
    ns = argparse.Namespace(dataset=args.dataset, noise_ratio=args.noise_ratio)
    train_dataset, _, _, _, _, params = get_dataset_config(ns)
    default_td, default_tr = PAPER_THRESHOLDS.get((args.dataset, ns.noise_ratio), (None, None))
    td, tr = args.td or default_td, args.tr or default_tr
    if td is None or tr is None:
        raise SystemExit('pass --td and --tr for this setting')

    df = load_predictions(params)
    index, labels, flagged = clean(df, td, tr)
    real = df.set_index('index')['real_label'].reindex(index).to_numpy()
    print(f'{args.dataset} {args.noise_ratio}: T_D={td}, T_R={tr}; flagged {flagged.sum():,}, '
          f'kept {len(index):,} of {len(df):,}, label noise in the cleaned set {np.mean(labels != real):.2%}')

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f'{args.dataset}_{args.noise_ratio}_{td}_{tr}'
    pkl = os.path.join(args.output_dir, f'cleaned_{tag}.pkl')
    images = np.asarray(train_dataset.data)
    with open(pkl, 'wb') as f:
        for i, y in zip(index, labels):
            pickle.dump({'data': np.asarray(images[i], dtype=np.uint8), 'label': int(y)}, f)

    d = config.DOWNSTREAM_PARAMS[args.dataset]
    accuracies = []
    for seed in args.seeds:
        set_global_seed(seed)
        tester = FinalModelTester(pkl, d['train_transform'], d['test_transform'],
                                  train_batch_size=d['train_batch_size'], val_batch_size=512,
                                  test_batch_size=512, lr=d['lr'], warmup_epochs=d['warmup_epochs'],
                                  patience=d['patience'], weight_decay=d['weight_decay'],
                                  use_lr_scheduler=False, smoothing=d['smoothing'], test=d['test'],
                                  val_ratio=d['val_ratio'], seed=seed)
        tester.model_checkpoint_path = os.path.join(args.output_dir, f'best_{tag}_s{seed}.pth')
        tester.train(epochs=args.epochs or d['epochs'])
        accuracies.append(100 * tester.test())
        print(f'seed {seed}: test accuracy {accuracies[-1]:.2f}%')

    spread = f' ± {statistics.stdev(accuracies):.2f}' if len(accuracies) > 1 else ''
    print(f'{tag}: {statistics.mean(accuracies):.2f}%{spread} over {len(accuracies)} run(s)')


if __name__ == '__main__':
    main()
