"""Noise detection as a function of ensemble size m' (no retraining).

For every subset S of the m = 10 inner-ensemble members (all 1,023 of them), a sample's disagreement
count is the number of members in S whose prediction differs from its observed label. Subsetting
keeps each member's training fraction fixed, so it isolates the effect of ensemble size. Reported per
m', averaged over all subsets of that size: ROC-AUC of the disagreement score, the best F1 over the
detection threshold, and the false-positive rate at unanimity (T_D = m').

Usage:
    uv run python scripts/ensemble_size.py                      # CIFAR-10 at 20/30/40% IDN
    uv run python scripts/ensemble_size.py --settings fashionmnist_20 --output results/ensemble_size_fmnist.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import snd.config as config

SETTINGS = {'cifar10_20': 'CIFAR10_20_PARAMS', 'cifar10_30': 'CIFAR10_30_PARAMS',
            'cifar10_40': 'CIFAR10_40_PARAMS', 'fashionmnist_20': 'FashionMNIST_20_PARAMS',
            'fashionmnist_30': 'FashionMNIST_30_PARAMS', 'fashionmnist_40': 'FashionMNIST_40_PARAMS'}
SHOWN = (1, 2, 4, 6, 8, 10)


def load(params):
    """Member-error matrix (N x m) and ground-truth noise flags from the committed predictions."""
    df = pd.concat([pd.read_csv(params['prediction_path'].format(k))
                    for k in range(1, params['outer_folds_num'] + 1)], ignore_index=True)
    preds = np.array([list(map(int, s.split('|'))) for s in df['preds']])
    errors = preds != df['noisy_label'].to_numpy()[:, None]
    assert (errors.sum(1) == df['mistakes'].to_numpy()).all()
    return errors, df['is_noisy'].astype(str).str.lower().eq('true').to_numpy()


def detection(flag, is_noisy):
    tp, fp = np.sum(flag & is_noisy), np.sum(flag & ~is_noisy)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / is_noisy.sum()
    return (2 * prec * rec / (prec + rec) if tp else 0.0), fp / np.sum(~is_noisy)


def analyse(setting, errors, is_noisy):
    rows = []
    m = errors.shape[1]
    for k in range(1, m + 1):
        for subset in itertools.combinations(range(m), k):
            count = errors[:, subset].sum(1)
            best_f1 = max(detection(count >= t, is_noisy)[0] for t in range(1, k + 1))
            rows.append({'setting': setting, 'm': k, 'subset': '-'.join(map(str, subset)),
                         'auc': roc_auc_score(is_noisy, count / k), 'best_f1': best_f1,
                         'fpr_unanimous': detection(count >= k, is_noisy)[1]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--settings', nargs='+', default=['cifar10_20', 'cifar10_30', 'cifar10_40'],
                   choices=sorted(SETTINGS))
    p.add_argument('--output', default='results/ensemble_size.csv')
    args = p.parse_args()

    rows = []
    for setting in args.settings:
        errors, is_noisy = load(getattr(config, SETTINGS[setting]))
        print(f'{setting}: N={len(is_noisy)}, noise rate {is_noisy.mean():.4f}, m={errors.shape[1]}')
        rows += analyse(setting, errors, is_noisy)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)

    summary = df.groupby(['setting', 'm'])[['auc', 'best_f1', 'fpr_unanimous']].mean()
    print(f"\nMean over all member subsets of size m' (m' = {', '.join(map(str, SHOWN))}):")
    for metric, label in (('auc', 'ROC-AUC'), ('best_f1', 'Best F1'), ('fpr_unanimous', "FPR at T_D=m'")):
        for setting in args.settings:
            values = ' & '.join(f'{summary.loc[(setting, k), metric]:.3f}' for k in SHOWN)
            print(f'{label} ({setting}) & {values}')
    summary.to_csv(args.output.replace('.csv', '_summary.csv'))


if __name__ == '__main__':
    main()
