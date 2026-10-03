"""R5.6 analysis: does the cleaning gain hold across downstream classifiers?

Reads every <job>.done written by r56_worker.py and reports, per (dataset, noise, arch):
test accuracy (mean +- std over seeds) for each training set, the gain of `ours` over
`noisy`, and the share of the achievable gain it recovers, (ours - noisy) / (oracle_clean - noisy).
`oracle_filtered` vs `oracle_clean` shows what removal alone costs.
"""
import glob
import json
import os

import pandas as pd

ORDER = ['noisy', 'ours', 'oracle_filtered', 'oracle_clean']


def load(out_dir):
    rows = []
    for path in glob.glob(os.path.join(out_dir, '*.done')):
        with open(path) as f:
            rows.append({k: v for k, v in json.load(f).items() if k != 'history'})
    return pd.DataFrame(rows)


def summarise(df):
    keys = ['dataset', 'noise', 'arch']
    acc = (df.groupby(keys + ['train_set'])['test_acc']
             .agg(['mean', 'std', 'count']).reset_index())
    acc['cell'] = acc.apply(lambda r: f"{100 * r['mean']:.2f} ± {100 * (r['std'] if r['count'] > 1 else 0):.2f}"
                            f" (n={int(r['count'])})", axis=1)
    table = acc.pivot_table(index=keys, columns='train_set', values='cell', aggfunc='first')
    table = table[[c for c in ORDER if c in table.columns]]

    means = acc.pivot_table(index=keys, columns='train_set', values='mean')
    gains = pd.DataFrame(index=means.index)
    if {'noisy', 'ours'} <= set(means.columns):
        gains['ours - noisy (pp)'] = 100 * (means['ours'] - means['noisy'])
    if {'noisy', 'ours', 'oracle_clean'} <= set(means.columns):
        gains['share of oracle gain'] = (means['ours'] - means['noisy']) / (means['oracle_clean'] - means['noisy'])
    if {'oracle_filtered', 'oracle_clean'} <= set(means.columns):
        gains['removal cost (pp)'] = 100 * (means['oracle_clean'] - means['oracle_filtered'])
    sets = (df.groupby(['dataset', 'noise', 'train_set'])[['n_train_total', 'label_noise_in_set']]
              .first().reset_index())
    return table, gains, sets


def report(out_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    df = load(out_dir)
    if df.empty:
        print('no finished jobs yet')
        return None
    df.to_csv(os.path.join(results_dir, 'r56_runs.csv'), index=False)
    table, gains, sets = summarise(df)
    table.to_csv(os.path.join(results_dir, 'r56_accuracy.csv'))
    gains.to_csv(os.path.join(results_dir, 'r56_gains.csv'))
    pd.set_option('display.width', 220)
    print('Training sets (size, share of wrong labels):')
    print(sets.round(4).to_string(index=False))
    print('\nTest accuracy (%), mean ± std over seeds:')
    print(table.to_string())
    print('\nGains:')
    print(gains.round(3).to_string())
    return df, table, gains
