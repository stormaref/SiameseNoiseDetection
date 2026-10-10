"""Recompute every paper number that comes from the saved prediction CSVs and diff it against main.tex.

Usage: python paper_repro_check.py <main.tex> [--fmnist-ref 513d833]

Reads preds/<name>/*/fold*_analysis.csv as committed at a git ref (CIFAR-10, CIFAR-10N: main;
F-MNIST: --fmnist-ref, default 513d833 = the round-3 paper run, 'main' = the 2026-10 re-run) and
replays the paper's rules exactly as snd.evaluation.cleaner_metrics does:
  detection   flagged = mistakes >= TD, scored against the CSV's is_noisy flag;
  relabeling  a flagged sample takes the first class with >= TR member votes, else it is removed;
              score in {-2,-1,0,1,2} per flagged sample, averaged over the flagged samples;
  after       noisy/clean counts and residual rate as in CleanerMetricsMixin.analyze_parameters.
Covered: Table 2 (combined best results), the before/after bar chart, both TD grids, both TR grids,
the ensemble-size table and the 'ours' row of the Co-teaching table. Prints every mismatch.
"""
import argparse
import glob
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402

PREDS = {'cifar10n': 'cifar10n', 'c20': 'cifar10(20)', 'c30': 'cifar10(30)', 'c40': 'cifar10(40)',
         'f20': 'fmnist(20)', 'f30': 'fmnist(30)', 'f40': 'fmnist(40)'}
REPO = K.setup_repo('main')


def load(ref, key):
    d = K.extract_preds(REPO, ref, PREDS[key])
    t = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(d, '*', 'fold*_analysis.csv')))])
    P = np.stack(t['preds'].astype(str).str.split('|').map(lambda x: np.array(x, dtype=int)).values)
    D = dict(noisy=t['noisy_label'].to_numpy(), true=t['real_label'].to_numpy(), P=P,
             nz=t['is_noisy'].astype(str).str.lower().eq('true').to_numpy(),
             mistakes=t['mistakes'].to_numpy(), index=t['index'].to_numpy())
    # consistency of the saved columns with each other
    D['checks'] = dict(n=len(t), unique_index=len(np.unique(D['index'])) == len(t),
                       mistakes_ok=bool((D['mistakes'] == (P != D['noisy'][:, None]).sum(1)).all()),
                       flag_ok=bool((D['nz'] == (D['noisy'] != D['true'])).all()))
    return D


def detect(D, td):
    fl, nz = D['mistakes'] >= td, D['nz']
    tp, fp = (fl & nz).sum(), (fl & ~nz).sum()
    fn, tn = (~fl & nz).sum(), (~fl & ~nz).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(acc=(tp + tn) / len(nz), p=p, r=r, f1=2 * p * r / max(p + r, 1e-12), fpr=fp / (fp + tn))


def relabel(D, td, tr):
    fl, nz, true = D['mistakes'] >= td, D['nz'], D['true']
    votes = np.stack([(D['P'] == c).sum(1) for c in range(10)], 1)
    ok = (votes >= tr).any(1)
    new = np.argmax(votes >= tr, axis=1)                       # first class with >= tr votes
    rel, rem = fl & ok, fl & ~ok
    c = {'2': (rel & nz & (new == true)).sum(), '0': (rel & nz & (new != true)).sum(),
         '-2': (rel & ~nz & (new != true)).sum(), '1': (rem & nz).sum(), '-1': (rem & ~nz).sum()}
    score = (2 * c['2'] + c['1'] - c['-1'] - 2 * c['-2']) / max(fl.sum(), 1)
    noisy, clean = nz.sum(), (~nz).sum()
    clean += -c['-1'] - c['-2'] + c['2']
    noisy += -c['1'] + c['-2'] - c['2']
    count = c['-2'] + c['2'] + c['0']
    return dict(c=c, count=count, racc=100 * c['2'] / max(count, 1), score=score,
                after=100 * noisy / (noisy + clean), remaining=noisy + clean,
                noisy_after=noisy, clean_after=clean, noisy_before=nz.sum(), clean_before=(~nz).sum())


# ----------------------------------------------------------------------------- paper parsing
def strip_cmd(s, cmd):
    """Delete every \\cmd{..}{..}{..}-style call (brace-matched, optional [..] args)."""
    out, i = [], 0
    while True:
        j = s.find('\\' + cmd, i)
        if j < 0:
            return ''.join(out) + s[i:]
        out.append(s[i:j])
        k = j + len(cmd) + 1
        nargs = {'multirow': 3}.get(cmd, 1)
        for _ in range(nargs):
            while k < len(s) and s[k] in ' [':
                if s[k] == '[':
                    k = s.index(']', k) + 1
                else:
                    k += 1
            depth = 0
            while k < len(s):
                depth += {'{': 1, '}': -1}.get(s[k], 0)
                k += 1
                if depth == 0:
                    break
        i = k


def table_text(tex, label):
    j = tex.index('\\label{%s}' % label)
    a, b = tex.rfind('\\begin{table}', 0, j), tex.index('\\end{table}', j)
    return tex[a:b]


NUM = re.compile(r'-?\d[\d,]*\.?\d*')


def rows(text):
    body = text.split('\\midrule', 1)[1] if '\\midrule' in text else text
    body = strip_cmd(body, 'multirow').replace('\\textbf', '').replace('\\%', '')
    out = []
    for r in body.split('\\\\'):
        r = re.sub(r'\\[a-zA-Z]+', ' ', r)
        nums = [float(x.replace(',', '')) for x in NUM.findall(r)]
        if len(nums) >= 4:
            out.append(nums)
    return out


# ----------------------------------------------------------------------------- comparison
MISMATCH = []
TRUNCATED = []          # printed by truncation instead of rounding (not an error)


def cmp(where, got, want, tol):
    if abs(got - want) > tol:
        MISMATCH.append(f'{where}: paper {want:g}, CSV {got:.6g}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('--fmnist-ref', default='513d833')
    a = ap.parse_args()
    tex = open(a.tex).read()
    refs = {k: (a.fmnist_ref if k.startswith('f') else 'main') for k in PREDS}
    data = {k: load(refs[k], k) for k in PREDS}
    for k, D in data.items():
        print(k, refs[k], D['checks'])

    # Table 2 + TD/TR used everywhere: read the paper's own thresholds from Table 2
    t2 = table_text(tex, 'tab:combined-best-results')
    order = ['cifar10n', 'c20', 'c30', 'c40', 'f20', 'f30', 'f40']
    r2 = {}
    for line in t2.split('\\\\'):
        for name in ('Detection Threshold', 'Relabeling Threshold', 'Noise Accuracy', 'Noise Precision',
                     'Noise Recall', 'Noise F1-Score', 'Relabeling Score'):
            if name in line:
                cells = [c.strip() for c in line.split('&')[1:]]
                r2[name] = [float(NUM.findall(re.sub(r'\\[a-zA-Z]+', ' ', c))[0]) for c in cells]
    TD = dict(zip(order, map(int, r2['Detection Threshold'])))
    TR = dict(zip(order, map(int, r2['Relabeling Threshold'])))
    print('paper thresholds TD', TD, 'TR', TR)
    for i, k in enumerate(order):
        d, rl = detect(data[k], TD[k]), relabel(data[k], TD[k], TR[k])
        cmp(f'Table 2 {k} acc', d['acc'], r2['Noise Accuracy'][i], 6e-5)
        cmp(f'Table 2 {k} precision', d['p'], r2['Noise Precision'][i], 6e-5)
        cmp(f'Table 2 {k} recall', d['r'], r2['Noise Recall'][i], 6e-5)
        cmp(f'Table 2 {k} F1', d['f1'], r2['Noise F1-Score'][i], 6e-5)
        cmp(f'Table 2 {k} relabeling score', rl['score'], r2['Relabeling Score'][i], 6e-3)

    # bar chart
    j = tex.index('\\label{fig:combined-bars}')
    fig = tex[tex.rfind('\\begin{figure}', 0, j):j]
    plots = re.findall(r'coordinates \{([^}]*)\}', fig)
    noisy = [int(v) for v in re.findall(r'\([\d.]+,(\d+)\)', plots[0])]
    clean = [int(v) for v in re.findall(r'\([\d.]+,(\d+)\)', plots[1])]
    pct = [float(x) for x in re.findall(r'(?:Before|After)\\\\([\d.]+)\\%', fig)]
    for i, k in enumerate(order):
        rl = relabel(data[k], TD[k], TR[k])
        cmp(f'Bars {k} noisy before', rl['noisy_before'], noisy[2 * i], 0)
        cmp(f'Bars {k} noisy after', rl['noisy_after'], noisy[2 * i + 1], 0)
        cmp(f'Bars {k} clean before', rl['clean_before'], clean[2 * i], 0)
        cmp(f'Bars {k} clean after', rl['clean_after'], clean[2 * i + 1], 0)
        cmp(f'Bars {k} residual %', rl['after'], pct[2 * i + 1], 6e-3)

    # TD grids (6..10, TR not involved)
    for label, keys in (('tab:cifar-10-hyperparameter-analysis', ['cifar10n', 'c20', 'c30', 'c40']),
                        ('tab:fashion-mnist-hyperparameter-analysis', ['f20', 'f30', 'f40'])):
        rr = rows(table_text(tex, label))
        if len(rr) != 5 * len(keys):
            MISMATCH.append(f'{label}: parsed {len(rr)} rows, expected {5 * len(keys)}')
            continue
        for b, k in enumerate(keys):
            for row in rr[5 * b:5 * b + 5]:
                td = int(row[0])
                d = detect(data[k], td)
                for name, got, want in zip(('acc', 'P', 'R', 'F1'), (d['acc'], d['p'], d['r'], d['f1']), row[1:5]):
                    cmp(f'{label} {k} TD={td} {name}', got, want, 6e-5)
        best = {k: max(range(6, 11), key=lambda td: detect(data[k], td)['f1']) for k in keys}
        for k in keys:
            if best[k] != TD[k]:
                MISMATCH.append(f'{label} {k}: oracle TD* (argmax F1) is {best[k]}, paper uses {TD[k]}')

    # TR grids at the paper's TD
    for label, keys in (('tab:cifar10-relabeling_threshold_results', ['cifar10n', 'c20', 'c30', 'c40']),
                        ('tab:fashion-mnist-relabeling_threshold_results', ['f20', 'f30', 'f40'])):
        rr = rows(table_text(tex, label))
        if len(rr) != 5 * len(keys):
            MISMATCH.append(f'{label}: parsed {len(rr)} rows, expected {5 * len(keys)}')
            continue
        for b, k in enumerate(keys):
            for row in rr[5 * b:5 * b + 5]:
                tr = int(row[0])
                rl = relabel(data[k], TD[k], tr)
                got = [rl['c'][s] for s in ('-2', '-1', '0', '1', '2')] + [rl['count']]
                for name, g, w in zip(('-2', '-1', '0', '1', '2', 'count'), got, row[1:7]):
                    cmp(f'{label} {k} TR={tr} {name}', g, w, 0)
                if abs(rl['racc'] - row[7]) > 6e-3 and abs(np.floor(rl['racc'] * 100) / 100 - row[7]) < 1e-9:
                    TRUNCATED.append(f'{label} {k} TR={tr} accuracy {rl["racc"]:.4f} printed {row[7]}')
                else:
                    cmp(f'{label} {k} TR={tr} accuracy', rl['racc'], row[7], 6e-3)
                cmp(f'{label} {k} TR={tr} score', rl['score'], row[8], 6e-3)
                cmp(f'{label} {k} TR={tr} after%', rl['after'], row[9], 6e-3)
                cmp(f'{label} {k} TR={tr} remaining', rl['remaining'], row[10], 0)

    # ensemble size (CIFAR-10), averaged over all member subsets
    ens = table_text(tex, 'tab:ensemble-size')
    sizes = [1, 2, 4, 6, 8, 10]
    for k, lvl in (('c20', '20'), ('c30', '30'), ('c40', '40')):
        D = data[k]
        acc = {m: [] for m in sizes}
        for m in sizes:
            for S in itertools.combinations(range(10), m):
                r = (D['P'][:, S] != D['noisy'][:, None]).sum(1)
                f1s = []
                for td in range(1, m + 1):
                    fl = r >= td
                    tp = (fl & D['nz']).sum()
                    p, q = tp / max(fl.sum(), 1), tp / D['nz'].sum()
                    f1s.append(2 * p * q / max(p + q, 1e-12))
                fpr = ((r >= m) & ~D['nz']).sum() / (~D['nz']).sum()
                acc[m].append((roc_auc_score(D['nz'], r), max(f1s), fpr))
        mean = {m: np.mean(acc[m], 0) for m in sizes}
        for metric, col in (('ROC-AUC', 0), ('Best F1', 1), ('FPR at', 2)):
            line = [l for l in ens.split('\\\\') if metric in l and f'({lvl}\\%)' in l][0]
            want = [float(x) for x in NUM.findall(re.sub(r'\\[a-zA-Z]+|\(\\TD=m\'\)', ' ', line.split('&', 1)[1]))]
            for m, w in zip(sizes, want):
                cmp(f'ensemble-size {metric} {lvl}% m={m}', mean[m][col], w, 6e-4)

    # Co-teaching table, 'ours' row (CIFAR-10 20%)
    co = table_text(tex, 'tab:coteaching')
    line = [l for l in co.split('\\\\') if l.strip().startswith('Ours')]
    if line:
        want = [float(x.replace(',', '')) for x in NUM.findall(re.sub(r'\\[a-zA-Z]+', ' ', line[0].split('&', 1)[1]))]
        d, rl = detect(data['c20'], TD['c20']), relabel(data['c20'], TD['c20'], TR['c20'])
        D = data['c20']
        got = [d['p'], d['r'], d['f1'], d['fpr'], roc_auc_score(D['nz'], D['mistakes']), rl['remaining'], rl['after']]
        for name, g, w, tol in zip(('P', 'R', 'F1', 'FPR', 'AUC', 'retained', 'residual%'), got, want,
                                   (6e-4, 6e-4, 6e-4, 6e-4, 6e-4, 0, 0.06)):
            cmp(f'Co-teaching ours {name}', g, w, tol)
    else:
        MISMATCH.append('Co-teaching table: no ours row found')

    for k in order:                                  # AUCs for the ROC figures / text
        print(f'AUC {k}: {roc_auc_score(data[k]["nz"], data[k]["mistakes"]):.4f}')
    print(f'\n{len(TRUNCATED)} values truncated instead of rounded (consistent with the CSVs)')
    print(f'{len(MISMATCH)} mismatches')
    for m in MISMATCH:
        print('  ' + m)


if __name__ == '__main__':
    main()
