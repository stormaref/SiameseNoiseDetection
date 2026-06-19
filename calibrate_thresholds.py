"""Algorithm-4 (no-ground-truth) vs oracle threshold calibration.

Reviewer R3.3 asks how much worse the detection/relabel thresholds (TD, TR) become when chosen by
Algorithm 4 (Threshold Calibration without Clean Labels -- pick (TD, TR) by the accuracy of a lightweight
downstream classifier, trained on the cleaned set, on a single FIXED held-out reference subset scored
against its NOISY labels) instead of an oracle tuned on the true noise labels. The same held-out subset
is reused for every (TD, TR), so its noisy-label bias is identical across candidates and cancels in the
ranking -- this avoids validating on each cleaned set's own (shrinking, easier) split, which made the
old signal anti-correlate with true test accuracy.

The ensemble predictions are already saved (preds/<ds>/<arch>/fold{1..k}_analysis.csv, with the
is_noisy / real_label ground truth), so the ORACLE grid is computed offline with zero training; only the
Algorithm-4 grid trains a small classifier per (TD, TR) pair.

Usage:
    # offline, instant -- validates the pipeline and finds the oracle thresholds
    python calibrate_thresholds.py --dataset fashionmnist --noise_ratio 20 --mode oracle

    # trains ~20 lightweight classifiers (needs GPU) -- the part you run
    python calibrate_thresholds.py --dataset fashionmnist --noise_ratio 20 --mode algo4 --epochs 15 --patience 4

    # instant -- prints oracle vs Algorithm-4 thresholds + degradation
    python calibrate_thresholds.py --dataset fashionmnist --noise_ratio 20 --mode compare

    # all three in sequence
    python calibrate_thresholds.py --dataset fashionmnist --noise_ratio 20 --mode all
"""
import argparse
import glob
import json
import math
import os
import pickle
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')  # headless: FinalModelTester imports pyplot

import numpy as np
import pandas as pd

from models.utils import set_global_seed
from models.config import FashionMNIST_TRAIN_TRANSFORMS, FashionMNIST_TEST_TRANSFORMS
from runner import get_dataset_config

set_global_seed(42)


# --------------------------------------------------------------------------------------
# Prediction loading + the cleaning/metrics logic (validated this session against the real
# cleaned pickles: it reproduces them exactly). Mirrors cleaner.advanced_clean's rules
# (flag if mistakes >= TD; relabel to the smallest label with vote count >= TR; else discard)
# without that method's in-place target mutation / matplotlib side effects.
# --------------------------------------------------------------------------------------
def load_preds(params):
    """Concatenate the per-fold analysis CSVs into one row-per-sample DataFrame."""
    dfs = []
    for fold in range(1, params['inner_folds_num'] + 1):
        dfs.append(pd.read_csv(params['prediction_path'].format(fold)))
    return pd.concat(dfs, ignore_index=True)


def build_vote_hist(df, num_classes):
    """(N, num_classes) histogram of the m ensemble votes per sample."""
    votes = np.array([[int(x) for x in str(s).split('|')] for s in df['preds']])
    hist = (votes[:, :, None] == np.arange(num_classes)).sum(axis=1)
    return hist, votes.shape[1]


def evaluate_pair(TD, TR, hist, noisy, real, mistakes):
    """Ground-truth detection metrics + residual corruption for one (TD, TR) pair.

    Detection metrics depend only on TD (flagging); TR only affects relabel/discard and residual.
    Returns scalar metrics plus the per-row cleaned labels + retained mask (used to write pickles).
    """
    is_noisy = noisy != real
    flagged = mistakes >= TD

    meets = hist >= TR                       # (N, num_classes): which labels clear the relabel vote count
    has_consensus = meets.any(axis=1)
    relabel = meets.argmax(axis=1)           # first/smallest label clearing TR (matches advanced_clean's found[0])

    retained = (~flagged) | (flagged & has_consensus)
    relabeled = flagged & has_consensus
    discarded = flagged & ~has_consensus
    cleaned = np.where(flagged, relabel, noisy)

    resid = retained & (cleaned != real)
    residual_pct = 100.0 * resid.sum() / max(int(retained.sum()), 1)
    # Clean yield = fraction of ALL samples that end up retained with a correct label. Unlike residual%
    # (which is trivially minimised by discarding everything), this rewards retention AND correctness, so
    # it is the non-degenerate offline proxy for downstream accuracy.
    correct_retained = int((retained & (cleaned == real)).sum())
    clean_yield_pct = 100.0 * correct_retained / len(noisy)

    tp = int((flagged & is_noisy).sum())
    fp = int((flagged & ~is_noisy).sum())
    fn = int((~flagged & is_noisy).sum())
    tn = int((~flagged & ~is_noisy).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(noisy)

    return {
        'td': TD, 'tr': TR,
        'det_precision': round(precision, 4), 'det_recall': round(recall, 4),
        'det_f1': round(f1, 4), 'det_accuracy': round(accuracy, 4),
        'residual_pct': round(residual_pct, 4), 'clean_yield_pct': round(clean_yield_pct, 4),
        'retained': int(retained.sum()), 'relabeled': int(relabeled.sum()),
        'discarded': int(discarded.sum()),
        '_cleaned': cleaned, '_retained_mask': retained,
    }


def make_grid(m):
    """Algorithm 4 grid (main.tex Algorithm 4): TD in {ceil(0.6m)..m}, TR in {ceil(0.5m)..m}.
    (The manuscript prose previously capped TR at TD; corrected in revision to match the pseudocode
    and the deployed thresholds, e.g. FMNIST-20 TD=9/TR=10.)"""
    td_lo, tr_lo = math.ceil(0.6 * m), math.ceil(0.5 * m)
    return [(td, tr) for td in range(td_lo, m + 1) for tr in range(tr_lo, m + 1)]


def pick_oracle(rows):
    """Oracle = the (TD, TR) maximising clean yield (correctly-labelled retained fraction) on the
    true labels -- the non-degenerate offline proxy for best achievable downstream accuracy."""
    return max(rows, key=lambda r: r['clean_yield_pct'])


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------
def run_oracle(grid, hist, noisy, real, mistakes, out_dir):
    rows = [evaluate_pair(td, tr, hist, noisy, real, mistakes) for (td, tr) in grid]
    public = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
    pd.DataFrame(public).to_csv(os.path.join(out_dir, 'oracle_grid.csv'), index=False)
    o = pick_oracle(public)
    f1_td = max(public, key=lambda r: r['det_f1'])['td']
    print(f"\n[oracle] {len(public)} pairs -> oracle_grid.csv  (F1-optimal detection TD={f1_td})")
    print(f"[oracle] best by clean-yield: TD={o['td']} TR={o['tr']}  yield={o['clean_yield_pct']}%  "
          f"det_F1={o['det_f1']}  residual={o['residual_pct']}%  retained={o['retained']}")
    return public


def write_clean_pickle(train_dataset, df_index, retained_mask, cleaned, path, exclude_mask=None):
    """Stream {'data','label'} dicts (the format CleanDatasetLoader reads) for retained samples.
    If exclude_mask is given, those rows are held out (never written) so they can serve as a fixed
    evaluation set that is never trained on by any candidate."""
    data = np.asarray(train_dataset.data)  # FMNIST: (N, 28, 28) uint8
    keep = retained_mask.copy()
    if exclude_mask is not None:
        keep = keep & ~exclude_mask
    with open(path, 'wb') as f:
        for r in np.where(keep)[0]:
            entry = {'data': np.asarray(data[df_index[r]], dtype=np.uint8), 'label': int(cleaned[r])}
            pickle.dump(entry, f)


def eval_on_fixed_set(model, images, labels, transform, device, batch_size=512):
    """Accuracy of a trained model on a FIXED held-out reference set, scored against the labels
    provided (here: the held-out samples' NOISY labels). The same set/labels are reused for every
    (TD, TR), so the noisy-label bias is identical across candidates and cancels in the ranking --
    removing the confound of validating on each cleaned set's own (shrinking, easier) split. It still
    tracks the true objective because predicting a clean sample's noisy label == predicting its true label."""
    import torch
    from PIL import Image
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            imgs = images[i:i + batch_size]
            lbls = labels[i:i + batch_size]
            x = torch.stack([transform(Image.fromarray(im)) for im in imgs]).to(device)
            pred = model(x).argmax(1).cpu().numpy()
            correct += int((pred == lbls).sum())
            total += len(lbls)
    return correct / max(total, 1)


def run_algo4(grid, hist, noisy, real, mistakes, df_index, train_dataset, args, out_dir,
              heldout_pos, heldout_images, heldout_labels):
    # Heavy import deferred so oracle/compare need no torch/GPU.
    import gc
    import torch
    from models.final_model_tester import FinalModelTester

    rows = []
    pairs = grid[:args.limit] if args.limit else grid
    tmpdir = tempfile.mkdtemp(prefix='algo4_calib_')
    try:
        for (td, tr) in pairs:
            m = evaluate_pair(td, tr, hist, noisy, real, mistakes)
            pkl = os.path.join(tmpdir, f'clean_td{td}_tr{tr}.pkl')
            write_clean_pickle(train_dataset, df_index, m['_retained_mask'], m['_cleaned'], pkl,
                               exclude_mask=heldout_pos)

            tester = FinalModelTester(
                train_dataset_path=pkl,
                train_transform=FashionMNIST_TRAIN_TRANSFORMS,
                test_transform=FashionMNIST_TEST_TRANSFORMS,
                test='fmnist', val_ratio=0.1, patience=args.patience,
                smoothing=0.1, cnn_size=512,
            )
            tester.train(epochs=args.epochs)
            # NEW Algorithm-4 signal: accuracy on the FIXED held-out reference set (noisy labels)
            val_fixed = eval_on_fixed_set(tester.model, heldout_images, heldout_labels,
                                          FashionMNIST_TEST_TRANSFORMS, tester.device)
            val_cleaned = float(tester.best_val_accuracy)   # OLD (confounded) signal -- kept for comparison
            test_acc = float(tester.test())                 # true downstream accuracy (FMNIST test = GT) -> oracle signal
            rows.append({'td': td, 'tr': tr,
                         'val_fixed': round(val_fixed, 4), 'val_cleaned': round(val_cleaned, 4),
                         'test_accuracy': round(test_acc, 4),
                         'det_f1': m['det_f1'], 'residual_pct': m['residual_pct'],
                         'clean_yield_pct': m['clean_yield_pct'], 'retained': m['retained'],
                         'relabeled': m['relabeled'], 'discarded': m['discarded']})
            print(f"[algo4] TD={td} TR={tr}  val_fixed={val_fixed:.4f}  val_cleaned={val_cleaned:.4f}  "
                  f"test_acc={test_acc:.4f}  (retained={m['retained']})")

            os.remove(pkl)
            del tester
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'algo4_grid.csv'), index=False)
    best_fixed = max(rows, key=lambda r: r['val_fixed'])
    best_test = max(rows, key=lambda r: r['test_accuracy'])
    print(f"\n[algo4] {len(rows)} pairs -> algo4_grid.csv")
    print(f"[algo4] Algorithm-4 pick (max FIXED held-out acc): TD={best_fixed['td']} TR={best_fixed['tr']}  "
          f"val_fixed={best_fixed['val_fixed']}  test_acc={best_fixed['test_accuracy']}")
    print(f"[algo4] oracle pick       (max true test acc):     TD={best_test['td']} TR={best_test['tr']}  "
          f"test_acc={best_test['test_accuracy']}")
    return rows


def run_compare(grid, hist, noisy, real, mistakes, out_dir):
    oracle_csv = os.path.join(out_dir, 'oracle_grid.csv')
    algo4_csv = os.path.join(out_dir, 'algo4_grid.csv')
    offline = (pd.read_csv(oracle_csv).to_dict('records') if os.path.exists(oracle_csv)
               else run_oracle(grid, hist, noisy, real, mistakes, out_dir))
    if not os.path.exists(algo4_csv):
        print(f"[compare] {algo4_csv} not found -- run `--mode algo4` first.")
        return None
    a4_rows = pd.read_csv(algo4_csv).to_dict('records')

    algo4 = max(a4_rows, key=lambda r: r['val_fixed'])        # Algorithm 4: fixed held-out, no clean labels
    oracle = max(a4_rows, key=lambda r: r['test_accuracy'])   # oracle: true held-out test accuracy
    f1_td = max(offline, key=lambda r: r['det_f1'])['td']     # paper's stated TD criterion (context)
    keys = ('td', 'tr', 'test_accuracy', 'det_f1', 'residual_pct', 'clean_yield_pct')

    # Does each label-free signal track the true objective? (the point of the fixed-reference fix)
    vf = np.array([r['val_fixed'] for r in a4_rows])
    vc = np.array([r['val_cleaned'] for r in a4_rows])
    ta = np.array([r['test_accuracy'] for r in a4_rows])
    corr_fixed = float(np.corrcoef(vf, ta)[0, 1])
    corr_cleaned = float(np.corrcoef(vc, ta)[0, 1])

    comparison = {
        'oracle_by_test_acc': {k: oracle[k] for k in keys},
        'algorithm4_by_val_fixed': {**{k: algo4[k] for k in keys}, 'val_fixed': algo4['val_fixed']},
        'downstream_test_acc_degradation_pp': round(100 * (oracle['test_accuracy'] - algo4['test_accuracy']), 3),
        'signal_corr_with_test_acc': {'fixed_heldout_NEW': round(corr_fixed, 3),
                                      'cleaned_split_OLD': round(corr_cleaned, 3)},
        'reference': {'offline_F1_optimal_TD': f1_td},
    }
    with open(os.path.join(out_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n[compare] oracle (true test acc) vs Algorithm-4 (fixed held-out, no clean labels)")
    print(f"  oracle      : TD={oracle['td']} TR={oracle['tr']}  test_acc={oracle['test_accuracy']}  "
          f"det_F1={oracle['det_f1']}  residual={oracle['residual_pct']}%")
    print(f"  Algorithm-4 : TD={algo4['td']} TR={algo4['tr']}  test_acc={algo4['test_accuracy']}  "
          f"(picked by val_fixed={algo4['val_fixed']})  det_F1={algo4['det_f1']}  residual={algo4['residual_pct']}%")
    print(f"  downstream test-accuracy degradation: {comparison['downstream_test_acc_degradation_pp']} pp")
    print(f"  signal->test-acc correlation:  NEW fixed held-out r={corr_fixed:+.3f}   OLD cleaned-split r={corr_cleaned:+.3f}")
    return comparison


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Algorithm-4 vs oracle threshold calibration")
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--noise_ratio', default='20')
    p.add_argument('--mode', choices=['oracle', 'algo4', 'compare', 'all'], default='all')
    p.add_argument('--epochs', type=int, default=15, help='downstream epochs per (TD,TR) for Algorithm 4')
    p.add_argument('--patience', type=int, default=4, help='early-stopping patience for the downstream classifier')
    p.add_argument('--limit', type=int, default=0, help='train only the first N grid pairs (0=all; for smoke-testing the run)')
    p.add_argument('--output_dir', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join('results', f'calibration_{args.dataset}_{args.noise_ratio}')
    os.makedirs(out_dir, exist_ok=True)

    train_dataset, _, _, _, _, params = get_dataset_config(args)
    df = load_preds(params)
    noisy = df['noisy_label'].to_numpy(int)
    real = df['real_label'].to_numpy(int)
    mistakes = df['mistakes'].to_numpy(int)
    df_index = df['index'].to_numpy(int)
    num_classes = int(max(noisy.max(), real.max())) + 1
    hist, m = build_vote_hist(df, num_classes)
    grid = make_grid(m)
    # Fixed held-out reference set for Algorithm-4 selection (same samples for every candidate;
    # excluded from each candidate's training set, scored on their noisy labels). See run_algo4.
    rng = np.random.RandomState(123)
    heldout_pos = np.zeros(len(df), dtype=bool)
    heldout_pos[rng.choice(len(df), size=int(0.1 * len(df)), replace=False)] = True
    heldout_images = np.asarray(train_dataset.data)[df_index[heldout_pos]].astype(np.uint8)
    heldout_labels = noisy[heldout_pos]
    print(f"dataset={args.dataset} noise={args.noise_ratio}  N={len(df)}  m={m}  classes={num_classes}  "
          f"grid={len(grid)} pairs  out={out_dir}")
    print(f"fixed held-out reference: {int(heldout_pos.sum())} samples (scored on noisy labels)")

    if args.mode in ('oracle', 'all'):
        run_oracle(grid, hist, noisy, real, mistakes, out_dir)
    if args.mode in ('algo4', 'all'):
        run_algo4(grid, hist, noisy, real, mistakes, df_index, train_dataset, args, out_dir,
                  heldout_pos, heldout_images, heldout_labels)
    if args.mode in ('compare', 'all'):
        run_compare(grid, hist, noisy, real, mistakes, out_dir)


if __name__ == '__main__':
    main()
