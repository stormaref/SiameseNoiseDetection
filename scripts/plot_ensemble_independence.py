"""Post-hoc analysis of ensemble error independence (Assumption 4.4 / Theorem FP)."""

import argparse
import json
import os

from snd.pipeline.cleaner import NoiseCleaner
from snd.evaluation.ensemble_independence import EnsembleIndependenceAnalyzer
from snd.utils import set_global_seed
from snd.cli import get_dataset_config

set_global_seed(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyze ensemble error independence for Assumption 4.4 validation')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10n',
        choices=['cifar10', 'fashionmnist', 'cifar10n'],
        help='Dataset to use')
    parser.add_argument(
        '--noise_ratio',
        type=str,
        default='n',
        help='Noise ratio: 20, 30, 40 (or n for cifar10n)')
    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help='Detection threshold tau_d (defaults to config mistakes_count)')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for JSON/CSV/plots (auto-generated if omitted)')
    parser.add_argument(
        '--include_noisy',
        action='store_true',
        help='Include noisy samples (default: clean samples only)')
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip PDF plot generation (JSON/CSV only)')
    return parser.parse_args()


def print_summary(summary: dict) -> None:
    print('\n=== Ensemble Independence Analysis (Assumption 4.4) ===')
    print(f"Clean samples: {summary['num_clean_samples']}")
    print(
        'Misclassified clean samples (Cov computed on these): '
        f"{summary['num_misclassified_clean_samples']}"
    )
    print(f"Ensemble size m: {summary['num_models']}")
    print(
        f"Threshold tau_d: {summary['threshold']} ({summary['threshold_fraction']:.2f} m)")
    print(f"p_C over all clean samples: {summary['p_c']:.4f}")
    print(
        'p_C on misclassified clean subset: '
        f"{summary['p_c_on_misclassified_clean']:.4f}"
    )
    print(
        'Mean pairwise prediction disagreement (all clean): '
        f"{summary['mean_pairwise_prediction_disagreement']:.4f}"
    )
    print(
        'Mean off-diagonal Cov(Xi, Xj) [Assumption 4.4]: '
        f"{summary['mean_off_diagonal_error_covariance']:.6f}"
    )
    print(
        'Mean off-diagonal Corr(Xi, Xj): '
        f"{summary['mean_off_diagonal_error_correlation']:.6f}"
    )
    print(
        f"Normalized rho_hat (on misclassified clean): {summary['rho_hat']:.6f}")
    print(
        f"Empirical FPR at threshold (all clean): {summary['empirical_false_positive_rate']:.6f}")
    print(
        f"Theoretical bound (rho=0): {summary['theoretical_bound_rho_0']:.6e}")
    print(
        f"Theoretical bound (rho=rho_hat): {summary['theoretical_bound_rho_hat']:.6e}")

    step2 = summary['step2_check_p_c_0_5']
    print('\n--- Theorem FP Step 2 check (p_C = 0.5, main.tex L1077-1079) ---')
    print(
        f"p_C assumed: {step2['p_c_assumed']:.2f}  |  p_C(1-p_C): {step2['bernoulli_variance']:.4f}")
    print(f"rho_max = max Cov(X_i,X_j) / (p_C(1-p_C)): {step2['rho_max']:.4f}")
    print(f"rho_mean (normalized pairwise): {step2['rho_mean']:.4f}")
    print(
        'All pairs satisfy Cov(X_i,X_j) <= rho_max p_C(1-p_C): '
        f"{step2['all_pairs_satisfy_weak_dependence']}"
    )
    print(f"Empirical Var(X): {step2['empirical_var_x']:.4f}")
    print(f"Theoretical Var(X) bound: {step2['theoretical_var_bound']:.4f}")
    print(f"Var bound satisfied: {step2['var_bound_satisfied']}")
    print(
        'FP bound at rho_max, p_C=0.5: '
        f"{step2['theoretical_fp_bound_at_rho_max']:.6e}"
    )
    print('========================================================\n')


def main() -> None:
    args = parse_args()

    train_dataset, _, train_transform, _, _, params = get_dataset_config(args)
    threshold = args.threshold if args.threshold is not None else params['mistakes_count']

    if args.output_dir is None:
        args.output_dir = os.path.join(
            'results',
            f"independence_{args.dataset}_{args.noise_ratio}",
        )

    noise_cleaner = NoiseCleaner(
        dataset=train_dataset,
        transform=train_transform,
        augmented_transform=train_transform,
        **params,
    )

    predictions = noise_cleaner.read_predictions()
    analyzer = EnsembleIndependenceAnalyzer.from_rows(
        predictions,
        threshold=threshold,
        clean_only=not args.include_noisy,
    )

    summary = analyzer.save_results(args.output_dir, plot=not args.no_plots)
    print_summary(summary)

    compact_path = os.path.join(args.output_dir, 'summary_compact.json')
    compact = {
        key: summary[key]
        for key in [
            'num_clean_samples',
            'num_misclassified_clean_samples',
            'num_models',
            'threshold',
            'p_c',
            'p_c_on_misclassified_clean',
            'mean_pairwise_prediction_disagreement',
            'mean_off_diagonal_error_covariance',
            'mean_off_diagonal_error_correlation',
            'rho_hat',
            'empirical_false_positive_rate',
            'theoretical_bound_rho_0',
            'theoretical_bound_rho_hat',
        ]
    }
    compact['step2_check_p_c_0_5'] = {
        key: summary['step2_check_p_c_0_5'][key]
        for key in [
            'p_c_assumed',
            'bernoulli_variance',
            'rho_max',
            'rho_mean',
            'all_pairs_satisfy_weak_dependence',
            'empirical_var_x',
            'theoretical_var_bound',
            'var_bound_satisfied',
            'theoretical_fp_bound_at_rho_max',
        ]
    }
    with open(compact_path, 'w', encoding='utf-8') as f:
        json.dump(compact, f, indent=2)

    print(f"Saved results to {args.output_dir}")


if __name__ == '__main__':
    main()

# conda run -n data python plot_ensemble_independence.py \
#   --dataset cifar10n --noise_ratio n
