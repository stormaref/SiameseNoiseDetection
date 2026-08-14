"""Empirical validation of Assumption 4.4 (ensemble error independence) for reviewer R3.2.

Assumption 4.4: on clean data s in Clean, define misclassification indicators
    X_j = 1{f_j(x) != y}  in {0, 1}
and study Cov(X_i, X_j) for i != j.

Covariance is computed on clean samples where at least one model misclassifies
(i.e. the subset of clean data where misclassification events occur).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Worst-case clean misclassification rate used in Theorem FP Step 2 checks (main.tex L1077).
STEP2_P_C_ASSUMED = 0.5


def _parse_preds(row: dict) -> np.ndarray:
    return np.array(str(row['preds']).split('|'), dtype=np.int32)


def _is_clean(row: dict) -> bool:
    return row['is_noisy'] == 'False'


@dataclass
class EnsembleIndependenceAnalyzer:
    """Analyze Cov(X_i, X_j) for misclassification indicators on clean data."""

    rows: list[dict]
    threshold: int
    clean_only: bool = True

    _error_matrix: np.ndarray | None = None
    _pred_matrix: np.ndarray | None = None
    _true_labels: np.ndarray | None = None
    _covariance_matrix: np.ndarray | None = None

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[dict],
        threshold: int,
        clean_only: bool = True,
    ) -> EnsembleIndependenceAnalyzer:
        return cls(list(rows), threshold=threshold, clean_only=clean_only)

    def _filtered_rows(self) -> list[dict]:
        if not self.clean_only:
            return self.rows
        return [row for row in self.rows if _is_clean(row)]

    def build_error_matrix(self) -> np.ndarray:
        """Return (n, m) matrix X with X[i,j] = 1{f_j(x_i) != y_i} on clean data."""
        if self._error_matrix is not None:
            return self._error_matrix

        filtered = self._filtered_rows()
        if not filtered:
            raise ValueError('No samples available after filtering.')

        preds_list = [_parse_preds(row) for row in filtered]
        m = len(preds_list[0])
        if any(len(p) != m for p in preds_list):
            raise ValueError('Inconsistent number of ensemble predictions across rows.')

        labels = np.array([int(row['real_label']) for row in filtered], dtype=np.int32)
        pred_matrix = np.stack(preds_list, axis=0)
        error_matrix = (pred_matrix != labels[:, None]).astype(np.float64)

        self._error_matrix = error_matrix
        self._pred_matrix = pred_matrix
        self._true_labels = labels
        return error_matrix

    def build_misclassified_clean_matrix(self) -> np.ndarray:
        """Return X restricted to clean samples with at least one misclassification."""
        x = self.build_error_matrix()
        misclassified_mask = np.sum(x, axis=1) >= 1
        if not np.any(misclassified_mask):
            raise ValueError('No misclassified clean samples found.')
        return x[misclassified_mask]

    @property
    def num_models(self) -> int:
        return self.build_error_matrix().shape[1]

    @property
    def num_clean_samples(self) -> int:
        return self.build_error_matrix().shape[0]

    @property
    def num_misclassified_clean_samples(self) -> int:
        return self.build_misclassified_clean_matrix().shape[0]

    def estimate_p_c(self) -> float:
        """E[X_j] over all clean samples (used for FPR / bounds)."""
        return float(np.mean(self.build_error_matrix()))

    def estimate_p_c_on_misclassified_clean(self) -> float:
        """E[X_j | clean sample misclassified by at least one model]."""
        return float(np.mean(self.build_misclassified_clean_matrix()))

    def pairwise_prediction_disagreement(self) -> np.ndarray:
        """Fraction of clean samples where model i and j predict different classes."""
        pred_matrix = self._pred_matrix
        if pred_matrix is None:
            self.build_error_matrix()
            pred_matrix = self._pred_matrix

        m = pred_matrix.shape[1]
        disagreement = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                rate = float(np.mean(pred_matrix[:, i] != pred_matrix[:, j]))
                disagreement[i, j] = rate
                disagreement[j, i] = rate
        return disagreement

    def mean_pairwise_prediction_disagreement(self) -> float:
        matrix = self.pairwise_prediction_disagreement()
        m = matrix.shape[0]
        if m < 2:
            return 0.0
        return float(np.mean(matrix[np.triu_indices(m, k=1)]))

    def pairwise_error_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        """Cov(X_i, X_j) on clean samples where at least one X_j = 1."""
        x = self.build_misclassified_clean_matrix()
        cov = np.cov(x.T, bias=False)
        std = np.sqrt(np.diag(cov))
        denom = np.outer(std, std)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        np.fill_diagonal(corr, 1.0)
        self._covariance_matrix = cov
        return cov, corr

    def mean_off_diagonal(self, matrix: np.ndarray) -> float:
        m = matrix.shape[0]
        if m < 2:
            return 0.0
        return float(np.mean(matrix[np.triu_indices(m, k=1)]))

    def estimate_rho(self) -> float:
        """rho_hat = mean_{i!=j} Cov(X_i, X_j) / (p_C (1 - p_C)) on misclassified clean data."""
        p_c = self.estimate_p_c_on_misclassified_clean()
        variance = p_c * (1.0 - p_c)
        if variance <= 0.0:
            return 0.0
        cov, _ = self.pairwise_error_covariance()
        return self.mean_off_diagonal(cov) / variance

    def check_theorem_fp_step2(self, p_c_assumed: float = STEP2_P_C_ASSUMED) -> dict:
        """Check Theorem FP Step 2 (main.tex L1077-1079) with fixed p_C.

        Verifies whether empirical pairwise covariances satisfy
            Cov(X_i, X_j) <= rho p_C(1-p_C)
        for rho = rho_max = max_{i!=j} Cov(X_i,X_j) / (p_C(1-p_C)), and whether
            Var(X) <= m p_C(1-p_C)(1 + rho_max m)
        holds for X = sum_j X_j on the same sample set.
        """
        cov, _ = self.pairwise_error_covariance()
        x_mis = self.build_misclassified_clean_matrix()
        m = self.num_models
        bernoulli_var = p_c_assumed * (1.0 - p_c_assumed)

        off_diag_indices = np.triu_indices(m, k=1)
        off_diag_cov = cov[off_diag_indices]
        rho_per_pair = off_diag_cov / bernoulli_var
        rho_max = float(np.max(rho_per_pair))
        rho_mean = float(np.mean(rho_per_pair))
        bound_at_rho_max = rho_max * bernoulli_var

        pairwise_checks = []
        for idx, (i, j) in enumerate(zip(*off_diag_indices)):
            cov_ij = float(off_diag_cov[idx])
            pairwise_checks.append({
                'model_i': int(i + 1),
                'model_j': int(j + 1),
                'cov_xi_xj': cov_ij,
                'rho_normalized': float(rho_per_pair[idx]),
                'bound_rho_max_p_c_1_minus_p_c': float(bound_at_rho_max),
                'satisfies_cov_le_rho_p_var': bool(cov_ij <= bound_at_rho_max + 1e-12),
            })

        x_total = np.sum(x_mis, axis=1)
        empirical_var_x = float(np.var(x_total, ddof=1))
        theoretical_var_bound = m * bernoulli_var * (1.0 + rho_max * m)

        return {
            'p_c_assumed': p_c_assumed,
            'bernoulli_variance': bernoulli_var,
            'rho_max': rho_max,
            'rho_mean': rho_mean,
            'all_pairs_satisfy_weak_dependence': all(
                pair['satisfies_cov_le_rho_p_var'] for pair in pairwise_checks
            ),
            'num_pairs': len(pairwise_checks),
            'num_pairs_violating': sum(
                1 for pair in pairwise_checks if not pair['satisfies_cov_le_rho_p_var']
            ),
            'pairwise_checks': pairwise_checks,
            'empirical_var_x': empirical_var_x,
            'theoretical_var_bound': theoretical_var_bound,
            'var_bound_satisfied': bool(empirical_var_x <= theoretical_var_bound + 1e-12),
            'theoretical_fp_bound_at_rho_max': self.theoretical_fp_bound(
                m, p_c_assumed, self.threshold, rho=rho_max),
        }

    def empirical_false_positive_rate(self, threshold: int | None = None) -> float:
        if threshold is None:
            threshold = self.threshold
        x = self.build_error_matrix()
        return float(np.mean(np.sum(x, axis=1) >= threshold))

    @staticmethod
    def theoretical_fp_bound(
        m: int,
        p_c: float,
        threshold: int,
        rho: float = 0.0,
    ) -> float:
        epsilon = threshold / m - p_c
        if epsilon <= 0.0:
            return float('nan')

        rho = max(0.0, rho)
        variance_term = 2.0 * p_c * (1.0 - p_c)
        if rho > 0.0:
            variance_term *= (1.0 + rho * m)

        exponent = -(m * epsilon ** 2) / (variance_term + (2.0 / 3.0) * epsilon)
        return float(np.exp(exponent))

    def sensitivity_analysis(
        self,
        rho_grid: np.ndarray | None = None,
    ) -> list[dict]:
        m = self.num_models
        p_c = self.estimate_p_c()
        rho_hat = self.estimate_rho()
        empirical_fpr = self.empirical_false_positive_rate()

        if rho_grid is None:
            rho_grid = np.array([0.0, 0.01, 0.05, 0.1, 0.2, max(rho_hat, 0.0), 0.5])
            rho_grid = np.unique(np.round(rho_grid, 6))

        rows = []
        for rho in rho_grid:
            rows.append({
                'rho': float(rho),
                'theoretical_bound': self.theoretical_fp_bound(
                    m, p_c, self.threshold, rho=float(rho)),
                'empirical_fpr': empirical_fpr,
            })
        return rows

    def summarize(self) -> dict:
        x = self.build_error_matrix()
        x_mis = self.build_misclassified_clean_matrix()
        cov, corr = self.pairwise_error_covariance()
        disagreement = self.pairwise_prediction_disagreement()
        p_c = self.estimate_p_c()
        p_c_mis = self.estimate_p_c_on_misclassified_clean()
        rho_hat = self.estimate_rho()
        step2_check = self.check_theorem_fp_step2(p_c_assumed=STEP2_P_C_ASSUMED)
        m = self.num_models
        threshold = self.threshold

        return {
            'num_clean_samples': self.num_clean_samples,
            'num_misclassified_clean_samples': self.num_misclassified_clean_samples,
            'num_samples': self.num_clean_samples,
            'num_models': m,
            'clean_only': self.clean_only,
            'covariance_computed_on': 'clean_samples_with_at_least_one_misclassification',
            'threshold': threshold,
            'threshold_fraction': threshold / m,
            'p_c': p_c,
            'p_c_on_misclassified_clean': p_c_mis,
            'p_c_variance': p_c * (1.0 - p_c),
            'p_c_variance_on_misclassified_clean': p_c_mis * (1.0 - p_c_mis),
            'mean_pairwise_prediction_disagreement': self.mean_pairwise_prediction_disagreement(),
            'mean_off_diagonal_error_covariance': self.mean_off_diagonal(cov),
            'mean_off_diagonal_error_correlation': self.mean_off_diagonal(corr),
            'rho_hat': rho_hat,
            'empirical_false_positive_rate': self.empirical_false_positive_rate(),
            'theoretical_bound_rho_0': self.theoretical_fp_bound(m, p_c, threshold, rho=0.0),
            'theoretical_bound_rho_hat': self.theoretical_fp_bound(m, p_c, threshold, rho=rho_hat),
            'mean_misclassification_count': float(np.mean(np.sum(x, axis=1))),
            'mean_misclassification_count_on_misclassified_clean': float(np.mean(np.sum(x_mis, axis=1))),
            'pairwise_prediction_disagreement': disagreement.tolist(),
            'error_covariance': cov.tolist(),
            'error_correlation': corr.tolist(),
            'sensitivity_analysis': self.sensitivity_analysis(),
            'step2_check_p_c_0_5': step2_check,
        }

    def plot_covariance_heatmap(self, output_path: str) -> None:
        cov, _ = self.pairwise_error_covariance()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cov, cmap='coolwarm', aspect='auto')
        ax.set_title(
            'Cov(Xi, Xj) on Clean Samples\nwith At Least One Misclassification'
        )
        ax.set_xlabel('Model index j')
        ax.set_ylabel('Model index i')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_sensitivity(self, output_path: str) -> None:
        rows = self.sensitivity_analysis()
        rhos = [row['rho'] for row in rows]
        bounds = [row['theoretical_bound'] for row in rows]
        empirical = rows[0]['empirical_fpr'] if rows else 0.0

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rhos, bounds, marker='o', label='Theorem FP bound')
        ax.axhline(empirical, color='tab:orange', linestyle='--',
                   label=f'Empirical FPR ({empirical:.4f})')
        ax.set_xlabel('Normalized correlation rho')
        ax.set_ylabel('False-positive rate / bound')
        ax.set_title('False-Positive Bound Sensitivity to rho')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_results(self, output_dir: str, plot: bool = True) -> dict:
        summary = self.summarize()
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        sensitivity_path = os.path.join(output_dir, 'sensitivity.csv')
        with open(sensitivity_path, 'w', encoding='utf-8') as f:
            f.write('rho,theoretical_bound,empirical_fpr\n')
            for row in summary['sensitivity_analysis']:
                f.write(
                    f"{row['rho']},{row['theoretical_bound']},{row['empirical_fpr']}\n"
                )

        step2_path = os.path.join(output_dir, 'step2_check_p_c_0_5.json')
        with open(step2_path, 'w', encoding='utf-8') as f:
            json.dump(summary['step2_check_p_c_0_5'], f, indent=2)

        if plot:
            self.plot_covariance_heatmap(
                os.path.join(output_dir, 'error_covariance_heatmap.pdf'))
            self.plot_sensitivity(
                os.path.join(output_dir, 'fp_bound_sensitivity.pdf'))

        return summary
