"""Aggregate the committed per-fold prediction CSVs into the README results tables.

Every number in the README's "Results" section comes from this script -- it reads only
``preds/<dataset>/<arch>/fold*_analysis.csv`` (ground truth included) and replays the
detection/correction decision rule offline, so it needs no GPU and no retraining.

Metric definitions match ``scripts/calibrate_thresholds.py``:

* detection P/R/F1  -- flagged (``mistakes >= TD``) vs. the true ``is_noisy`` flag
* residual noise %  -- retained samples whose (possibly corrected) label is still wrong
* clean yield %     -- share of the *original* dataset that is retained *and* correctly labelled
* relabeling score  -- the ground-truth-free {-2,-1,0,1,2} score, normalised per detection

Usage:
    uv run python scripts/summarize_results.py                  # markdown tables for the README
    uv run python scripts/summarize_results.py --td 8 --tr 9    # a different operating point
    uv run python scripts/summarize_results.py --sweep cifar10_30
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter
from dataclasses import dataclass

# (label, preds directory, whether the CSVs carry usable ground-truth labels)
BENCHMARKS: dict[str, tuple[str, str, bool]] = {
    "cifar10_20": ("CIFAR-10, IDN 20%", "preds/cifar10(20)/resnet50", True),
    "cifar10_30": ("CIFAR-10, IDN 30%", "preds/cifar10(30)/resnet50", True),
    "cifar10_40": ("CIFAR-10, IDN 40%", "preds/cifar10(40)/resnet50", True),
    "cifar10n": ("CIFAR-10N (real)", "preds/cifar10n/resnet50", True),
    "fmnist_20": ("Fashion-MNIST, IDN 20%", "preds/fmnist(20)/resnet34", True),
    "fmnist_30": ("Fashion-MNIST, IDN 30%", "preds/fmnist(30)/resnet34", True),
    "fmnist_40": ("Fashion-MNIST, IDN 40%", "preds/fmnist(40)/resnet34", True),
    "fmnist_60": ("Fashion-MNIST, IDN 60%", "preds/fmnist(60)/resnet34", True),
    # ANIMAL-10N has no clean labels, so detection P/R/F1 are undefined for it.
    "animal10n": ("ANIMAL-10N", "preds/animal10n/efficientnetv2", False),
}


@dataclass
class Result:
    name: str
    samples: int
    folds: int
    ensemble: int
    noise_pct: float
    precision: float
    recall: float
    f1: float
    fpr: float
    relabeled: int
    discarded: int
    retained_pct: float
    noise_before_pct: float
    residual_pct: float
    clean_yield_pct: float
    score: float


def load_folds(directory: str) -> tuple[list[dict], int, int]:
    """Read every ``fold*_analysis.csv`` in `directory` into one list of rows."""
    files = sorted(glob.glob(os.path.join(directory, "fold*_analysis.csv")))
    rows: list[dict] = []
    for path in files:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    ensemble = len(rows[0]["preds"].split("|")) if rows else 0
    return rows, len(files), ensemble


def evaluate(rows: list[dict], td: int, tr: int) -> tuple[dict, int]:
    """Replay the detect -> relabel/remove rule at thresholds (`td`, `tr`)."""
    tp = fp = fn = tn = 0
    relabeled = discarded = retained = 0
    correct_before = correct_after = 0
    score = detected = 0

    for row in rows:
        is_noisy = row["is_noisy"] == "True"
        real_label = int(row["real_label"])
        mistakes = int(row["mistakes"])
        votes = Counter(int(p) for p in str(row["preds"]).split("|"))

        correct_before += not is_noisy

        if mistakes < td:
            tn += not is_noisy
            fn += is_noisy
            retained += 1
            correct_after += not is_noisy
            continue

        tp += is_noisy
        fp += not is_noisy
        detected += 1

        consensus = sorted(label for label, count in votes.items() if count >= tr)
        if consensus:
            new_label = consensus[0]
            relabeled += 1
            retained += 1
            correct_after += new_label == real_label
            if is_noisy:
                score += 2 if new_label == real_label else 0
            elif new_label != real_label:
                score -= 2
        else:
            discarded += 1
            score += 1 if is_noisy else -1

    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    stats = dict(
        samples=total,
        noise_pct=100 * (tp + fn) / total,
        precision=100 * precision,
        recall=100 * recall,
        f1=100 * f1,
        fpr=100 * fp / (fp + tn) if fp + tn else 0.0,
        relabeled=relabeled,
        discarded=discarded,
        retained_pct=100 * retained / total,
        noise_before_pct=100 * (1 - correct_before / total),
        residual_pct=100 * (1 - correct_after / retained) if retained else 0.0,
        clean_yield_pct=100 * correct_after / total,
        score=score / detected if detected else 0.0,
    )
    return stats, detected


def summarize(key: str, td: int, tr: int) -> Result | None:
    name, directory, has_truth = BENCHMARKS[key]
    if not os.path.isdir(directory):
        return None
    rows, folds, ensemble = load_folds(directory)
    stats, _ = evaluate(rows, td, tr)
    if not has_truth:
        # Without clean labels every ground-truth-derived number is meaningless.
        for field in ("noise_pct", "precision", "recall", "f1", "fpr",
                      "noise_before_pct", "residual_pct", "clean_yield_pct", "score"):
            stats[field] = float("nan")
    return Result(name=name, folds=folds, ensemble=ensemble, **stats)


def markdown_table(results: list[Result]) -> str:
    head = ("| Dataset | Noise | Precision | Recall | F1 | FPR | Relabeled | Removed | "
            "Retained | Residual noise | Clean yield | Score |")
    rule = "|" + "---|" * 12
    lines = [head, rule]
    for r in results:
        def fmt(value: float, suffix: str = "") -> str:
            return "n/a" if value != value else f"{value:.2f}{suffix}"
        lines.append(
            f"| {r.name} | {fmt(r.noise_pct, '%')} | {fmt(r.precision)} | {fmt(r.recall)} | "
            f"**{fmt(r.f1)}** | {fmt(r.fpr)} | {r.relabeled:,} | {r.discarded:,} | "
            f"{r.retained_pct:.2f}% | {fmt(r.residual_pct, '%')} | {fmt(r.clean_yield_pct, '%')} | "
            f"{fmt(r.score)} |"
        )
    return "\n".join(lines)


def sweep_table(key: str, tr: int) -> str:
    name, directory, _ = BENCHMARKS[key]
    rows, _, ensemble = load_folds(directory)
    lines = [f"| T_D | Precision | Recall | F1 | Retained | Residual noise | Clean yield |",
             "|" + "---|" * 7]
    for td in range(1, ensemble + 1):
        s, _ = evaluate(rows, td, tr)
        lines.append(
            f"| {td} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | "
            f"{s['retained_pct']:.2f}% | {s['residual_pct']:.2f}% | {s['clean_yield_pct']:.2f}% |"
        )
    return f"### {name} -- detection threshold sweep (T_R = {tr})\n\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--td", type=int, default=10, help="detection threshold (ensemble disagreements)")
    parser.add_argument("--tr", type=int, default=9, help="relabeling threshold (ensemble agreement)")
    parser.add_argument("--sweep", choices=sorted(BENCHMARKS), help="print a T_D sweep for one benchmark")
    args = parser.parse_args()

    if args.sweep:
        print(sweep_table(args.sweep, args.tr))
        return

    results = [r for r in (summarize(k, args.td, args.tr) for k in BENCHMARKS) if r]
    missing = [k for k in BENCHMARKS if not os.path.isdir(BENCHMARKS[k][1])]
    print(f"Operating point: T_D = {args.td}, T_R = {args.tr}\n")
    print(markdown_table(results))
    if missing:
        print(f"\nskipped (no predictions on disk): {', '.join(missing)}")


if __name__ == "__main__":
    main()
