"""Generate qualitative examples figure for noise detection and relabeling."""

import argparse
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from snd.pipeline.cleaner import NoiseCleaner
from snd.utils import set_global_seed
from snd.cli import get_dataset_config, get_raw_dataset

set_global_seed(42)

CASE_ORDER = ['A', 'B', 'C']
CASE_TITLES = {
    'A': 'Detected & corrected',
    'B': 'Missed noise',
    'C': 'False alarm',
}


@dataclass
class SampleCase:
    index: int
    noisy_label: int
    real_label: int
    mistakes: int
    relabeled_label: int | None
    consensus_votes: int
    case: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot qualitative examples for noise detection and relabeling')
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
        '--mistakes_count',
        type=int,
        default=None,
        help='Number of model disagreements required to flag a sample as noisy '
             '(defaults to config mistakes_count)')
    parser.add_argument(
        '--relabel_threshold',
        type=int,
        default=None,
        help='Minimum vote count required for consensus relabeling (default: 9)')
    parser.add_argument(
        '--num_examples',
        type=int,
        default=4,
        help='Number of examples to show per case')
    parser.add_argument(
        '--output',
        type=str,
        default='figures/qualitative_cifar10n.pdf',
        help='Output path for the figure')
    parser.add_argument(
        '--dpi',
        type=int,
        default=600,
        help='Output resolution for rasterized PDF content')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sample selection')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the figure interactively')
    return parser.parse_args()


def get_consensus_label(preds: np.ndarray, relabel_threshold: int) -> int | None:
    unique, counts = np.unique(preds, return_counts=True)
    found = unique[counts >= relabel_threshold]
    return int(found[0]) if len(found) > 0 else None


def vote_margin(preds: np.ndarray, label: int) -> int:
    return int(np.sum(preds == label))


def classify_sample(
        row: dict,
        mistakes_count: int,
        relabel_threshold: int) -> SampleCase | None:
    is_noisy = row['is_noisy'] == 'True'
    real_label = int(row['real_label'])
    noisy_label = int(row['noisy_label'])
    mistakes = int(row['mistakes'])
    preds = np.array(str(row['preds']).split('|'), dtype=np.int32)
    relabeled_label = get_consensus_label(preds, relabel_threshold)
    detected = mistakes >= mistakes_count

    if is_noisy and detected and relabeled_label == real_label:
        case = 'A'
    elif is_noisy and not detected:
        case = 'B'
    elif not is_noisy and detected:
        case = 'C'
    else:
        return None

    return SampleCase(
        index=int(row['index']),
        noisy_label=noisy_label,
        real_label=real_label,
        mistakes=mistakes,
        relabeled_label=relabeled_label,
        consensus_votes=vote_margin(
            preds, relabeled_label) if relabeled_label is not None else 0,
        case=case,
    )


def collect_cases(
        predictions: list[dict],
        mistakes_count: int,
        relabel_threshold: int) -> dict[str, list[SampleCase]]:
    cases: dict[str, list[SampleCase]] = {'A': [], 'B': [], 'C': []}
    for row in predictions:
        sample = classify_sample(row, mistakes_count, relabel_threshold)
        if sample is not None:
            cases[sample.case].append(sample)
    return cases


def rank_cases(cases: list[SampleCase], case: str) -> list[SampleCase]:
    if case == 'A':
        return sorted(
            cases,
            key=lambda s: (s.mistakes, s.consensus_votes),
            reverse=True,
        )
    return sorted(cases, key=lambda s: s.mistakes, reverse=True)


def select_samples(
        cases: list[SampleCase],
        case: str,
        seed: int,
        num_examples: int) -> list[SampleCase]:
    ranked = rank_cases(cases, case)
    pool_size = max(num_examples, len(ranked) // 10)
    pool = ranked[:pool_size]

    rng = random.Random(seed + ord(case))
    if len(pool) <= num_examples:
        chosen = pool
    else:
        chosen = rng.sample(pool, num_examples)

    return sorted(chosen, key=lambda s: s.mistakes, reverse=True)


def class_name(classes: list[str], label: int) -> str:
    return classes[label]


def disagreement_text(mistakes: int, num_models: int) -> str:
    return f'Disagree: {mistakes}/{num_models}'


def build_caption_lines(
        sample: SampleCase,
        case_key: str,
        classes: list[str],
        num_models: int) -> list[str]:
    given = class_name(classes, sample.noisy_label)
    true = class_name(classes, sample.real_label)
    disagree = disagreement_text(sample.mistakes, num_models)

    if case_key == 'A':
        relabeled = class_name(classes, sample.relabeled_label)
        return [
            f'Given: {given}  |  True: {true}',
            f'Relabeled: {relabeled}  |  {disagree}',
        ]
    if case_key == 'B':
        return [
            f'Given: {given}  |  True: {true}',
            f'Missed  |  {disagree}',
        ]
    return [
        f'Label: {true} (clean)',
        f'Flagged  |  {disagree}',
    ]


def plot_qualitative_figure(
        dataset,
        selected: dict[str, list[SampleCase]],
        classes: list[str],
        num_examples: int,
        num_models: int,
        output_path: str,
        dpi: int,
        show: bool) -> None:
    row_count = len(CASE_ORDER)
    col_count = num_examples

    fig_width = 2.5 * col_count
    fig_height = 3.0 * row_count + 0.5

    fig = plt.figure(figsize=(fig_width, fig_height))

    height_ratios = []
    for case_idx in range(row_count):
        height_ratios.extend([0.4, 6.0, 1.15])
        if case_idx < row_count - 1:
            height_ratios.append(0.55)

    grid_rows = len(height_ratios)

    side_margin = 0.07
    outer_gs = GridSpec(
        1,
        1,
        figure=fig,
        left=side_margin,
        right=1 - side_margin,
        top=0.90,
        bottom=0.05,
    )

    gs = GridSpecFromSubplotSpec(
        grid_rows,
        col_count,
        subplot_spec=outer_gs[0, 0],
        height_ratios=height_ratios,
        wspace=0.30,
        hspace=0.18,
    )

    grid_row = 0
    for case_idx, case_key in enumerate(CASE_ORDER):
        title_row = grid_row
        image_row = grid_row + 1
        caption_row = grid_row + 2
        grid_row += 3
        if case_idx < row_count - 1:
            grid_row += 1

        title_ax = fig.add_subplot(gs[title_row, :])
        title_ax.axis('off')
        title_ax.text(
            0.5,
            0.5,
            f'Case {case_key}: {CASE_TITLES[case_key]}',
            ha='center',
            va='center',
            fontsize=11,
            fontweight='bold',
        )

        for col_idx, sample in enumerate(selected[case_key]):
            img_ax = fig.add_subplot(gs[image_row, col_idx])
            img, _ = dataset[sample.index]
            img_ax.imshow(np.array(img), interpolation='nearest')
            img_ax.set_xticks([])
            img_ax.set_yticks([])
            img_ax.set_aspect('equal')
            for spine in img_ax.spines.values():
                spine.set_visible(False)

            cap_ax = fig.add_subplot(gs[caption_row, col_idx])
            cap_ax.axis('off')
            caption_lines = build_caption_lines(
                sample, case_key, classes, num_models)
            cap_ax.text(
                0.5,
                0.5,
                '\n'.join(caption_lines),
                ha='center',
                va='center',
                fontsize=8,
                linespacing=1.5,
            )

    fig.suptitle(
        'Qualitative Examples of Noise Detection and Relabeling',
        fontsize=13,
        y=0.97,
        ha='center',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.rcParams['pdf.compression'] = 0
    fig.savefig(output_path, dpi=dpi, pad_inches=0.15)
    print(f"Saved figure to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_selected_samples(
        selected: dict[str, list[SampleCase]],
        classes: list[str]) -> None:
    for case_key in CASE_ORDER:
        print(f'Case {case_key}:')
        for sample in selected[case_key]:
            print(
                f"  index={sample.index}, "
                f"given={class_name(classes, sample.noisy_label)}, "
                f"true={class_name(classes, sample.real_label)}, "
                f"mistakes={sample.mistakes}, "
                f"relabeled={sample.relabeled_label}"
            )


def main() -> None:
    args = parse_args()

    if args.num_examples < 1:
        raise ValueError('--num_examples must be at least 1')

    train_dataset, _, train_transform, _, classes, params = get_dataset_config(
        args)
    raw_dataset = get_raw_dataset(args)
    num_models = params['inner_folds_num']
    mistakes_count = (
        args.mistakes_count
        if args.mistakes_count is not None
        else params['mistakes_count']
    )
    relabel_threshold = (
        args.relabel_threshold
        if args.relabel_threshold is not None
        else 9
    )

    noise_cleaner = NoiseCleaner(
        dataset=train_dataset,
        transform=train_transform,
        augmented_transform=train_transform,
        **params,
    )

    predictions = noise_cleaner.read_predictions()
    cases = collect_cases(predictions, mistakes_count, relabel_threshold)

    for case_key in CASE_ORDER:
        print(f"Case {case_key} candidates: {len(cases[case_key])}")

    missing = [
        case_key for case_key in CASE_ORDER
        if len(cases[case_key]) < args.num_examples
    ]
    if missing:
        raise RuntimeError(
            f"Not enough candidates for case(s) {missing} "
            f"(need {args.num_examples} each). "
            f"Try lowering --num_examples, or adjust --mistakes_count "
            f"(currently {mistakes_count}) / --relabel_threshold "
            f"(currently {relabel_threshold})."
        )

    selected = {
        case_key: select_samples(
            cases[case_key], case_key, args.seed, args.num_examples)
        for case_key in CASE_ORDER
    }

    print_selected_samples(selected, classes)
    plot_qualitative_figure(
        raw_dataset,
        selected,
        classes,
        args.num_examples,
        num_models,
        args.output,
        args.dpi,
        args.show,
    )


if __name__ == '__main__':
    main()

# conda run -n data python plot_qualitative_examples.py --mistakes_count 8 --output figures/qualitative_cifar10n.pdf --seed 913
