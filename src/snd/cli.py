"""Command-line entry point: detect noisy labels, then write a cleaned dataset.

Run as ``snd`` (installed console script) or ``python -m snd.cli``. Per-fold
predictions under ``preds/`` are reused when present, so re-running with different
``--mistakes_count`` / ``--relabel_threshold`` costs nothing but I/O.
"""
import argparse
import os

import torch

import snd.config as config
from snd.pipeline.cleaner import NoiseCleaner
from snd.utils import set_global_seed, CIFAR10_CLASSES, FashionMNIST_CLASSES

# Set global seed for reproducibility
set_global_seed(42)

# Datasets are referenced by attribute *name* so that snd.config can build them
# lazily -- selecting Fashion-MNIST must not download CIFAR-10.
DATASETS = {
    'cifar10': {
        'train': 'CIFAR10_TRAIN_DATASET',
        'test': 'CIFAR10_TEST_DATASET',
        'train_transform': 'CIFAR10_TRAIN_TRANSFORMS',
        'test_transform': 'CIFAR10_TEST_TRANSFORMS',
        'classes': CIFAR10_CLASSES,
        'params': {'20': 'CIFAR10_20_PARAMS',
                   '30': 'CIFAR10_30_PARAMS',
                   '40': 'CIFAR10_40_PARAMS'},
    },
    'cifar10n': {
        'train': 'CIFAR10_TRAIN_DATASET',
        'test': 'CIFAR10_TEST_DATASET',
        'train_transform': 'CIFAR10_TRAIN_TRANSFORMS',
        'test_transform': 'CIFAR10_TEST_TRANSFORMS',
        'classes': CIFAR10_CLASSES,
        'params': {'n': 'CIFAR10N_PARAMS'},
    },
    'fashionmnist': {
        'train': 'FashionMNIST_TRAIN_DATASET',
        'test': 'FashionMNIST_TEST_DATASET',
        'train_transform': 'FashionMNIST_TRAIN_TRANSFORMS',
        'test_transform': 'FashionMNIST_TEST_TRANSFORMS',
        'classes': FashionMNIST_CLASSES,
        'params': {'20': 'FashionMNIST_20_PARAMS',
                   '30': 'FashionMNIST_30_PARAMS',
                   '40': 'FashionMNIST_40_PARAMS'},
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset and noise configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and evaluate noise detection model')
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=sorted(DATASETS),
                        help='Dataset to use: cifar10, fashionmnist, or cifar10n')
    parser.add_argument('--noise_ratio',
                        type=str,
                        required=True,
                        help='Noise ratio: 20, 30, 40 (or n for cifar10n)')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results',
                        help='Directory to save results')
    parser.add_argument('--mistakes_count',
                        type=int,
                        default=8,
                        help='Number of mistakes to count as noise')
    parser.add_argument('--relabel_threshold',
                        type=int,
                        default=9,
                        help='Relabel threshold')

    return parser.parse_args()


def _resolve_params(dataset: str, noise_ratio: str) -> dict:
    """Look up the hyperparameter dict for a (dataset, noise ratio) pair."""
    available = DATASETS[dataset]['params']
    if dataset == 'cifar10n' and noise_ratio != 'n':
        print(f"Warning: For CIFAR-10N, noise ratio should be 'n'. "
              f"Ignoring provided value: {noise_ratio}")
        noise_ratio = 'n'
    if noise_ratio not in available:
        raise ValueError(f'Invalid noise ratio for {dataset}: {noise_ratio}. '
                         f'Choose from {", ".join(sorted(available))}.')
    return getattr(config, available[noise_ratio])


def get_dataset_config(args: argparse.Namespace) -> tuple:
    """Get dataset configuration based on command line arguments.

    Args:
        args: Command line arguments

    Returns:
        tuple: (train_dataset, test_dataset, train_transform, test_transform, classes, params)
    """
    entry = DATASETS[args.dataset]
    return (getattr(config, entry['train']),
            getattr(config, entry['test']),
            getattr(config, entry['train_transform']),
            getattr(config, entry['test_transform']),
            entry['classes'],
            _resolve_params(args.dataset, args.noise_ratio))


def get_raw_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    """Get the raw (untransformed) training dataset for the selected benchmark.

    Args:
        args: Command line arguments

    Returns:
        torch.utils.data.Dataset: Raw training dataset
    """
    return getattr(config, DATASETS[args.dataset]['train'])


def main() -> None:
    """Main function to run the noise detection and cleaning pipeline."""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get dataset configuration
    train_dataset, test_dataset, train_transform, test_transform, classes, params = get_dataset_config(
        args)
    mistakes_count = args.mistakes_count
    relabel_threshold = args.relabel_threshold
    # Initialize noise cleaner
    noise_cleaner = NoiseCleaner(
        dataset=train_dataset,
        transform=train_transform,
        augmented_transform=train_transform,
        **params
    )

    # Run noise cleaning pipeline
    print(
        f"Starting training for {args.dataset} with noise ratio {args.noise_ratio}")
    noise_cleaner.clean()

    # Save cleaned dataset
    dataset = get_raw_dataset(args)
    manual_cleaned = noise_cleaner.advanced_clean(
        dataset=dataset, mistakes_count=mistakes_count, relabel_threshold=relabel_threshold)
    noise_cleaner.save_cleaned_cifar_dataset_manual(
        manual_cleaned,
        args.output_dir,
        f'cleaned_{args.dataset}_{args.noise_ratio}_{mistakes_count}_{relabel_threshold}.pth'
    )


if __name__ == "__main__":
    main()
