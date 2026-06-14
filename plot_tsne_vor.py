"""t-SNE visualization with Volume of Overlapping Region (VOR / F2) analysis."""

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.config import *
from models.siamese import SiameseNetwork
from models.utils import set_global_seed
from runner import get_dataset_config

set_global_seed(42)


@dataclass
class DatasetResult:
    name: str
    tsne_2d: np.ndarray
    labels: np.ndarray
    classes: list[str]
    vor_matrix: np.ndarray
    overall_vor: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot t-SNE embeddings and VOR (F2) for CIFAR-10 and Fashion-MNIST')
    parser.add_argument(
        '--cifar_noise_ratio',
        type=str,
        default='20',
        help='Noise ratio for CIFAR-10 config: 20, 30, 40, or n for cifar10n')
    parser.add_argument(
        '--fmnist_noise_ratio',
        type=str,
        default='20',
        help='Noise ratio for Fashion-MNIST config: 20, 30, 40')
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5000,
        help='Number of images to sample per dataset (stratified by class)')
    parser.add_argument(
        '--output',
        type=str,
        default='figures/tsne_vor.pdf',
        help='Output path for the figure')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling and t-SNE')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for feature extraction')
    parser.add_argument(
        '--perplexity',
        type=float,
        default=30.0,
        help='t-SNE perplexity (must be < n_samples)')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the figure interactively')
    parser.add_argument(
        '--no_cifar',
        action='store_true',
        help='Skip CIFAR-10 panel')
    parser.add_argument(
        '--no_fmnist',
        action='store_true',
        help='Skip Fashion-MNIST panel')
    parser.add_argument(
        '--pairwise',
        action='store_true',
        help='Include pairwise VOR heatmaps below each t-SNE panel')
    return parser.parse_args()


def stratified_indices(labels: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    per_class = max(1, n_samples // num_classes)
    selected: list[int] = []

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            continue
        count = min(per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=count, replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected, dtype=np.int64)
    if len(selected) > n_samples:
        selected = rng.choice(selected, size=n_samples, replace=False)
    return np.sort(selected)


def get_labels(dataset) -> np.ndarray:
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    if hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(label)
    return np.array(labels)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: np.ndarray, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[int(self.indices[idx])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def extract_backbone_features(
        dataset,
        transform,
        n_samples: int,
        backbone: str,
        pretrained: bool,
        device: torch.device,
        batch_size: int,
        seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = get_labels(dataset)
    indices = stratified_indices(labels, n_samples, seed)
    subset = TransformedSubset(dataset, indices, transform)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SiameseNetwork(
        model=backbone,
        pre_trained=pretrained,
        trainable=False,
    ).to(device)
    model.eval()

    features: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc=f'Extracting {backbone} features'):
            images = images.to(device)
            batch_features = model.feature_extractor(images)
            if batch_features.ndim > 2:
                batch_features = batch_features.view(batch_features.size(0), -1)
            features.append(batch_features.cpu().numpy())
            sampled_labels.append(batch_labels.numpy())

    return np.concatenate(features, axis=0), np.concatenate(sampled_labels, axis=0)


def compute_pairwise_f2(
        points_i: np.ndarray,
        points_j: np.ndarray) -> float:
    """Volume of Overlapping Region (F2) for one class pair in 2D."""
    ratios = []
    for dim in range(points_i.shape[1]):
        min_i, max_i = points_i[:, dim].min(), points_i[:, dim].max()
        min_j, max_j = points_j[:, dim].min(), points_j[:, dim].max()

        overlap = max(0.0, min(max_i, max_j) - max(min_i, min_j))
        value_range = max(max_i, max_j) - min(min_i, min_j)
        ratios.append(overlap / value_range if value_range > 0 else 0.0)

    return float(np.prod(ratios))


def compute_vor_matrix(
        embedding_2d: np.ndarray,
        labels: np.ndarray,
        num_classes: int) -> np.ndarray:
    vor_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    for class_i in range(num_classes):
        mask_i = labels == class_i
        if not np.any(mask_i):
            continue
        points_i = embedding_2d[mask_i]

        for class_j in range(class_i + 1, num_classes):
            mask_j = labels == class_j
            if not np.any(mask_j):
                continue
            points_j = embedding_2d[mask_j]
            f2 = compute_pairwise_f2(points_i, points_j)
            vor_matrix[class_i, class_j] = f2
            vor_matrix[class_j, class_i] = f2

    return vor_matrix


def compute_overall_vor(vor_matrix: np.ndarray) -> float:
    upper = vor_matrix[np.triu_indices_from(vor_matrix, k=1)]
    if len(upper) == 0:
        return 0.0
    return float(np.mean(upper))


def run_tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    effective_perplexity = min(perplexity, max(5.0, (len(features) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=effective_perplexity,
        init='pca',
        learning_rate='auto',
    )
    return tsne.fit_transform(features)


def analyze_dataset(
        dataset_name: str,
        dataset,
        transform,
        classes: list[str],
        params: dict,
        n_samples: int,
        seed: int,
        batch_size: int,
        perplexity: float,
        device: torch.device) -> DatasetResult:
    pretrained = params.get('pre_trained', True)
    features, labels = extract_backbone_features(
        dataset=dataset,
        transform=transform,
        n_samples=n_samples,
        backbone=params['model'],
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    tsne_2d = run_tsne(features, seed=seed, perplexity=perplexity)
    num_classes = len(classes)
    vor_matrix = compute_vor_matrix(tsne_2d, labels, num_classes)
    overall_vor = compute_overall_vor(vor_matrix)

    print(f'\n=== {dataset_name} ===')
    print(f'Samples: {len(labels)}')
    print(f'Backbone: {params["model"]} (pretrained={pretrained})')
    print(f'Overall VOR (mean pairwise F2 in t-SNE space): {overall_vor:.4f}')

    return DatasetResult(
        name=dataset_name,
        tsne_2d=tsne_2d,
        labels=labels,
        classes=classes,
        vor_matrix=vor_matrix,
        overall_vor=overall_vor,
    )


def plot_tsne_panel(
        ax,
        tsne_2d: np.ndarray,
        labels: np.ndarray,
        classes: list[str],
        overall_vor: float,
        title: str) -> None:
    scatter = ax.scatter(
        tsne_2d[:, 0],
        tsne_2d[:, 1],
        c=labels,
        cmap='tab10',
        s=8,
        alpha=0.65,
        linewidths=0,
    )
    ax.set_title(f'{title}\nOverall VOR = {overall_vor:.4f}', fontsize=11)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(classes)))
    cbar.ax.set_yticklabels(classes, fontsize=7)


def plot_vor_heatmap(
        ax,
        vor_matrix: np.ndarray,
        classes: list[str],
        title: str) -> None:
    im = ax.imshow(vor_matrix, cmap='YlOrRd', vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)

    for i in range(len(classes)):
        for j in range(len(classes)):
            value = vor_matrix[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(
                j,
                i,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=5,
                color=text_color if i != j else 'gray',
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='F2 (VOR)')


def plot_figure(
        results: list[DatasetResult],
        output_path: str,
        show: bool,
        include_pairwise: bool) -> None:
    n_cols = len(results)
    if n_cols == 0:
        raise ValueError('No datasets selected for plotting.')

    n_rows = 2 if include_pairwise else 1
    fig_height = 10 if include_pairwise else 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, fig_height))
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)
    elif n_rows == 1:
        axes = np.array(axes).reshape(1, n_cols)

    for col_idx, result in enumerate(results):
        plot_tsne_panel(
            axes[0, col_idx],
            result.tsne_2d,
            result.labels,
            result.classes,
            result.overall_vor,
            f't-SNE: {result.name}',
        )
        if include_pairwise:
            plot_vor_heatmap(
                axes[1, col_idx],
                result.vor_matrix,
                result.classes,
                f'Pairwise VOR: {result.name}',
            )

    title = 't-SNE Embeddings and Volume of Overlapping Region (VOR / F2)'
    if not include_pairwise:
        title = 't-SNE Embeddings with Overall VOR'
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, pad_inches=0.15)
    print(f'Saved figure to {output_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_namespace(dataset: str, noise_ratio: str) -> argparse.Namespace:
    return argparse.Namespace(dataset=dataset, noise_ratio=noise_ratio)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.n_samples < 30:
        raise ValueError('--n_samples must be at least 30 for stable t-SNE.')
    if args.no_cifar and args.no_fmnist:
        raise ValueError('At least one dataset must be enabled.')

    device = get_device()
    print(f'Using device: {device}')

    results: list[DatasetResult] = []

    if not args.no_cifar:
        cifar_args = build_namespace('cifar10', args.cifar_noise_ratio)
        train_dataset, _, train_transform, _, classes, params = get_dataset_config(
            cifar_args)
        results.append(
            analyze_dataset(
                dataset_name='CIFAR-10',
                dataset=train_dataset,
                transform=train_transform,
                classes=classes,
                params=params,
                n_samples=args.n_samples,
                seed=args.seed,
                batch_size=args.batch_size,
                perplexity=args.perplexity,
                device=device,
            )
        )

    if not args.no_fmnist:
        fmnist_args = build_namespace('fashionmnist', args.fmnist_noise_ratio)
        train_dataset, _, train_transform, _, classes, params = get_dataset_config(
            fmnist_args)
        results.append(
            analyze_dataset(
                dataset_name='Fashion-MNIST',
                dataset=train_dataset,
                transform=train_transform,
                classes=classes,
                params=params,
                n_samples=args.n_samples,
                seed=args.seed + 1,
                batch_size=args.batch_size,
                perplexity=args.perplexity,
                device=device,
            )
        )

    plot_figure(results, args.output, args.show, include_pairwise=args.pairwise)


if __name__ == '__main__':
    main()

# conda run -n data python plot_tsne_vor.py \
#   --cifar_noise_ratio 20 \
#   --fmnist_noise_ratio 20 \
#   --n_samples 5000 \
#   --output figures/tsne_vor.pdf
#
# conda run -n data python plot_tsne_vor.py \
#   --pairwise \
#   --output figures/tsne_vor_pairwise.pdf
