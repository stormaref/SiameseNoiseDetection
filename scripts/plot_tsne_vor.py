"""t-SNE visualization + class-overlap measures computed in the REAL feature space.

Rationale (revision, 2026-06): measuring class overlap on the 2-D t-SNE projection is
not a faithful difficulty measure (t-SNE distorts distances/volumes and is stochastic).
Instead we map BOTH datasets through ONE common, fixed, pretrained backbone (resnet50,
ImageNet) and measure class overlap in that high-dimensional feature space. t-SNE is then
used ONLY to visualize the same features. Reported measures (all in the real feature space):

  * VOR  (mean F2) : mean over class pairs of the mean per-dimension range-overlap ratio.
                     (The classic F2 is a PRODUCT over dims, which underflows to ~0 in high
                     dimensions and is not comparable across different dimensionalities; we
                     use the dimension-normalized mean so it stays in [0,1] and is comparable.)
  * N2            : ratio of summed intra-class to summed inter-class nearest-neighbor
                    distance. Lower => better separated (easier).
  * silhouette    : mean silhouette over true classes. Higher => better separated (easier).

For contrast we also print the OLD-style F2-product measured on the 2-D t-SNE coordinates.

Run (from the project root, so cached datasets in ./data are reused):
    conda run -n data python plot_tsne_vor.py
    conda run -n data python plot_tsne_vor.py --backbone resnet50 --n_samples 5000
"""

import argparse
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
FMNIST_CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetResult:
    name: str
    tsne_2d: np.ndarray
    labels: np.ndarray
    classes: list
    measures: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='t-SNE + real-feature-space class-overlap (VOR/N2/silhouette)')
    p.add_argument('--backbone', type=str, default='resnet50',
                   help='Common pretrained backbone for BOTH datasets (resnet50/resnet34/resnet18)')
    p.add_argument('--data_root', type=str, default='data', help='torchvision dataset root')
    p.add_argument('--n_samples', type=int, default=5000,
                   help='Total images per dataset (stratified by class)')
    p.add_argument('--output', type=str, default='figures/tsne_vor.pdf')
    p.add_argument('--dpi', type=int, default=600)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--perplexity', type=float, default=30.0)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--show', action='store_true')
    p.add_argument('--no_cifar', action='store_true')
    p.add_argument('--no_fmnist', action='store_true')
    return p.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_backbone(name: str, device: torch.device) -> torch.nn.Module:
    from torchvision import models
    weight_map = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
    }
    if name not in weight_map:
        raise ValueError(f'Unsupported backbone: {name}')
    ctor, weights = weight_map[name]
    model = ctor(weights=weights)
    model.fc = torch.nn.Identity()      # expose the penultimate feature vector
    model.eval()
    return model.to(device)


def make_transform(grayscale: bool) -> T.Compose:
    ops = []
    if grayscale:
        ops.append(T.Grayscale(num_output_channels=3))
    ops += [T.Resize(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def stratified_indices(targets: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_classes = int(targets.max()) + 1
    per_class = max(1, n_samples // num_classes)
    selected = []
    for c in range(num_classes):
        ci = np.where(targets == c)[0]
        if len(ci) == 0:
            continue
        selected.extend(rng.choice(ci, size=min(per_class, len(ci)), replace=False).tolist())
    return np.sort(np.array(selected, dtype=np.int64))


@torch.no_grad()
def extract_features(dataset, indices, backbone, device, batch_size, desc):
    feats, labels = [], []
    for k in tqdm(range(0, len(indices), batch_size), desc=desc, unit='batch'):
        batch_idx = indices[k:k + batch_size]
        imgs = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        f = backbone(imgs).flatten(1).cpu().numpy()
        feats.append(f)
        labels.extend([int(dataset[i][1]) for i in batch_idx])
    return np.concatenate(feats, axis=0), np.asarray(labels)


# ---------------------------- overlap measures ----------------------------
def _pair_overlap_ratios(Xi, Xj):
    mn_i, mx_i = Xi.min(0), Xi.max(0)
    mn_j, mx_j = Xj.min(0), Xj.max(0)
    overlap = np.maximum(0.0, np.minimum(mx_i, mx_j) - np.maximum(mn_i, mn_j))
    rng = np.maximum(mx_i, mx_j) - np.minimum(mn_i, mn_j)
    ratios = np.zeros_like(rng)
    np.divide(overlap, rng, out=ratios, where=rng > 0)
    return ratios


def vor(X, y, agg, desc='VOR'):
    classes = np.unique(y)
    pairs = [(a, b) for i, a in enumerate(classes) for b in classes[i + 1:]]
    vals = [agg(_pair_overlap_ratios(X[y == a], X[y == b]))
            for a, b in tqdm(pairs, desc=desc, unit='pair', leave=False)]
    return float(np.mean(vals))


def n2(X, y, desc='N2'):
    nn = NearestNeighbors(n_neighbors=min(50, len(X))).fit(X)
    dist, ind = nn.kneighbors(X)
    intra = inter = 0.0
    for p in tqdm(range(len(X)), desc=desc, unit='pt', leave=False):
        same = diff = None
        for q in range(1, ind.shape[1]):
            nb = ind[p, q]
            if same is None and y[nb] == y[p]:
                same = dist[p, q]
            if diff is None and y[nb] != y[p]:
                diff = dist[p, q]
            if same is not None and diff is not None:
                break
        if same is not None and diff is not None:
            intra += same
            inter += diff
    return intra / inter if inter > 0 else float('nan')


def analyze_dataset(name, dataset, grayscale, classes, backbone, device, args) -> DatasetResult:
    print(f'\n=== {name} ===')
    targets = np.asarray(dataset.targets)
    idx = stratified_indices(targets, args.n_samples, args.seed)
    X, y = extract_features(dataset, idx, backbone, device, args.batch_size,
                            desc=f'Extracting {args.backbone} features [{name}]')

    print(f'  computing real-space measures in {X.shape[1]}-d ...')
    measures = {
        'dim': X.shape[1],
        'VOR_mean_hi': vor(X, y, np.mean, desc=f'VOR(mean) [{name}]'),
        'VOR_prod_hi': vor(X, y, np.prod, desc=f'VOR(prod) [{name}]'),
        'N2_hi': n2(X, y, desc=f'N2 [{name}]'),
        'silhouette_hi': float(silhouette_score(X, y)),
    }

    print(f'  running t-SNE for visualization [{name}] ...')
    Z = TSNE(n_components=2, random_state=args.seed, perplexity=args.perplexity,
             init='pca', learning_rate='auto').fit_transform(X)
    measures['VOR_prod_tsne'] = vor(Z, y, np.prod, desc=f'VOR(prod,tSNE) [{name}]')
    measures['silhouette_tsne'] = float(silhouette_score(Z, y))

    for k, v in measures.items():
        print(f'    {k:<16}= {v:.4f}')
    return DatasetResult(name=name, tsne_2d=Z, labels=y, classes=classes, measures=measures)


# ---------------------------- plotting ----------------------------
def plot_panel(ax, res: DatasetResult):
    sc = ax.scatter(res.tsne_2d[:, 0], res.tsne_2d[:, 1], c=res.labels,
                    cmap='tab10', s=8, alpha=0.65, linewidths=0)
    m = res.measures
    ax.set_title(f'{res.name}\nVOR = {m["VOR_mean_hi"]:.4f}', fontsize=11)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(sc, ax=ax, ticks=range(len(res.classes)))
    cbar.ax.set_yticklabels(res.classes, fontsize=7)


def plot_figure(results, output_path, show, dpi):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        plot_panel(ax, res)
    fig.suptitle('t-SNE of pretrained-ResNet features (class overlap measured in real feature space)',
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.rcParams['pdf.compression'] = 0
    fig.savefig(output_path, dpi=dpi, pad_inches=0.15)
    print(f'\nSaved figure to {output_path}')
    plt.show() if show else plt.close(fig)


def print_summary(results):
    keys = ['dim', 'VOR_mean_hi', 'VOR_prod_hi', 'N2_hi', 'silhouette_hi',
            'VOR_prod_tsne', 'silhouette_tsne']
    print('\n================  COMMON pretrained backbone — real-feature-space overlap  ================')
    hdr = f'{"measure":<16}' + ''.join(f'{r.name:>16}' for r in results)
    print(hdr); print('-' * len(hdr))
    for k in keys:
        print(f'{k:<16}' + ''.join(f'{r.measures[k]:>16.4f}' for r in results))
    print('\nLower N2 / higher silhouette / lower VOR  =>  better separated (easier dataset).')
    print('VOR_prod_tsne is the OLD-style number (F2 product on 2-D t-SNE) shown only for contrast.')


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.no_cifar and args.no_fmnist:
        raise ValueError('At least one dataset must be enabled.')
    device = get_device()
    print(f'Using device: {device}')
    print(f'Building common pretrained backbone: {args.backbone}')
    backbone = build_backbone(args.backbone, device)

    results = []
    if not args.no_cifar:
        cifar = torchvision.datasets.CIFAR10(
            args.data_root, train=True, download=True, transform=make_transform(False))
        results.append(analyze_dataset('CIFAR-10', cifar, False, CIFAR10_CLASSES,
                                        backbone, device, args))
    if not args.no_fmnist:
        fmnist = torchvision.datasets.FashionMNIST(
            args.data_root, train=True, download=True, transform=make_transform(True))
        results.append(analyze_dataset('Fashion-MNIST', fmnist, True, FMNIST_CLASSES,
                                        backbone, device, args))

    print_summary(results)
    plot_figure(results, args.output, args.show, args.dpi)


if __name__ == '__main__':
    main()

# ---------------------------------------------------------------------------
# HOW TO RUN (from the project root, so ./data is reused and ./figures exists)
#
# Recommended (env activated -> live tqdm progress in the terminal):
#   conda activate data
#   python plot_tsne_vor.py --n_samples 5000
#
# Or call the env's python directly (also streams progress, no activation):
#   /Users/storm/miniconda3/envs/data/bin/python plot_tsne_vor.py --n_samples 5000
#
# Note: `conda run -n data python ...` works too but tends to BUFFER stdout,
#   so the tqdm bars may only appear at the very end.
#
# Useful flags:
#   --n_samples 10000        # more samples -> steadier numbers (slower)
#   --backbone resnet50      # common backbone for BOTH datasets (default)
#   --output figures/tsne_vor.pdf   # overwrite the paper figure (default path)
#   --no_fmnist  /  --no_cifar      # run a single dataset
# ---------------------------------------------------------------------------
