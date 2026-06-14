# Label Noise Correction with Siamese Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

Real-world training sets contain mislabeled examples, and the noise is often **instance-dependent** — harder, more ambiguous samples are more likely to be wrong. This framework **detects** those mislabeled samples and **corrects** them (relabel or remove) using an ensemble of Siamese networks trained with nested cross-validation, scoring every decision *without* access to ground-truth labels.


## 🔑 Key Features
- **Siamese architecture** — a twin-branch network jointly trained with a contrastive loss (on embedding pairs) and cross-entropy, over a configurable backbone (ResNet, PreAct-ResNet, VGG, DLA, EfficientNetV2, …).
- **Nested cross-validation** — stratified outer/inner folds prevent data leakage during detection.
- **Ensemble disagreement** — a sample is flagged when enough held-out ensemble members disagree with its current label.
- **Consensus relabeling** — flagged samples are relabeled when the ensemble agrees on an alternative class, otherwise removed.
- **Ground-truth-free quality metric** — the *relabeling score* grades each correction in {-2,-1,0,1,2}, so cleaning quality can be measured on real datasets where the truth is unknown.
- **Multiple datasets** — CIFAR-10, Fashion-MNIST (synthetic 20/30/40% noise), and CIFAR-10N (real human label noise).


## ⚙️ How It Works

The pipeline is a three-layer nesting (`runner.py` → `cleaner.py` → `detector.py` → `siamese.py`/`trainer.py`):

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1. Orchestrate | `NoiseCleaner` (`models/cleaner.py`) | Injects synthetic noise, runs the **outer** k-fold loop, then produces the cleaned dataset. |
| 2. Detect | `NoiseDetector` (`models/detector.py`) | Per outer fold, trains an **inner** ensemble of Siamese models via `StratifiedKFold`. Each sample is scored by every model that did *not* train on it; the number of disagreements with its (noisy) label is its **`mistakes`** score. |
| 3. Model | `SiameseNetwork` (`models/siamese.py`) | Backbone → contrastive embedding head → classifier, trained by `Trainer` with a weighted **contrastive + cross-entropy** objective. |

**Decision rule:**
- A sample is flagged **noisy** when `mistakes ≥ mistakes_count` (default `8`).
- A flagged sample is **relabeled** when some class appears `≥ relabel_threshold` times across the ensemble's predictions (default `9`); otherwise it is **removed**.


## 🚀 Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which provisions Python 3.11 and all packages from the pinned `uv.lock`.

```bash
uv sync                    # create .venv and install pinned dependencies
uv sync --extra notebook   # also install ipykernel/jupyter for the notebooks
```

PyTorch 2.4 is installed automatically; CUDA/MPS are used when available and fall back to CPU. Run anything inside the environment with `uv run`, e.g. `uv run python runner.py …`.


## 🏋️ Usage

Datasets download automatically via `torchvision` into `data/` on first run.

```bash
# CIFAR-10, 30% synthetic instance-dependent noise
uv run python runner.py --dataset cifar10 --noise_ratio 30 --output_dir results/cifar10_30

# Fashion-MNIST, 40% synthetic noise
uv run python runner.py --dataset fashionmnist --noise_ratio 40 --output_dir results/fmnist_40

# CIFAR-10N, real-world human label noise
uv run python runner.py --dataset cifar10n --noise_ratio n --output_dir results/cifar10n
```

`runner.py` selects the matching config, runs detection per outer fold, applies the relabel/remove decisions, and writes the cleaned dataset to `--output_dir`.

### CLI arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset` | ✓ | — | `cifar10`, `fashionmnist`, or `cifar10n` |
| `--noise_ratio` | ✓ | — | `20`, `30`, `40` (synthetic) or `n` (CIFAR-10N) |
| `--output_dir` | | `results/` | Where the cleaned dataset is saved |
| `--mistakes_count` | | `8` | Disagreement threshold to flag a sample as noisy |
| `--relabel_threshold` | | `9` | Ensemble agreement needed to relabel rather than remove |

> **No retraining needed.** Per-fold predictions are committed under `preds/`. If they exist for a config, `runner.py` skips training and regenerates a cleaned dataset directly from the cached predictions — so you can sweep `--mistakes_count`/`--relabel_threshold` cheaply.


## 🛠️ Configuration

All hyperparameters live in `models/config.py` as per-dataset/per-noise-level dictionaries. `runner.py` picks one by `--dataset`/`--noise_ratio` and splats it into `NoiseCleaner`.

| Dataset | Config dicts |
|---------|--------------|
| CIFAR-10 | `CIFAR10_20_PARAMS`, `CIFAR10_30_PARAMS`, `CIFAR10_40_PARAMS` |
| Fashion-MNIST | `FashionMNIST_20_PARAMS`, `FashionMNIST_30_PARAMS`, `FashionMNIST_40_PARAMS` |
| CIFAR-10N | `CIFAR10N_PARAMS` |

Each dict controls the backbone architecture, outer/inner fold counts, training settings (batch size, learning rate, epochs, contrastive weighting), embedding size, detection/relabel thresholds, and output paths. Edit these dicts to change behavior — the CLI deliberately exposes only the flags above.


## 📊 Reproducing the paper figures

Aggregated results and analysis are in the notebooks; the standalone scripts regenerate individual figures from the committed prediction CSVs:

```bash
uv run python plot_tsne_vor.py                 # t-SNE / Voronoi embeddings
uv run python plot_ensemble_independence.py    # ensemble independence analysis
uv run python plot_qualitative_examples.py     # qualitative detection/relabel examples
```


## 📂 Repository Structure
```
.
├── models/
│   ├── cleaner.py            # Orchestrator: noise injection, outer folds, cleaning
│   ├── detector.py          # Inner ensemble training + mistakes scoring
│   ├── trainer.py           # Joint contrastive + cross-entropy training loop
│   ├── siamese.py           # Siamese network (backbone + embedding + classifier)
│   ├── config.py            # Per-dataset/per-noise hyperparameter dicts
│   ├── noise.py / dataset.py / cifar10n.py   # Noise adders and dataset wrappers
│   ├── tta_cleaner.py       # Test-time-augmentation cleaning variant (notebook use)
│   ├── resnet.py / preact.py / dla.py / cnn.py / predefined.py   # Backbones
│   └── …                    # Testers, predictors, contrastive loss, utils
├── preds/                   # Committed per-fold prediction CSVs (enable retraining-free cleaning)
├── results/                 # Cleaned datasets and experiment outputs
├── plot_*.py                # Standalone figure-generation scripts
├── runner.py                # CLI entry point
├── main.ipynb / paper.ipynb # Experimentation and aggregated results
├── pyproject.toml           # Project metadata and dependencies (uv)
└── uv.lock                  # Pinned, reproducible lockfile
```

`*.pth`, `data/`, `cleaned/`, and `figures/` are gitignored; `preds/` CSVs are tracked so cleaned datasets can be reproduced without retraining.

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
