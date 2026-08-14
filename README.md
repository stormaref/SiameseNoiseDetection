# Label Noise Correction with Siamese Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)

Real-world training sets contain mislabeled examples, and the noise is often **instance-dependent** — harder, more ambiguous samples are more likely to be wrong. This framework **detects** those mislabeled samples and **corrects** them (relabel or remove) using an ensemble of Siamese networks trained with nested cross-validation, scoring every decision *without* access to ground-truth labels.


## 🔑 Key Features
- **Siamese architecture** — a twin-branch network jointly trained with a contrastive loss (on embedding pairs) and cross-entropy, over a configurable backbone (ResNet, PreAct-ResNet, EfficientNetV2).
- **Nested cross-validation** — stratified outer/inner folds prevent data leakage during detection.
- **Ensemble disagreement** — a sample is flagged when enough held-out ensemble members disagree with its current label.
- **Consensus relabeling** — flagged samples are relabeled when the ensemble agrees on an alternative class, otherwise removed.
- **Ground-truth-free quality metric** — the *relabeling score* grades each correction in {-2,-1,0,1,2}, so cleaning quality can be measured on real datasets where the truth is unknown.
- **Multiple datasets** — CIFAR-10 and Fashion-MNIST (synthetic 20/30/40% noise), CIFAR-10N (real human label noise) and ANIMAL-10N.


## ⚙️ How It Works

The pipeline is a three-layer nesting (`cli.py` → `cleaner.py` → `detector.py` → `siamese.py`/`trainer.py`):

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1. Orchestrate | `NoiseCleaner` (`snd/pipeline/cleaner.py`) | Injects synthetic noise, runs the **outer** k-fold loop, then produces the cleaned dataset. |
| 2. Detect | `NoiseDetector` (`snd/pipeline/detector.py`) | Per outer fold, trains an **inner** ensemble of Siamese models via `StratifiedKFold`. Each sample is scored by every model that did *not* train on it; the number of disagreements with its (noisy) label is its **`mistakes`** score. |
| 3. Model | `SiameseNetwork` (`snd/models/siamese.py`) | Backbone → contrastive embedding head → classifier, trained by `Trainer` with a weighted **contrastive + cross-entropy** objective. |

**Decision rule** — with a 10-model inner ensemble and thresholds `T_D` (detection) and `T_R` (relabeling):
- A sample is flagged **noisy** when `mistakes ≥ T_D`.
- A flagged sample is **relabeled** when some class appears `≥ T_R` times across the ensemble's predictions; otherwise it is **removed**.


## 📊 Results

All numbers below are computed from the committed per-fold predictions in `preds/` at the operating point **`T_D = 10`, `T_R = 9`** (the value in each config dict), and are reproducible offline in seconds:

```bash
uv run python scripts/summarize_results.py                    # the table below
uv run python scripts/summarize_results.py --td 8 --tr 9      # any other operating point
uv run python scripts/summarize_results.py --sweep cifar10_30 # threshold sweep
```

| Dataset | Noise | Precision | Recall | F1 | FPR | Relabeled | Removed | Retained | Residual noise | Clean yield | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10, IDN 20% | 20.42% | 89.47 | 89.50 | **89.49** | 2.70 | 8,003 | 2,208 | 95.58% | 4.15% | 91.61% | 1.39 |
| CIFAR-10, IDN 30% | 29.95% | 91.07 | 74.79 | **82.13** | 3.13 | 9,258 | 3,040 | 93.92% | 10.01% | 84.51% | 1.42 |
| CIFAR-10, IDN 40% | 39.61% | 91.51 | 63.29 | **74.83** | 3.85 | 9,342 | 4,354 | 91.29% | 17.71% | 75.12% | 1.39 |
| CIFAR-10N (real) | 9.01% | 69.99 | 68.48 | **69.22** | 2.91 | 3,012 | 1,396 | 97.21% | 4.90% | 92.44% | 0.65 |
| Fashion-MNIST, IDN 20% | 20.83% | 87.14 | 83.25 | **85.15** | 3.23 | 8,557 | 3,381 | 94.36% | 5.36% | 89.31% | 1.30 |
| Fashion-MNIST, IDN 30% | 30.37% | 90.44 | 75.97 | **82.57** | 3.50 | 10,765 | 4,543 | 92.43% | 9.59% | 83.56% | 1.39 |
| Fashion-MNIST, IDN 40% | 40.31% | 90.33 | 59.02 | **71.40** | 4.27 | 9,848 | 5,957 | 90.07% | 19.79% | 72.25% | 1.33 |
| Fashion-MNIST, IDN 60%¹ | 60.12% | 70.89 | 68.20 | **69.52** | 42.21 | 21,509 | 13,192 | 78.01% | 40.66% | 46.29% | 0.60 |

*Precision / Recall / F1 / FPR* grade **detection** against the true noise flags. *Residual noise* is the share of the retained set whose (possibly corrected) label is still wrong; *clean yield* is the share of the **original** dataset that ends up both retained and correctly labeled; *score* is the ground-truth-free relabeling score. Backbone: ResNet-50 for CIFAR-10/10N, ResNet-34 for Fashion-MNIST, 10 outer × 10 inner folds.

**The headline:** on CIFAR-10 with 20% instance-dependent noise, label accuracy goes from **79.6% → 95.9%** while keeping 95.6% of the data. At 30% noise, **70.1% → 90.0%**; at 40%, **60.4% → 82.3%**. On real CIFAR-10N human noise, **91.0% → 95.1%** while discarding under 3% of the training set.

¹ Fashion-MNIST 60% uses a 15-fold ensemble, so its thresholds are not directly comparable to the 10-fold rows. It is included as a breakdown point: past ~50% noise the ensemble's own errors dominate and the false-positive rate explodes.

**ANIMAL-10N** predictions are committed under `preds/animal10n/efficientnetv2/`, but the dataset ships no clean labels, so detection precision/recall cannot be measured. At this operating point the pipeline flags 2,844 of 50,000 samples (1,607 relabeled, 1,237 removed).

### Threshold sensitivity

`T_D` trades detection precision against recall, and the best value for *downstream accuracy* is not the F1-optimal one — a lower `T_D` finds more noise but throws away more clean data. CIFAR-10 at 30% noise (`T_R = 9`):

| T_D | Precision | Recall | F1 | Retained | Residual noise | Clean yield |
|---|---|---|---|---|---|---|
| 6 | 76.35 | 98.91 | 86.18 | 83.31% | 3.34% | 80.53% |
| 7 | 79.76 | 98.04 | 87.96 | 85.30% | 3.57% | 82.26% |
| 8 | 83.50 | 96.47 | 89.52 | 87.51% | 4.02% | 84.00% |
| 9 | 87.17 | 91.97 | 89.50 | 90.52% | 5.37% | **85.65%** |
| 10 | 91.07 | 74.79 | 82.13 | 93.92% | 10.01% | 84.51% |

The CLI defaults (`--mistakes_count 8 --relabel_threshold 9`) sit near the F1 optimum; the config dicts use `T_D = 10`, which maximizes precision and retention.

### Choosing thresholds without clean labels

`scripts/calibrate_thresholds.py` picks `(T_D, T_R)` by the accuracy of a lightweight probe classifier on a single **fixed** held-out subset scored against its *noisy* labels — no ground truth anywhere. On Fashion-MNIST 20% it costs **0.27 pp** of downstream test accuracy versus an oracle tuned on the true noise flags (89.12% vs 89.39%), and the signal correlates with true test accuracy at **ρ = 0.954**.

### Ensemble independence

`scripts/plot_ensemble_independence.py` validates the weak-dependence assumption behind the false-positive bound. On CIFAR-10N the ensemble's errors are correlated but not strongly so: mean pairwise error correlation **ρ̂ = 0.330**, every pair satisfies the covariance condition, and the empirical error variance (9.72) stays under the theoretical bound (11.33). The empirical false-positive rate of **2.91%** sits far inside the theorem's bound at ρ_max with p_C = 0.5 (**38.2%**) — though it does exceed the much tighter bound obtained by plugging ρ̂ in directly (0.77%), so the guarantee is the conservative one.


## 🚀 Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which provisions Python 3.13 and all packages from the pinned `uv.lock`.

```bash
uv sync                    # create .venv and install pinned dependencies
uv sync --extra notebook   # also install ipykernel/jupyter for the notebooks
```

PyTorch 2.13 is installed automatically; CUDA/MPS are used when available and fall back to CPU. The project installs as a package, exposing the `snd` console script.


## 🏋️ Usage

Datasets download automatically via `torchvision` into `data/` on first use.

```bash
# CIFAR-10, 30% synthetic instance-dependent noise
uv run snd --dataset cifar10 --noise_ratio 30 --output_dir results/cifar10_30

# Fashion-MNIST, 40% synthetic noise
uv run snd --dataset fashionmnist --noise_ratio 40 --output_dir results/fmnist_40

# CIFAR-10N, real-world human label noise
uv run snd --dataset cifar10n --noise_ratio n --output_dir results/cifar10n
```

`snd` (equivalently `python -m snd.cli`) selects the matching config, runs detection per outer fold, applies the relabel/remove decisions, and writes the cleaned dataset to `--output_dir`.

### CLI arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset` | ✓ | — | `cifar10`, `fashionmnist`, or `cifar10n` |
| `--noise_ratio` | ✓ | — | `20`, `30`, `40` (synthetic) or `n` (CIFAR-10N) |
| `--output_dir` | | `results/` | Where the cleaned dataset is saved |
| `--mistakes_count` | | `8` | Disagreement threshold `T_D` to flag a sample as noisy |
| `--relabel_threshold` | | `9` | Ensemble agreement `T_R` needed to relabel rather than remove |

> **No retraining needed.** Per-fold predictions are committed under `preds/`. If they exist for a config, the pipeline skips training and regenerates a cleaned dataset directly from the cached predictions — so you can sweep `--mistakes_count`/`--relabel_threshold` cheaply.


## 🛠️ Configuration

All hyperparameters live in `src/snd/config.py` as per-dataset/per-noise-level dictionaries. `snd.cli` picks one by `--dataset`/`--noise_ratio` and splats it into `NoiseCleaner`.

| Dataset | Config dicts |
|---------|--------------|
| CIFAR-10 | `CIFAR10_20_PARAMS`, `CIFAR10_30_PARAMS`, `CIFAR10_40_PARAMS` |
| Fashion-MNIST | `FashionMNIST_20_PARAMS`, `FashionMNIST_30_PARAMS`, `FashionMNIST_40_PARAMS` |
| CIFAR-10N | `CIFAR10N_PARAMS` |

Each dict controls the backbone architecture, outer/inner fold counts, training settings (batch size, learning rate, epochs, contrastive weighting), embedding size, detection/relabel thresholds, and output paths. Edit these dicts to change behavior — the CLI deliberately exposes only the flags above.

The four dataset handles in `config.py` are built lazily on first access, so importing the module to read a hyperparameter dict does not download ~200 MB into `data/`.


## 📈 Reproducing the paper figures

Aggregated results and analysis are in the notebooks; the standalone scripts regenerate individual figures and tables from the committed prediction CSVs:

```bash
uv run python scripts/summarize_results.py           # the results tables above
uv run python scripts/plot_tsne_vor.py               # t-SNE / Voronoi embeddings
uv run python scripts/plot_ensemble_independence.py  # ensemble independence analysis
uv run python scripts/plot_qualitative_examples.py   # qualitative detection/relabel examples
uv run python scripts/calibrate_thresholds.py --dataset fashionmnist --noise_ratio 20 --mode all
```


## 📂 Repository Structure
```
.
├── src/snd/
│   ├── cli.py                   # CLI entry point (`snd`)
│   ├── config.py                # Per-dataset/per-noise hyperparameter dicts (lazy datasets)
│   ├── utils.py                 # Seeding and class-name constants
│   ├── data/                    # Datasets, noise injection, fold splitting
│   │   ├── base.py              #   NoiseAdder interface
│   │   ├── noise.py             #   Uniform (instance-independent) noise
│   │   ├── instance_dependent.py#   Instance-dependent noise
│   │   ├── cifar10n.py          #   Real human labels (CIFAR-10N)
│   │   ├── dataset.py           #   Pair/single dataset wrappers
│   │   └── fold.py              #   Cross-validation splitter
│   ├── models/                  # Networks
│   │   ├── siamese.py           #   Backbone + embedding head + classifier
│   │   └── preact.py            #   PreAct-ResNet backbones
│   ├── training/                # Optimization
│   │   ├── trainer.py           #   Joint contrastive + cross-entropy loop
│   │   └── contrastive.py       #   Contrastive loss
│   ├── pipeline/                # Detection & correction
│   │   ├── cleaner.py           #   Orchestrator: noise injection, outer folds, cleaning
│   │   ├── detector.py          #   Inner ensemble training + mistakes scoring
│   │   └── tta_cleaner.py       #   Test-time-augmentation variant (notebook use)
│   └── evaluation/              # Analysis
│       ├── cleaner_report.py    #   NoiseCleaner's analysis/plot methods (mixin)
│       ├── final_model_tester.py#   Downstream classifier evaluation
│       ├── ensemble_independence.py # Weak-dependence diagnostics
│       ├── tester.py            #   Per-fold inference
│       └── visualizer.py        #   Embedding visualization
├── scripts/                     # Standalone figure/table generation
├── preds/                       # Committed per-fold prediction CSVs (enable retraining-free cleaning)
├── results/                     # Calibration and independence outputs
├── main.ipynb / paper.ipynb     # Experimentation and aggregated results
├── pyproject.toml               # Project metadata and dependencies (uv)
└── uv.lock                      # Pinned, reproducible lockfile
```

`*.pth`, `data/`, `cleaned/`, and `figures/` are gitignored; `preds/` CSVs are tracked so cleaned datasets can be reproduced without retraining.

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
