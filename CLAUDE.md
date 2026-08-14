# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research implementation for the paper *"Instance-Dependent Label Noise Correction via Contrastive Learning and Ensemble Disagreement"*. The framework detects and corrects mislabeled training samples in image datasets (CIFAR-10, Fashion-MNIST, CIFAR-10N, ANIMAL-10N) using Siamese networks trained with nested cross-validation, then flags noise via ensemble disagreement and relabels via prediction consensus.

This is a research codebase — most experimentation happens in `main.ipynb` and `paper.ipynb` (aggregated results); `snd.cli` is the only CLI entry point. There is no test suite, linter config, or build step.

## Environment & Commands

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (Python 3.13, PyTorch 2.13, CUDA/MPS optional — falls back to CPU). Versions are pinned in `pyproject.toml`/`uv.lock`; do not use `pip` or `conda` here.

```bash
uv sync                                  # create .venv and install pinned deps (uv.lock)
uv sync --extra notebook                 # also install ipykernel/jupyter for the notebooks

# Run the full detect → relabel → save-cleaned-dataset pipeline:
uv run snd --dataset cifar10      --noise_ratio 30 --output_dir results/cifar10_30
uv run snd --dataset fashionmnist --noise_ratio 40 --output_dir results/fmnist_40
uv run snd --dataset cifar10n     --noise_ratio n  --output_dir results/cifar10n
# optional: --mistakes_count N (detection threshold, default 8) --relabel_threshold N (default 9)

# Regenerate the README results tables from the committed predictions (offline, seconds):
uv run python scripts/summarize_results.py
```

The project is an installed package (`src/` layout, hatchling), so `snd` is on PATH inside the venv and `import snd...` works from anywhere. Datasets download automatically via `torchvision` into `data/`. The `scripts/plot_*.py` scripts regenerate paper figures from saved prediction CSVs.

## Package layout

```
src/snd/
├── cli.py         # CLI entry point (`snd` console script)
├── config.py      # Per-dataset/per-noise hyperparameter dicts + transforms + lazy dataset handles
├── utils.py       # set_global_seed, class-name constants
├── data/          # base.py (NoiseAdder ABC), noise.py, instance_dependent.py, cifar10n.py, dataset.py, fold.py
├── models/        # siamese.py, preact.py
├── training/      # trainer.py, contrastive.py, hyperparams.py (TrainingConfig)
├── pipeline/      # cleaner.py, detector.py, tta_cleaner.py
└── evaluation/    # cleaner_report.py (= metrics + plots mixins), final_model_tester.py,
                   # ensemble_independence.py, tester.py, visualizer.py
scripts/           # summarize_results.py, calibrate_thresholds.py, plot_*.py
```

## Pipeline Architecture

The flow is a three-layer nesting. Read `cli.py` → `cleaner.py` → `detector.py` → `trainer.py`/`siamese.py` to follow it top-down.

1. **`NoiseCleaner` (`snd/pipeline/cleaner.py`)** — the orchestrator. Injects synthetic noise (see noise types below), runs the **outer** k-fold loop (`outer_folds_num`), and after detection produces the cleaned dataset. `clean()` runs detection per outer fold; `advanced_clean()` applies relabel/removal decisions.

2. **`CleanerReportingMixin` (`snd/evaluation/cleaner_report.py`)** — the analysis half of `NoiseCleaner`, split out to keep the orchestration readable. It is just the composition of `CleanerMetricsMixin` (`cleaner_metrics.py`, computes detection metrics and the relabeling score from the prediction CSVs) and `CleanerPlotsMixin` (`cleaner_plots.py`, the matplotlib output). `NoiseCleaner` inherits it, so `analyze*`/`plot*`/`calculate_relabeling_score` are still called on the cleaner instance (notebooks rely on this).

3. **`NoiseDetector` (`snd/pipeline/detector.py`)** — per outer fold, trains an **inner** ensemble of `inner_folds_num` Siamese models via `StratifiedKFold`. Each held-out sample is classified by every model in which it was not trained; the count of models that disagree with the current (noisy) label is the sample's **"mistakes"** score.

4. **`SiameseNetwork` (`snd/models/siamese.py`)** — twin-branch model: a backbone feature extractor (`resnet18/34/50/101`, `preact-resnet18/34/50`, `efficientnetv2`, `custom`) → `fc_embedding` (contrastive head, ends in Sigmoid) → `fc_classifier`. The `parallel` flag routes the classifier off the backbone features instead of the embedding.

5. **`Trainer` (`snd/training/trainer.py`)** — jointly optimizes **contrastive loss** (on embedding pairs, weighted by `contrastive_ratio`) **+ cross-entropy** (per branch). `freeze_epoch` switches between phased training (contrastive-only, then classifier-only) and combined loss when `None`.

### Detection & relabeling logic (the core idea)
- A sample is flagged **noisy** when its `mistakes` ≥ `mistakes_count` (ensemble disagreement).
- A flagged sample is **relabeled** when some class appears ≥ `relabel_threshold` times across the ensemble's predictions; otherwise it is **removed**. See `advanced_clean()` in `cleaner.py` and `calculate_relabeling_score()` in `cleaner_report.py`.
- The **relabeling score** (`calculate_relabeling_score`) is the paper's ground-truth-free quality metric, scoring each decision in {-2,-1,0,1,2}.
- `ContrastiveLoss` takes `same_label = 1` for same-class pairs and `0` otherwise (`Trainer.calc_loss` computes it as `(label1 == label2).float()`): same-class pairs are pulled together, different-class pairs pushed apart beyond `margin`. Only the Euclidean metric is supported.

## Configuration

Model/optimization hyperparameters travel as a single **`TrainingConfig`** (`snd/training/hyperparams.py`) instead of being re-declared on every constructor. `NoiseCleaner` accepts either `config=TrainingConfig(...)` or the flat keyword form (what the config dicts and notebooks use) and forwards the config object to `NoiseDetector`. Both classes expose the fields as plain attributes via `__getattr__` (`self.margin`, `self.model`, plus the legacy aliases `num_class`, `training_batch_size`, `optimzer`), so existing call sites keep working. **To add a hyperparameter, add one field to `TrainingConfig`** — not three constructors.

Per-run settings live on **`src/snd/config.py`** as per-dataset/per-noise-level dicts (`CIFAR10_20_PARAMS`, `CIFAR10_30_PARAMS`, `CIFAR10_40_PARAMS`, `CIFAR10N_PARAMS`, `FashionMNIST_{20,30,40}_PARAMS`). `cli.py` selects one by `--dataset`/`--noise_ratio` and splats it into `NoiseCleaner(**params)`. To change architecture, folds, noise level, paths, or thresholds, edit these dicts — do not add CLI flags unless asked.

Key config conventions:
- `model_save_path`, `noisy_indices_path`, `prediction_path` are **format strings** with a `{}` for the fold number (e.g. `"preds/cifar10(30)/resnet50/fold{}_analysis.csv"`).
- `noise_type`: `'idn'` (instance-dependent, `InstanceDependentNoiseAdder`), `'iin'` (uniform/instance-independent, `LabelNoiseAdder`), `'cifar10n'` (real human labels, `CIFAR10N`), or `'none'`.
- The four dataset handles (`CIFAR10_TRAIN_DATASET`, …) are built lazily via a module `__getattr__`, so importing `snd.config` does not download data. Do **not** reintroduce `from snd.config import *` — star imports bypass PEP 562 and break lazy loading. Use `import snd.config as config` + attribute access.

## Persistence & resumption

Training is expensive and **checkpoint-driven**:
- Per-fold model weights (`*.pth`) are saved to `model_save_path`. If a checkpoint exists, that fold is skipped on rerun.
- If a fold's `noisy_indices_path` CSV exists, `clean()` skips training entirely and loads cached results — so re-running with existing `preds/` regenerates a cleaned dataset **without retraining** (the predictions in `preds/` are committed for this reason).
- `prediction_path` CSVs store per-sample `index, noisy_label, is_noisy, real_label, mistakes, label_pred, preds` (the `preds` column is a `|`-joined list of each ensemble member's prediction) and drive all downstream `analyze`/`plot` methods and `scripts/summarize_results.py`.

`*.pth`, `data/`, `cleaned/`, and `figures/` are gitignored; `preds/` CSVs are tracked.

## Conventions & gotchas

- `set_global_seed(42)` is called at import of `snd.cli` and `snd.config` for reproducibility; `InstanceDependentNoiseAdder.add_noise()` nonetheless re-randomizes its own seed.
- Backbones live in the `BACKBONES` registry in `snd/models/siamese.py`, mapping name → `(builder(num_classes, pre_trained), feature width)`. Add new ones there with the correct width (e.g. resnet50 → 2048).
- `NoiseAdder` subclasses expose `orginal_labels` (note the spelling), `noisy_labels`, `noisy_indices`, and reporting helpers (`report`, `ravel`, `calculate_metrics`) consumed throughout `cleaner.py`.
- `snd/pipeline/tta_cleaner.py` (`TTACleaner`) is a separate test-time-augmentation cleaning variant used from the notebooks, not from the CLI. It is *not* a copy of `NoiseCleaner` — it has its own train/evaluate path and shares only `get_image_size` — but it takes an overlapping 28-parameter constructor, so hyperparameter changes usually need mirroring there.
- ANIMAL-10N is driven from `main.ipynb` with inline params (there is no `ANIMAL10N_PARAMS`), and it uses the `efficientnetv2` backbone — the only reason `timm` is a dependency.
- After changing imports, verify with an import sweep of every `snd.*` module plus an end-to-end `uv run snd --dataset cifar10 --noise_ratio 30` (which runs from cached `preds/` in seconds and should print `6243 removed from dataset and 11059 relabled` / `4.02% noise remained in 43757 data`).
