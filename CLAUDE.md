# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research implementation for the paper *"Instance-Dependent Label Noise Correction via Contrastive Learning and Ensemble Disagreement"*. The framework detects and corrects mislabeled training samples in image datasets (CIFAR-10, Fashion-MNIST, CIFAR-10N) using Siamese networks trained with nested cross-validation, then flags noise via ensemble disagreement and relabels via prediction consensus.

This is a research codebase — most experimentation happens in `main.ipynb` and `paper.ipynb` (aggregated results); `runner.py` is the only CLI entry point. There is no test suite, linter config, or build step.

## Environment & Commands

```bash
pip install -r requirements.txt          # Python 3.10+, PyTorch 2.6, CUDA optional (falls back to CPU)

# Run the full detect → relabel → save-cleaned-dataset pipeline:
python runner.py --dataset cifar10     --noise_ratio 30 --output_dir results/cifar10_30
python runner.py --dataset fashionmnist --noise_ratio 40 --output_dir results/fmnist_40
python runner.py --dataset cifar10n    --noise_ratio n  --output_dir results/cifar10n
# optional: --mistakes_count N (detection threshold, default 8) --relabel_threshold N (default 9)
```

Datasets download automatically via `torchvision` into `data/`. The standalone `plot_*.py` scripts regenerate paper figures from saved prediction CSVs.

## Pipeline Architecture

The flow is a three-layer nesting. Read `runner.py` → `cleaner.py` → `detector.py` → `trainer.py`/`siamese.py` to follow it top-down.

1. **`NoiseCleaner` (`models/cleaner.py`)** — the orchestrator. Injects synthetic noise (see noise types below), runs the **outer** k-fold loop (`outer_folds_num`), and after detection produces the cleaned dataset. `clean()` runs detection per outer fold; `advanced_clean()` applies relabel/removal decisions; `analyze*`/`plot*`/`calculate_relabeling_score` produce the paper's metrics and figures.

2. **`NoiseDetector` (`models/detector.py`)** — per outer fold, trains an **inner** ensemble of `inner_folds_num` Siamese models via `StratifiedKFold`. Each held-out sample is classified by every model in which it was not trained; the count of models that disagree with the current (noisy) label is the sample's **"mistakes"** score.

3. **`SiameseNetwork` (`models/siamese.py`)** — twin-branch model: a backbone feature extractor (`resnet18/34/50/101`, `preact-resnet*`, `vgg*-bn`, `dla`, `efficientnetv2`, `custom`) → `fc_embedding` (contrastive head, ends in Sigmoid) → `fc_classifier`. The `parallel` flag routes the classifier off the backbone features instead of the embedding.

4. **`Trainer` (`models/trainer.py`)** — jointly optimizes **contrastive loss** (on embedding pairs, weighted by `contrastive_ratio`) **+ cross-entropy** (per branch). `freeze_epoch` switches between phased training (contrastive-only, then classifier-only) and combined loss when `None`.

### Detection & relabeling logic (the core idea)
- A sample is flagged **noisy** when its `mistakes` ≥ `mistakes_count` (ensemble disagreement).
- A flagged sample is **relabeled** when some class appears ≥ `relabel_threshold` times across the ensemble's predictions; otherwise it is **removed**. See `advanced_clean()` and `calculate_relabeling_score()` in `cleaner.py`.
- The **relabeling score** (`calculate_relabeling_score`) is the paper's ground-truth-free quality metric, scoring each decision in {-2,-1,0,1,2}.

## Configuration

All hyperparameters live in **`models/config.py`** as per-dataset/per-noise-level dicts (`CIFAR10_20_PARAMS`, `CIFAR10_30_PARAMS`, `CIFAR10_40_PARAMS`, `CIFAR10N_PARAMS`, `FashionMNIST_{20,30,40}_PARAMS`). `runner.py` selects one by `--dataset`/`--noise_ratio` and splats it into `NoiseCleaner(**params)`. To change architecture, folds, noise level, paths, or thresholds, edit these dicts — do not add CLI flags unless asked.

Key config conventions:
- `model_save_path`, `noisy_indices_path`, `prediction_path` are **format strings** with a `{}` for the fold number (e.g. `"preds/cifar10(30)/resnet50/fold{}_analysis.csv"`).
- `noise_type`: `'idn'` (instance-dependent, `InstanceDependentNoiseAdder`), `'iin'` (uniform/instance-independent, `LabelNoiseAdder`), `'cifar10n'` (real human labels, `CIFAR10N`), or `'none'`.

## Persistence & resumption

Training is expensive and **checkpoint-driven**:
- Per-fold model weights (`*.pth`) are saved to `model_save_path`. If a checkpoint exists, that fold is skipped on rerun.
- If a fold's `noisy_indices_path` CSV exists, `clean()` skips training entirely and loads cached results — so re-running with existing `preds/` regenerates a cleaned dataset **without retraining** (the predictions in `preds/` are committed for this reason).
- `prediction_path` CSVs store per-sample `index, noisy_label, is_noisy, real_label, mistakes, label_pred, preds` (the `preds` column is a `|`-joined list of each ensemble member's prediction) and drive all downstream `analyze`/`plot` methods.

`*.pth`, `data/`, `cleaned/`, and `figures/` are gitignored; `preds/` CSVs are tracked.

## Conventions & gotchas

- `set_global_seed(42)` is called at import of `runner.py` and `config.py` for reproducibility; `InstanceDependentNoiseAdder.add_noise()` nonetheless re-randomizes its own seed.
- Backbone-specific output widths are hardcoded in `SiameseNetwork.__init__` (e.g. resnet50 → 2048); add new backbones there with the correct `cnn_output`.
- `NoiseAdder` subclasses expose `orginal_labels` (note the spelling), `noisy_labels`, `noisy_indices`, and reporting helpers (`report`, `ravel`, `calculate_metrics`) consumed throughout `cleaner.py`.
- `models/tta_cleaner.py` (`TTACleaner`) is a separate test-time-augmentation cleaning variant used from the notebooks, not from `runner.py`.
