# Round-4 revision experiments (Kaggle)

Self-contained notebooks for the experiments Reviewer 5 asked for. Each `.ipynb` embeds its
helper modules (`src/*.py`, written to the working directory by `%%writefile` cells), clones
this public repo at runtime for the `snd` package and the committed prediction CSVs, and
rebuilds the **exact published noisy labels and outer folds** from those CSVs. The IDN
generator re-randomises, so the CSVs are the only faithful record of which labels were
corrupted.

| notebook | reviewer item | trains |
|---|---|---|
| `notebooks/r51_detector_isolation.ipynb` | R5.1 (+ R4.3 support, + ensemble filters for R5.2) | inner-ensemble members: `siamese` (our objective, corrected contrastive loss) and `ce` (same network and data, no contrastive term) |
| `notebooks/r52_ensemble_baselines.ipynb` (not yet added) | R5.2 | Co-teaching, DivideMix on our noisy labels; detection + correction metrics |
| `notebooks/r56_downstream_classifiers.ipynb` | R5.6 | 4 classifiers × {noisy, ours, oracle-filtered, oracle-clean} × seeds |

## Running on Kaggle

1. *Create → New Notebook → File → Import Notebook*, upload one `.ipynb`.
2. Settings: Accelerator **GPU T4 x2**, Internet **On**, Persistence **Files only**.
3. Edit the configuration cell (dataset, noise level, `preds_ref`, folds/variants or classifiers).
4. **Save Version → Save & Run All (Commit)**. The workers use both GPUs and stop taking new
   jobs before the 12-hour limit.
5. To continue: *Add Input → Your Work →* this notebook (the previous version's output), commit
   again. Finished jobs are restored and skipped. The launch cell prints how many jobs remain.
6. Results: the version's *Output* tab → `results/*.csv` (and `logs/`, raw `r5x_out/`).

`preds_ref`: `'main'` = the CIFAR-10 re-run with the corrected contrastive loss, `'513d833'` =
the round-3 run in the current manuscript. The noise draw and outer folds are identical in both,
so training-based experiments do not depend on this choice. It only changes the published-detector
reference row (R5.1) and the `ours` cleaned set (R5.6).

## Budget (rough; the first job's log gives the real per-epoch time)

| experiment | unit | T4 estimate |
|---|---|---|
| R5.1 CIFAR-10, 1 outer fold, 2 variants × 10 members | 20 ResNet-50 members | ~28 GPU-h ≈ 14 h on T4×2 (2 commits) |
| R5.1 Fashion-MNIST, 1 outer fold, 2 × 10 | 20 ResNet-34 members | ~7 GPU-h ≈ 3–4 h on T4×2 |
| R5.6 CIFAR-10, one noise level, 4 × 4 × 3 seeds | 48 classifier runs | ~30 GPU-h ≈ 15 h on T4×2 |

The Kaggle weekly GPU quota (~30 h) is the binding constraint. Suggested order:
R5.1 Fashion-MNIST 20% (fast; shows the effect early), R5.6 CIFAR-10 20%, R5.1 CIFAR-10 20%, then
the remaining noise levels as quota allows.

## Faithfulness notes

- Siamese protocol taken from the `NoiseCleaner(...)` calls in `main.ipynb` (cells 16/25/35,
  59/71/80), which differ from `snd/config.py` and the CLI. Examples: scoring uses the
  un-augmented transform, CIFAR-10 40% used a different augmentation from 20/30%, F-MNIST
  patience is 12. Downstream protocol from the `FinalEvaluator(...)` calls (cells 106–124).
- Inner splits: the published runs used an unseeded `StratifiedKFold`, so they cannot be
  reproduced. Here they are seeded and **shared by all variants**, as are the 200k training pairs
  and the initial weights. Variant comparisons are paired on identical data.
- Mixed precision (fp16 autocast) is used on CUDA for speed. All variants use it, so
  comparisons stay internal. The contrastive term is computed in fp32.
- Each training job writes `<job>.done` only after its results are saved. A job interrupted
  by the session limit is simply re-run.

## Local smoke test

```bash
SND_REPO=/path/to/SiameseNoiseDetection SND_WORK=/tmp/w \
SND_CONFIG_OVERRIDES='{"subsample": 1500, "loader_workers": 0, "poll": 10, "overrides": {"max_epochs": 2}}' \
uv run --with nbclient --with ipykernel --with nbformat python -c "..."   # execute with nbclient
```
`python build_notebooks.py notebooks/` regenerates the `.ipynb` files from `src/`.
