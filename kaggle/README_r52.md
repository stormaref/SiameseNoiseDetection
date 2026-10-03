## README

### What each job does and what it saves

* `coteaching_<ds>_<noise>`: Han et al. (2018), reimplemented (the official code targets PyTorch
  0.3). Two PreAct-ResNet18 networks from `snd.models.preact`. In every mini-batch each network keeps
  its `R(T)` smallest-loss samples and the *other* network is updated on them. The forget rate is the
  actual noise rate of the labels, ramped linearly over `Tk=10` epochs. Training runs 200 epochs with
  batch 128 and Adam lr 1e-3; from epoch 80 the lr decays linearly to 0 and beta1 drops from 0.9 to
  0.1. Augmentation is random crop (pad 4) plus horizontal flip, done on the GPU. Fashion-MNIST is
  fed as 3-channel grayscale at 28x28. `final.npz` holds `sel1`/`sel2` (selected during the final
  epoch), the per-sample losses, both networks' softmax and CE on the un-augmented training images,
  and a per-epoch history (label precision, flagged count, test accuracy).
* `dividemix_<ds>_<noise>`: the official code (MIT licence) at commit `d9d3058`, patched by
  `r52_worker.patch_dividemix`. Every change is marked `# r52` and listed in `r52_patch.diff`, which
  is copied into each job folder. The patches:
  * run on current PyTorch: `next(it)`, no `torchnet`, any device;
  * AMP;
  * read our images and labels: `r52_{train,test}.npz` hold the images with their true labels, and
    the noisy labels come through the official `noise_file` JSON. The loader now refuses to draw
    new noise;
  * write the co-divide GMM clean probabilities of both networks every epoch
    (`gmm_history.npz`);
  * checkpoint every 5 epochs and before the deadline (`ckpt.pt`);
  * export `final.npz` at the end: the GMM division on the final weights plus both networks'
    softmax on the un-augmented images.

  The hyperparameters are the official CIFAR-10 ones: 300(+1) epochs, SGD lr 0.02 divided by 10 at
  epoch 150, batch 64, warm-up 10, T=0.5, alpha=4, p_threshold=0.5. Two settings follow the paper's
  choices for structured noise: the warm-up confidence penalty (`noise_mode='asym'`) and
  `lambda_u=0` (Table 7: 0 for CIFAR-10 at 20% sym and 40% asym). Both are configurable.
  **Fashion-MNIST is our adaptation**: there is no official config, so it uses the same
  hyperparameters with 28x28 grayscale replicated to 3 channels, RandomCrop(28, pad 4) plus flip,
  and FMNIST mean/std.
* Analysis-only filters (`r52_filters.py`, no training) run on the R5.1 member files. Each sample
  is scored by the m members that did not see its outer fold:
  * majority: flag if more than m/2 members misclassify it;
  * consensus: flag if all m members misclassify it;
  * Confident Learning: `cleanlab.filter.find_label_issues` (defaults) on the mean softmax.

  They are evaluated on the samples the R5.1 runs cover.

### Detection / correction rules used in the table

| method | flagged as noisy | cleaned label (primary row) | AUC score |
|---|---|---|---|
| ours | mistakes >= TD (paper oracle TD) | class with >= TR votes, else removed | mistakes / 10 |
| coteaching | not selected as small-loss by net1 **and** not by net2 in the final epoch | removed (secondary: joint argmax if both nets' argmax agree, else removed) | mean un-augmented CE of both nets |
| coteaching-global | in the top tau fraction of un-augmented CE for both nets | removed | same |
| dividemix | (p1 + p2)/2 < 0.5, GMM on the final weights' losses | argmax of (softmax1 + softmax2)/2 on the un-augmented image (secondary: removed) | 1 - (p1 + p2)/2 |
| majority / consensus | r > m/2 / r = m | removed (secondary: strict-majority class, else removed) | r / m |
| confident_learning | cleanlab default | removed (secondary: argmax of the mean softmax) | 1 - mean p(given label) |

Metrics are the same for every row:
* precision, recall, F1 and FPR against `is_noisy`; ROC-AUC and AP;
* residual noise = wrong labels kept / size of the cleaned set;
* relabel accuracy = relabelled to the true class / relabelled;
* the paper's relabeling score.

Each metric is computed over two scopes:
* `all`: the full 50k/60k training set (for the filters, the samples R5.1 covers);
* `folds`: only the outer folds listed in `outer_folds`.

### Expected runtime on one T4 (estimates; the logs print seconds per epoch, so re-plan after the first hour)

| job | per epoch | total |
|---|---|---|
| Co-teaching CIFAR-10 (200 ep) | ~30-45 s | ~2-2.5 h |
| Co-teaching FMNIST (200 ep) | ~30-45 s | ~2-2.5 h |
| DivideMix CIFAR-10 (301 ep) | ~60-90 s (warm-up epochs ~4x cheaper) | ~5.5-7.5 h |
| DivideMix FMNIST (301 ep) | similar to CIFAR-10 | ~5.5-7.5 h |

The full grid is roughly 6 x 2.2 + 6 x 6.5, about 50 T4-hours. On 2xT4 that is about 25 h of wall
time, i.e. 3 sessions of 11.5 h, which is more than one week's 30 h GPU quota. Suggested order: all
CIFAR-10 jobs plus Co-teaching FMNIST first (2 sessions), then DivideMix FMNIST. To save quota you
could also drop to the 20%/40% settings for FMNIST.

### Resuming across 12-hour sessions

1. Each session stops taking new jobs once less than `min_job_hours` is left. A running job
   checkpoints and exits cleanly if the next epoch would overrun `session_start + session_hours`.
   DivideMix exits with code 3, which means paused, and its queue claim is released.
2. Next session: *Edit* the notebook, then *Add Input -> Your Work -> this notebook* and pick the
   **latest** version only (if several versions are attached, the first match wins for each file).
   Then *Save & Run All* again. `restore_previous_outputs('r52_out')` copies the `.done` files,
   checkpoints and histories back. Finished jobs are skipped; paused jobs continue from `ckpt.pt`
   (weights, optimiser, AMP scaler, RNG, epoch).
3. When everything is `.done`, run cell 4 (it also works on a partial set: it scores whatever has a
   `final.npz`).

### Local smoke test (CPU/MPS)

```bash
SND_REPO=/path/to/SiameseNoiseDetection SND_WORK=/tmp/r52work uv run --no-sync --project $SND_REPO \
  python r52_worker.py smoke.json --prepare   # then the same with no flag (repeat until all .done), then r52_analysis.py
# cleanlab for the CL filter without touching the project: uv run --no-sync --with cleanlab ...
```

With `"subsample": 2000, "test_subsample": 500`, small `epochs`, and for DivideMix
`"warm_up": 1, "num_workers": 0` (macOS spawn), plus `"max_epochs_this_run"` to exercise
pause/resume, and `"r51_min_members": 2` for a 2-member R5.1 smoke run. Set `R52_MPS=1` to use
Apple MPS instead of the CPU.
