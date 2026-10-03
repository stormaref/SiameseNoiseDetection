"""Shared plumbing for the round-4 revision experiments (Kaggle or any CUDA box).

What lives here:
  * repo setup: clone the public SiameseNoiseDetection repo, import `snd` from it;
  * the exact noisy labels and outer folds of a published run, rebuilt from its
    committed prediction CSVs (the IDN generator re-randomises, so the CSVs are the
    only faithful record of which labels were corrupted);
  * the per-dataset protocol as actually run (main.ipynb cells 16/25/35, 59/71/80,
    106-124), which differs in places from snd/config.py and the CLI;
  * a crash-safe job queue that keeps every visible GPU busy and survives Kaggle's
    12-hour session limit by resuming from a previous version's output.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO_URL = 'https://github.com/stormaref/SiameseNoiseDetection.git'
ON_KAGGLE = os.path.isdir('/kaggle/working')
WORK = '/kaggle/working' if ON_KAGGLE else os.path.abspath(os.environ.get('SND_WORK', 'work'))
# repo clone, extracted preds and datasets are re-creatable: keep them out of the saved output
SCRATCH = '/tmp/snd' if ON_KAGGLE else WORK
DATA_ROOT = os.path.join(SCRATCH, 'data')

CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

# Siamese-ensemble protocol per published run (main.ipynb; values that differ
# from snd/config.py are noted). `aug` names an entry of `augmentation()` below.
PROTOCOL = {
    ('cifar10', 20): dict(preds='cifar10(20)', backbone='resnet50', pre_trained=True, emb=64,
                          wd=5e-4, patience=8, margin=2, aug='cifar_crop'),
    ('cifar10', 30): dict(preds='cifar10(30)', backbone='resnet50', pre_trained=True, emb=64,
                          wd=5e-4, patience=10, margin=2, aug='cifar_crop'),
    ('cifar10', 40): dict(preds='cifar10(40)', backbone='resnet50', pre_trained=True, emb=64,
                          wd=5e-4, patience=15, margin=2, aug='cifar_affine'),
    ('fashionmnist', 20): dict(preds='fmnist(20)', backbone='resnet34', pre_trained=False, emb=128,
                               wd=1e-3, patience=12, margin=2, aug='fmnist_plain'),
    ('fashionmnist', 30): dict(preds='fmnist(30)', backbone='resnet34', pre_trained=False, emb=128,
                               wd=1e-3, patience=12, margin=2, aug='fmnist_plain'),
    ('fashionmnist', 40): dict(preds='fmnist(40)', backbone='resnet34', pre_trained=False, emb=128,
                               wd=1e-3, patience=12, margin=2, aug='fmnist_norm'),
}
# Shared by every run above.
SIAMESE_COMMON = dict(lr=5e-5, batch_size=2048, train_pairs=200_000, val_pairs=20_000,
                      dropout=0.5, label_smoothing=0.1, max_epochs=1000, inner_folds=10)

# Downstream-classifier protocol (FinalEvaluator calls, main.ipynb cells 106-124).
DOWNSTREAM = {
    'cifar10': dict(lr=1e-3, wd=5.46e-5, batch_size=256, patience=20, warmup=5,
                    smoothing=0.1, val_ratio=0.1, max_epochs=200),
    'fashionmnist': dict(lr=2e-4, wd=1.12e-6, batch_size=256, patience=10, warmup=5,
                         smoothing=0.1, val_ratio=0.05, max_epochs=200),
}


# --------------------------------------------------------------------------- repo
def setup_repo(code_ref='main', repo_dir=None):
    """Clone (or reuse) the code repo, check out `code_ref`, make `snd` importable."""
    repo_dir = repo_dir or os.environ.get('SND_REPO') or os.path.join(SCRATCH, 'SiameseNoiseDetection')
    if not os.path.isdir(os.path.join(repo_dir, '.git')):
        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
        subprocess.run(['git', 'clone', '--quiet', REPO_URL, repo_dir], check=True)
    if not os.environ.get('SND_REPO'):      # never move the HEAD of a local working copy
        subprocess.run(['git', '-C', repo_dir, 'checkout', '--quiet', code_ref], check=True)
    # local testing: SND_SRC points at an extracted `src/` of code_ref when the working
    # copy in SND_REPO is on another branch
    src = os.environ.get('SND_SRC') or os.path.join(repo_dir, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    return repo_dir


def _resolve_ref(repo_dir, ref):
    for candidate in (f'origin/{ref}', ref):
        ok = subprocess.run(['git', '-C', repo_dir, 'rev-parse', '--verify', '--quiet', candidate],
                            capture_output=True).returncode == 0
        if ok:
            return candidate
    raise ValueError(f'git ref {ref!r} not found in {repo_dir}')


def extract_preds(repo_dir, preds_ref, preds_name):
    """Materialise preds/<preds_name> as committed on branch `preds_ref`; return its dir."""
    out = os.path.join(SCRATCH, 'preds', preds_ref, preds_name)
    if not glob.glob(os.path.join(out, '*', 'fold*_analysis.csv')):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ref = _resolve_ref(repo_dir, preds_ref)
        archive = subprocess.run(['git', '-C', repo_dir, 'archive', ref, f'preds/{preds_name}'],
                                 capture_output=True, check=True).stdout
        tmp = out + '.tmp'
        os.makedirs(tmp, exist_ok=True)
        subprocess.run(['tar', '-x', '-C', tmp], input=archive, check=True)
        shutil.move(os.path.join(tmp, 'preds', preds_name), out)
        shutil.rmtree(tmp)
    return out


# --------------------------------------------------------------------------- labels
def load_label_table(preds_dir):
    """One row per training sample: noisy/true label, noise flag, outer fold, saved preds."""
    frames = []
    for path in glob.glob(os.path.join(preds_dir, '*', 'fold*_analysis.csv')):
        fold = int(re.search(r'fold(\d+)_analysis', path).group(1))
        df = pd.read_csv(path)
        df['outer_fold'] = fold
        frames.append(df)
    table = pd.concat(frames).set_index('index').sort_index()
    table['is_noisy'] = table['is_noisy'].astype(str).str.lower().eq('true')
    if not table.index.is_unique:
        raise ValueError(f'{preds_dir}: a sample appears in more than one outer fold')
    return table


def base_dataset(name, train=True, attempts=5):
    from torchvision.datasets import CIFAR10, FashionMNIST
    cls = {'cifar10': CIFAR10, 'fashionmnist': FashionMNIST}[name]
    for attempt in range(attempts):             # the mirrors fail transiently now and then
        try:
            return cls(root=DATA_ROOT, train=train, download=True)
        except RuntimeError as error:
            if attempt == attempts - 1:
                raise
            print(f'download failed ({error}); retrying in 30 s', flush=True)
            time.sleep(30)


def prefetch_backbone(dataset, noise, attempts=5):
    """Download the pretrained backbone once, before the workers start (both would otherwise
    fetch it at the same time, and download.pytorch.org can be slow from Kaggle)."""
    proto = PROTOCOL.get((dataset, noise))
    if not proto or not proto['pre_trained']:
        return
    from torchvision import models
    for attempt in range(attempts):
        try:
            getattr(models, proto['backbone'])(weights='DEFAULT')
            return
        except Exception as error:                  # network errors surface as several types
            if attempt == attempts - 1:
                raise
            print(f'weights download failed ({error}); retrying in 30 s', flush=True)
            time.sleep(30)


def noisy_train_set(name, table):
    """The torchvision training set with its targets replaced by the published noisy labels."""
    ds = base_dataset(name, train=True)
    if len(table) != len(ds) or (table.index.to_numpy() != np.arange(len(ds))).any():
        raise ValueError('prediction CSVs do not cover the training set exactly once')
    true = np.asarray(ds.targets)
    if (table['real_label'].to_numpy() != true).any():
        raise ValueError('real_label in the CSVs does not match torchvision targets')
    ds.targets = [int(y) for y in table['noisy_label']]
    return ds


def smoke_subsample(table, cfg):
    """Local smoke tests only: a fixed random subset, re-indexed 0..n-1 (no-op on real runs)."""
    if not cfg.get('subsample'):
        return table, None
    keep = np.sort(np.random.default_rng(0).choice(len(table), cfg['subsample'], replace=False))
    return table.iloc[keep].reset_index(drop=True), keep


def augmentation(name):
    """Transforms exactly as in main.ipynb (scoring always uses `eval`)."""
    from torchvision import transforms as T
    cifar_norm = [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    gray = [T.Grayscale(num_output_channels=3), T.ToTensor()]
    table = {
        'cifar_crop': (T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(0.5),
                                  T.RandomRotation(15), *cifar_norm]),
                       T.Compose(cifar_norm)),
        'cifar_affine': (T.Compose([T.RandomRotation(15), T.RandomHorizontalFlip(0.5),
                                    T.RandomAffine(0, translate=(0.1, 0.1)),
                                    T.RandomResizedCrop(32, scale=(0.9, 1.0)), *cifar_norm]),
                         T.Compose(cifar_norm)),
        'fmnist_plain': (T.Compose(gray), T.Compose(gray)),
        'fmnist_norm': (T.Compose(gray + [T.Normalize((0.5,), (0.5,))]),
                        T.Compose(gray + [T.Normalize((0.5,), (0.5,))])),
        # downstream classifiers (FinalEvaluator cells)
        'down_cifar': (T.Compose([T.RandomRotation(15), T.RandomHorizontalFlip(0.5),
                                  T.RandomAffine(0, translate=(0.1, 0.1)),
                                  T.RandomResizedCrop(32, scale=(0.9, 1.0)), *cifar_norm]),
                       T.Compose(cifar_norm)),
        'down_fmnist': (T.Compose([T.Grayscale(num_output_channels=3), T.RandomHorizontalFlip(0.5),
                                   T.RandomRotation(10), T.RandomAffine(0, translate=(0.1, 0.1)),
                                   T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
                        T.Compose(gray + [T.Normalize((0.5,), (0.5,))])),
    }
    return table[name]


def cleaned_labels(table, td, tr):
    """Apply the published DetectAndRelabel rule (NoiseCleaner.advanced_clean).

    Returns an int array of new labels with -1 for removed samples: flagged if
    mistakes >= td; relabelled to a class predicted by >= tr members, else removed.
    """
    labels = table['noisy_label'].to_numpy().copy()
    for pos, (mistakes, preds) in enumerate(zip(table['mistakes'], table['preds'])):
        if mistakes < td:
            continue
        votes = np.bincount(np.fromstring(preds, dtype=int, sep='|'), minlength=10)
        winners = np.flatnonzero(votes >= tr)
        labels[pos] = winners[0] if len(winners) else -1
    return labels


# --------------------------------------------------------------------------- jobs
class JobQueue:
    """File-backed queue: a job is done when its result file exists.

    Workers claim a job by atomically creating `<result>.claim`; claims left behind by
    a killed session are ignored once older than `stale_hours`.
    """

    def __init__(self, out_dir, stale_hours=13):
        self.out_dir = out_dir
        self.stale = stale_hours * 3600
        os.makedirs(out_dir, exist_ok=True)

    def done(self, job_id):
        return os.path.exists(os.path.join(self.out_dir, job_id + '.done'))

    def claim(self, job_id):
        path = os.path.join(self.out_dir, job_id + '.claim')
        if os.path.exists(path) and time.time() - os.path.getmtime(path) > self.stale:
            os.remove(path)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def finish(self, job_id, summary):
        with open(os.path.join(self.out_dir, job_id + '.done'), 'w') as f:
            json.dump(summary, f, indent=1)
        claim = os.path.join(self.out_dir, job_id + '.claim')
        if os.path.exists(claim):
            os.remove(claim)

    def release(self, job_id):
        claim = os.path.join(self.out_dir, job_id + '.claim')
        if os.path.exists(claim):
            os.remove(claim)

    def summaries(self):
        rows = []
        for path in sorted(glob.glob(os.path.join(self.out_dir, '*.done'))):
            with open(path) as f:
                rows.append(json.load(f))
        return rows


def restore_previous_outputs(out_name):
    """Copy `<out_name>/` from any attached earlier notebook version into WORK.

    On Kaggle: Add Input -> Your Work -> this notebook (a previous version); its
    /kaggle/working appears under /kaggle/input/<slug>/.
    """
    target = os.path.join(WORK, out_name)
    os.makedirs(target, exist_ok=True)
    copied = 0
    for src in glob.glob(f'/kaggle/input/*/{out_name}') + glob.glob(f'/kaggle/input/*/*/{out_name}'):
        for path in glob.glob(os.path.join(src, '**', '*'), recursive=True):
            if os.path.isdir(path) or path.endswith('.claim'):
                continue
            dest = os.path.join(target, os.path.relpath(path, src))
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(path, dest)
                copied += 1
    print(f'restored {copied} files into {target}')
    return target


def launch_workers(script, config_path, n_gpus=None, log_dir=None, poll=60):
    """Run `python script config_path` once per GPU and stream a progress line per poll."""
    import torch
    n_gpus = n_gpus or max(torch.cuda.device_count(), 1)
    log_dir = log_dir or os.path.join(WORK, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for gpu in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), SND_WORKER=str(gpu))
        log = open(os.path.join(log_dir, f'worker{gpu}.log'), 'a')
        procs.append((subprocess.Popen([sys.executable, '-u', script, config_path], env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(20)                       # stagger dataset downloads / pair generation
    while any(p.poll() is None for p, _ in procs):
        time.sleep(poll)
        for gpu in range(n_gpus):
            tail = _last_line(os.path.join(log_dir, f'worker{gpu}.log'))
            print(f'[{time.strftime("%H:%M")}] gpu{gpu}: {tail}', flush=True)
    for p, log in procs:
        log.close()
    return [p.returncode for p, _ in procs]


def _last_line(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - 4000, 0))
            lines = f.read().decode(errors='replace').replace('\r', '\n').strip().splitlines()
            return lines[-1][-160:] if lines else ''
    except OSError:
        return ''


class Deadline:
    """Stop taking new jobs when the session is close to Kaggle's 12-hour limit."""

    def __init__(self, hours):
        self.end = time.time() + hours * 3600

    def allows(self, expected_hours):
        return time.time() + expected_hours * 3600 < self.end
