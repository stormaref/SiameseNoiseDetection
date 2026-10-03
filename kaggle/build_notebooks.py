"""Assemble the self-contained Kaggle notebooks from src/*.py.

Each notebook embeds its helper modules as %%writefile cells, so uploading the single
.ipynb to Kaggle is enough. Usage: python build_notebooks.py <out_dir>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')

KAGGLE_HOWTO = """\
**Kaggle settings:** Accelerator *GPU T4 x2* (P100 also works, one worker), Internet *On*
(phone-verified account), Persistence *Files only*. Run with **Save Version -> Save & Run All
(Commit)** so it runs in the background for up to 12 h.

**Resuming:** jobs are saved as they finish. If the session ends with jobs left, open the
notebook, *Add Input -> Your Work -> this notebook* (the previous version's output), and
commit again: finished jobs are restored and skipped. Repeat until the launch cell prints
that nothing is left; the analysis cell then reports the final tables.

**Outputs** (in the version's Output tab): `{out}/` raw per-job results, `results/` CSV
tables, `logs/` per-GPU training logs."""


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n').splitlines(True)}


def writefile(name):
    with open(os.path.join(SRC, name)) as f:
        return code(f'%%writefile {name}\n' + f.read())


SETUP = """
import json, os, subprocess, sys, time
SESSION_START = time.time()
os.chdir('/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd())
subprocess.run('nvidia-smi -L', shell=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + PIP, check=False)
sys.path.insert(0, os.getcwd())
import snd_kaggle as K
repo = K.setup_repo(CONFIG['code_ref'])
print('code:', subprocess.run(['git', '-C', repo, 'log', '-1', '--format=%h %ad %s', '--date=short'],
                              capture_output=True, text=True).stdout)
"""

LAUNCH = """
for train in (True, False):                     # download once, before the workers start
    K.base_dataset(CONFIG['dataset'], train)
CONFIG['out_dir'] = K.restore_previous_outputs(OUT_NAME)
CONFIG['session_start'] = SESSION_START
with open('config.json', 'w') as f:
    json.dump(CONFIG, f, indent=1)
codes = K.launch_workers(WORKER, 'config.json', poll=CONFIG.get('poll', 300))
print('worker exit codes:', codes)
done = len(K.JobQueue(CONFIG['out_dir']).summaries())
print(f'{done} of {N_JOBS} jobs finished' + ('' if done >= N_JOBS else
      ' -- commit a new version with this one attached as input to continue'))
"""

R51_INTRO = """
# R5.1 — Which part of the detector carries the signal?

Reviewer 5, comment 1: compare prediction disagreement of (i) ordinary separately trained models,
(ii) contrastive Siamese models, and (iii) an explicit embedding-level measure.

Both variants share the published noisy labels and outer folds (rebuilt from the committed
prediction CSVs), the same seeded inner split, the same 200k training pairs per member, the same
initial weights, schedule and early stopping. They differ only in the objective:

| variant | objective |
|---|---|
| `siamese` | ours: CE on both branches + the contrastive loss `same·d² + (1-same)·[m-d]₊²` (`snd.training.contrastive`, the corrected form used for the CIFAR-10 runs) |
| `ce` | CE only (contrastive weight 0): an ordinary classifier ensemble on identical data |
| `ce_linear` | optional: backbone + plain linear head, CE only |

Each member saves, for the held-out outer fold, its softmax outputs and embeddings, and the
embeddings of its own training subset. The analysis then scores every held-out sample by
disagreement (the published detector), mean confidence in the observed label, and three
embedding-level scores computed inside each member (centroid margin, k-NN label disagreement,
across-member instability of the distance to the observed-label centroid; embedding spaces of
separately trained members are not aligned, so raw embedding variance is not meaningful).
It also reports the per-member misclassification rate on noisy vs clean samples (R4.3) and
paired-bootstrap CIs for every AUC difference.

**Budget (rough, refine from the first member's log):** CIFAR-10 ResNet-50 ~1.2–1.6 h per member
on a T4 with AMP, so one outer fold × 2 variants × 10 members ≈ 28 GPU-h ≈ 14 h on T4×2
(two commits). Fashion-MNIST ResNet-34 ≈ 0.3 h per member, so one outer fold ≈ 3–4 h on T4×2.
"""

R56_INTRO = """
# R5.6 — Is the gain from better labels or from the downstream model?

Reviewer 5, comment 6: evaluate the cleaned datasets with several classifiers under one
identical protocol, against the noisy data and an oracle-clean dataset.

Training sets, all from the same published noisy labels:
`noisy` (as observed) · `ours` (DetectAndRelabel at the oracle thresholds `td`, `tr`) ·
`oracle_filtered` (every truly noisy sample removed: separates *fewer samples* from *better
labels*) · `oracle_clean` (all ground-truth labels: the ceiling).

Classifiers: PreAct-ResNet34 (the paper's), ResNet-18, VGG-16-BN, MobileNetV2 (32×32 stems).
Protocol = the paper's `FinalModelTester` settings for the dataset (Adam, 5-epoch linear warm-up
then constant rate, label smoothing 0.1, stratified validation split, early stopping on
validation accuracy, best weights, clean test set), applied unchanged to every classifier and
training set.

The `ours` × `preact_resnet34` cells follow the paper's Table 3 protocol, so with
`preds_ref='main'` they are also the downstream numbers for the re-run CIFAR-10 predictions
(the paper's Table 3 still reports the round-3 run, `preds_ref='513d833'`).

**Budget (rough):** CIFAR-10 ≈ 0.4–0.8 h per job on a T4; 4 sets × 4 classifiers × 3 seeds = 48 jobs
≈ 30 GPU-h ≈ 15 h on T4×2. Jobs run seed by seed, so each commit leaves complete seeds.
"""

R51_CONFIG = """
import json, os
CONFIG = dict(
    dataset='cifar10', noise=20,      # ('cifar10', 20|30|40) or ('fashionmnist', 20|30|40)
    preds_ref='main',                 # run whose predictions are the reference row: 'main' = the
                                      # CIFAR-10 re-run, '513d833' = the round-3 paper run (same
                                      # noise draw and folds, so training here is unaffected)
    code_ref='main',                  # branch providing the snd package
    outer_folds=[1],                  # outer fold(s) to score; fold 1 is unaffected by the
                                      # inner-model reuse in the CIFAR-10 re-run (folds 4, 5, 8)
    variants=['siamese', 'ce'],       # add 'ce_linear' if budget allows
    members=10, seed=0, amp=True,
    session_hours=11.5, member_hours=1.6,        # don't start a member that would cross 11.5 h
    save_checkpoints=True, loader_workers=2,
)
CONFIG.update(json.loads(os.environ.get('SND_CONFIG_OVERRIDES', '{}')))   # local smoke tests only
OUT_NAME, WORKER, PIP = 'r51_out', 'r51_worker.py', ['timm']
N_JOBS = len(CONFIG['outer_folds']) * CONFIG['members'] * len(CONFIG['variants'])
"""

R51_ANALYSIS = """
import r51_analysis as A
table = K.load_label_table(K.extract_preds(repo, CONFIG['preds_ref'],
                                           K.PROTOCOL[(CONFIG['dataset'], CONFIG['noise'])]['preds']))
table, _ = K.smoke_subsample(table, CONFIG)
scores, thresholds, members, boot = A.report(CONFIG['out_dir'], table, 'results',
                                             min_members=CONFIG['members'])
"""

R56_CONFIG = """
import json, os
# Oracle thresholds (td, tr) per run. 'main' = the CIFAR-10 re-run (best noise-F1 on the paper's
# 6..10 grid with tr=10 -- confirm before use); '513d833' = the round-3 paper run (its Table 2).
# Fashion-MNIST predictions are the same in both.
THRESHOLDS = {
    ('main', 'cifar10', 20): (10, 10), ('main', 'cifar10', 30): (8, 10), ('main', 'cifar10', 40): (8, 10),
    ('513d833', 'cifar10', 20): (8, 10), ('513d833', 'cifar10', 30): (7, 10), ('513d833', 'cifar10', 40): (6, 10),
    ('main', 'fashionmnist', 20): (9, 10), ('main', 'fashionmnist', 30): (9, 10), ('main', 'fashionmnist', 40): (7, 10),
}
CONFIG = dict(
    dataset='cifar10', noise=20, preds_ref='main', code_ref='main',
    train_sets=['noisy', 'ours', 'oracle_filtered', 'oracle_clean'],
    archs=['preact_resnet34', 'resnet18', 'vgg16_bn', 'mobilenet_v2'],
    seeds=[0, 1, 2], amp=True,
    session_hours=11.5, job_hours=0.9, loader_workers=2,
)
CONFIG.update(json.loads(os.environ.get('SND_CONFIG_OVERRIDES', '{}')))   # local smoke tests only
CONFIG['td'], CONFIG['tr'] = THRESHOLDS[(CONFIG['preds_ref'], CONFIG['dataset'], CONFIG['noise'])]
OUT_NAME, WORKER, PIP = 'r56_out', 'r56_worker.py', ['timm']
N_JOBS = len(CONFIG['train_sets']) * len(CONFIG['archs']) * len(CONFIG['seeds'])
"""

R56_ANALYSIS = """
import r56_analysis as A
_ = A.report(CONFIG['out_dir'], 'results')
"""


def notebook(cells):
    return {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                         'language_info': {'name': 'python'},
                         'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True,
                                    'isGpuEnabled': True}}}


def build(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    specs = {
        'r51_detector_isolation.ipynb': (R51_INTRO, 'r51_out', R51_CONFIG,
                                         ['snd_kaggle.py', 'r51_worker.py', 'r51_analysis.py'], R51_ANALYSIS),
        'r56_downstream_classifiers.ipynb': (R56_INTRO, 'r56_out', R56_CONFIG,
                                             ['snd_kaggle.py', 'r56_worker.py', 'r56_analysis.py'], R56_ANALYSIS),
    }
    for name, (intro, out, config, files, analysis) in specs.items():
        cells = [md(intro), md(KAGGLE_HOWTO.format(out=out)), md('## 1. Configuration'), code(config),
                 md('## 2. Helper modules (written to the working directory)')]
        cells += [writefile(f) for f in files]
        cells += [md('## 3. Setup'), code(SETUP), md('## 4. Train (all GPUs, resumable)'), code(LAUNCH),
                  md('## 5. Analysis'), code(analysis)]
        with open(os.path.join(out_dir, name), 'w') as f:
            json.dump(notebook(cells), f, indent=1)
        print('wrote', os.path.join(out_dir, name))


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'notebooks'))
