"""Merge R5.1 outputs from several Kaggle runs and run the analysis locally.

Usage: python merge_r51.py <dataset> <noise> <out_results_dir> <run_dir> [<run_dir> ...]
Each run_dir is a downloaded Kaggle output with r51_out/; npz files are symlinked into one
folder so every variant pair is compared on identical samples.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kaggle', 'src'))
os.environ.setdefault('SND_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import snd_kaggle as K  # noqa: E402
import r51_analysis as A  # noqa: E402

dataset, noise, results = sys.argv[1], int(sys.argv[2]), sys.argv[3]
merged = os.path.join(results, 'merged_r51_out')
os.makedirs(merged, exist_ok=True)
for run in sys.argv[4:]:
    for path in glob.glob(os.path.join(run, 'r51_out', '*.npz')):
        link = os.path.join(merged, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(path), link)
repo = K.setup_repo('main')
table = K.load_label_table(K.extract_preds(repo, '513d833' if dataset == 'fashionmnist' else 'main', K.PROTOCOL[(dataset, noise)]['preds']))
A.report(merged, table, results)
