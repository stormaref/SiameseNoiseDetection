"""Derive a run-specific notebook from a built one by pinning CONFIG overrides.

Usage: python make_run.py <base.ipynb> <out.ipynb> '<json overrides>'
The overrides are applied right after the CONFIG dict (before N_JOBS / thresholds).
"""
import json
import sys

base, out, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
nb = json.load(open(base))
hook = "CONFIG.update(json.loads(os.environ.get('SND_CONFIG_OVERRIDES', '{}')))"
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if hook in src:
        pinned = f'CONFIG.update({overrides!r})   # this run\n'
        src = src.replace(hook, pinned + hook)
        cell['source'] = src.splitlines(True)
        break
else:
    raise SystemExit('config hook not found')
json.dump(nb, open(out, 'w'), indent=1)
print(out, len(json.dumps(nb)), 'bytes')
