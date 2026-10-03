"""Compact run notebook (code cells only) with pinned overrides, printed as one JSON line."""
import json, sys
base, title, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
nb = json.load(open(base))
hook = "CONFIG.update(json.loads(os.environ.get('SND_CONFIG_OVERRIDES', '{}')))"
cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [f'# {title}\n', 'Built from SiameseNoiseDetection/kaggle (see kaggle/README.md).']}]
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if hook in src:
        src = src.replace(hook, f'CONFIG.update({overrides!r})   # this run\n' + hook)
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': src.splitlines(True)})
out = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
       'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                    'language_info': {'name': 'python'}}}
print(json.dumps(out))
