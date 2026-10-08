"""Download every file listed in a saved `list_notebook_session_output` result.

Usage: python fetch_kaggle_output.py <listing.json|.txt> <dest_dir> [--skip-pt]
The listing holds signed kaggleusercontent URLs; files land under dest_dir/<file_name>,
and the session log (stdout+stderr) is written to dest_dir/session_log.txt.
"""
import json
import os
import sys
import subprocess
import time

listing, dest = sys.argv[1], sys.argv[2]
skip_pt = '--skip-pt' in sys.argv
d = json.load(open(listing))
os.makedirs(dest, exist_ok=True)
n = 0
failed = []
for f in d.get('files', []):
    name = f['file_name']
    if skip_pt and name.endswith('.pt'):
        continue
    path = os.path.join(dest, name)
    if os.path.exists(path) and not path.endswith('.done'):
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    for attempt in range(8):                         # the CDN resets connections now and then
        r = subprocess.run(['curl', '-sS', '--fail', '--http1.1', '--connect-timeout', '20',
                            '-o', tmp, f['url']], capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp, path)
            n += 1
            break
        time.sleep(2 + 3 * attempt)
    else:
        failed.append(name)
log = d.get('log') or '[]'
log = json.loads(log) if isinstance(log, str) else log
with open(os.path.join(dest, 'session_log.txt'), 'w') as fh:
    for e in log:
        fh.write(e.get('data', ''))
print(f'{n} files downloaded to {dest}; {len(d.get("files", []))} listed; failed: {failed}')
