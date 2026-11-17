#!/usr/bin/env python3
"""Verify sha256 and byte length of package files listed in SHA256SUMS.json."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
entries=json.loads((root/'SHA256SUMS.json').read_text())['files']
failed=[]
for r in entries:
    path=root/r['path']
    if not path.is_file() or path.stat().st_size!=r['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
        failed.append(r['path'])
print(f'Verified {len(entries)-len(failed)}/{len(entries)} release files')
if failed:print('Failed:',failed)
raise SystemExit(bool(failed))
