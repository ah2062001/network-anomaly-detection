#!/usr/bin/env python3
"""Read-only validation of three local input CSVs against archived manifest.

This script is not a reconstruction of upstream flow extraction or labels.
"""
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('.'))
p.add_argument('--skip-row-count',action='store_true')
args=p.parse_args()
root=args.root.resolve()
manifest=json.loads((Path(__file__).resolve().parents[1]/'docs/input_fingerprints.json').read_text())
failed=[]
for key,spec in manifest.items():
    path=root/spec['path']
    if not path.is_file():
        print(f'MISSING {key}: {path}');failed.append(key);continue
    digest=hashlib.sha256();nl=0;last=b''
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024),b''):
            digest.update(chunk)
            if not args.skip_row_count: nl += chunk.count(b'\n')
            last=chunk[-1:] if chunk else last
    got=digest.hexdigest();ok=got==spec['sha256']
    if not args.skip_row_count:
        # Compare physical data-line counts after excluding the header.
        rows=nl-(1 if last==b'\n' else 0)
        ok=ok and rows==spec['rows']
        print(f'{key}: sha256={"PASS" if got==spec["sha256"] else "FAIL"}; '
              f'physical_data_rows={rows} (expected {spec["rows"]})')
    else:print(f'{key}: sha256={"PASS" if ok else "FAIL"}')
    if not ok:failed.append(key)
print('PASS' if not failed else f'FAIL: {failed}')
raise SystemExit(bool(failed))
