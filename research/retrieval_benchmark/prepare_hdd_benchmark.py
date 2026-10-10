"""Copy only read-only serving files for two representative indexes to the HDD.

Does not touch OpenAlex downloads. Copies to a dedicated benchmark directory and
uses temporary names before publishing each completed file.
"""
import json
import shutil
from pathlib import Path

from bench.common import write_json
from bench.serve_bench import NAME, ROOT, configs, search_files

DEST = Path('/mnt/bulk/scholarsreadinglist-retrieval-bench/index2')
if not Path('/mnt/bulk').is_mount():
    raise RuntimeError('The bulk drive is not mounted')
manifest = []
for name in ('lateon-pool4-2bit-np32', 'denseon-ivfsq8-np64'):
    cfg = configs()[name]
    files = [Path(p) for p in search_files(cfg)]
    if cfg['engine'] == 'warp':
        files.append(cfg['dir'].parent / f'{NAME}.ids.json')
    else:
        files.append(cfg['dir'] / f'{NAME}.ids.json')
    for src in files:
        dst = DEST / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + '.part')
        shutil.copyfile(src, tmp)
        assert tmp.stat().st_size == src.stat().st_size
        tmp.replace(dst)
        manifest.append({'source': str(src), 'destination': str(dst), 'bytes': src.stat().st_size})
write_json(DEST / 'copy-manifest.json', manifest)
print(f'Copied {len(manifest)} files, {sum(x["bytes"] for x in manifest) / 1e9:.2f} GB to {DEST}')
