"""Export bounded GDN source evidence from the pinned image; no device access."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    source = Path('/opt/tt-metal')
    destination = Path('/results')
    kernel = 'ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule'
    paths = sorted(path for path in (source / kernel).rglob('*') if path.is_file())
    if not paths:
        raise RuntimeError('Fused decode source absent from image')
    paths += [source / name for name in (
        'models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py',
        'models/demos/blackhole/qwen36/tt/gdn/tp.py',
        'models/demos/blackhole/qwen36/tt/attention/tp.py',
        'models/demos/blackhole/qwen36/tt/model.py',
        'models/demos/blackhole/qwen36/tt/qwen36_vllm.py',
        'ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp',
    )]
    manifest = dict(scope='Source parity only, not binary equivalence or performance', files=[])
    manifest['revision'] = subprocess.check_output(
        ['git', '-c', 'safe.directory=/opt/tt-metal', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    for path in paths:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest['files'].append(dict(path=str(relative), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
