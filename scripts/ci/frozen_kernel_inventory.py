"""Read-only bounded kernel/profiler inventory; never opens accelerator devices."""

import argparse
import hashlib
import json
from pathlib import Path


KERNELS = (
    'ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device/kernels/compute/decode_gated_delta_rule.cpp',
    'ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device/kernels/dataflow/reader_decode_gated_delta_rule.cpp',
    'ttnn/cpp/ttnn/operations/transformer/decode_gated_delta_rule/device/kernels/dataflow/writer_decode_gated_delta_rule.cpp',
    'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp',
)
MARKERS = ('DeviceZoneScopedN', 'PROFILE_KERNEL', 'PROFILER', 'counter', 'cb_wait', 'cb_reserve')


def inventory(root, output):
    root, output = Path(root).resolve(strict=True), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'inventory.json').exists():
        raise ValueError('Fresh inventory output required')
    report = dict(scope='Read-only source capability inventory, not measured performance',
        devices_opened=False, files=[], missing=[], profiler_candidates=[])

    def retain(path):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError('Source escaped the pinned runtime root')
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if len(data) > 512 * 1024:
            raise ValueError('Unexpectedly large source: ' + relative)
        destination = output / 'sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        report['files'].append(dict(path=relative, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))

    for relative in KERNELS:
        path = root / relative
        if path.is_file():
            retain(path)
        else:
            report['missing'].append(relative)
    for relative in ('tt_metal/tools/profiler', 'tt_metal/hw/inc'):
        directory = root / relative
        if not directory.is_dir():
            report['missing'].append(relative)
            continue
        candidates = sorted(path for path in directory.rglob('*') if path.is_file()
            and path.suffix in ('.h', '.hpp', '.cpp', '.py') and 'profil' in path.as_posix().lower())
        if len(candidates) > 120:
            raise ValueError('Profiler inventory exceeds reviewed file budget')
        for path in candidates:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError('Profiler source escaped runtime root')
            if path.stat().st_size > 512 * 1024:
                continue
            lines = path.read_text(errors='replace').splitlines()
            matches = [dict(line=index, text=line[:240]) for index, line in enumerate(lines, 1)
                if any(marker in line for marker in MARKERS)]
            if matches:
                report['profiler_candidates'].append(dict(path=path.relative_to(root).as_posix(),
                    matches=matches[:30], total_matches=len(matches)))
                retain(path)
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/opt/tt-metal'))
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    report = inventory(options.root, options.output)
    print(json.dumps(dict(files=len(report['files']), missing=report['missing'], devices_opened=False)))


if __name__ == '__main__':
    main()
