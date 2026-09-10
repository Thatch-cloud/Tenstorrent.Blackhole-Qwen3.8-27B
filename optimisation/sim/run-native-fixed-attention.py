"""Own the pinned simulator-only packer compatibility change for one native attention probe."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ORIGINAL = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
PATCHED = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'


def restore_owned_sources(directory, original, patched, lock):
    current = {name: (directory / name).read_bytes() for name in original}
    if current == original and not lock.exists():
        return
    if current != patched or not lock.exists():
        raise ValueError('Owned SDPA sources changed unexpectedly; refusing to overwrite')
    for name, data in original.items():
        (directory / name).write_bytes(data)
    if {name: (directory / name).read_bytes() for name in original} != original:
        raise ValueError('SDPA restoration failed')
    lock.unlink()


def patched_bytes(original, patch):
    if hashlib.sha256(original).hexdigest() != ORIGINAL:
        raise ValueError('Original pinned Blackhole packer required')
    lines = patch.splitlines(keepends=True)
    body = lines[next(index for index, line in enumerate(lines) if line.startswith(b'@@')) + 1:]
    before = b''.join(line[1:] for line in body if line.startswith((b' ', b'-')))
    after = b''.join(line[1:] for line in body if line.startswith((b' ', b'+')))
    if original.count(before) != 1:
        raise ValueError('Unique pinned packer patch context required')
    result = original.replace(before, after)
    if hashlib.sha256(result).hexdigest() != PATCHED:
        raise ValueError('Packer compatibility transformation differs from pinned result')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learned-layer', action='store_true')
    parser.add_argument('--norm-scatter', action='store_true')
    parser.add_argument('--target-attention', action='store_true')
    parser.add_argument('--approx-draft', action='store_true')
    parser.add_argument('--dram-projection', choices=('gate', 'up', 'down'))
    parser.add_argument('--dram-replay', action='store_true')
    parser.add_argument('--dram-mlp', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    options = parser.parse_args()
    if options.dram_mlp and (options.dram_projection or options.dram_replay or options.approx_draft
            or options.target_attention or options.norm_scatter or options.learned_layer or options.checkpoint):
        raise ValueError('Complete DRAM MLP is an isolated probe')
    if options.dram_replay and not options.dram_projection:
        raise ValueError('DRAM replay requires an explicit projection')
    if options.dram_projection and (options.approx_draft or options.target_attention
            or options.norm_scatter or options.learned_layer or options.checkpoint is not None):
        raise ValueError('DRAM projection is an isolated probe')
    if options.approx_draft and (options.target_attention or options.norm_scatter
            or options.learned_layer or options.checkpoint is not None):
        raise ValueError('Approximate draft attention is an isolated probe')
    if options.target_attention and (options.norm_scatter or options.learned_layer or options.checkpoint is not None):
        raise ValueError('Target attention is an isolated probe')
    if options.norm_scatter and (options.learned_layer or options.checkpoint is not None):
        raise ValueError('Norm scatter is an isolated probe')
    if options.learned_layer != (options.checkpoint is not None):
        raise ValueError('Learned-layer probe requires an explicit checkpoint')
    directory = Path(__file__).resolve().parent
    root = Path(os.environ.get('SIM_ROOT', '/opt/ttsim'))
    sys.path.insert(0, str(directory.parents[1] / 'scripts/ci'))
    from native_draft_sdpa import KERNEL_DIRECTORY, patched_sources
    native_directory = root / 'tt-metal' / KERNEL_DIRECTORY
    native_lock = native_directory / '.qwen-precise-draft.lock'
    if native_lock.exists():
        raise ValueError('Another precise SDPA owner is active')
    from native_draft_sdpa import SOURCE_HASHES
    native_original = {name: (native_directory / name).read_bytes() for name in SOURCE_HASHES}
    native_patched = patched_sources(native_original)
    packer = root / 'tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
    lock = packer.with_name('.qwen-native-fixed-packer.lock')
    original = packer.read_bytes()
    patched = patched_bytes(original, (directory / 'blackhole-packer-zero-flags.patch').read_bytes().replace(b'\r\n', b'\n'))
    with lock.open('x') as owner:
        owner.write(str(os.getpid()) + '\n')
    changed = False
    try:
        if packer.read_bytes() != original:
            raise ValueError('Packer changed before ownership was established')
        packer.write_bytes(patched)
        changed = True
        environment = dict(os.environ, QWEN_SIM_PACKER_ZERO_GRAFT='1', QWEN_SIM_SHARED_BDF='1',
            QWEN_SIM_BOUNDED_MEMORY='1', QWEN_SIM_DISPATCH_PROBE='dspark-native-cached-layer-probe'
            if options.learned_layer else 'dspark-native-fixed-attention-probe',
            OMP_NUM_THREADS='1', KERNEL_TIMEOUT='1800')
        arguments = ['--checkpoint', str(options.checkpoint.resolve())] if options.learned_layer else []
        wrapper = 'run-native-layer-dispatch-probe.sh' if options.learned_layer else 'run-dispatch-probe.sh'
        if options.norm_scatter:
            wrapper = 'run-native-layer-dispatch-probe.sh'
            environment['QWEN_SIM_LAYER_PROBE'] = 'gdn-norm-scatter-probe'
        if options.target_attention:
            wrapper = 'run-native-layer-dispatch-probe.sh'
            environment['QWEN_SIM_LAYER_PROBE'] = 'target-t16-attention-probe'
        if options.approx_draft:
            wrapper = 'run-native-layer-dispatch-probe.sh'
            environment['QWEN_SIM_LAYER_PROBE'] = 'dspark-approx-fixed-attention-probe'
        if options.dram_projection:
            wrapper = 'run-native-layer-dispatch-probe.sh'
            environment['QWEN_SIM_LAYER_PROBE'] = 'dram-sharded-projection-probe'
            arguments = ['--projection', options.dram_projection]
            if options.dram_replay:
                arguments.append('--replay')
        if options.dram_mlp:
            wrapper = 'run-native-layer-dispatch-probe.sh'
            environment['QWEN_SIM_LAYER_PROBE'] = 'dram-mlp-probe'
        result = subprocess.run(['bash', str(directory / wrapper), *arguments], env=environment)
        return result.returncode
    finally:
        try:
            restore_owned_sources(native_directory, native_original, native_patched, native_lock)
        finally:
            if changed:
                if packer.read_bytes() != patched:
                    raise ValueError('Owned packer changed externally; refusing to overwrite it')
                packer.write_bytes(original)
                if hashlib.sha256(packer.read_bytes()).hexdigest() != ORIGINAL:
                    raise ValueError('Original packer restoration failed')
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
