"""Own the pinned simulator-only packer compatibility change for one native attention probe."""

import hashlib
import os
from pathlib import Path
import subprocess


ORIGINAL = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
PATCHED = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'


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
    directory = Path(__file__).resolve().parent
    root = Path(os.environ.get('SIM_ROOT', '/opt/ttsim'))
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
            QWEN_SIM_BOUNDED_MEMORY='1', QWEN_SIM_DISPATCH_PROBE='dspark-native-fixed-attention-probe',
            OMP_NUM_THREADS='1', KERNEL_TIMEOUT='1800')
        result = subprocess.run(['bash', str(directory / 'run-dispatch-probe.sh')], env=environment)
        return result.returncode
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
