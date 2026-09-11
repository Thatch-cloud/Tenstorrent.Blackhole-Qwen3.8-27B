"""Run captured T16 fusion with owned simulator-only packer compatibility."""

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    options = parser.parse_args()
    directory = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('packer_compat', directory / 'run-native-fixed-attention.py')
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    root = Path(os.environ.get('SIM_ROOT', '/opt/ttsim'))
    packer = root / 'tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
    original = packer.read_bytes()
    patched = compatibility.patched_bytes(original,
        (directory / 'blackhole-packer-zero-flags.patch').read_bytes().replace(b'\r\n', b'\n'))
    lock = packer.with_name('.qwen-native-fixed-packer.lock')
    with lock.open('x') as owner:
        owner.write(str(os.getpid()) + '\n')
    changed = False
    try:
        if packer.read_bytes() != original:
            raise ValueError('Packer changed before ownership was established')
        packer.write_bytes(patched)
        changed = True
        environment = dict(os.environ, QWEN_SIM_PACKER_ZERO_GRAFT='1', QWEN_SIM_SHARED_BDF='1',
            QWEN_SIM_DISPATCH_PROBE='fused-batch-probe', OMP_NUM_THREADS='1', KERNEL_TIMEOUT='3600')
        return subprocess.run(['bash', str(directory / 'run-dispatch-probe.sh'),
            '--fixture', str(options.fixture.resolve()), '--device-weight-check', '--trace-replay',
            '--trace-t16'], env=environment).returncode
    finally:
        if changed:
            if packer.read_bytes() != patched:
                raise ValueError('Owned packer changed externally; refusing to overwrite it')
            packer.write_bytes(original)
            if hashlib.sha256(packer.read_bytes()).hexdigest() != compatibility.ORIGINAL:
                raise ValueError('Original packer restoration failed')
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
