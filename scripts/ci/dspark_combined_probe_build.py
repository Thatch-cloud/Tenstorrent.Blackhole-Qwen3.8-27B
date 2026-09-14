"""Validate numerical probes against actual combined-runtime build evidence."""

from pathlib import Path

from dspark_64k_build import validate_build
from dspark_hardware_gate import digest


def validate_combined(root, unused_output):
    return validate_build(root, '/experiment/results/dspark-64k-hardware-build.json',
        Path(__file__).parent)


def fingerprints(root, *, packer, expected_packer, audit_kernel,
        packer_compat=False, precise_native=False):
    if packer_compat or not precise_native:
        raise ValueError('Precise hardware kernel and original packer required')
    root = Path(root)
    audit_kernel(root)
    evidence = validate_combined(root, None)
    directory = root / 'ttnn/cpp/ttnn/operations/transformer/sdpa'
    sources = [path.relative_to(root) for path in directory.rglob('*')
        if path.is_file() and path.suffix in ('.cpp', '.hpp', '.h')]
    if not sources:
        raise ValueError('Native SDPA sources required')
    sources.extend(map(Path, (
        'tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h',
        'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h',
        'tt_metal/hw/inc/api/compute/reduce.h')))
    result = {str(path): digest(root / path) for path in sorted(
        [Path(packer), *map(Path, evidence['binaries']), *sources])}
    if result[packer] != expected_packer or any(
            result[name] != checksum for name, checksum in evidence['binaries'].items()):
        raise ValueError('Combined binary or original hardware packer changed')
    return result
