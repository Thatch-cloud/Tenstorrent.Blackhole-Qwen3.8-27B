"""Independent DSpark attention simulator reconciliation, not learned-model or hardware qualification."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from dspark_attention import CONTEXTS, POLICY
from dspark_markov_gate import coordinates
from native_draft_sdpa import KERNEL_DIRECTORY, SIGNATURE, SOURCE_HASHES, patched_sources


def qualify(report, *, sources, native, exit_status, packer_compat=False, precise_native=False, key_chunk_size=32):
    if (type(key_chunk_size) is not int or key_chunk_size not in (32, 64)
            or type(report.get('key_chunk_size', 32)) is not int or report.get('key_chunk_size', 32) != key_chunk_size):
        raise ValueError('Explicit matching native key chunk size required; old evidence cannot qualify chunk64')
    if (type(packer_compat) is not bool or report.get('packer_compat') is not packer_compat
            or type(precise_native) is not bool or report.get('precise_native') is not precise_native
            or exit_status.strip() != '0' or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('error') or report.get('backend') != 'simulator'
            or report.get('accuracy_policy') != POLICY or report.get('target_integrated') is not False
            or report.get('eligible_for_hardware') is not False or report.get('contexts') != list(CONTEXTS)
            or not sources or not native or report.get('sources') != sources or report.get('sources_after') != sources
            or report.get('native_sources') != native or report.get('native_sources_after') != native):
        raise ValueError('Complete clean source-bound simulator matrix and explicit packer scope required')
    audit = report.get('kernel_audit')
    if precise_native:
        expected = {name: native.get(f'{KERNEL_DIRECTORY}/{name}') for name in SOURCE_HASHES}
        if (not isinstance(audit, dict) or audit.get('original') != SOURCE_HASHES or audit.get('patched') != expected
                or audit.get('signature') != {str(index): value for index, value in SIGNATURE.items()}
                or not audit.get('runtime_sources') or any(native.get(name) != checksum
                    for name, checksum in audit['runtime_sources'].items())):
            raise ValueError('Reviewed precise exponential signature, original sources and matching runtime required')
    elif audit is not None:
        raise ValueError('Unmodified native policy must not carry a precise-graft audit')
    if any(type(context) is not int for context in report['contexts']):
        raise ValueError('Integer context coordinates required')
    coordinates(report.get('eager_checks'), ('context', 'pattern', 'chip'),
        {(context, pattern, chip) for context in CONTEXTS for pattern in range(5) for chip in range(2)}, ('full_padded_close',))
    coordinates(report.get('replay_checks'), ('context', 'ordinal', 'pattern', 'chip'),
        {(context, ordinal, pattern, chip) for context in CONTEXTS for ordinal, pattern in enumerate((0, 1, 2, 3, 4, 0))
            for chip in range(2)}, ('exact', 'bindings_stable'))
    coordinates(report.get('input_checks'), ('context', 'phase', 'ordinal', 'tensor', 'chip'),
        {(context, phase, ordinal, tensor, chip) for context in CONTEXTS for phase, count in (('eager', 5), ('replay', 6))
            for ordinal in range(count) for tensor in range(4) for chip in range(2)}, ('exact',))
    coordinates(report.get('dependency_controls'), ('context', 'control', 'chip'),
        {(context, control, chip) for context in CONTEXTS for control in ('padding', 'oldest_history', 'future_proposal')
            for chip in range(2)}, ('passed',))
    coordinates(report.get('stale_controls'), ('context', 'chip'),
        {(context, chip) for context in CONTEXTS for chip in range(2)}, ('missing_update_detected',))
    for entry in report['eager_checks']:
        errors = (entry.get('max_abs'), entry.get('valid_max_abs'))
        if (any(type(error) not in (int, float) or not math.isfinite(error) or error < 0 for error in errors)
                or errors[1] > errors[0] or type(entry.get('failed_elements')) is not int or entry['failed_elements'] != 0):
            raise ValueError('Finite full and live-row numerical diagnostics required')
    counts = {name: len(report[name]) for name in
        ('eager_checks', 'replay_checks', 'input_checks', 'dependency_controls', 'stale_controls')}
    return dict(passed=True, checks=sum(counts.values()), counts=counts, packer_compat=packer_compat, precise_native=precise_native,
        key_chunk_size=key_chunk_size,
        worst_cpu_error=max(entry['max_abs'] for entry in report['eager_checks']),
        worst_valid_cpu_error=max(entry['valid_max_abs'] for entry in report['eager_checks']),
        target_integrated=False, eligible_for_hardware=False,
        scope='Synthetic full-context proposal attention only; not learned backbone, target correctness, quality or TG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--metal-root', type=Path, required=True)
    parser.add_argument('--packer-compat', action='store_true')
    parser.add_argument('--precise-native', action='store_true')
    parser.add_argument('--key-chunk-size', type=int, choices=(32, 64), default=32)
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('dspark_attention_probe', Path(__file__).with_name('dspark-attention-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    native = probe.fingerprints(options.metal_root)
    if options.packer_compat:
        native[probe.PACKER] = probe.COMPAT_PACKER
    if options.precise_native:
        original = {name: (options.metal_root / KERNEL_DIRECTORY / name).read_bytes() for name in SOURCE_HASHES}
        native.update({f'{KERNEL_DIRECTORY}/{name}': hashlib.sha256(source).hexdigest()
            for name, source in patched_sources(original).items()})
    result = qualify(json.loads(options.report.read_text()), sources=probe.source_hashes(), native=native,
        exit_status=options.exit_status.read_text(), packer_compat=options.packer_compat, precise_native=options.precise_native,
        key_chunk_size=options.key_chunk_size)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
