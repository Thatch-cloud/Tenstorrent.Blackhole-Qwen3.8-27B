"""Simulator-only DSpark full attention: seven live queries, full history, exact mask controls and replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_attention import CONTEXTS, POLICY, execute, fixtures, numerical_difference, reference, validate_mask
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from native_draft_sdpa import audit_active_kernel, run_precise_probe


SOURCES = ('dspark-attention-probe.py', 'dspark_attention.py', 'draft_attention.py',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py', 'native_draft_sdpa.py')
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
COMPAT_PACKER = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'
BINARY_SHA256 = 'd2652fc01a6836b4d567a788a9c11d8f6cb238bb480bbf68d0e32ee4037c3e24'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def fingerprints(root, *, packer_compat=False, precise_native=False):
    binaries = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
    directory = root / 'ttnn/cpp/ttnn/operations/transformer/sdpa'
    paths = [path.relative_to(root) for path in directory.rglob('*')
        if path.is_file() and path.suffix in ('.cpp', '.hpp', '.h')]
    if type(precise_native) is not bool:
        raise ValueError('Explicit precise-native scope required')
    if precise_native:
        audit_active_kernel(root)
    if not paths or (not precise_native and (directory / 'device/kernels/compute/.qwen-precise-draft.lock').exists()):
        raise ValueError('Unowned native SDPA source tree required')
    result = {str(path): digest(root / path) for path in sorted([Path(PACKER), *map(Path, binaries), *paths])}
    if (type(packer_compat) is not bool or result[PACKER] != (COMPAT_PACKER if packer_compat else ORIGINAL_PACKER)
            or any(result[name] != BINARY_SHA256 for name in binaries)):
        raise ValueError('Explicit pinned packer scope and original native binaries required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precise-native', action='store_true', help='Use the existing owned precise exponential graft')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    kernel_audit = run_precise_probe(__file__) if options.precise_native else None
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    packer_compat = os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') == '1'
    report = dict(passed=False, closed_cleanly=False, backend='simulator', target_integrated=False,
        eligible_for_hardware=False, accuracy_policy=POLICY, contexts=list(CONTEXTS), scope=__doc__,
        sources=source_hashes(), native_sources=fingerprints(root, packer_compat=packer_compat, precise_native=options.precise_native),
        packer_compat=packer_compat, precise_native=options.precise_native, kernel_audit=kernel_audit,
        eager_checks=[], replay_checks=[],
        input_checks=[], dependency_controls=[], stale_controls=[])
    mesh = trace = None
    persistent, transient = [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    def equal_bits(actual, expected):
        return actual.shape == expected.shape and torch.equal(actual.contiguous().view(torch.int16),
            expected.contiguous().view(torch.int16))

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        for context in CONTEXTS:
            patterns = fixtures(context)
            for values in patterns:
                validate_mask(values[3], context)

            def upload(value, index, device=True):
                mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if index < 3 else ttnn.ReplicateTensorToMesh(mesh)
                return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
                    **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))

            persistent.extend(upload(value, index) for index, value in enumerate(patterns[0]))
            payloads = [[upload(value, index, False) for index, value in enumerate(values)] for values in patterns]
            bindings = [addresses(ttnn, value) for value in persistent]

            def update(pattern):
                validate_mask(patterns[pattern][3], context)
                for source, destination in zip(payloads[pattern], persistent, strict=True):
                    ttnn.copy_host_to_device_tensor(source, destination)
                ttnn.synchronize_device(mesh)

            def host(value, chip):
                return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

            def audit_inputs(phase, ordinal, pattern):
                for index, (expected, value) in enumerate(zip(patterns[pattern], persistent, strict=True)):
                    for chip in range(2):
                        if not equal_bits(host(value, chip), expected[chip:chip + 1] if index < 3 else expected):
                            raise AssertionError('Attention changed a borrowed input')
                        report['input_checks'].append(dict(context=context, phase=phase, ordinal=ordinal,
                            tensor=index, chip=chip, exact=True))

            def run():
                output = execute(ttnn, *persistent, context_rows=context, mask_validated=True)
                transient.append(output)
                return output

            references = []
            for pattern, values in enumerate(patterns):
                progress(f'context_{context}_eager_{pattern}')
                update(pattern)
                output = run()
                ttnn.synchronize_device(mesh)
                observed = [host(output, chip) for chip in range(2)]
                for chip, actual in enumerate(observed):
                    expected = reference(values, chip)
                    diagnostics = numerical_difference(actual, expected)
                    if not diagnostics['full_padded_close'] and 'failure_operands' not in report:
                        path = options.output.with_suffix('.operands.pt')
                        torch.save(dict(context=context, pattern=pattern, chip=chip, values=values,
                            actual=actual, expected=expected), path)
                        report['failure_operands'] = dict(path=str(path), sha256=digest(path))
                    report['eager_checks'].append(dict(context=context, pattern=pattern, chip=chip, **diagnostics))
                references.append(observed)
                audit_inputs('eager', pattern, pattern)
                release_owned(ttnn, transient)
                transient.clear()
            for chip in range(2):
                if not equal_bits(references[0][chip], references[2][chip]):
                    raise AssertionError('Fully masked key/value padding changed attention output')
                for control, pattern in (('oldest_history', 3), ('future_proposal', 4)):
                    if equal_bits(references[0][chip][..., :1, :], references[pattern][chip][..., :1, :]):
                        raise AssertionError(f'Full noncausal context dependency missing: {control}')
                    report['dependency_controls'].append(dict(context=context, chip=chip, control=control, passed=True))
                report['dependency_controls'].append(dict(context=context, chip=chip, control='padding', passed=True))
            update(0)
            progress(f'context_{context}_capture')
            trace, output = capture_operation(ttnn, mesh, run)
            output_bindings = addresses(ttnn, output)
            for ordinal, pattern in enumerate((0, 1, 2, 3, 4, 0)):
                progress(f'context_{context}_replay_{ordinal}')
                update(pattern)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                stable = bindings == [addresses(ttnn, value) for value in persistent] and output_bindings == addresses(ttnn, output)
                for chip in range(2):
                    if not stable or not equal_bits(host(output, chip), references[pattern][chip]):
                        raise AssertionError('Captured attention changed its bindings or padded eager result')
                    report['replay_checks'].append(dict(context=context, ordinal=ordinal, pattern=pattern,
                        chip=chip, exact=True, bindings_stable=True))
                audit_inputs('replay', ordinal, pattern)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for chip in range(2):
                actual = host(output, chip)
                if not equal_bits(actual, references[0][chip]) or equal_bits(actual, references[1][chip]):
                    raise AssertionError('Omitted input update was not distinguishable')
                report['stale_controls'].append(dict(context=context, chip=chip, missing_update_detected=True))
            ttnn.release_trace(mesh, trace)
            trace = None
            release_owned(ttnn, transient)
            release_owned(ttnn, persistent)
            transient.clear()
            persistent.clear()
            progress(f'context_{context}_complete')
        failed = sum(not entry['full_padded_close'] for entry in report['eager_checks'])
        if failed:
            raise AssertionError(f'{failed} numerical comparisons fail the unchanged 0.01/0.01 gate; controls do not override them')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, transient)
                release_owned(ttnn, persistent)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
            report['sources_after'] = source_hashes()
            report['native_sources_after'] = fingerprints(root, packer_compat=packer_compat, precise_native=options.precise_native)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Attention probe or native sources changed during execution')
            progress('complete' if report['passed'] else 'failed')
        finally:
            if (not report['closed_cleanly'] or report.get('sources_after') != report['sources']
                    or report.get('native_sources_after') != report['native_sources']):
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
