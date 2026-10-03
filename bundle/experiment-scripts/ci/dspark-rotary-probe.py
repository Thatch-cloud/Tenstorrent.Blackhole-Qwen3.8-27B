"""Simulator-only DSpark rotary on synthetic heads, absolute YaRN positions and exact own-policy replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_head_preparation import rope_reference
from dspark_intake import FILES
from dspark_rope_tables import DSparkRotary
from dspark_rotary_device import CASES, COMPOSED_POLICY, POLICY, execute, fixtures
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dspark-rotary-probe.py', 'dspark_rotary_device.py', 'dspark_rope_tables.py', 'dspark_intake.py',
    'attention_batch.py', 'draft_head_preparation.py', 'feature_projection.py', 'gdn_multitoken_conv.py')
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
BINARY_SHA256 = 'd2652fc01a6836b4d567a788a9c11d8f6cb238bb480bbf68d0e32ee4037c3e24'
CPU_REPORT_SHA256 = '97424ecd6a1355d24c7f4c73bfd07c9ca3c8035e452b032cf91d20d6412043f9'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def fingerprints(root):
    binaries = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
    paths = [Path(PACKER), *map(Path, binaries)]
    for name in ('experimental/transformer/rotary_embedding_hf', 'copy/typecast',
            'data_movement/slice', 'data_movement/concat', 'eltwise/unary',
            'eltwise/binary', 'eltwise/binary_ng'):
        selected = [path.relative_to(root) for path in (root / 'ttnn/cpp/ttnn/operations' / name).rglob('*')
            if path.suffix in ('.cpp', '.hpp', '.h') and path.is_file()]
        if not selected:
            raise ValueError(f'Native rotary dependency source tree missing: {name}')
        paths.extend(selected)
    result = {str(path): digest(root / path) for path in sorted(paths)}
    if result[PACKER] != ORIGINAL_PACKER or any(result[name] != BINARY_SHA256 for name in binaries):
        raise ValueError('Original reviewed native runtime and packer required')
    return result


def configuration(path):
    if path.stat().st_size != FILES['config.json'][0] or digest(path) != FILES['config.json'][1]:
        raise ValueError('Pinned DSpark configuration required')
    report_path = Path(__file__).with_name('dspark-yarn-cpu-reference.json')
    if digest(report_path) != CPU_REPORT_SHA256:
        raise ValueError('Reviewed upstream CPU table comparison required')
    cpu_report = json.loads(report_path.read_text())
    if cpu_report.get('cpu_reference_passed') is not True or any(
            digest(Path(__file__).with_name(name)) != checksum for name, checksum in cpu_report['sources'].items()):
        raise ValueError('CPU table implementation changed since upstream comparison')
    return DSparkRotary(json.loads(path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--composed', action='store_true', help='Test separate elementwise path with bitwise CPU gate')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    rotary = configuration(options.config)
    root = Path(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', target_integrated=False,
        eligible_for_hardware=False, scope=__doc__, accuracy_policy=COMPOSED_POLICY if options.composed else POLICY,
        cases=[list(case) for case in CASES],
        config_sha256=FILES['config.json'][1], cpu_report_sha256=CPU_REPORT_SHA256,
        sources=source_hashes(), native_sources=fingerprints(root), eager_checks=[], replay_checks=[],
        input_checks=[], dependency_controls=[], stale_controls=[])
    mesh = trace = None
    persistent, transient = [], []

    def progress(stage):
        report['stage'] = stage
        print(json.dumps(dict(stage=stage)), flush=True)

    def equal_bits(actual, expected):
        return actual.shape == expected.shape and torch.equal(actual.contiguous().view(torch.int16),
            expected.contiguous().view(torch.int16))

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        for case, (heads, live_rows, padded_rows) in enumerate(CASES):
            patterns = fixtures(rotary, heads, live_rows, padded_rows)

            def upload(value, index, device=True):
                mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if index == 0 else ttnn.ReplicateTensorToMesh(mesh)
                return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
                    **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))

            persistent.extend(upload(value, index) for index, value in enumerate(patterns[0]))
            payloads = [[upload(value, index, False) for index, value in enumerate(pattern)] for pattern in patterns]
            bindings = [addresses(ttnn, value) for value in persistent]

            def update(pattern):
                for source, destination in zip(payloads[pattern], persistent, strict=True):
                    ttnn.copy_host_to_device_tensor(source, destination)
                ttnn.synchronize_device(mesh)

            def host(value, chip):
                return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

            def audit_inputs(phase, ordinal, pattern):
                for index, (source, value) in enumerate(zip(patterns[pattern], persistent, strict=True)):
                    for chip in range(2):
                        expected = source[chip:chip + 1] if index == 0 else source
                        if not equal_bits(host(value, chip), expected):
                            raise AssertionError('Rotary changed a borrowed head or position-table input')
                        report['input_checks'].append(dict(case=case, phase=phase, ordinal=ordinal,
                            tensor=index, chip=chip, exact=True))

            def run():
                return execute(ttnn, *persistent, transient, composed=options.composed)

            references = []
            for pattern, values in enumerate(patterns):
                progress(f'case_{case}_eager_{pattern}')
                update(pattern)
                output = run()
                ttnn.synchronize_device(mesh)
                observed = [host(output, chip) for chip in range(2)]
                for chip, actual in enumerate(observed):
                    expected = rope_reference(values[0][chip:chip + 1], *values[1:])
                    try:
                        torch.testing.assert_close(actual.float(), expected.float(), rtol=.01, atol=.01)
                        if options.composed and not equal_bits(actual, expected):
                            raise AssertionError('Composed rotary differs from the unchanged bitwise CPU reference')
                    except AssertionError:
                        path = options.output.with_suffix('.operands.pt')
                        torch.save(dict(case=case, pattern=pattern, chip=chip, live_rows=live_rows,
                            heads=values[0][chip:chip + 1], cosine=values[1], sine=values[2],
                            wide_inputs=[host(value, chip) for value in transient[:3]],
                            native_wide_output=host(transient[-2], chip), actual=actual, expected=expected), path)
                        report['failure_operands'] = dict(path=str(path), sha256=digest(path))
                        raise
                    report['eager_checks'].append(dict(case=case, pattern=pattern, chip=chip,
                        full_padded_close=True, cpu_bitwise_exact=equal_bits(actual, expected),
                        max_abs=float((actual.float() - expected.float()).abs().max()),
                        valid_max_abs=float((actual[..., :live_rows, :].float() - expected[..., :live_rows, :].float()).abs().max())))
                references.append(observed)
                audit_inputs('eager', pattern, pattern)
                release_owned(ttnn, transient)
                transient.clear()
            for chip in range(2):
                valid = [reference[chip][..., :live_rows, :] for reference in references]
                if equal_bits(valid[0], valid[1]) or not equal_bits(valid[2], valid[3]):
                    raise AssertionError('Absolute position or padding-isolation negative control failed')
                report['dependency_controls'].extend([
                    dict(case=case, chip=chip, control='positions', detected=True),
                    dict(case=case, chip=chip, control='padding', isolated=True)])
            update(0)
            progress(f'case_{case}_capture')
            trace, output = capture_operation(ttnn, mesh, run)
            output_bindings = addresses(ttnn, output)
            for repetition, pattern in enumerate((0, 1, 2, 3, 0)):
                progress(f'case_{case}_replay_{repetition}')
                update(pattern)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                stable = bindings == [addresses(ttnn, value) for value in persistent] and output_bindings == addresses(ttnn, output)
                for chip in range(2):
                    if not stable or not equal_bits(host(output, chip), references[pattern][chip]):
                        raise AssertionError('Rotary trace changed its bindings or own full padded eager result')
                    report['replay_checks'].append(dict(case=case, repetition=repetition, pattern=pattern,
                        chip=chip, exact=True, bindings_stable=True))
                audit_inputs('replay', repetition, pattern)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for chip in range(2):
                actual = host(output, chip)
                if not equal_bits(actual, references[0][chip]) or equal_bits(actual, references[1][chip]):
                    raise AssertionError('Omitted absolute-position update was not distinguishable')
                report['stale_controls'].append(dict(case=case, chip=chip, missing_update_detected=True))
            ttnn.release_trace(mesh, trace)
            trace = None
            release_owned(ttnn, transient)
            release_owned(ttnn, persistent)
            transient.clear()
            persistent.clear()
            progress(f'case_{case}_complete')
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
            report['sources_after'], report['native_sources_after'] = source_hashes(), fingerprints(root)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Rotary probe or native sources changed during execution')
            progress('complete' if report['passed'] else 'failed')
        finally:
            if (not report['closed_cleanly'] or report.get('sources_after') != report['sources']
                    or report.get('native_sources_after') != report['native_sources']):
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
