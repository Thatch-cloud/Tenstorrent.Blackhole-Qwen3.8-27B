"""Paired24-worker versus96-worker captured GDN cost, not model throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from gdn_multitoken import execute as control_execute, load_kernels
from gdn_multitoken_conv import release_owned
from gdn_multitoken_timing import measure
from gdn_vsplit_prepared import PreparedVSplit
import gdn_vsplit


def validate_matrix(fixtures):
    expected = {(seed, rows) for seed in (0, 1, 2) for rows in (1, 2, 4, 8, 16, 32)}
    if ({(record['seed'], record['rows']) for record in fixtures} != expected
            or len(fixtures) != len(expected)
            or any(record.get('exact') is not True or record['timed_replays'] != 120
                   or record.get('refreshed_checks') != 2 for record in fixtures)):
        raise AssertionError('Incomplete exact paired width/seed matrix')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefetch-inputs', action='store_true')
    args = parser.parse_args()
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit allocation of both hardware cards required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch hardware required')
    import torch
    import ttnn
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_decode_packed_ttnn

    path = Path('/experiment/results/gdn-vsplit-prefetch-timing.json' if args.prefetch_inputs
                else '/experiment/results/gdn-vsplit-timing.json')
    report = dict(passed=False, fixtures=[], prerequisite_run=34046231437,
        scope='Paired captured24-worker FNG versus96-worker recurrence plus24-worker FNG; synthetic kernel only',
        output_placement='Both arms BF16 TILE L1; all recurrent prefixes BF16 TILE DRAM',
        value_split=gdn_vsplit.audit(Path('/opt/tt-metal')),
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('gdn_vsplit_prepared.py', 'gdn_vsplit.py', 'gdn_multitoken.py', 'gdn-vsplit-timing.py')})
    if args.prefetch_inputs:
        import gdn_vsplit_prefetch
        report['prefetch'] = gdn_vsplit_prefetch.audit(Path('/opt/tt-metal'))
        report['prerequisite_run'] = 34048090329
        report['scope'] += '; recurrence input prefetch enabled'
    mesh = None
    kernels = load_kernels(fuse_norm_gate=True)
    report['control_kernels'] = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in kernels.items()}
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(tensor):
            parts = ttnn.get_device_tensors(tensor)
            if len(parts) != 2:
                raise AssertionError('Both chip-local values required')
            return [ttnn.to_torch(value).clone() for value in parts]

        for seed in (0, 1, 2):
            for rows in (1, 2, 4, 8, 16, 32):
                report['last_stage'] = dict(seed=seed, rows=rows, stage='fixture')
                path.write_text(json.dumps(report, indent=2))
                torch.manual_seed(seed)
                values = [torch.randn(1, rows, 5120).bfloat16(), torch.rand(1, rows, 24).bfloat16(),
                    (-torch.rand(1, rows, 24)).bfloat16(), torch.randn(1, rows, 3072).bfloat16()]
                initial_host = (torch.randn(1, 24, 128, 128) * .05).bfloat16()
                norm_w = upload((1 + torch.randn(1, 1, 128) * .1).bfloat16())
                initial = upload(initial_host)
                packed = [upload(value) for value in values]
                owned = [norm_w, initial, *packed]
                prepared = None
                traces = {}
                outputs = {}
                try:
                    def native_expected(source_values):
                        expected_outputs, expected_states = [[], []], [[], []]
                        state = initial
                        try:
                            for index in range(rows):
                                tokens = [upload(value[:, index:index + 1]) for value in source_values]
                                output = None
                                try:
                                    output, after = recurrent_gated_delta_rule_decode_packed_ttnn(*tokens[:3], 8, 24, 128, 128,
                                        initial_state=state, device=mesh, high_precision=False, inplace_state=False,
                                        return_row_major=True, z=tokens[3], norm_w=norm_w)
                                    if state is not initial:
                                        ttnn.deallocate(state)
                                    state = after
                                    for chip, value in enumerate(host(output)):
                                        expected_outputs[chip].append(value.reshape(1, 1, 24, 128))
                                    for chip, value in enumerate(host(state)):
                                        expected_states[chip].append(value)
                                finally:
                                    release_owned(ttnn, tokens + ([output] if output is not None else []))
                            return ([torch.cat(chip_values, dim=0) for chip_values in expected_outputs],
                                    [torch.cat(chip_values, dim=0) for chip_values in expected_states])
                        finally:
                            if state is not initial:
                                ttnn.deallocate(state)

                    expected_outputs, expected_states = native_expected(values)
                    refreshed_values = [-values[0], 1 - values[1], values[2] * .5, -values[3]]
                    refreshed_expected = native_expected(refreshed_values)
                    if all(torch.equal(before, after) for before, after in zip(expected_states, refreshed_expected[1], strict=True)):
                        raise AssertionError('Refresh fixture failed to change native recurrent history')
                    refreshed_inputs = [upload(value) for value in refreshed_values]
                    owned.extend(refreshed_inputs)
                    prepared = PreparedVSplit(mesh, *packed[:3], initial, z=packed[3], norm_w=norm_w,
                        experimental=True, output_memory=ttnn.L1_MEMORY_CONFIG, prefetch_inputs=args.prefetch_inputs)

                    def control():
                        return control_execute(mesh, *packed[:3], initial, kernels, z=packed[3], norm_w=norm_w)

                    warm = control()
                    prepared.run()
                    ttnn.synchronize_device(mesh)
                    release_owned(ttnn, warm)
                    traces['serial'], outputs['serial'] = capture_operation(ttnn, mesh, control)
                    traces['multitoken'], outputs['multitoken'] = capture_operation(ttnn, mesh, prepared.run)

                    def replay(arm):
                        ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=True)

                    def validate(arm):
                        actual_output, actual_states = host(outputs[arm][0]), host(outputs[arm][1])
                        for chip in range(2):
                            if (not torch.equal(actual_output[chip].reshape(rows, 1, 24, 128), expected_outputs[chip])
                                    or not torch.equal(actual_states[chip], expected_states[chip])):
                                raise AssertionError(f'Native oracle mismatch {seed=} {rows=} {arm=} {chip=}')
                        if not all(torch.equal(value, initial_host) for value in host(initial)):
                            raise AssertionError('Timed kernel mutated initial state')
                        for tensor, expected in zip(packed, values, strict=True):
                            if not all(torch.equal(value, expected) for value in host(tensor)):
                                raise AssertionError('Timed kernel mutated fixed input')
                        if arm == 'multitoken' and not all(torch.isfinite(value).all() for value in host(outputs[arm][2])):
                            raise AssertionError('Nonfinite FP32 bridge')

                    report['last_stage'] = dict(seed=seed, rows=rows, stage='paired-replay')
                    path.write_text(json.dumps(report, indent=2))
                    timing = measure(replay, validate, lambda: ttnn.synchronize_device(mesh))
                    for source, destination in zip(refreshed_inputs, packed, strict=True):
                        ttnn.copy(source, destination)
                    ttnn.synchronize_device(mesh)
                    values = refreshed_values
                    expected_outputs, expected_states = refreshed_expected
                    for arm in ('serial', 'multitoken'):
                        replay(arm)
                        validate(arm)
                    timing['refreshed_checks'] = 2
                    timing['control24_median_ms'] = timing.pop('serial_median_ms')
                    timing['candidate96_median_ms'] = timing.pop('multitoken_median_ms')
                    timing['paired_blocks'] = [dict(control24_ms=block['serial_ms'], candidate96_ms=block['multitoken_ms'],
                                                   speedup=block['speedup']) for block in timing['paired_blocks']]
                    for sample in timing['samples']:
                        sample['arm'] = 'control24' if sample['arm'] == 'serial' else 'candidate96'
                    timing['scope'] = report['scope']
                    result = dict(seed=seed, rows=rows, **timing)
                    report['fixtures'].append(result)
                    path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(result), flush=True)
                finally:
                    for trace in traces.values():
                        ttnn.release_trace(mesh, trace)
                    if prepared is not None:
                        prepared.close()
                    if 'serial' in outputs:
                        release_owned(ttnn, outputs['serial'])
                    release_owned(ttnn, owned)
        validate_matrix(report['fixtures'])
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
