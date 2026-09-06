"""Hardware recurrence-only exactness and paired cost; no model throughput claim."""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path

from gdn_multitoken import HANDOFF_HASHES, HASHES, cb_plan, execute as execute_kernel, load_kernels
from gdn_multitoken_timing import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm-gate', action='store_true')
    args = parser.parse_args()
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch silicon required')
    import torch
    import ttnn
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_decode_packed_ttnn

    kernels = load_kernels(fuse_norm_gate=args.norm_gate)
    report = dict(passed=False, checks=[], continuations=[], negative_controls=[], timings=[], native_hashes=HASHES,
                  generated_hashes={name: hashlib.sha256(value.encode()).hexdigest() for name, value in kernels.items()},
                  scope='Synthetic packed recurrence only: no conv, norm/gate, real weights or model throughput claim',
                  cores_per_chip=24, feedback_dtype='BF16', feedback_bytes_per_core=32768,
                  cb_bytes_per_core=sum(cb_plan(args.norm_gate)[0].values()) * 2048 + sum(cb_plan(args.norm_gate)[1].values()) * 4096)
    if args.norm_gate:
        report.update(scope='Synthetic recurrence plus native fused norm/gate; no conv, real weights, model or timing',
                      handoff_runtime_hashes=HANDOFF_HASHES,
                      feedback_ring='CB5 initial-state ring; one full-ring reader push then compute-only feedback',
                      output_layout='[1,T,3072] TILE L1; head-local assembly, zero padded rows',
                      state_placement='Candidate all-prefix DRAM; native oracle state L1; correctness only')
    path = Path('/experiment/results/gdn-multitoken-norm.json' if args.norm_gate else '/experiment/results/gdn-multitoken.json')
    stack_file = None
    context = {}

    def stage(name, **details):
        if args.norm_gate:
            report['last_stage'] = dict(stage=name, **context, **details)
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps(report['last_stage']), flush=True)

    if args.norm_gate:
        stack_file = path.with_suffix('.stacks.log').open('w')
        faulthandler.dump_traceback_later(120, repeat=True, file=stack_file)
    mesh = None
    trace = None
    try:
        stage('mesh-open')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        stage('mesh-ready')

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(value):
            shards = ttnn.get_device_tensors(value)
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            return [ttnn.to_torch(shard).clone() for shard in shards]

        def exact(actual, expected):
            return len(actual) == len(expected) == 2 and all(torch.equal(value, reference)
                for value, reference in zip(actual, expected, strict=True))

        def release(values):
            for value in values:
                ttnn.deallocate(value)

        def native(values, state):
            return recurrent_gated_delta_rule_decode_packed_ttnn(*values[:3], 8, 24, 128, 128,
                initial_state=state, device=mesh, high_precision=False, inplace_state=False, return_row_major=True,
                **(dict(z=values[3], norm_w=norm_w) if args.norm_gate else {}))

        def execute(values, state):
            return execute_kernel(mesh, *values[:3], state, kernels,
                                  **(dict(z=values[3], norm_w=norm_w) if args.norm_gate else {}))

        def host_output(value):
            return [shard.reshape(-1, 1, 24, 128) for shard in host(value)]

        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            for rows in (1, 2, 4, 8, 16):
                context.update(seed=seed, rows=rows)
                stage('fixture-upload')
                values = [torch.randn(1, rows, 5120).bfloat16(), torch.rand(1, rows, 24).bfloat16(),
                          (-torch.rand(1, rows, 24)).bfloat16()]
                norm_w = None
                if args.norm_gate:
                    values.append(torch.randn(1, rows, 3072).bfloat16())
                    norm_w = upload((1 + torch.randn(1, 1, 128) * 0.1).bfloat16())
                initial_host = (torch.randn(1, 24, 128, 128) * 0.05).bfloat16()
                initial = upload(initial_host)
                packed = [upload(value) for value in values]
                tokens = [[upload(value[:, index:index + 1]) for value in values] for index in range(rows)]
                correction = [upload(torch.randn(1, 1, 5120).bfloat16()), upload(torch.rand(1, 1, 24).bfloat16()),
                              upload((-torch.rand(1, 1, 24)).bfloat16())]
                if args.norm_gate:
                    correction.append(upload(torch.randn(1, 1, 3072).bfloat16()))
                reference = [initial]
                expected_output = []
                expected_state = [host(initial)]
                for index, token in enumerate(tokens):
                    stage('native-oracle', token=index)
                    output, state = native(token, reference[-1])
                    expected_output.append(host_output(output))
                    expected_state.append(host(state))
                    reference.append(state)
                    release([output])
                    stage('native-oracle-validated', token=index)
                expected_continuations = []
                for prefix, state in enumerate(reference):
                    stage('native-continuation-oracle', prefix=prefix)
                    output, after = native(correction, state)
                    expected_continuations.append((host(output), host(after)))
                    release([output, after])
                stage('candidate-warm-submit')
                warm = execute(packed, initial)
                stage('candidate-warm-submitted')
                release(warm)
                ttnn.synchronize_device(mesh)
                stage('candidate-warm-complete')
                stage('candidate-capture')
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                captured = execute(packed, initial)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                stage('candidate-captured')
                for mode in ('eager', 'trace'):
                    stage('candidate-evaluate', mode=mode)
                    if mode == 'trace':
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        actual = captured
                    else:
                        actual = execute(packed, initial)
                    output_host, states_host = host_output(actual[0]), host(actual[1])
                    stage('candidate-readback-complete', mode=mode)
                    if not exact(host(initial), expected_state[0]):
                        raise AssertionError('Kernel modified initial state')
                    for index in range(rows):
                        if not exact([value[index:index + 1] for value in output_host], expected_output[index]):
                            raise AssertionError(f'Output mismatch {seed=} {rows=} {mode=} {index=}')
                        if not exact([value[index:index + 1] for value in states_host], expected_state[index + 1]):
                            raise AssertionError(f'Prefix mismatch {seed=} {rows=} {mode=} {index=}')
                    for prefix in range(rows + 1):
                        stage('restored-continuation', mode=mode, prefix=prefix)
                        restored = upload(initial_host if prefix == 0 else states_host[0][prefix - 1:prefix])
                        output, after = native(correction, restored)
                        expected = expected_continuations[prefix]
                        if not exact(host(output), expected[0]) or not exact(host(after), expected[1]):
                            raise AssertionError('Restored prefix continuation mismatch')
                        release([output, after, restored])
                        report['continuations'].append(dict(seed=seed, rows=rows, mode=mode, prefix=prefix, exact=True))
                    report['checks'].append(dict(seed=seed, rows=rows, mode=mode, exact=True))
                    if mode == 'eager':
                        release(actual)
                if exact(expected_state[0], expected_state[-1]):
                    raise AssertionError('Stale-state control is ineffective')
                output, after = native(correction, initial)
                detected = not exact(host(output), expected_continuations[-1][0])
                release([output, after])
                if not detected:
                    raise AssertionError('Stale state failed to change continuation')
                report['negative_controls'].append(dict(seed=seed, rows=rows, stale_state_detected=True))
                if args.norm_gate:
                    stage('fixture-cleanup')
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    release([*captured, *packed, *reference, *correction, norm_w,
                             *[value for token in tokens for value in token]])
                    stage('fixture-complete')
                    path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(dict(seed=seed, rows=rows, norm_gate=True, exact=True)), flush=True)
                    continue
                serial_trace = None
                serial_values = []
                try:
                    ttnn.synchronize_device(mesh)
                    serial_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    state = initial
                    for token in tokens:
                        output, state = native(token, state)
                        serial_values.append((output, state))
                    ttnn.end_trace_capture(mesh, serial_trace, cq_id=0)

                    def replay(arm):
                        ttnn.execute_trace(mesh, serial_trace if arm == 'serial' else trace, cq_id=0, blocking=True)

                    def validate_timing(arm):
                        if not exact(host(initial), expected_state[0]):
                            raise AssertionError('Timed kernel modified initial state')
                        if arm == 'serial':
                            outputs = [host(pair[0]) for pair in serial_values]
                            states = [host(pair[1]) for pair in serial_values]
                        else:
                            packed_outputs, packed_states = host(captured[0]), host(captured[1])
                            outputs = [[value[index:index + 1] for value in packed_outputs] for index in range(rows)]
                            states = [[value[index:index + 1] for value in packed_states] for index in range(rows)]
                        for index in range(rows):
                            if not exact(outputs[index], expected_output[index]) or not exact(states[index], expected_state[index + 1]):
                                raise AssertionError(f'Timing output/prefix mismatch {arm=} {index=}')

                    timing = measure(replay, validate_timing, lambda: ttnn.synchronize_device(mesh))
                    report['timings'].append(dict(seed=seed, rows=rows, **timing))
                finally:
                    if serial_trace is not None:
                        ttnn.release_trace(mesh, serial_trace)
                    release([value for pair in serial_values for value in pair])
                ttnn.release_trace(mesh, trace)
                trace = None
                release([*captured, *packed, *reference, *correction, *[value for token in tokens for value in token]])
                path.write_text(json.dumps(report, indent=2))
                print(json.dumps(dict(seed=seed, rows=rows, exact=True)), flush=True)
        if len(report['checks']) != 30 or len(report['continuations']) != 216 or len(report['negative_controls']) != 15:
            raise AssertionError('Incomplete recurrence gate')
        if not args.norm_gate and (len(report['timings']) != 15 or sum(timing['timed_replays'] for timing in report['timings']) != 1800):
            raise AssertionError('Incomplete paired recurrence timing')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            stage('mesh-close')
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
        if stack_file is not None:
            faulthandler.cancel_dump_traceback_later()
            stack_file.close()


if __name__ == '__main__':
    main()
