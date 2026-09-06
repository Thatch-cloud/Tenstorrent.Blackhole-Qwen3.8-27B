"""Hardware recurrence-only multi-token correctness gate; no model throughput claim."""

import hashlib
import json
import os
from pathlib import Path

from gdn_multitoken import HASHES, cb_plan, execute, load_kernels


def main():
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch silicon required')
    import torch
    import ttnn
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_decode_packed_ttnn

    kernels = load_kernels()
    report = dict(passed=False, checks=[], continuations=[], negative_controls=[], native_hashes=HASHES,
                  generated_hashes={name: hashlib.sha256(value.encode()).hexdigest() for name, value in kernels.items()},
                  scope='Synthetic packed recurrence only: no conv, norm/gate, real weights, full model or speed claim',
                  cores_per_chip=24, feedback_dtype='BF16', feedback_bytes_per_core=32768,
                  cb_bytes_per_core=sum(cb_plan()[0].values()) * 2048 + sum(cb_plan()[1].values()) * 4096)
    path = Path('/experiment/results/gdn-multitoken.json')
    mesh = None
    trace = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()

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
            return recurrent_gated_delta_rule_decode_packed_ttnn(*values, 8, 24, 128, 128,
                initial_state=state, device=mesh, high_precision=False, inplace_state=False, return_row_major=True)

        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            for rows in (1, 2, 4, 8, 16):
                values = [torch.randn(1, rows, 5120).bfloat16(), torch.rand(1, rows, 24).bfloat16(),
                          (-torch.rand(1, rows, 24)).bfloat16()]
                initial_host = (torch.randn(1, 24, 128, 128) * 0.05).bfloat16()
                initial = upload(initial_host)
                packed = [upload(value) for value in values]
                tokens = [[upload(value[:, index:index + 1]) for value in values] for index in range(rows)]
                correction = [upload(torch.randn(1, 1, 5120).bfloat16()), upload(torch.rand(1, 1, 24).bfloat16()),
                              upload((-torch.rand(1, 1, 24)).bfloat16())]
                reference = [initial]
                expected_output = []
                expected_state = [host(initial)]
                for token in tokens:
                    output, state = native(token, reference[-1])
                    expected_output.append(host(output))
                    expected_state.append(host(state))
                    reference.append(state)
                    release([output])
                expected_continuations = []
                for state in reference:
                    output, after = native(correction, state)
                    expected_continuations.append((host(output), host(after)))
                    release([output, after])
                warm = execute(mesh, *packed, initial, kernels)
                release(warm)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                captured = execute(mesh, *packed, initial, kernels)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for mode in ('eager', 'trace'):
                    if mode == 'trace':
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        actual = captured
                    else:
                        actual = execute(mesh, *packed, initial, kernels)
                    output_host, states_host = host(actual[0]), host(actual[1])
                    if not exact(host(initial), expected_state[0]):
                        raise AssertionError('Kernel modified initial state')
                    for index in range(rows):
                        if not exact([value[index:index + 1] for value in output_host], expected_output[index]):
                            raise AssertionError(f'Output mismatch {seed=} {rows=} {mode=} {index=}')
                        if not exact([value[index:index + 1] for value in states_host], expected_state[index + 1]):
                            raise AssertionError(f'Prefix mismatch {seed=} {rows=} {mode=} {index=}')
                    for prefix in range(rows + 1):
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
                ttnn.release_trace(mesh, trace)
                trace = None
                release([*captured, *packed, *reference, *correction, *[value for token in tokens for value in token]])
                path.write_text(json.dumps(report, indent=2))
                print(json.dumps(dict(seed=seed, rows=rows, exact=True)), flush=True)
        if len(report['checks']) != 30 or len(report['continuations']) != 216 or len(report['negative_controls']) != 15:
            raise AssertionError('Incomplete recurrence gate')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
