"""Real-weight projected-input convolution/recurrence gate; no throughput claim."""

import faulthandler
import hashlib
import json
import os
from pathlib import Path

from gdn_prefix import gated_decode
from gdn_multitoken import HANDOFF_HASHES, load_kernels, validate_handoff_runtime
from gdn_multitoken_conv import addresses, release_owned, run_projected


def main():
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch silicon required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_gdn_layer
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    root = Path('/opt/tt-metal')
    source = root / 'models/demos/blackhole/qwen36/tt/gdn/tp.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'f767d0648ae01b0b1c0bb7bf601f5490661707b845c11ce6dbebb86ba0f84dc9':
        raise ValueError('Audited native GDN changed')
    flags = ('QWEN_GDN_CONV_GATES', 'QWEN_GDN_PROJ_DIRECT', 'QWEN_GDN_PACKED_QKV', 'QWEN_GDN_NORM_GATE',
             'QWEN35_GDN_STATE_BF16', 'QWEN35_GDN_DECODE_BF16', 'QWEN_GDN_FUSED_DECODE', 'QWEN_GDN_FUSED_INPLACE')
    if any(os.environ.get(flag) != '1' for flag in flags):
        raise ValueError('Pinned native GDN flags required')
    validate_handoff_runtime(root)
    kernels = load_kernels(root, True)
    path = Path('/experiment/results/gdn-multitoken-conv.json')
    report = dict(passed=False, checks=[], projection_calls=0, handoff_runtime_hashes=HANDOFF_HASHES,
                  generated_hashes={name: hashlib.sha256(value.encode()).hexdigest() for name, value in kernels.items()},
                  adapter_sha256=hashlib.sha256(Path(__file__).with_name('gdn_multitoken_conv.py').read_bytes()).hexdigest(),
                  scope='One real-weight TP2 GDN projected-input/conv/gates/recurrence/norm path; excludes output projection, attention, full model, rollback continuation and timing')
    context = {}

    def stage(name, **details):
        report['last_stage'] = dict(stage=name, **context, **details)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['last_stage']), flush=True)

    mesh, trace = None, None
    with path.with_suffix('.stacks.log').open('w') as stacks:
        faulthandler.dump_traceback_later(120, repeat=True, file=stacks)
        try:
            stage('mesh-open')
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
            mesh.enable_program_cache()
            stage('real-weights-load')
            args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=256)
            layer = next(index for index, kind in enumerate(args.attention_type_list) if kind == 'linear_attention')
            gdn = TPGatedDeltaNet(mesh, args, load_gdn_weights_tp(mesh, load_gdn_layer(args.CKPT_DIR, layer), args), TT_CCL(mesh))
            gdn.B = 1
            gdn.reset_state()
            gdn._stable_state = True
            forward = gated_decode(gdn)
            report['layer'] = layer
            report['checkpoint_revision'] = Path(args.CKPT_DIR).name

            def upload(value):
                return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

            def host(value):
                shards = ttnn.get_device_tensors(value)
                if len(shards) != 2:
                    raise AssertionError('Both chips required')
                return [ttnn.to_torch(shard).clone() for shard in shards]

            def exact(actual, expected):
                return len(actual) == len(expected) == 2 and all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))

            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                for rows in (1, 2, 4, 8, 16):
                    context.update(seed=seed, rows=rows)
                    stage('fixture-initialize')
                    live = [gdn.rec_state, *gdn.conv_states]
                    entry = [upload((torch.randn(tuple(value.shape)) * 0.05).bfloat16()) for value in live]
                    for initial, destination in zip(entry, live, strict=True):
                        ttnn.copy(initial, destination)
                    entry_host = [host(value) for value in entry]
                    inputs = [upload((torch.randn(1, 1, 5120) * 0.1).bfloat16()) for token in range(rows)]
                    projected_rows, expected = [], []
                    original = gdn._project_qkvzab_raw

                    def record_projection(value, batch, memory):
                        projected = original(value, batch, memory)
                        projected_rows.append(ttnn.clone(projected, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                        report['projection_calls'] += 1
                        return projected

                    gdn._project_qkvzab_raw = record_projection
                    try:
                        for token, value in enumerate(inputs):
                            stage('native-reference', token=token)
                            output = forward(value)
                            expected.append((host(output), host(gdn.rec_state), [host(state) for state in gdn.conv_states]))
                            ttnn.deallocate(output)
                    finally:
                        gdn._project_qkvzab_raw = original
                    if len(projected_rows) != rows:
                        raise AssertionError('Native direct projection not engaged')
                    projected = ttnn.concat(projected_rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    working = [ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in entry[1:]]
                    working_addresses = [addresses(ttnn, value) for value in working]

                    def restore():
                        for initial, destination in zip(entry[1:], working, strict=True):
                            ttnn.copy(initial, destination)
                        ttnn.synchronize_device(mesh)

                    def candidate():
                        return run_projected(mesh, projected, entry[0], working, list(gdn.tw['conv_taps']),
                            gdn.tw['dt_bias'], gdn.tw['neg_exp_A'], gdn.tw['norm_w'], kernels)

                    def validate(result, mode):
                        outputs, states = host(result['output']), host(result['states'])
                        for token in range(rows):
                            if not exact([value[:, token:token + 1] for value in outputs], expected[token][0]):
                                raise AssertionError(f'Gated output mismatch {token=} {mode=}')
                            if not exact([value[token:token + 1] for value in states], expected[token][1]):
                                raise AssertionError(f'Recurrent prefix mismatch {token=} {mode=}')
                            for tap, value in enumerate(result['conv_prefixes'][token]):
                                if not exact(host(value), expected[token][2][tap]):
                                    raise AssertionError(f'Convolution prefix mismatch {token=} {tap=} {mode=}')
                        if not all(exact(host(value), saved) for value, saved in zip(entry, entry_host, strict=True)):
                            raise AssertionError('Initial state snapshots modified')
                        if [addresses(ttnn, value) for value in working] != working_addresses:
                            raise AssertionError('Working convolution addresses changed')
                        report['checks'].append(dict(seed=seed, rows=rows, mode=mode, exact=True))

                    stage('candidate-eager')
                    restore()
                    result = candidate()
                    validate(result, 'eager')
                    release_owned(ttnn, result['owned'])
                    restore()
                    stage('candidate-capture')
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    captured = candidate()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    restore()
                    stage('candidate-replay')
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    validate(captured, 'trace')
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    release_owned(ttnn, captured['owned'])
                    release_owned(ttnn, [*entry, *inputs, *projected_rows, projected, *working])
                    stage('fixture-complete')
            if len(report['checks']) != 30 or report['projection_calls'] != 93:
                raise AssertionError('Incomplete real-weight integration gate')
            stage('mesh-close')
            ttnn.close_mesh_device(mesh)
            mesh = None
            report['passed'] = True
            stage('complete')
        except BaseException as error:
            report['error'] = f'{type(error).__name__}: {error}'
            path.write_text(json.dumps(report, indent=2))
            raise
        finally:
            if mesh is not None:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                ttnn.close_mesh_device(mesh)
            faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
