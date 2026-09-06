"""Real-weight projected-input convolution/recurrence gate; no throughput claim."""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path

from gdn_prefix import gated_decode
from gdn_pair_timing import paired_replays
from gdn_multitoken import HANDOFF_HASHES, load_kernels, validate_handoff_runtime
from gdn_multitoken_conv import addresses, finish_output, release_owned, restore_prefix, run_projected
from gdn_batched_conv import run_batched_projected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuation', action='store_true')
    parser.add_argument('--full-layer', action='store_true')
    parser.add_argument('--paired-timing', action='store_true')
    parser.add_argument('--batch-conv', action='store_true')
    parser.add_argument('--dma-windows', action='store_true')
    parser.add_argument('--packed-checkpoints', action='store_true')
    parser.add_argument('--max-rows', type=int, choices=(16, 32), default=16)
    options = parser.parse_args()
    widths = (1, 2, 4, 8, 16, 32) if options.max_rows == 32 else (1, 2, 4, 8, 16)
    if options.dma_windows and not options.batch_conv:
        parser.error('--dma-windows requires --batch-conv')
    if options.packed_checkpoints and not options.dma_windows:
        parser.error('--packed-checkpoints requires --dma-windows')
    if options.paired_timing and not options.full_layer:
        parser.error('--paired-timing requires --full-layer')
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch silicon required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_gdn_layer
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

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
    report = dict(passed=False, checks=[], continuation_checks=[], stale_controls=[], projection_calls=0, handoff_runtime_hashes=HANDOFF_HASHES,
                  generated_hashes={name: hashlib.sha256(value.encode()).hexdigest() for name, value in kernels.items()},
                  adapter_sha256=hashlib.sha256(Path(__file__).with_name('gdn_multitoken_conv.py').read_bytes()).hexdigest(),
                  continuation_enabled=options.continuation,
                  full_layer=options.full_layer, paired_timing=[],
                  batched_convolution=options.batch_conv,
                  dma_windows=options.dma_windows,
                  packed_checkpoints=options.packed_checkpoints,
                  prefix_copy_hashes={suffix: hashlib.sha256(Path(__file__).with_name('gdn_conv_prefix_copy.' + suffix).read_bytes()).hexdigest() for suffix in ('py', 'cpp')} if options.packed_checkpoints else {},
                  window_dma_hashes={suffix: hashlib.sha256(Path(__file__).with_name('gdn_conv_windows.' + suffix).read_bytes()).hexdigest() for suffix in ('py', 'cpp')} if options.dma_windows else {},
                  batched_adapter_sha256=hashlib.sha256(Path(__file__).with_name('gdn_batched_conv.py').read_bytes()).hexdigest() if options.batch_conv else None,
                  scope='One real-weight TP2 GDN projected-input/conv/gates/recurrence/norm path; optional all-prefix two-step native continuation; excludes output projection, attention, full model and timing')
    context = {}
    if options.full_layer:
        report['scope'] = 'One full real-weight TP2 GDN layer with batched input/output projections, fabric reduce, every state prefix and final-state commit; excludes attention, full model and committed token throughput'

    def stage(name, **details):
        report['last_stage'] = dict(stage=name, **context, **details)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['last_stage']), flush=True)

    mesh, trace = None, None
    timing_traces = {}
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
            forward = gdn.forward_decode if options.full_layer else gated_decode(gdn)
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
                for rows in widths:
                    context.update(seed=seed, rows=rows)
                    stage('fixture-initialize')
                    live = [gdn.rec_state, *gdn.conv_states]
                    live_addresses = [addresses(ttnn, value) for value in live]
                    entry = [upload((torch.randn(tuple(value.shape)) * 0.05).bfloat16()) for value in live]
                    for initial, destination in zip(entry, live, strict=True):
                        ttnn.copy(initial, destination)
                    entry_host = [host(value) for value in entry]
                    inputs = [upload((torch.randn(1, 1, 5120) * 0.1).bfloat16()) for token in range(rows)]
                    packed_input = ttnn.concat(inputs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if options.full_layer else None
                    projected_rows, expected, oracle_prefixes = [], [], []
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
                            if options.continuation:
                                oracle_prefixes.append([ttnn.clone(state, memory_config=ttnn.DRAM_MEMORY_CONFIG) for state in live])
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
                        if options.full_layer:
                            for initial, destination in zip(entry, live, strict=True):
                                ttnn.copy(initial, destination)
                        ttnn.synchronize_device(mesh)

                    def candidate(use_batched=options.batch_conv, use_packed=options.packed_checkpoints):
                        source = original(packed_input, rows, ttnn.L1_MEMORY_CONFIG) if options.full_layer else projected
                        operation = run_batched_projected if use_batched else run_projected
                        result = operation(mesh, source, entry[0], working, list(gdn.tw['conv_taps']),
                            gdn.tw['dt_bias'], gdn.tw['neg_exp_A'], gdn.tw['norm_w'], kernels,
                            **(dict(packed_checkpoints=True) if use_batched and use_packed else {}),
                            **(dict(dma_windows=True) if use_batched and options.dma_windows else {}))
                        if options.full_layer:
                            result['owned'].append(source)
                            finish_output(gdn, result, ttnn, tt_all_reduce)
                            restore_prefix(ttnn, result, entry, live, rows)
                        return result

                    def validate(result, mode, record=True):
                        outputs, states = host(result['layer_output'] if options.full_layer else result['output']), host(result['states'])
                        packed_history = [host(value) for value in result['packed_conv_states']] if result.get('packed_checkpoints') else None
                        for token in range(rows):
                            actual_output = [value[:, :, token:token + 1] if options.full_layer else value[:, token:token + 1] for value in outputs]
                            if not exact(actual_output, expected[token][0]):
                                raise AssertionError(f'Gated output mismatch {token=} {mode=}')
                            if not exact([value[token:token + 1] for value in states], expected[token][1]):
                                raise AssertionError(f'Recurrent prefix mismatch {token=} {mode=}')
                            for tap in range(4):
                                actual_history = [value[:, token:token + 1] for value in packed_history[tap]] if packed_history is not None else host(result['conv_prefixes'][token][tap])
                                if not exact(actual_history, expected[token][2][tap]):
                                    raise AssertionError(f'Convolution prefix mismatch {token=} {tap=} {mode=}')
                        if not all(exact(host(value), saved) for value, saved in zip(entry, entry_host, strict=True)):
                            raise AssertionError('Initial state snapshots modified')
                        if [addresses(ttnn, value) for value in working] != working_addresses:
                            raise AssertionError('Working convolution addresses changed')
                        if options.full_layer:
                            if not exact(host(gdn.rec_state), expected[-1][1]) or not all(
                                    exact(host(value), saved) for value, saved in zip(gdn.conv_states, expected[-1][2], strict=True)):
                                raise AssertionError('Final-state commit differs from native')
                        if record:
                            report['checks'].append(dict(seed=seed, rows=rows, mode=mode, exact=True))

                    def continuation_values():
                        values = []
                        for value in corrections:
                            output = forward(value)
                            values.append([host(output), *[host(state) for state in live]])
                            ttnn.deallocate(output)
                        if [addresses(ttnn, value) for value in [gdn.rec_state, *gdn.conv_states]] != live_addresses:
                            raise AssertionError('Native continuation changed stable state addresses')
                        return values

                    def continuation_equal(actual, control):
                        return all(exact(actual_value, control_value)
                            for actual_step, control_step in zip(actual, control, strict=True)
                            for actual_value, control_value in zip(actual_step, control_step, strict=True))

                    def validate_continuations(result, mode):
                        if not options.continuation:
                            return
                        for accepted in range(rows + 1):
                            stage('restored-native-continuation', mode=mode, accepted=accepted)
                            sources = entry if accepted == 0 else oracle_prefixes[accepted - 1]
                            for source, destination in zip(sources, live, strict=True):
                                ttnn.copy(source, destination)
                            control = continuation_values()
                            restore_prefix(ttnn, result, entry, live, accepted)
                            actual = continuation_values()
                            if not continuation_equal(actual, control):
                                raise AssertionError(f'Restored native continuation mismatch {accepted=} {mode=}')
                            report['continuation_checks'].append(dict(seed=seed, rows=rows, mode=mode, accepted=accepted, steps=2, exact=True))
                            if accepted == 0 and mode == 'eager':
                                restore_prefix(ttnn, result, entry, live, rows)
                                stale = continuation_values()
                                if continuation_equal(stale, control):
                                    raise AssertionError('Stale-state negative control was not detected')
                                report['stale_controls'].append(dict(seed=seed, rows=rows, detected=True))
                        if not all(exact(host(value), saved) for value, saved in zip(entry, entry_host, strict=True)):
                            raise AssertionError('Continuation modified entry snapshots')

                    corrections = [upload((torch.randn(1, 1, 5120) * 0.1).bfloat16()) for step in range(2)] if options.continuation else []
                    stage('candidate-eager')
                    restore()
                    result = candidate()
                    validate(result, 'eager')
                    validate_continuations(result, 'eager')
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
                    validate_continuations(captured, 'trace')
                    release_owned(ttnn, captured['owned'])
                    if options.paired_timing and seed == 0:
                        def control():
                            if options.packed_checkpoints:
                                return candidate(True, False)
                            if options.batch_conv:
                                return candidate(False)
                            owned, outputs, states, conv_prefixes = [], [], [], []
                            for value in inputs:
                                output = forward(value)
                                state = ttnn.clone(gdn.rec_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                                prefix = [ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG) for value in gdn.conv_states]
                                outputs.append(output)
                                states.append(state)
                                conv_prefixes.append(prefix)
                                owned.extend([output, state, *prefix])
                            output = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            packed_states = ttnn.concat(states, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                            owned.extend([output, packed_states])
                            return dict(layer_output=output, states=packed_states, conv_prefixes=conv_prefixes, owned=owned)

                        stage('paired-full-layer-capture')
                        arms, results = dict(control=control, candidate=candidate), {}
                        for arm, operation in arms.items():
                            restore()
                            warm = operation()
                            validate(warm, arm + '-warm', record=False)
                            release_owned(ttnn, warm['owned'])
                        for arm, operation in arms.items():
                            restore()
                            timing_traces[arm] = ttnn.begin_trace_capture(mesh, cq_id=0)
                            results[arm] = operation()
                            ttnn.end_trace_capture(mesh, timing_traces[arm], cq_id=0)
                        stage('paired-full-layer-replay')
                        timing = paired_replays(restore, lambda: ttnn.synchronize_device(mesh),
                            lambda arm: ttnn.execute_trace(mesh, timing_traces[arm], cq_id=0, blocking=False),
                            lambda arm: validate(results[arm], arm + '-timing', record=False))
                        timing.update(seed=seed, rows=rows, checkpoint_policy='all',
                            scope='One GDN layer: native serial input/output projections versus batched projections and device-loop recurrence; both return every DRAM prefix and commit final native state; restore outside timing; not full-model tok/s',
                            control_recurrence_state='native L1', candidate_recurrence_state='immutable initial DRAM plus device-local loop')
                        if options.batch_conv:
                            timing.update(scope='Paired full GDN layer: serial versus parallel-window native convolution; both batched projections, device-loop recurrence, all DRAM prefixes and native final commit; not full-model throughput',
                                          control_recurrence_state='immutable initial DRAM plus device-local loop')
                        if options.packed_checkpoints:
                            timing.update(scope='Paired full GDN layer: packed versus separately materialized convolution prefixes; both DMA-built causal windows, native batched convolution, device-loop recurrence and native final commit; every logical prefix available; not full-model throughput',
                                          checkpoint_policy='all logical prefixes; packed candidate and separate control tensors')
                        report['paired_timing'].append(timing)
                        for arm in arms:
                            ttnn.release_trace(mesh, timing_traces.pop(arm))
                            release_owned(ttnn, results[arm]['owned'])
                    release_owned(ttnn, [*entry, *inputs, *projected_rows, projected, *working, *corrections,
                                         *([packed_input] if packed_input is not None else []),
                                         *[state for prefix in oracle_prefixes for state in prefix]])
                    stage('fixture-complete')
            if len(report['checks']) != 6 * len(widths) or report['projection_calls'] != 3 * sum(widths):
                raise AssertionError('Incomplete real-weight integration gate')
            if options.continuation and (len(report['continuation_checks']) != 216 or len(report['stale_controls']) != 15):
                raise AssertionError('Incomplete composed-prefix continuation gate')
            if options.paired_timing and len(report['paired_timing']) != 5:
                raise AssertionError('Incomplete paired full-layer timing matrix')
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
                for captured_trace in timing_traces.values():
                    ttnn.release_trace(mesh, captured_trace)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                ttnn.close_mesh_device(mesh)
            faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
