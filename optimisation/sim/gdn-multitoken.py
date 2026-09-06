"""Simulator-only first pass for the exact generated device-loop kernels."""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from gdn_multitoken import HANDOFF_HASHES, HASHES, execute, load_kernels
from gdn_multitoken_conv import finish_output, release_owned, restore_prefix, run_projected


def require_simulator(environment):
    simulator = environment.get('TT_METAL_SIMULATOR', '')
    if not simulator or not Path(simulator).is_file() or environment.get('TT_METAL_SLOW_DISPATCH_MODE') != '1':
        raise RuntimeError('Simulator library and slow dispatch mandatory; no hardware fallback')
    if environment.get('QWEN_HARDWARE_TESTS') or environment.get('QWEN_CARDS_ALLOCATED'):
        raise RuntimeError('Hardware authorization must not be set for this simulator entry point')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--rows', type=int, choices=(1, 2, 4, 8, 16), default=2)
    parser.add_argument('--norm-gate', action='store_true')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--continuation', action='store_true')
    parser.add_argument('--output-projection', action='store_true')
    parser.add_argument('--model-adapter', action='store_true')
    args = parser.parse_args()
    if args.conv and not args.norm_gate:
        parser.error('--conv requires --norm-gate')
    if args.continuation and not args.conv:
        parser.error('--continuation requires --conv')
    if args.output_projection and not args.conv:
        parser.error('--output-projection requires --conv')
    if args.model_adapter and not args.conv:
        parser.error('--model-adapter requires --conv')
    require_simulator(os.environ)
    path = Path(os.environ['QWEN_SIM_REPORT'])
    kernels = load_kernels(args.source_root, args.norm_gate)
    report = dict(passed=False, backend='ttsim', rows=args.rows, norm_gate=args.norm_gate,
                  seed=args.seed, handoff_runtime_hashes=HANDOFF_HASHES if args.norm_gate else {},
                  convolution=args.conv,
                  continuation_enabled=args.continuation,
                  output_projection=args.output_projection,
                  model_adapter_checks=0,
                  model_adapter_sha256=hashlib.sha256((Path(__file__).resolve().parents[2] / 'scripts/ci/gdn_device_loop_state.py').read_bytes()).hexdigest() if args.model_adapter else None,
                  output_projection_scope='Synthetic 3072x128 local projection only; no real weights or inter-chip collective' if args.output_projection else None,
                  adapter_sha256=hashlib.sha256((Path(__file__).resolve().parents[2] / 'scripts/ci/gdn_multitoken_conv.py').read_bytes()).hexdigest() if args.conv else None,
                  continuation_checks=0, stale_controls=0,
                  native_hashes=HASHES,
                  generated_hashes={name: hashlib.sha256(source.encode()).hexdigest() for name, source in kernels.items()},
                  scope='Slow-dispatch functional liveness/exactness against serial T1 of same kernel; not native oracle certification or performance')

    def stage(name, **details):
        report['last_stage'] = dict(stage=name, **details)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['last_stage']), flush=True)

    mesh = None
    with path.with_suffix('.stacks.log').open('w') as stacks:
        faulthandler.dump_traceback_later(30, repeat=True, file=stacks)
        try:
            stage('import-runtime')
            import torch
            import ttnn
            report['ttnn_path'] = ttnn.__file__
            report['extension_sha256'] = hashlib.sha256(Path(ttnn._ttnn.__file__).read_bytes()).hexdigest()
            if not Path(ttnn.__file__).resolve().is_relative_to(Path(os.environ['TT_METAL_HOME']).resolve()):
                raise RuntimeError('Expected local source-built TTNN')
            stage('mesh-open')
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)

            def upload(value):
                return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

            def host(value):
                shards = ttnn.get_device_tensors(value)
                if len(shards) != 2:
                    raise AssertionError('Two simulated chips required')
                return [ttnn.to_torch(shard).clone() for shard in shards]

            torch.manual_seed(args.seed)
            values = [torch.randn(1, args.rows, 5120).bfloat16(), torch.rand(1, args.rows, 24).bfloat16(),
                      (-torch.rand(1, args.rows, 24)).bfloat16()]
            if args.norm_gate:
                values.append(torch.randn(1, args.rows, 3072).bfloat16())
            stage('fixture-upload')
            norm_w = upload((1 + torch.randn(1, 1, 128) * 0.1).bfloat16()) if args.norm_gate else None
            initial_host = (torch.randn(1, 24, 128, 128) * 0.05).bfloat16()
            initial = upload(initial_host)
            if args.conv:
                projected = (torch.randn(1, args.rows, 8256) * 0.1).bfloat16()
                taps = [upload((torch.randn(1, 1, 5120) * 0.1).bfloat16()) for index in range(4)]
                dt_bias = upload(torch.randn(1, 1, 24).bfloat16())
                neg_exp_A = upload((-torch.rand(1, 1, 24)).bfloat16())
                conv_host = [(torch.randn(1, 1, 5120) * 0.05).bfloat16() for index in range(4)]
                serial_conv = [upload(value) for value in conv_host]
                candidate_conv = [upload(value) for value in conv_host]
                serial_results, expected = [], []
                state = initial
                for token in range(args.rows):
                    stage('conv-serial-t1', token=token)
                    result = run_projected(mesh, upload(projected[:, token:token + 1]), state, serial_conv,
                                           taps, dt_bias, neg_exp_A, norm_w, kernels)
                    serial_results.append(result)
                    state = result['states']
                    expected.append((host(result['output']), host(state), [host(value) for value in serial_conv]))
                stage('conv-multitoken-submit')
                result = run_projected(mesh, upload(projected), initial, candidate_conv,
                                       taps, dt_bias, neg_exp_A, norm_w, kernels)
                stage('conv-multitoken-readback')
                outputs, states = host(result['output']), host(result['states'])
                for token in range(args.rows):
                    prefix = [host(value) for value in result['conv_prefixes'][token]]
                    for chip in range(2):
                        if not torch.equal(outputs[chip][:, token:token + 1], expected[token][0][chip]):
                            raise AssertionError(f'Convolution-chain output mismatch {token=} {chip=}')
                        if not torch.equal(states[chip][token:token + 1], expected[token][1][chip]):
                            raise AssertionError(f'Convolution-chain recurrent state mismatch {token=} {chip=}')
                        for tap in range(4):
                            if not torch.equal(prefix[tap][chip], expected[token][2][tap][chip]):
                                raise AssertionError(f'Convolution prefix mismatch {token=} {chip=} {tap=}')
                if not all(torch.equal(value, initial_host) for value in host(initial)):
                    raise AssertionError('Initial recurrent state modified')
                if args.model_adapter:
                    from gdn_device_loop_state import DeviceLoopState
                    from gdn_snapshot import ActiveSnapshot
                    full_host = [torch.cat([initial_host, (torch.randn(7, 24, 128, 128) * 0.05).bfloat16()], dim=0)]
                    full_host.extend(torch.cat([value, (torch.randn(1, 7, 5120) * 0.05).bfloat16()], dim=1) for value in conv_host)
                    full_entry = [upload(value) for value in full_host]
                    live = [upload(value) for value in full_host]
                    projected_device = upload(projected)
                    dummy = upload(torch.zeros(1, args.rows, 5120).bfloat16())
                    def slice_along(value, dimension, start, end):
                        begins, ends = [0] * len(value.shape), list(value.shape)
                        begins[dimension], ends[dimension] = start, end
                        return ttnn.slice(value, begins, ends, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    layer = SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], mesh=mesh,
                        _slice_along=slice_along,
                        _project_qkvzab_raw=lambda *unused: ttnn.clone(projected_device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                        tw=dict(conv_taps=taps, dt_bias=dt_bias, neg_exp_A=neg_exp_A, norm_w=norm_w))
                    active = ActiveSnapshot(layer, ttnn, direct=True)
                    adapter = DeviceLoopState(active, ttnn, kernels)
                    checkpoint = active.allocate()
                    expected_values = [host(result['output']), host(result['states'])]
                    for accepted in range(args.rows + 1):
                        stage('model-adapter-active-slot', accepted=accepted)
                        for source, destination in zip(full_entry, live, strict=True):
                            ttnn.copy(source, destination)
                        actual = adapter.decode(dummy, checkpoint, accepted)
                        for value, reference in zip((actual['output'], actual['states']), expected_values, strict=True):
                            if not all(torch.equal(left, right) for left, right in zip(host(value), reference, strict=True)):
                                raise AssertionError('Model adapter output/recurrent prefix mismatch')
                        for index, (destination, snapshot) in enumerate(zip(live, checkpoint, strict=True)):
                            final_reference = expected[-1][1] if index == 0 else expected[-1][2][index - 1]
                            prefix_reference = [initial_host if index == 0 else conv_host[index - 1]] * 2 if accepted == 0 else (
                                expected[accepted - 1][1] if index == 0 else expected[accepted - 1][2][index - 1])
                            for chip, (actual_live, saved) in enumerate(zip(host(destination), host(snapshot), strict=True)):
                                active_row = actual_live[:1] if index == 0 else actual_live[:, :1]
                                inactive = actual_live[1:] if index == 0 else actual_live[:, 1:]
                                original_inactive = full_host[index][1:] if index == 0 else full_host[index][:, 1:]
                                if not torch.equal(active_row, final_reference[chip]) or not torch.equal(saved, prefix_reference[chip]):
                                    raise AssertionError(f'Model adapter publication mismatch {accepted=} {index=} {chip=}')
                                if not torch.equal(inactive, original_inactive):
                                    raise AssertionError('Model adapter modified an inactive slot')
                        release_owned(ttnn, actual['owned'])
                        report['model_adapter_checks'] += 1
                    adapter.close()
                    release_owned(ttnn, [*checkpoint, *full_entry, *live, projected_device, dummy])
                if args.output_projection:
                    stage('local-output-projection')
                    weights = upload((torch.randn(3072, 128) * 0.01).bfloat16())
                    gdn = SimpleNamespace(mesh=mesh, tt_ccl=None, tw={'out': weights},
                        args=SimpleNamespace(ccl_topology=lambda: None),
                        _row_proj=lambda value, weight: ttnn.linear(value, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                    def local_only(partial, *unused, **kwargs):
                        return partial
                    finish_output(gdn, result, ttnn, local_only)
                    projected_outputs = host(result['layer_output'])
                    for token, serial in enumerate(serial_results):
                        finish_output(gdn, serial, ttnn, local_only)
                        for chip, expected_output in enumerate(host(serial['layer_output'])):
                            if not torch.equal(projected_outputs[chip][:, :, token:token + 1], expected_output):
                                raise AssertionError(f'Local output projection mismatch {token=} {chip=}')
                    report['output_projection_exact'] = True
                if args.continuation:
                    entry = [initial, *[upload(value) for value in conv_host]]
                    destinations = [upload(torch.zeros_like(initial_host)), *[upload(torch.zeros_like(value)) for value in conv_host]]
                    reference_conv = [upload(torch.zeros_like(value)) for value in conv_host]
                    correction = upload((torch.randn(1, 2, 8256) * 0.1).bfloat16())
                    for accepted in range(args.rows + 1):
                        stage('conv-restored-continuation', accepted=accepted)
                        sources = entry if accepted == 0 else [serial_results[accepted - 1]['states'],
                                                               *serial_results[accepted - 1]['conv_prefixes'][0]]
                        for source, destination in zip(sources[1:], reference_conv, strict=True):
                            ttnn.copy(source, destination)
                        control = run_projected(mesh, correction, sources[0], reference_conv,
                                                taps, dt_bias, neg_exp_A, norm_w, kernels)
                        restore_prefix(ttnn, result, entry, destinations, accepted)
                        actual = run_projected(mesh, correction, destinations[0], destinations[1:],
                                               taps, dt_bias, neg_exp_A, norm_w, kernels)
                        control_values = [control['output'], control['states'], *[value for prefix in control['conv_prefixes'] for value in prefix]]
                        actual_values = [actual['output'], actual['states'], *[value for prefix in actual['conv_prefixes'] for value in prefix]]
                        for actual_value, control_value in zip(actual_values, control_values, strict=True):
                            if not all(torch.equal(left, right) for left, right in zip(host(actual_value), host(control_value), strict=True)):
                                raise AssertionError(f'Restored continuation mismatch {accepted=}')
                        report['continuation_checks'] += 1
                        if accepted == 0:
                            restore_prefix(ttnn, result, entry, destinations, args.rows)
                            stale = run_projected(mesh, correction, destinations[0], destinations[1:],
                                                  taps, dt_bias, neg_exp_A, norm_w, kernels)
                            if all(torch.equal(left, right) for left, right in zip(host(stale['states']), host(control['states']), strict=True)):
                                raise AssertionError('Stale-state control was not detected')
                            report['stale_controls'] += 1
                            release_owned(ttnn, stale['owned'])
                        release_owned(ttnn, [*control['owned'], *actual['owned']])
                stage('mesh-close')
                ttnn.close_mesh_device(mesh)
                mesh = None
                report['passed'] = True
                stage('complete')
                return
            packed = [upload(value) for value in values]

            def run(inputs, state):
                return execute(mesh, *inputs[:3], state, kernels,
                    **(dict(z=inputs[3], norm_w=norm_w) if args.norm_gate else {}))

            expected_outputs, expected_states = [], []
            state = initial
            for token in range(args.rows):
                stage('serial-t1', token=token)
                inputs = [upload(value[:, token:token + 1]) for value in values]
                output, state = run(inputs, state)
                expected_outputs.append([value.reshape(1, 1, 24, 128) for value in host(output)])
                expected_states.append(host(state))
                stage('serial-t1-complete', token=token)
            stage('multitoken-submit')
            output, states = run(packed, initial)
            stage('multitoken-readback')
            actual_outputs = [value.reshape(args.rows, 1, 24, 128) for value in host(output)]
            actual_states = host(states)
            for token in range(args.rows):
                for chip in range(2):
                    if not torch.equal(actual_outputs[chip][token:token + 1], expected_outputs[token][chip]):
                        raise AssertionError(f'Output mismatch {token=} {chip=}')
                    if not torch.equal(actual_states[chip][token:token + 1], expected_states[token][chip]):
                        raise AssertionError(f'State mismatch {token=} {chip=}')
            if not all(torch.equal(value, initial_host) for value in host(initial)):
                raise AssertionError('Initial state modified')
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
                ttnn.close_mesh_device(mesh)
            faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
