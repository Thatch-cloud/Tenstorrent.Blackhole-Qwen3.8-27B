"""Simulator-only first pass for the exact generated device-loop kernels."""

import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from gdn_multitoken import HASHES, execute, load_kernels


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
    args = parser.parse_args()
    require_simulator(os.environ)
    path = Path(os.environ['QWEN_SIM_REPORT'])
    kernels = load_kernels(args.source_root, args.norm_gate)
    report = dict(passed=False, backend='ttsim', rows=args.rows, norm_gate=args.norm_gate,
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

            torch.manual_seed(0)
            values = [torch.randn(1, args.rows, 5120).bfloat16(), torch.rand(1, args.rows, 24).bfloat16(),
                      (-torch.rand(1, args.rows, 24)).bfloat16()]
            if args.norm_gate:
                values.append(torch.randn(1, args.rows, 3072).bfloat16())
            stage('fixture-upload')
            norm_w = upload((1 + torch.randn(1, 1, 128) * 0.1).bfloat16()) if args.norm_gate else None
            initial_host = (torch.randn(1, 24, 128, 128) * 0.05).bfloat16()
            initial = upload(initial_host)
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
