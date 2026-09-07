"""Simulator-first learned MLP trace replay with changing input and stale-input negative control."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_convolution_fixture import load_convolution
from draft_mlp_branch import execute_mlp_branch, prepare_mlp_branch, validate_mlp_branch
from draft_mlp_fixture import load_mlp
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--convolution-fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    manifest, weights = load_mlp(options.fixture)
    convolution_manifest, convolution = load_convolution(options.convolution_fixture)
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, convolution_checkpoint=convolution_manifest,
        eager_checks=[], trace_checks=[], negative_controls=[], sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('draft_mlp_branch.py', 'draft-mlp-trace-probe.py', 'draft_convolution.py',
                'draft_mlp.py', 'feature_collective.py', 'attention_batch.py')})
    mesh, trace, persistent, transient = None, None, [], []
    def retain_persistent(value):
        persistent.append(value)
        return value
    def retain_transient(value):
        transient.append(value)
        return value
    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip])
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        collective = TT_CCL(mesh)
        parameters = prepare_mlp_branch(ttnn, mesh, weights, convolution, retain_persistent)
        hidden_blocks = []
        for seed in (731, 732):
            value = torch.zeros((1, 1, 32, 5120), dtype=torch.bfloat16)
            value[..., :8, :] = torch.randn((1, 1, 8, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()
            hidden_blocks.append(value)
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in hidden_blocks]
        device_hidden = retain_persistent(ttnn.from_torch(hidden_blocks[0], device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        original_addresses = [addresses(ttnn, value) for value in persistent]
        def execute():
            return execute_mlp_branch(ttnn, mesh, collective, device_hidden, weights, convolution,
                retain_transient, parameters=parameters)
        references = []
        for pattern, source in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(source, device_hidden)
            state = execute()
            outputs = []
            for chip in range(2):
                check = validate_mlp_branch(state, host, chip, checkpoint=manifest)
                if not torch.equal(host(device_hidden, chip), hidden_blocks[pattern]):
                    raise AssertionError('MLP changed caller-owned input')
                check['pattern'] = pattern
                report['eager_checks'].append(check)
                outputs.append(host(state['output'], chip).clone())
            references.append(outputs)
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_reference_{pattern}_complete')
        if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
            raise AssertionError('Negative-control inputs must produce distinct outputs')
        ttnn.copy_host_to_device_tensor(host_inputs[0], device_hidden)
        ttnn.synchronize_device(mesh)
        progress('capturing')
        trace, captured = capture_operation(ttnn, mesh, execute)
        output_addresses = addresses(ttnn, captured['output'])
        for repetition, pattern in enumerate((0, 1, 0)):
            ttnn.copy_host_to_device_tensor(host_inputs[pattern], device_hidden)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if ([addresses(ttnn, value) for value in persistent] != original_addresses
                    or addresses(ttnn, captured['output']) != output_addresses):
                raise AssertionError('Trace input, parameters or output moved')
            for chip in range(2):
                if not torch.equal(host(captured['output'], chip), references[pattern][chip]):
                    raise AssertionError('Captured MLP output differs from validated eager output')
                if not torch.equal(host(device_hidden, chip), hidden_blocks[pattern]):
                    raise AssertionError('Captured MLP modified its input')
                report['trace_checks'].append(dict(repetition=repetition, pattern=pattern, chip=chip, exact=True))
            if repetition == 0:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for chip in range(2):
                    actual = host(captured['output'], chip)
                    if not torch.equal(actual, references[0][chip]) or torch.equal(actual, references[1][chip]):
                        raise AssertionError('Missing-input-update negative control failed')
                    report['negative_controls'].append(dict(chip=chip, stale_input_detected=True))
            progress(f'trace_replay_{repetition}_complete')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, transient)
            release_owned(ttnn, persistent)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = len(report['eager_checks']) == 4 and len(report['trace_checks']) == 6 and len(report['negative_controls']) == 2
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
