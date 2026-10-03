"""Simulator-only synthetic MTP shortlist projection and A/B/A selection gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_shortlist_device import prepare_head, select_token
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def fixture(width):
    import torch

    if width not in (32768, 65536):
        raise ValueError('32K or 64K shortlist required')
    tokens = tuple(range(width - 2)) + (248044, 248046)
    weight = torch.zeros((248320, 5120), dtype=torch.bfloat16)
    weight[tokens[123], 0] = weight[tokens[231], 0] = 4
    weight[tokens[-1], 1] = 8
    hidden = torch.zeros((2, 1, 1, 1, 5120), dtype=torch.bfloat16)
    hidden[0, ..., 0] = hidden[1, ..., 1] = 1
    return weight, tokens, hidden, (tokens[123], tokens[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, choices=(32768, 65536), required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, width=options.width, checks=[],
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('draft-shortlist-probe.py', 'draft_shortlist_device.py',
                                        'draft_vocabulary.py')})
    mesh, trace = None, None
    persistent, transient = [], []
    options.output.parent.mkdir(parents=True, exist_ok=True)

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)

    try:
        weight, tokens, inputs, expected = fixture(options.width)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
                                     trace_region_size=268435456)
        mesh.enable_program_cache()
        head, identifiers = prepare_head(ttnn, mesh, weight, tokens, persistent)
        del weight
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       mesh_mapper=mapper) for value in inputs]
        hidden = ttnn.from_torch(inputs[0], device=mesh, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                mesh_mapper=mapper)
        persistent.append(hidden)
        original_addresses = [addresses(ttnn, value) for value in persistent]

        def execute():
            return select_token(ttnn, hidden, head, identifiers, transient)

        def check(output, pattern, stage):
            shards = ttnn.get_device_tensors(output)
            if len(shards) != 2:
                raise AssertionError('Both chip outputs required')
            for chip, shard in enumerate(shards):
                actual = ttnn.to_torch(shard)
                if actual.numel() != 1 or int(actual.item()) != expected[pattern]:
                    raise AssertionError(f'{stage} chip={chip} expected={expected[pattern]} actual={actual}')
                actual_hidden = ttnn.to_torch(ttnn.get_device_tensors(hidden)[chip])
                if not torch.equal(actual_hidden, inputs[pattern]):
                    raise AssertionError('Caller input mutated')
                report['checks'].append(dict(stage=stage, chip=chip, pattern=pattern, exact=True))

        for pattern in (0, 1):
            ttnn.copy_host_to_device_tensor(host_inputs[pattern], hidden)
            check(execute(), pattern, 'eager')
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_{pattern}_complete')
        ttnn.copy_host_to_device_tensor(host_inputs[0], hidden)
        ttnn.synchronize_device(mesh)
        trace, output = capture_operation(ttnn, mesh, execute)
        output_addresses = addresses(ttnn, output)
        for repetition, pattern in enumerate((0, 1, 0)):
            ttnn.copy_host_to_device_tensor(host_inputs[pattern], hidden)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(output, pattern, f'replay_{repetition}')
            if ([addresses(ttnn, value) for value in persistent] != original_addresses
                    or addresses(ttnn, output) != output_addresses):
                raise AssertionError('Trace buffers moved')
            if repetition == 0:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                check(output, 0, 'stale_input_control')
                if expected[0] == expected[1]:
                    raise AssertionError('Stale-input control cannot distinguish patterns')
            progress(f'replay_{repetition}_complete')
        report['passed'] = len(report['checks']) == 12
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


if __name__ == '__main__':
    main()
