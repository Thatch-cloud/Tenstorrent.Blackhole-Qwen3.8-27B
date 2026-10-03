"""Learned MLP parameter reuse and changing-input replay; optional isolated hardware latency."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

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
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--timing', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.timing and not options.hardware:
        parser.error('Latency measurements require allocated hardware')
    if not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required for TP2 output reduction')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    manifest, weights = load_mlp(options.fixture)
    convolution_manifest, convolution = load_convolution(options.convolution_fixture)
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, convolution_checkpoint=convolution_manifest,
        backend='hardware' if options.hardware else 'simulator', timings_ms=[],
        timing_scope='Post-warmup eager branch execution, activation allocation and TP2 synchronization; excludes input/weight uploads and validation; not full drafter or committed throughput',
        checks=[], sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('draft_mlp_branch.py', 'draft-mlp-replay-probe.py')})
    mesh, parameters_owned, temporary = None, [], []
    def retain_parameter(value):
        parameters_owned.append(value)
        return value
    def retain_temporary(value):
        temporary.append(value)
        return value
    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip])
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        collective = TT_CCL(mesh)
        parameters = prepare_mlp_branch(ttnn, mesh, weights, convolution, retain_parameter)
        original_addresses = [addresses(ttnn, value) for value in parameters_owned]
        if len(parameters_owned) != 9:
            raise AssertionError('Expected exactly nine reusable parameter tensors')
        first_outputs = None
        seeds = (731, 732, 731, 732, 731, 732) if options.timing else (731, 732, 731)
        for repetition, seed in enumerate(seeds):
            hidden = torch.zeros((1, 1, 32, 5120), dtype=torch.bfloat16)
            hidden[..., :8, :] = torch.randn((1, 1, 8, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()
            device_hidden = retain_temporary(ttnn.from_torch(hidden, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
            if options.timing:
                ttnn.synchronize_device(mesh)
                started = time.perf_counter()
            state = execute_mlp_branch(ttnn, mesh, collective, device_hidden, weights, convolution,
                retain_temporary, parameters=parameters)
            if options.timing:
                ttnn.synchronize_device(mesh)
                elapsed_ms = (time.perf_counter() - started) * 1000
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise AssertionError('Positive hardware latency required')
                if repetition:
                    report['timings_ms'].append(elapsed_ms)
            current_outputs = []
            for chip in range(2):
                check = validate_mlp_branch(state, host, chip, checkpoint=manifest)
                check.update(repetition=repetition, seed=seed)
                current = host(state['output'], chip).clone()
                if first_outputs is not None and torch.equal(current, first_outputs[chip]) != (seed == 731):
                    raise AssertionError('Changing-input or repeat-input MLP output gate failed')
                current_outputs.append(current)
                report['checks'].append(check)
            if first_outputs is None:
                first_outputs = current_outputs
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, temporary)
            temporary.clear()
            if [addresses(ttnn, value) for value in parameters_owned] != original_addresses:
                raise AssertionError('Reusable parameter ownership changed')
            report.update(repetitions_completed=repetition + 1, reusable_parameter_tensors=len(parameters_owned))
            options.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(dict(repetition=repetition, seed=seed, checked=True)), flush=True)
        report['passed'] = len(report['checks']) == 2 * len(seeds)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, temporary)
            release_owned(ttnn, parameters_owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
