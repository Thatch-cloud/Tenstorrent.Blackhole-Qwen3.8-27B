"""Native logical-row argmax: simulator reduction gate, hardware full-sampler gate."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from force_argmax import sample_rows
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    import torch
    import ttnn

    path = Path(__file__).with_name('sampling-kernel.py')
    spec = importlib.util.spec_from_file_location('sampling_oracle', path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    report = dict(passed=False, backend='hardware' if options.hardware else 'simulator',
        scope='Complete native TP2 sampler including four-link fabric' if options.hardware else
              'Native untilize/argmax on replicated post-gather logits; fabric not simulated',
        checks=[], source_checks=[], stale_controls=0,
        implementation_sha256=hashlib.sha256(Path(__file__).with_name('force_argmax.py').read_bytes()).hexdigest())
    selections = (False, True) if options.hardware else (True,)
    mesh, sampler = None, None
    owned, traces = [], []
    cases = ('boundaries', 'cross-shard-tie', 'near-tie', 'all-equal', 'random', 'boundaries')
    widths = (1, 2, 4, 8)
    try:
        if options.hardware:
            from sampling_link_policy import audit
            report['fabric_sources'] = audit('/opt/tt-metal', os.environ)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        if options.hardware:
            from models.common.sampling.generator import SamplingGenerator
            from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
            from models.tt_transformers.tt.ccl import TT_CCL
            from sampling_link_policy import sampler_links
            sampler = SamplingGenerator(args=Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536),
                mesh_device=mesh, tt_ccl=TT_CCL(mesh))
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3) if options.hardware else ttnn.ReplicateTensorToMesh(mesh)
        for rows in widths:
            values = oracle.logits_case(248320, rows, cases[0])
            tensor = ttnn.from_torch(values, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            owned.append(tensor)
            original_ids = addresses(ttnn, tensor)

            def execute(native_rows):
                if options.hardware:
                    with sampler_links(sampler.tt_sampling, 4):
                        axis = sampler.tt_sampling._get_sampling_cluster_axis()
                        links, topology = sampler.tt_sampling._get_force_argmax_all_gather_config(axis)
                        if links != 4 or topology != ttnn.Topology.Linear:
                            raise AssertionError('Native sampler did not select four physical-pair links')
                        report['sampler_num_links'] = links
                        output = sample_rows(sampler, tensor, rows, ttnn, native_rows=native_rows)
                else:
                    linear = ttnn.untilize(tensor, use_multicore=True)
                    try:
                        output = ttnn.argmax(linear, dim=-1, keepdim=False)
                    finally:
                        ttnn.deallocate(linear)
                owned.append(output)
                return output

            for native_rows in selections:
                warm = execute(native_rows)
                ttnn.synchronize_device(mesh)
                release_owned(ttnn, [warm])
                owned.pop()
            captures = {}
            for native_rows in selections:
                trace, output = capture_operation(ttnn, mesh, lambda native_rows=native_rows: execute(native_rows))
                traces.append(trace)
                captures[native_rows] = trace, output, addresses(ttnn, output)
            for repetition, kind in enumerate(cases):
                values = oracle.logits_case(248320, rows, kind)
                expected = values.argmax(-1).reshape(-1)
                staged = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(staged, tensor)
                for native_rows, (trace, output, output_ids) in captures.items():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    if addresses(ttnn, output) != output_ids or addresses(ttnn, tensor) != original_ids:
                        raise AssertionError('Captured sampler input/output addresses changed')
                    parts = ttnn.get_device_tensors(output)
                    if len(parts) != 2:
                        raise AssertionError('Both chips required')
                    for chip, part in enumerate(parts):
                        actual = ttnn.to_torch(part).reshape(-1).long()
                        if actual.numel() != (rows if native_rows else 32) or not torch.equal(actual[:rows], expected):
                            raise AssertionError(f'Wrong token or logical output rows: T{rows}, {kind}, native={native_rows}, chip={chip}')
                        if kind == 'near-tie':
                            stale = oracle.logits_case(248320, rows, 'cross-shard-tie').argmax(-1).reshape(-1)
                            if torch.equal(actual[:rows], stale):
                                raise AssertionError('Stale sampler output was accepted')
                            report['stale_controls'] += 1
                        report['checks'].append(dict(rows=rows, kind=kind, repetition=repetition,
                            native_rows=native_rows, chip=chip, exact=True))
                for chip, part in enumerate(ttnn.get_device_tensors(tensor)):
                    expected_input = values[..., chip * 124160:(chip + 1) * 124160] if options.hardware else values
                    if not torch.equal(ttnn.to_torch(part), expected_input):
                        raise AssertionError('Sampling modified its logits')
                    report['source_checks'].append(dict(rows=rows, repetition=repetition, chip=chip, exact=True))
                options.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(dict(rows=rows, kind=kind, exact=True)), flush=True)
            for trace, _, _ in captures.values():
                ttnn.release_trace(mesh, trace)
                traces.remove(trace)
            release_owned(ttnn, owned)
            owned.clear()
        report['passed'] = (len(report['checks']) == 48 * len(selections) and len(report['source_checks']) == 48
                            and report['stale_controls'] == 8 * len(selections))
        if not report['passed']:
            raise AssertionError('Incomplete sampler matrix')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, owned)
            if sampler is not None:
                sampler.reset_trace()
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
