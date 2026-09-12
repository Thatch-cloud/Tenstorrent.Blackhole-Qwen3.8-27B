"""Native force-argmax one/two/four-link experiment; no model throughput claim."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

from sampling_link_policy import audit, sampler_links


def main():
    sources = audit('/opt/tt-metal', os.environ)
    import torch
    import ttnn
    from models.common.sampling.generator import SamplingGenerator
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    specification = importlib.util.spec_from_file_location('sampling_cases', Path(__file__).with_name('sampling-kernel.py'))
    cases = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(cases)
    path = Path('/experiment/results/sampling-links.json')
    report = dict(passed=False, scope=__doc__, native_sources=sources, checks=[], blocks=[],
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('sampling-links.py', 'sampling_link_policy.py', 'sampling-kernel.py')},
        physical_capacity_basis='Operator four-link cabling and successful four-channel mesh initialization; not get_num_links discovery')
    mesh = sampler = tensor = None

    def checkpoint(stage):
        report['stage'] = stage
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage, checks=len(report['checks']), blocks=len(report['blocks']))), flush=True)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=67108864)
        mesh.enable_program_cache()
        collective = TT_CCL(mesh)
        report['unmodified_helper_counts'] = {str(axis): collective.get_num_links(axis) for axis in (None, 0, 1)}
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536)
        if args.vocab_size != 248320 or args.padded_vocab_size != args.vocab_size:
            raise ValueError('Pinned unpadded vocabulary required')
        sampler = SamplingGenerator(args=args, mesh_device=mesh, tt_ccl=collective)
        sampler.set_trace_bucket(1)
        if not sampler.tt_sampling.force_argmax_sampling or sampler.seed_manager.has_active_request_seed():
            raise ValueError('Unseeded force argmax path required')
        report['configured_default_links'] = sampler.tt_sampling.num_argmax_gather_links
        tensor = ttnn.from_torch(cases.logits_case(args.vocab_size, 32, 'random'), device=mesh,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))

        def stage_input(kind):
            values = cases.logits_case(args.vocab_size, 32, kind)
            payload = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
            ttnn.copy_host_to_device_tensor(payload, tensor)
            ttnn.synchronize_device(mesh)
            return values

        def verify(result, values, links, kind):
            output = result[0] if isinstance(result, tuple) else result
            parts = ttnn.get_device_tensors(output)
            if len(parts) != 2:
                raise AssertionError('Both chip outputs required')
            expected = values.argmax(dim=-1).reshape(-1)
            for chip, part in enumerate(parts):
                torch.testing.assert_close(ttnn.to_torch(part).reshape(-1).long(), expected, rtol=0, atol=0)
                if not torch.equal(ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]), values.chunk(2, dim=3)[chip]):
                    raise AssertionError('Sampling mutated caller logits')
                report['checks'].append(dict(links=links, kind=kind, chip=chip, exact=True, input_unchanged=True))

        for links in (1, 2, 4):
            with sampler_links(sampler.tt_sampling, links):
                axis = sampler.tt_sampling._get_sampling_cluster_axis()
                actual, topology = sampler.tt_sampling._get_force_argmax_all_gather_config(axis)
                if actual != links or topology != ttnn.Topology.Linear:
                    raise AssertionError('Requested link count was clamped or topology changed')
                try:
                    for kind in ('random', 'boundaries', 'cross-shard-tie', 'near-tie', 'all-equal'):
                        values = stage_input(kind)
                        verify(sampler.sample(tensor, enable_trace=True), values, links, kind)
                finally:
                    sampler.reset_trace()
            checkpoint(f'links_{links}_correctness_complete')
        values = stage_input('random')
        for comparison in (2, 4):
            for repetition in range(3):
                for links in (1, comparison, comparison, 1):
                    with sampler_links(sampler.tt_sampling, links):
                        try:
                            verify(sampler.sample(tensor, enable_trace=True), values, links, 'warm')
                            timings = []
                            for _ in range(20):
                                started = time.perf_counter()
                                result = sampler.sample(tensor, enable_trace=True)
                                ttnn.synchronize_device(mesh)
                                timings.append((time.perf_counter() - started) * 1000)
                                verify(result, values, links, 'timed')
                            report['blocks'].append(dict(comparison=comparison, repetition=repetition,
                                links=links, samples_ms=timings, mean_ms=statistics.mean(timings)))
                        finally:
                            sampler.reset_trace()
                checkpoint(f'comparison_{comparison}_abba_{repetition}_complete')
        if sampler.tt_sampling.num_argmax_gather_links != report['configured_default_links']:
            raise AssertionError('Sampler configuration did not restore')
        report['passed'] = True
        checkpoint('complete')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if sampler is not None:
            sampler.reset_trace()
        if tensor is not None:
            ttnn.deallocate(tensor)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
