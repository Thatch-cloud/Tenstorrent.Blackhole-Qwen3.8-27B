"""Simulator-only projected-history append, abort and sliding-window ownership checks."""

import argparse
import json
import os
from pathlib import Path

from dflash_device import DFlashDevice
from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[])
    mesh = device = source = trace = trace_output = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        def project(features, prefix):
            sliced = ttnn.slice(features[0], (0, 0, 0, 0), (1, 1, prefix, 5120))
            result = ttnn.clone(sliced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if addresses(ttnn, sliced) != addresses(ttnn, features[0]):
                ttnn.deallocate(sliced)
            return result
        for initial_rows in (170, 2047):
            expected = (torch.arange(initial_rows).reshape(1, 1, initial_rows, 1) % 64).expand(1, 1, initial_rows, 5120).bfloat16()
            device = DFlashDevice.__new__(DFlashDevice)
            device.operations, device.mesh = ttnn, mesh
            device.closed, device.pending, device.owned = False, None, []
            device.position = device.history_rows = initial_rows
            device.published_rows = 0
            padded = torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16)
            padded[..., :initial_rows, :] = expected
            device.history = upload(padded)
            device.spare_history = upload(torch.zeros_like(padded))
            device.project_features = project
            pool = {addresses(ttnn, device.history), addresses(ttnn, device.spare_history)}
            trace_input = device.history
            def operation():
                temporary = ttnn.add(trace_input, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                result = ttnn.add(temporary, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(temporary)
                return result
            warm = operation()
            ttnn.deallocate(warm)
            trace, trace_output = capture_operation(ttnn, mesh, operation)
            for prefix in (1, 2, 7, 8, 32):
                additions = (torch.arange(32).reshape(1, 1, 32, 1) + prefix * 5).expand(1, 1, 32, 5120).bfloat16()
                source = upload(additions)
                original = addresses(ttnn, device.history)
                publication = device.prepare_publication((source,), prefix, position=device.position)
                if addresses(ttnn, device.history) != original or addresses(ttnn, publication.history) == original:
                    raise AssertionError('Preparation changed or aliased committed history')
                device.discard_publication(publication)
                if addresses(ttnn, device.history) != original:
                    raise AssertionError('Abort changed committed history')
                publication = device.prepare_publication((source,), prefix, position=device.position)
                device.commit_publication(publication)
                ttnn.deallocate(source)
                source = None
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if {addresses(ttnn, device.history), addresses(ttnn, device.spare_history)} != pool:
                    raise AssertionError('Persistent history must stay in the two pre-trace allocations')
                expected = torch.cat((expected, additions[..., :prefix, :]), dim=2)[..., -2048:, :]
                for chip, shard in enumerate(ttnn.get_device_tensors(device.history)):
                    actual = ttnn.to_torch(shard)
                    if not torch.equal(actual[..., :device.history_rows, :], expected) or torch.any(actual[..., device.history_rows:, :]):
                        raise AssertionError('Projected history contains padding, rejected rows or stale data')
                    report['checks'].append(dict(initial_rows=initial_rows, prefix=prefix, chip=chip,
                        position=device.position, rows=device.history_rows, exact=True, abort_unchanged=True,
                        preallocated_pool=True, survived_trace_replay=True))
            ttnn.release_trace(mesh, trace)
            trace = None
            ttnn.deallocate(trace_output)
            trace_output = None
            device.close()
            device = None
        report['passed'] = len(report['checks']) == 20
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if trace_output is not None:
            ttnn.deallocate(trace_output)
        if device is not None:
            device.close()
        if source is not None:
            ttnn.deallocate(source)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
