"""Native argmax-to-embedding feedback in one captured chain; no MTP speed claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from mtp_device_chain import collect_tokens, feedback_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, backend='simulator', scope=__doc__, checks=[], stale_controls=0,
        physical_fabric=False, vocabulary=256, hidden=5120,
        implementation_sha256=hashlib.sha256(Path(__file__).with_name('mtp_device_chain.py').read_bytes()).hexdigest())
    mesh, source, table = None, None, None
    retained, traces = [], []
    weights = torch.arange(5120).remainder(64).repeat(256, 1).to(torch.bfloat16)
    weights[:, :256] = 0
    weights[torch.arange(256), (torch.arange(256) + 1) % 256] = 200
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        table = ttnn.from_torch(weights.reshape(1, 1, 256, 5120), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        source = ttnn.from_torch(torch.tensor([[[[31]]]], dtype=torch.int32), device=mesh, dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        source_ids, table_ids = addresses(ttnn, source), addresses(ttnn, table)

        def embed(identifiers, *, memory_config):
            return ttnn.embedding(identifiers, table, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)

        def operation(count, owned):
            identifiers, selected, embedded_rows = source, [], []
            for _ in range(count):
                embedded = feedback_embedding(ttnn, embed, lambda value: value, identifiers, owned)
                embedded_rows.append(embedded)
                logits = ttnn.slice(embedded, (0, 0, 0, 0), (1, 1, 1, 256), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                owned.append(logits)
                row = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                owned.append(row)
                identifiers = ttnn.argmax(row, dim=-1, keepdim=False)
                owned.append(identifiers)
                selected.append(identifiers)
            return collect_tokens(ttnn, selected, owned), embedded_rows

        for count in range(1, 8):
            warm = []
            operation(count, warm)
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, warm)
        for count in range(1, 8):
            trace, (selected, embedded_rows) = capture_operation(ttnn, mesh, lambda count=count: operation(count, retained))
            traces.append(trace)
            for repetition, seed in enumerate((31, 255, 31)):
                staged = ttnn.from_torch(torch.tensor([[[[seed]]]], dtype=torch.int32), dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(staged, source)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                expected = [(seed + offset + 1) % 256 for offset in range(count)]
                for chip, part in enumerate(ttnn.get_device_tensors(selected)):
                    actual = ttnn.to_torch(part).reshape(-1).tolist()
                    if actual != expected:
                        raise AssertionError(f'Device feedback tokens differ: {count=}, {seed=}, {chip=}, {actual=}')
                    if repetition == 1:
                        if actual == [(31 + offset + 1) % 256 for offset in range(count)]:
                            raise AssertionError('Stale seed was not rejected')
                        report['stale_controls'] += 1
                    report['checks'].append(dict(count=count, seed=seed, chip=chip, exact=True))
                for offset, embedded in enumerate(embedded_rows):
                    for part in ttnn.get_device_tensors(embedded):
                        if not torch.equal(ttnn.to_torch(part).reshape(5120), weights[(seed + offset) % 256]):
                            raise AssertionError('Native feedback embedding differs from host lookup')
                if addresses(ttnn, source) != source_ids or addresses(ttnn, table) != table_ids:
                    raise AssertionError('Feedback replaced a captured input or borrowed embedding')
                print(json.dumps(dict(count=count, seed=seed, exact=True)), flush=True)
        for part in ttnn.get_device_tensors(table):
            if not torch.equal(ttnn.to_torch(part).reshape(256, 5120), weights):
                raise AssertionError('Feedback mutated its embedding table')
        report['passed'] = len(report['checks']) == 42 and report['stale_controls'] == 14
        if not report['passed']:
            raise AssertionError('Incomplete feedback chain matrix')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, retained)
            release_owned(ttnn, [value for value in (source, table) if value is not None])
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
