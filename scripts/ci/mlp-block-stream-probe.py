"""Weight-free transport test only; no MLP numerical or throughput qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from feature_projection import require_projection_environment
from mlp_block_stream import BLOCK_TILES, BLOCK_BYTES, geometry, source_pages, pack


SOURCES = ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp-block-stream-probe.py')


def fingerprints():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if os.environ.get('QWEN_SIM_ONLY') != '1' or Path('/dev/tenstorrent').exists():
        parser.error('Device-free simulator required')
    import torch
    import ttnn

    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=[], sources=fingerprints(),
        scope=__doc__, mlp_qualified=False, performance_qualified=False, hardware_qualified=False)
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=0)
        mesh.enable_program_cache()
        for pairs, blocks in ((8, 2), (272, 1)):
            shape = geometry(pairs, blocks)
            for pattern in (0, 1):
                words = torch.arange(shape['source_pages'] * 144, dtype=torch.int64)
                source = ((words * 2654435761 + 1013904223 + pattern * 65537) & 0xffffffff).to(torch.uint32)
                source = source.reshape(1, 1, shape['source_pages'], 144)
                expected = torch.zeros((shape['stream_pages'], BLOCK_TILES, 144), dtype=torch.uint32)
                for block in range(blocks):
                    for worker in range(shape['workers']):
                        for tile, page in enumerate(source_pages(worker, block, pairs=pairs, blocks=blocks)):
                            if page is not None:
                                expected[block * shape['workers'] + worker, tile] = source[0, 0, page]
                expected = expected.reshape(1, 1, shape['stream_pages'], BLOCK_BYTES // 4)
                device_source = device_output = None
                try:
                    device_source = ttnn.from_torch(source, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    device_output = pack(mesh, device_source, pairs=pairs, blocks=blocks, raw_tile_fixture=True)
                    ttnn.synchronize_device(mesh)
                    for chip, (input_shard, output_shard) in enumerate(zip(ttnn.get_device_tensors(device_source),
                            ttnn.get_device_tensors(device_output), strict=True)):
                        if not torch.equal(ttnn.to_torch(input_shard), source):
                            raise AssertionError('Transport mutated source bytes')
                        if not torch.equal(ttnn.to_torch(output_shard), expected):
                            raise AssertionError('Transport changed packed words, tile order or padding')
                        report['checks'].append(dict(pairs=pairs, blocks=blocks, pattern=pattern, chip=chip,
                            source_unchanged=True, exact_all_words=True, pages=shape['stream_pages']))
                    print(json.dumps(dict(stage='transport_exact', pairs=pairs, blocks=blocks, pattern=pattern)), flush=True)
                finally:
                    for tensor in (device_output, device_source):
                        if tensor is not None:
                            ttnn.deallocate(tensor)
        if len(report['checks']) != 8:
            raise AssertionError('Both patterns, geometries and chips required')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            report['sources_after'] = fingerprints()
            if report['sources_after'] != report['sources'] or not report['closed_cleanly']:
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
