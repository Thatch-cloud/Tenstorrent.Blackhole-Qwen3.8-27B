"""Real TTNN hidden-row extraction and changing-input replay; no throughput measurement."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from mtp_hidden_rows import MTPHiddenRows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    import torch
    import ttnn

    report = dict(passed=False, backend='hardware' if options.hardware else 'simulator',
        scope=__doc__, checks=[], source_checks=[], stale_controls=0,
        implementation_sha256=hashlib.sha256(Path(__file__).with_name('mtp_hidden_rows.py').read_bytes()).hexdigest())
    mesh, reader, sources = None, None, {}
    widths, patterns = (1, 2, 4, 8), (-19, 23, -19)

    def fixture(rows, pattern):
        return (torch.arange(5120).remainder(63).reshape(1, 1, 1, 5120)
                + torch.arange(rows).reshape(1, 1, rows, 1) * 64 + pattern).to(torch.bfloat16)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        for rows in widths:
            sources[rows] = ttnn.from_torch(fixture(rows, patterns[0]), device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        original_ids = {rows: addresses(ttnn, value) for rows, value in sources.items()}
        reader = MTPHiddenRows(ttnn, mesh, sources.values())
        reader.prepare()
        for repetition, pattern in enumerate(patterns):
            for rows, source in sources.items():
                staged = ttnn.from_torch(fixture(rows, pattern), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(staged, source)
                for row in range(rows):
                    output = reader(source, row)
                    if addresses(ttnn, output) != reader.destination_ids or addresses(ttnn, source) != original_ids[rows]:
                        raise AssertionError('Prepared row source or destination moved')
                    parts = ttnn.get_device_tensors(output)
                    if len(parts) != 2:
                        raise AssertionError('Both chips required')
                    for chip, part in enumerate(parts):
                        actual = ttnn.to_torch(part)
                        expected = fixture(rows, pattern)[:, :, row:row + 1]
                        if not torch.equal(actual, expected):
                            raise AssertionError(f'Hidden row differs: pattern={pattern}, rows={rows}, row={row}, chip={chip}')
                        if repetition == 1:
                            if torch.equal(actual, fixture(rows, patterns[0])[:, :, row:row + 1]):
                                raise AssertionError('Stale hidden row was not rejected')
                            report['stale_controls'] += 1
                        report['checks'].append(dict(repetition=repetition, rows=rows, row=row, chip=chip, exact=True))
                for chip, part in enumerate(ttnn.get_device_tensors(source)):
                    if not torch.equal(ttnn.to_torch(part), fixture(rows, pattern)):
                        raise AssertionError('Row extraction modified the source')
                    report['source_checks'].append(dict(repetition=repetition, rows=rows, chip=chip, exact=True))
                options.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(dict(repetition=repetition, rows=rows, exact=True)), flush=True)
        report['passed'] = (len(report['checks']) == 90 and len(report['source_checks']) == 24
                            and report['stale_controls'] == 30)
        if not report['passed']:
            raise AssertionError('Incomplete MTP hidden row matrix')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if reader is not None:
            reader.close()
        if mesh is not None:
            release_owned(ttnn, list(sources.values()))
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
