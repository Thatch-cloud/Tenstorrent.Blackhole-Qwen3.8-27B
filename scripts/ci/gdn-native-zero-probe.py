"""T16 zero-prefix publication refreshes stale entry from unchanged native state."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from gdn_native_slot_publication import prepare
from gdn_publication_fixture import host_layer, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh simulator report required')
    import torch
    import ttnn

    directory = Path(__file__).parent
    sources = ('gdn-native-zero-probe.py', 'gdn_native_slot_publication.py', 'gdn_commit_dma.py',
        'gdn_commit_dma.cpp', 'gdn_state_copy.py', 'gdn_state_copy.cpp', 'gdn_publication_fixture.py',
        'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')
    def hashes():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sources}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(), checks=[],
        prefix=0, layers=1, hardware_qualified=False, padding_audited=False, scope=__doc__)
    mesh, trace, owned = None, None, []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        def upload(value, device=False):
            return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
                **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
        owned = [upload(value, True) for value in host_layer(0, 0)]
        bindings = [addresses(ttnn, value) for value in owned]
        publication = prepare(mesh, [owned], 0, experimental=True)
        def refresh(pattern):
            values = host_layer(pattern, 0)
            for source, target in zip(values, owned, strict=True):
                ttnn.copy_host_to_device_tensor(upload(source), target)
            entry = [torch.cat([values[10][chip * 8:chip * 8 + 1] for chip in range(2)])]
            entry += [value[:, :1].clone() for value in values[11:15]]
            if torch.equal(values[0], entry[0]):
                raise AssertionError('Fixture must distinguish stale entry from native slot zero')
            return expected(entry + values[5:], 0)
        def check(values, pattern, mode):
            for operand, (tensor, reference) in enumerate(zip(owned, values, strict=True)):
                for chip, (part, target) in enumerate(zip(ttnn.get_device_tensors(tensor), reference.chunk(2), strict=True)):
                    if not torch.equal(ttnn.to_torch(part), target):
                        raise AssertionError(f'Zero publication mismatch {pattern=} {mode=} {operand=} {chip=}')
                    report['checks'].append(dict(pattern=pattern, mode=mode, operand=operand, chip=chip, exact=True))
            if [addresses(ttnn, value) for value in owned] != bindings:
                raise AssertionError('Publication bindings changed')
        save('eager')
        reference = refresh(0)
        publication()
        ttnn.synchronize_device(mesh)
        check(reference, 0, 'eager')
        trace, unused = capture_operation(ttnn, mesh, publication)
        for pattern in (1, 0):
            save('replay_' + str(pattern))
            reference = refresh(pattern)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(reference, pattern, 'replay')
        if len(report['checks']) != 120:
            raise AssertionError('Complete two-chip zero-publication matrix required')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
