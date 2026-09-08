"""Simulator-only real tail-copy kernels and capture ownership, not learned-model or throughput evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from dflash_prefill_window import prefill_window, snapshot_prefill_tail
from dflash_request_runtime import TARGET_TAPS
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses
from target_features import LayerOutputCapture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[], ownership_checks=[], unchanged_inputs=[], negative_controls=[],
        hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('dflash_prefill_window.py', 'target_features.py', 'dflash-prefill-window-probe.py')})
    mesh = source = captured = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        for position in (170, 2048, 2049, 4093, 4096):
            window = prefill_window(position)
            padded_rows = math.ceil(position / 32) * 32 + 32
            host = torch.cat([((torch.arange(padded_rows).reshape(1, 1, padded_rows, 1) % 97
                + torch.arange(2560).reshape(1, 1, 1, 2560) % 13 + chip * 128) / 4).bfloat16() for chip in range(2)], dim=0)
            host[..., position:, :] = -500
            expected = host[..., window['start']:position, :].clone()
            source = ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            model = SimpleNamespace(layers=[SimpleNamespace(forward=lambda value: value) for layer in range(64)])
            captured = LayerOutputCapture(model, TARGET_TAPS,
                snapshot=lambda value: snapshot_prefill_tail(ttnn, value, position, checks=report['checks']),
                release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))))
            with captured.capture():
                for tap in TARGET_TAPS:
                    model.layers[tap].forward(source)
            for chip, shard in enumerate(ttnn.get_device_tensors(source)):
                if not torch.equal(ttnn.to_torch(shard), host[chip:chip + 1]):
                    raise AssertionError('Tail capture modified borrowed full-context features')
                report['unchanged_inputs'].append(dict(position=position, chip=chip, exact=True))
            changed = ttnn.from_torch(torch.full_like(host, -200), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(changed, source)
            for tap, output in zip(TARGET_TAPS, captured.outputs(), strict=True):
                for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                    actual = ttnn.to_torch(shard).contiguous()
                    if not torch.equal(actual.view(torch.int16), expected[chip:chip + 1].contiguous().view(torch.int16)):
                        raise AssertionError('Owned tail changed after the borrowed source was overwritten')
                    report['ownership_checks'].append(dict(position=position, tap=tap, chip=chip, exact=True))
            for name, wrong in [('padded-tail', host[..., -window['rows']:, :]),
                    *([('first-window', host[..., :window['rows'], :])] if window['start'] else [])]:
                if torch.equal(expected, wrong):
                    raise AssertionError('Fixture failed to distinguish a wrong feature window')
                report['negative_controls'].append(dict(position=position, mutation=name, detected=True))
            captured.close()
            captured = None
            ttnn.deallocate(source)
            source = None
        report['passed'] = (len(report['checks']) == 50 and len(report['ownership_checks']) == 50
            and len(report['unchanged_inputs']) == 10 and len(report['negative_controls']) == 8)
        if not report['passed']:
            raise AssertionError('Incomplete simulator tail-window matrix')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if captured is not None:
            captured.close()
        if source is not None:
            ttnn.deallocate(source)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
