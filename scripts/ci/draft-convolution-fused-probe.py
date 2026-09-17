"""Exact fused/composed convolution, changed-input replay and ownership; not model throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from draft_convolution import convolution_reference, grouped_causal_convolution
from draft_convolution_fused import fused_convolution, checked_convolution
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, choices=(1, 8, 32), nargs='+', default=[1, 8, 32])
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[], wrapper_checks=[], stale_controls=0, rounding_controls=0,
        borrowed_checks=0, sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('draft-convolution-fused-probe.py', 'draft_convolution_fused.py',
                'draft_convolution_fused_compute.cpp', 'draft_convolution_fused_io.cpp', 'draft_convolution.py')})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=33554432)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        for rows in options.rows:
            owned, trace, captured = [], None, None
            shapes = [(2, 1, rows, 5120), (2, 1, rows, 320), (2, 1, rows, 320),
                (2, 1, 1, 5120), (2, 1, 1, 5120)]
            tensors, binding = [], []
            try:
                for seed in range(2):
                    print(json.dumps(dict(stage='convolution', rows=rows, seed=seed)), flush=True)
                    generator = torch.Generator().manual_seed(8039 + rows + seed)
                    host = [(torch.randint(-8, 9, shape, generator=generator).float() / 16 if seed == 0
                        else torch.randn(shape, generator=generator) * 3).bfloat16() for shape in shapes]
                    expected = [convolution_reference(host[0][chip:chip + 1],
                        [value[chip:chip + 1] for value in host[1:3]], [value[chip:chip + 1] for value in host[3:]])
                        for chip in range(2)]
                    if seed:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        if any(torch.equal(ttnn.to_torch(shard), target) for shard, target in
                                zip(ttnn.get_device_tensors(captured), expected, strict=True)):
                            raise AssertionError('Changed-input fixture must detect stale trace inputs on both chips')
                        report['stale_controls'] += 1
                    for index, value in enumerate(host):
                        if seed == 0:
                            tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
                            owned.append(tensor)
                            tensors.append(tensor)
                        else:
                            payload = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                            ttnn.copy_host_to_device_tensor(payload, tensors[index])
                    current_binding = [addresses(ttnn, value) for value in tensors]
                    if seed == 0:
                        binding = current_binding
                    elif current_binding != binding:
                        raise AssertionError('Input bindings changed across replay')
                    composed = grouped_causal_convolution(ttnn, mesh, tensors[0], tensors[1:3], tensors[3:],
                        fp32_intermediates=True)
                    owned.append(composed)
                    eager = checked_convolution(ttnn, mesh, tensors[0], tensors[1:3], tensors[3:],
                        fp32_intermediates=True, retain_temporaries=owned.append, audit=True,
                        checks=report['wrapper_checks'], context=dict(position=seed * rows, layer=0))
                    ttnn.synchronize_device(mesh)
                    if seed == 0:
                        trace, captured = capture_operation(ttnn, mesh,
                            lambda: fused_convolution(ttnn, mesh, tensors[0], tensors[1:3], tensors[3:]))
                        owned.append(captured)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for path, output in (('composed', composed), ('fused', eager), ('replay', captured)):
                        for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                            actual = ttnn.to_torch(shard)
                            exact = torch.equal(actual.view(torch.int16), expected[chip].view(torch.int16))
                            report['checks'].append(dict(rows=rows, seed=seed, chip=chip, path=path, exact=exact))
                            if not exact:
                                differences = actual.view(torch.int16) != expected[chip].view(torch.int16)
                                report['first_difference'] = dict(indices=differences.nonzero()[:16].tolist(),
                                    actual=actual[differences][:16].float().tolist(),
                                    expected=expected[chip][differences][:16].float().tolist())
                                torch.save(dict(host=host, actual=actual, expected=expected[chip], chip=chip, path=path),
                                    options.output.with_suffix('.failure.pt'))
                                raise AssertionError(f'{path} differs at {int(differences.sum())} BF16 elements, rows={rows}, seed={seed}, chip={chip}')
                    for tensor, value in zip(tensors, host, strict=True):
                        for chip, shard in enumerate(ttnn.get_device_tensors(tensor)):
                            if not torch.equal(ttnn.to_torch(shard).view(torch.int16), value[chip:chip + 1].view(torch.int16)):
                                raise AssertionError('Borrowed convolution operand changed')
                            report['borrowed_checks'] += 1
                    if seed:
                        for chip in range(2):
                            hidden, dynamic0, dynamic1, base0, base1 = [value[chip:chip + 1].float() for value in host]
                            previous = torch.cat((torch.zeros_like(hidden[..., :1, :]), hidden[..., :-1, :]), dim=2)
                            unrounded = (base0 * hidden + dynamic0.repeat_interleave(16, dim=-1) * hidden
                                + base1 * previous + dynamic1.repeat_interleave(16, dim=-1) * previous).bfloat16()
                            if torch.equal(unrounded, expected[chip]):
                                raise AssertionError('Fixture must detect removed BF16 rounding boundaries')
                            report['rounding_controls'] += 1
                report['last_rows'] = rows
                options.output.write_text(json.dumps(report, indent=2))
            finally:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
        report['passed'] = (len(report['checks']) == 12 * len(options.rows)
            and len(report['wrapper_checks']) == 4 * len(options.rows)
            and report['stale_controls'] == len(options.rows) and report['rounding_controls'] == 2 * len(options.rows))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
