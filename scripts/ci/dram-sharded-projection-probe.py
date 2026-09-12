"""Simulator-only native DRAM-sharded T16 projection screen; no speed claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dram_sharded_projection import configurations, execute
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from tiny_tile_matmul import PROJECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection', choices=('gate', 'up', 'down'), default='gate')
    parser.add_argument('--replay', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists() or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1':
        raise ValueError('Fresh isolated simulator report and owned packer compatibility required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

    names = ('dram-sharded-projection-probe.py', 'dram_sharded_projection.py', 'dram_projection_reload.py',
        'dram_projection_replay.py', 'tensix_projection_raw.py', 'tensix_projection_raw.cpp',
        'attention_batch.py', 'gdn_multitoken_conv.py')
    def hashes():
        return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=hashes(), checks=[], timing_qualified=False, replay_qualified=False)
    mesh = None
    owned = []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728)
        mesh.enable_program_cache()
        config = configurations(ttnn, mesh, options.projection)
        inner, width, cores, dtype_name, activated = PROJECTIONS[options.projection]
        report['projection'] = options.projection
        report['geometry'] = config['plan']
        generator = torch.Generator().manual_seed(383950)
        host_weight = (torch.randint(-8, 9, (2, 1, inner, width), generator=generator).float() / 32).bfloat16()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        def retain(value):
            owned.append(value)
            return value
        weight = retain(ttnn.from_torch(host_weight, device=mesh, dtype=getattr(ttnn, dtype_name),
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        save('reshard_weights')
        sharded = retain(ttnn.to_memory_config(weight, config['weights']))
        restored = retain(ttnn.to_memory_config(sharded, ttnn.DRAM_MEMORY_CONFIG))
        for original, current in zip(ttnn.get_device_tensors(weight), ttnn.get_device_tensors(restored), strict=True):
            if not torch.equal(ttnn.to_torch(original), ttnn.to_torch(current)):
                raise AssertionError('Weight resharding changed dequantized values')
        report['weight_roundtrip_exact'] = True
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)
        control = create_matmul_1d_decode_progcfg(16, inner, width, cores,
            fused_activation=ttnn.UnaryOpType.SILU if activated else None, fp32_acc=True, grid_w=11)
        host_patterns = []
        for pattern in range(2):
            save('eager_' + str(pattern))
            persistent_count = len(owned)
            host = (torch.randn((2, 1, 16, inner), generator=generator) * .125).bfloat16()
            host_patterns.append(host)
            source = retain(ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
            golden = retain(ttnn.linear(source, weight, program_config=control,
                compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG))
            candidate = execute(ttnn, source, sharded, config, compute, retain, preserve_partials=True)
            for chip in range(2):
                expected = ttnn.to_torch(ttnn.get_device_tensors(golden)[chip])
                actual = ttnn.to_torch(ttnn.get_device_tensors(candidate)[chip])
                exact = torch.equal(actual, expected)
                report['checks'].append(dict(pattern=pattern, chip=chip, exact=exact,
                    failed_elements=int((actual != expected).sum()),
                    max_abs=float((actual.float() - expected.float()).abs().max())))
                if not exact:
                    raise AssertionError('Native DRAM-sharded projection changes target output')
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, owned[persistent_count:])
            del owned[persistent_count:]
        if options.replay:
            from dram_projection_replay import check
            save('replay')
            source = retain(ttnn.from_torch(host_patterns[0], device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
            def reference(keep):
                return keep(ttnn.linear(source, weight, program_config=control,
                    compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG))
            report['replay'] = check(ttnn, torch, mesh, source, sharded, weight, config,
                compute, reference, host_patterns, mapper, retain)
            report['replay_qualified'] = len(report['replay']['checks']) == 6
            if not report['replay_qualified']:
                raise AssertionError('All changed-input replay checks required')
        report['passed'] = len(report['checks']) == 4
        save('complete')
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        report['sources_after'] = hashes()
        if report['sources_after'] != report['sources']:
            report['passed'] = False
        save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
