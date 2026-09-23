"""G0 for verify-trace T1: the device byte compares that must pass before any token gate.

Two class-B* cuts of QWEN_FAST_VERIFY_T1 rest on a property only the device can show:

  #11 (lever_n_m3native_patch section A2) re-partitions the attn_qkv and MLP gate M = 64
      matmul configs over N. Claim: with in0_block_w, the fused activation and the call
      site's compute config unchanged, every output tile is reduced over K in the same
      blocks on one core, so the outputs are the same bytes. Checked here per projection:
      the old and the new config on the same device weight (the model's dtype: bf8 qkv, bf4
      gate) and the same activations (64 rows; several input kinds, or activations captured
      from a verify round via --activations), torch.equal on the whole output, plus warm
      device-synchronised timing of each.
  #8a (verify_trace_t1.sample_shards / combine_shards) replaces the pinned sampler with a
      per-chip argmax + max and a host combine. Claim: equal to torch.argmax's first
      occurrence over the full 248320-wide row, which the pinned sampler matches
      (sampling-links.py). Checked on the injected cases the sampler probes use (random,
      boundaries, cross-shard tie, near tie, all equal) plus same-shard ties and a
      +0/-0 pair across the shard boundary, every row, ids and the chosen value.

It runs where the model tree and ttnn are (inside the serving image), with this checkout's
scripts/ci first on the path - it needs verify_trace_t1.py (in the image only from the wave-2
build on) and lever_n_m3native_patch.py (never in the image). One card (a 1x1 mesh: the two
vocab shards run one after the other on the same chip) or the pair (1x2: each chip its own
shard, as in the model). Nothing here is imported by the serving path. Run it through
verify-t1-g0-rig.sh, which mounts this checkout's scripts/ci, passes the explicit allocation
(QWEN_HARDWARE_TESTS=1 QWEN_CARDS_ALLOCATED=1) into the container and refuses while any
container holds a card.

Pinned to what the model runs, not to what the card reports: the worker grid must be the
model's decode_grid_w = 11 (the model reads it from the same compute_with_storage_grid_size,
and every MLP probe pins 11) with at least 8 rows, and every config's grid, per_core_N and
subblock must be the shape the graft builds there (PLAN_SHAPES; the new qkv config's last
core holds 2 tiles with subblock width 3, the case no other config exercises). The compute
configs are the call sites': attention/tp.py uses tpc.COMPUTE_HIFI2, and the MLP gate uses
its decode config (the 64-row decode input is [1, 1, 64, 5120], so T = x.shape[1] = 1), which
the MLP probes pin as LoFi, math_approx_mode, fp32_dest_acc_en and packer_l1_acc. The gate's
input is L1-interleaved, as the call site's to_memory_config(x, L1) makes it.

A 1x1 run proves the per-shard kernels; only a 1x2 run (the report's chip_order_proven) also
proves the chip-to-vocab-offset order the host combine assumes. Run 1x2 once before G1.

The report's `passed` is true only when every comparison is byte-exact.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import lever_n_m3native_patch as patcher
import verify_trace_t1

ROWS = 64
DIM = 5120
INPUT_KINDS = ('randn', 'wide', 'outliers', 'tiny')
ARGMAX_KINDS = ('random', 'boundaries', 'cross-shard-tie', 'near-tie', 'all-equal', 'same-shard-tie', 'signed-zero')


DECODE_GRID_W = 11
MIN_GRID_ROWS = 8
# (grid, per_core_N, (out_subblock_h, out_subblock_w)) the graft builds at decode_grid_w = 11
# (test_verify_trace_t1_graft.MatmulConfigGraftTests holds the graft to the same numbers).
PLAN_SHAPES = {
    'attn_qkv': dict(before=((8, 8), 4, (1, 4)), after=((11, 4), 6, (1, 3))),
    'mlp_w1': dict(before=((11, 4), 7, (2, 1)), after=((11, 8), 4, (1, 4))),
}


def plan(decode_grid_w=DECODE_GRID_W):
    """The #11 comparisons, as model_config builds them before and after the graft's A2 block
    (the builder arguments of NATIVE_64_BLOCK and of the verify t1 helper)."""
    return [
        dict(name='attn_qkv', m=ROWS, k=DIM, n=7168, weight='bfloat8_b', silu=False, compute='hifi2',
             input='dram', output='dram', before=dict(num_cores=64),
             after=dict(num_cores=patcher.VERIFY_T1_ATTN_QKV_CORES, grid_w=decode_grid_w)),
        dict(name='mlp_w1', m=ROWS, k=DIM, n=8704, weight='bfloat4_b', silu=True, compute='lofi_decode',
             input='l1', output='l1', before=dict(num_cores=44, grid_w=decode_grid_w),
             after=dict(num_cores=patcher.VERIFY_T1_MLP_W1_CORES, grid_w=decode_grid_w)),
    ]


def model_grid_w(grid):
    """The model's decode_grid_w from the opened device's worker grid, or why G0 would not test
    what the model runs."""
    if grid.x != DECODE_GRID_W or grid.y < MIN_GRID_ROWS:
        raise ValueError('G0 needs the model worker grid (decode_grid_w %d, at least %d rows); the device reports '
                         '%dx%d' % (DECODE_GRID_W, MIN_GRID_ROWS, grid.x, grid.y))
    return grid.x


def config_shape(config):
    """(grid, per_core_N, subblock) of a 1D matmul program config, whether its grid is a
    CoreCoord or a tuple."""
    grid = config.compute_with_storage_grid_size
    grid = (grid.x, grid.y) if hasattr(grid, 'x') else tuple(grid)
    return grid, config.per_core_N, (config.out_subblock_h, config.out_subblock_w)


def compute_configs(ttnn, tpc):
    """The call sites' compute kernel configs (see the module docstring)."""
    return dict(hifi2=tpc.COMPUTE_HIFI2,
                lofi_decode=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                                             math_approx_mode=True, fp32_dest_acc_en=True,
                                                             packer_l1_acc=True))


def same_values(found, expected):
    """Bit-equal, except that +0 and -0 are one value: the combine and every argmax treat them
    as equal, so a device max returning +0 for a -0 maximum is not a G0 failure."""
    import torch

    found, expected = found.float().reshape(-1), expected.float().reshape(-1)
    if found.shape != expected.shape:
        return False
    bits = found.view(torch.int32) == expected.view(torch.int32)
    zeros = (found == 0) & (expected == 0)
    return bool((bits | zeros).all())


def activations(kind, generator):
    import torch

    values = torch.randn(ROWS, DIM, generator=generator)
    if kind == 'wide':
        values = values * 16
    elif kind == 'outliers':
        mask = torch.rand(ROWS, DIM, generator=generator) < 0.01
        values = torch.where(mask, values * 100, values)
    elif kind == 'tiny':
        values = values * 1e-3
    elif kind != 'randn':
        raise ValueError(kind)
    return values.to(torch.bfloat16)


def argmax_case(kind, rows=ROWS, width=verify_trace_t1.SHARD_WIDTH):
    """(1, 1, rows, 2 * width) bf16 logits: sampling-kernel.logits_case's kinds, plus a tie
    inside one shard and a +0 / -0 pair straddling the shard boundary."""
    import torch

    vocab = 2 * width
    generator = torch.Generator().manual_seed(123)
    if kind == 'random':
        return torch.randn(1, 1, rows, vocab, generator=generator).to(torch.bfloat16)
    logits = torch.full((1, 1, rows, vocab), -10.0, dtype=torch.bfloat16)
    boundaries = (0, 31, 32, width - 1, width, vocab - 1)
    for row in range(rows):
        if kind == 'boundaries':
            logits[0, 0, row, boundaries[row % len(boundaries)]] = 100.0
        elif kind == 'cross-shard-tie':
            logits[0, 0, row, 31] = 100.0
            logits[0, 0, row, width] = 100.0
        elif kind == 'near-tie':
            logits[0, 0, row, 31] = 1.0
            logits[0, 0, row, width] = 1.0078125
        elif kind == 'all-equal':
            logits.zero_()
            break
        elif kind == 'same-shard-tie':
            shard = row % 2
            logits[0, 0, row, shard * width + 7 + row] = 50.0
            logits[0, 0, row, shard * width + 4000 + row] = 50.0
        elif kind == 'signed-zero':
            logits[0, 0, row, width - 1 - row] = -0.0
            logits[0, 0, row, width + row] = 0.0
        else:
            raise ValueError(kind)
    return logits


def reference_ids(logits):
    """What the pinned sampler returns (sampling-links.py checks it against this exactly)."""
    return logits.reshape(logits.shape[-2], logits.shape[-1]).float().argmax(dim=-1)


def compare_matmuls(ttnn, mesh, tpc, iterations, captured=None):
    import torch

    compute = compute_configs(ttnn, tpc)
    grid_w = model_grid_w(mesh.compute_with_storage_grid_size())
    results = []
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    generator = torch.Generator().manual_seed(2026)
    for projection in plan(grid_w):
        silu = dict(fused_activation=ttnn.UnaryOpType.SILU) if projection['silu'] else {}
        configs = {side: tpc.create_matmul_1d_decode_progcfg(projection['m'], projection['k'], projection['n'],
                                                             **dict(projection[side], **silu))
                   for side in ('before', 'after')}
        for field in patcher.VERIFY_T1_KEPT_FIELDS:
            if getattr(configs['before'], field, None) != getattr(configs['after'], field, None):
                raise AssertionError('%s: %s differs between the configs' % (projection['name'], field))
        for side, config in configs.items():
            if config_shape(config) != PLAN_SHAPES[projection['name']][side]:
                raise AssertionError('%s %s config is %r, not the %r the model runs' % (
                    projection['name'], side, config_shape(config), PLAN_SHAPES[projection['name']][side]))
        weight = ttnn.from_torch((torch.randn(projection['k'], projection['n'], generator=generator) * 0.02),
                                 dtype=getattr(ttnn, projection['weight']), layout=ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=replicate)
        memory = ttnn.DRAM_MEMORY_CONFIG if projection['output'] == 'dram' else ttnn.L1_MEMORY_CONFIG
        kinds = list(INPUT_KINDS) + (['captured'] if captured and projection['name'] in captured else [])
        for kind in kinds:
            host = captured[projection['name']].reshape(ROWS, DIM).to(torch.bfloat16) if kind == 'captured' \
                else activations(kind, generator)
            x = ttnn.from_torch(host.reshape(1, 1, ROWS, DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=mesh, mesh_mapper=replicate,
                                memory_config=ttnn.L1_MEMORY_CONFIG if projection['input'] == 'l1'
                                else ttnn.DRAM_MEMORY_CONFIG)
            outputs, timings = {}, {}
            for side, config in configs.items():
                def run():
                    return tpc.matmul_1d_decode(x, weight, config, compute[projection['compute']],
                                                out_memory_config=memory)

                result = run()
                ttnn.synchronize_device(mesh)
                outputs[side] = [ttnn.to_torch(part) for part in ttnn.get_device_tensors(result)]
                ttnn.deallocate(result)
                samples = []
                for unused in range(iterations):
                    started = time.perf_counter()
                    result = run()
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - started) * 1e6)
                    ttnn.deallocate(result)
                timings[side] = statistics.median(samples) if samples else None
            exact = all(torch.equal(old.view(torch.int16), new.view(torch.int16))
                        for old, new in zip(outputs['before'], outputs['after'], strict=True))
            results.append(dict(projection=projection['name'], kind=kind, exact=exact,
                                median_us=timings, chips=len(outputs['before']),
                                shapes={side: config_shape(config) for side, config in configs.items()}))
            ttnn.deallocate(x)
        ttnn.deallocate(weight)
    return results


def compare_argmax(ttnn, mesh):
    import torch

    chips = mesh.get_num_devices()
    width = verify_trace_t1.SHARD_WIDTH
    results = []
    for kind in ARGMAX_KINDS:
        logits = argmax_case(kind)
        expected = reference_ids(logits)
        if chips == 2:
            device = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
            ids, values = verify_trace_t1.sample_shards(ttnn, device, ROWS)
            chip_ids = [ttnn.to_torch(part).reshape(-1)[:ROWS] for part in ttnn.get_device_tensors(ids)]
            chip_values = [ttnn.to_torch(part).reshape(-1)[:ROWS] for part in ttnn.get_device_tensors(values)]
            for value in (device, ids, values):
                ttnn.deallocate(value)
        else:
            chip_ids, chip_values = [], []
            for shard in range(2):
                half = logits[..., shard * width:(shard + 1) * width].contiguous()
                device = ttnn.from_torch(half, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ids, values = verify_trace_t1.sample_shards(ttnn, device, ROWS)
                chip_ids.append(ttnn.to_torch(ids).reshape(-1)[:ROWS])
                chip_values.append(ttnn.to_torch(values).reshape(-1)[:ROWS])
                for value in (device, ids, values):
                    ttnn.deallocate(value)
        combined = verify_trace_t1.combine_shards(chip_ids, chip_values)
        shard_maxima = [logits[0, 0, :, shard * width:(shard + 1) * width].float().amax(dim=-1) for shard in range(2)]
        values_exact = all(same_values(chip_values[shard], shard_maxima[shard]) for shard in range(2))
        mismatched = [row for row in range(ROWS) if int(combined[row]) != int(expected[row])]
        results.append(dict(kind=kind, exact=not mismatched and values_exact, ids_exact=not mismatched,
                            values_exact=values_exact, mismatched_rows=mismatched[:8]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', choices=('1x1', '1x2'), default='1x1')
    parser.add_argument('--part', choices=('matmul', 'argmax', 'all'), default='all')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--activations', type=Path, default=None,
                        help="torch.save'd {'attn_qkv': [64, 5120], 'mlp_w1': [64, 5120]} captured activations")
    options = parser.parse_args()
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    report = dict(passed=False, scope='G0 for QWEN_FAST_VERIFY_T1 #11 and #8a; not a token gate', mesh=options.mesh,
                  chip_order_proven=options.mesh == '1x2')
    mesh = None
    try:
        chips = 2 if options.mesh == '1x2' else 1
        if chips == 2:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, chips), l1_small_size=24576)
        mesh.enable_program_cache()
        captured = torch.load(options.activations) if options.activations else None
        if options.part in ('matmul', 'all'):
            report['matmul'] = compare_matmuls(ttnn, mesh, tpc, options.iterations, captured)
        if options.part in ('argmax', 'all'):
            report['argmax'] = compare_argmax(ttnn, mesh)
        report['passed'] = all(entry['exact'] for part in ('matmul', 'argmax') for entry in report.get(part, ()))
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
        raise
    finally:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
