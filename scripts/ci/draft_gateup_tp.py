"""The drafter MLP's gate and up projections as ONE matmul launch at four cards (QWEN_FAST_DRAFT_GATEUP1, default off; tp4/fx-wp6, the gate|up
half of F-F3c).

WHY. Per layer the MLP branch runs the gate and the up projection as two matmuls over the same input (`prepared`, the conv output): 75.6 and
76.1 us on 68 cores (v676 quad: 10 launches, 0.753 ms a quad). Two launches read one input and stream two weight matrices of the same shape.

WHAT. At load (draft_mlp_branch.prepare_mlp_branch, flag on) each chip's gate and up weights are uploaded as ONE (5120, 2 * 4352) matrix,
gate columns first - replacing the two separate uploads, so DRAM does not grow - and the branch runs one matmul with per_core_N twice the
served one (4 where the served gate and up use 2: 272 output tiles on 68 cores, the pair's own gate/up program shape), the same
in0_block_w, fp32 output and compute config. Output columns [0, 4352) are the gate projection and [4352, 8704) the up projection. QWEN_FAST_DRAFT_TAIL
reads the two halves in place (draft_tail_tp.swiglu_fused); without it the branch cuts them with two tile-aligned slices (exact copies)
for the served SwiGLU, which pays for itself only barely - the lever is meant to be run with the tail.

EXACTNESS. Bit-identical: a matmul output element is its own K reduction, the same sequence of in0_block_w-blocked multiply-accumulates into
an fp32 destination whatever the other columns of the same core are, so output column j of the fused launch equals column j of the separate
launch; per_core_N changes only which core owns the column. Bfloat8_b weights quantise in 16-element face rows inside a tile, and the fused
matrix concatenates whole tile columns, so each tile's bytes are the separate matrix's. CPU: the fused weight's halves equal the separate
weights, the program geometry is checked (whole columns per core, cores within the grid), and the fused matmul is compared column for column
with the separate ones on a fake that reduces in the served K order (test_draft_gateup_tp). Card: QWEN_FAST_DRAFT_GATEUP1_AUDIT=1 uploads the
separate weights (temporarily, per audited call, then frees them), runs the served matmuls beside the fused one and compares every chip's bytes
column for column (draft_fusion_tp's eager audit).

NOT A KERNEL: this lever is a weight layout and a program config; it has no .cpp.

Stdlib only at import (torch inside functions), py 3.7. The flag is strict 0 or 1 (draft_fusion_tp) and refused at the pair.
"""

import math

import draft_fusion_tp as fusion

GRID = (8, 10)                  # the served gate/up program grid (draft_mlp_branch.execute_mlp_branch)
GRID_CORES = GRID[0] * GRID[1]
MAX_SERVED_CORES = 80           # the cores the served gate/up spread their tiles over: ceil(shard tiles / 80) columns each

RUNTIME_FILES = ('draft_gateup_tp.py', 'draft_fusion_tp.py')


def enabled(environ=None):
    return fusion.enabled(fusion.GATEUP1, environ)


def served_columns(shard_tiles):
    """Per-core output columns of the served gate / up matmuls: ceil(shard tiles / 80) (draft_mlp_branch.gate_up_columns_of)."""
    return math.ceil(shard_tiles / MAX_SERVED_CORES)


def plan(shard_tiles):
    """The fused matmul's geometry for a gate (= up) shard of `shard_tiles` tile columns: per-core columns (twice the served), the cores, and
    whether the geometry is legal (whole columns per core, within the 8 x 10 grid). ValueError when it is not."""
    if type(shard_tiles) is not int or shard_tiles < 1:
        raise ValueError('shard_tiles must be a positive integer, got %r' % (shard_tiles,))
    columns = 2 * served_columns(shard_tiles)
    total = 2 * shard_tiles
    if total % columns:
        raise ValueError('%d fused output tiles do not split into whole per-core columns of %d' % (total, columns))
    cores = total // columns
    if cores > GRID_CORES:
        raise ValueError('%d cores exceed the %d x %d program grid' % (cores, GRID[0], GRID[1]))
    return dict(shard_tiles=shard_tiles, columns=columns, tiles=total, cores=cores, half_tiles=shard_tiles)


def shard_problem(shards):
    """Why these per-rank (gate, up, down) shards cannot take the fused layout (a short reason), or None."""
    try:
        gate, up = shards[0][0], shards[0][1]
    except (IndexError, KeyError, TypeError):
        return 'no weight shards'
    if tuple(gate.shape) != tuple(up.shape) or len(gate.shape) != 2:
        return 'gate %r and up %r shards differ' % (tuple(getattr(gate, 'shape', ())), tuple(getattr(up, 'shape', ())))
    if gate.shape[1] % 32:
        return 'the shard width %d is not whole tiles' % gate.shape[1]
    try:
        plan(gate.shape[1] // 32)
    except ValueError as failure:
        return str(failure)
    return None


def fused_host_weight(shards):
    """The host matrix whose dim-0 shard for rank r is [gate_r | up_r] (5120, 2 * W): the per-rank column concatenation, ranks stacked on dim 0
    exactly as the served upload stacks the separate gate and up (torch.cat([rank[index] for rank in shards], dim=0))."""
    import torch

    return torch.cat([torch.cat([rank[0], rank[1]], dim=1) for rank in shards], dim=0)


def separate_host_weight(shards, index):
    """The served upload's host matrix for projection `index` (0 gate, 1 up, 2 down)."""
    import torch

    return torch.cat([rank[index] for rank in shards], dim=0)


def split_halves(host):
    """The gate and the up half of a fused projection read back to the host: the columns [0, W) and [W, 2W)."""
    half = host.shape[-1] // 2
    return host[..., :half], host[..., half:]


def served_projection(operations, parameters, value, weight, rows, columns):
    """The served gate / up matmul (draft_mlp_branch's `project` with the 8 x 10 grid and `columns` per core), WITHOUT a lifetime owner: the audit
    owns what it makes and frees it itself."""
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=GRID,
        in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32, per_core_N=columns,
        fuse_batch=True, fused_activation=None, mcast_in0=True)
    return operations.matmul(value, weight, dtype=operations.float32, program_config=program,
                             compute_kernel_config=parameters['kernel'], memory_config=operations.DRAM_MEMORY_CONFIG)


def audit(operations, mesh, parameters, prepared, fused, rows, site='mlp'):
    """Eager: upload the separate gate and up weights from the host shards, run the served matmuls (program config of the served spread) on the
    same input, and compare every chip's bytes with the fused output's halves. Frees what it made. Raises AssertionError on a difference."""
    import torch

    shards = parameters['shards']
    dtype = parameters['device_projections'][0].dtype
    columns = served_columns(tuple(shards[0][0].shape)[1] // 32)
    operations.synchronize_device(mesh)
    held = []
    try:
        references = []
        for index in (0, 1):
            weight = operations.from_torch(separate_host_weight(shards, index), device=mesh, dtype=dtype,
                layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
                mesh_mapper=operations.ShardTensorToMesh(mesh, dim=0))
            held.append(weight)
            references.append(served_projection(operations, parameters, prepared, weight, rows, columns))
            held.append(references[-1])
        operations.synchronize_device(mesh)
        mine = fusion.chip_bits(operations, fused)
        gate, up = [fusion.chip_bits(operations, reference) for reference in references]
        bad = []
        for chip in range(len(mine)):
            left, right = split_halves(mine[chip])
            if (left.shape != gate[chip].shape or right.shape != up[chip].shape
                    or not torch.equal(left, gate[chip]) or not torch.equal(right, up[chip])):
                bad.append(chip)
    finally:
        for tensor in reversed(held):
            try:
                operations.deallocate(tensor)
            except Exception:  # noqa: BLE001 - a temporary of an audit: never mask the verdict
                pass
    fusion.audit_result(fusion.GATEUP1, fusion.GATEUP1_AUDIT_LINE, fusion.GATEUP1_MISMATCH, site, rows, bad,
                        fusion.audited_total(fusion.GATEUP1), columns=2 * columns)


def project_fused(operations, mesh, parameters, prepared, project, rows, *, site='mlp'):
    """The fused gate|up matmul on `prepared`: the (1, 1, rows, 2 * W) fp32 projection. `project(value, weight, grid, columns)` is the
    branch's own matmul closure (the served program shape with `columns` per core)."""
    shard_tiles = tuple(parameters['shards'][0][0].shape)[1] // 32 if parameters.get('shards') else None
    found = plan(shard_tiles)
    fused = project(prepared, parameters['device_projections'][0], GRID, found['columns'])
    fusion.STATS['gateup1'] += 1
    fusion.note(fusion.GATEUP1_ENGAGED, 'site=%s rows=%d per_core_N=%d cores=%d tiles=%d' % (
        site, rows, found['columns'], found['cores'], found['tiles']))
    if fusion.audit_enabled(fusion.GATEUP1_AUDIT) and fusion.audit_due(fusion.GATEUP1, site, rows):
        audit(operations, mesh, parameters, prepared, fused, rows, site)
    return fused

