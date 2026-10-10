"""The drafter's matmuls on program grids sized from the device instead of fixed 8-wide ones (QWEN_FAST_DRAFT_MM_GRID, default off; tp4/fx-wp6, R2).

WHY. v678 (run 38034191271) ran the same job as v676 on the unlocked cards (13 x 10 worker grid, no harvested column). Matmuls whose program grid
is a fixed 8-wide rectangle lost 17-36 %: the drafter's gate and up (68 cores of an (8, 10) grid) 75.3 -> 88.1 us, its down projection (80 cores of
(8, 10)) 67.4 -> 91.4 us, its wo projection +21 %, the fused commit's 80-core projection 92.8 -> 124.7 us; the matmuls the target builds with
decode_grid_w (the device grid's width) moved -4 % to +3 %, and the drafter's (8, 5) and (8, 8) ones, which stop above row 5, did not move at
all (plan section 8.3). The cause is not established (the logical-to-physical column map and the distance to DRAM changed when no column is harvested).

WHAT. The MLP branch's matmuls keep everything that defines their arithmetic - the program kind (1D multicast, mcast_in0), in0_block_w, the output
subblock, per_core_M, per_core_N, fuse_batch, the fp32 output and the compute config - and change one field, compute_with_storage_grid_size, from the
fixed (8, y) to (min(device grid width, cores), ceil(cores / that)): the same number of cores (ceil(N tiles / per_core_N)), laid out row-major in
rows of the device's width, as decode_grid_w lays the target's out. Read from the device every time: 13 x 10 now, 11 x 10 before the firmware unlock;
nothing here names 11 or 13.

EXACTNESS. Bit-identical. A matmul output tile is the same sequence of in0_block_w-blocked multiply-accumulates into an fp32 destination whatever
cores run it: in a 1D multicast program each core owns per_core_N output columns for every row and accumulates K block by block in the same order, and
only WHICH core owns a column, and which core multicasts which in0 block, moves with the grid. CPU: the program configs differ in the grid field
alone (test_draft_mmgrid_tp), the core count is unchanged, and the branch's whole output is equal with the lever on and off on the functional fake.
Card: QWEN_FAST_DRAFT_MM_GRID_AUDIT=1 runs each matmul on both grids (the served one and the wide one) on the very same operands and compares every
chip's bytes (draft_fusion_tp's eager audit, on the warm pass); then the draft-singles audit and the position-keyed prefix compare as for the other
levers.

WHAT IS NOT HERE. The drafter's wo projection is in draft_attention_branch.py and the fused commit's 80-core projection in fused_commit_tp.py (neither
is this package's file): both can call wide_grid() / program_config() below at their own matmul with the same one-line change. grid_for() is the
hook the MLP branch uses; it needs only the weight's shape.

Stdlib only at import, py 3.7. The flag is strict 0 or 1 (draft_fusion_tp) and refused at the pair.
"""

import draft_fusion_tp as fusion

IN0_BLOCK_W = 4                 # the served drafter matmuls' (draft_mlp_branch.project); never changed by this lever

RUNTIME_FILES = ('draft_mmgrid_tp.py', 'draft_fusion_tp.py')


def enabled(environ=None):
    return fusion.enabled(fusion.MM_GRID, environ)


def cores_for(weight_shape, columns):
    """How many cores a 1D multicast matmul over a (K, N) weight takes at `columns` output tile columns per core: ceil(N tiles / columns)."""
    if len(weight_shape) < 2 or weight_shape[-1] % 32 or type(columns) is not int or columns < 1:
        raise ValueError('weight %r and %r columns per core are not a tiled 1D matmul' % (tuple(weight_shape), columns))
    return -(-(weight_shape[-1] // 32) // columns)


def wide_grid(mesh, cores):
    """(columns, rows) of the program grid for `cores` cores laid out row-major in rows as wide as the device's worker grid. ValueError when the
    device grid cannot be read or is too short to hold them."""
    grid = fusion.grid_of(mesh)
    if grid is None:
        raise ValueError('the compute grid cannot be read')
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer, got %r' % (cores,))
    width = min(grid[0], cores)
    rows = -(-cores // width)
    if rows > grid[1]:
        raise ValueError('%d cores need %d rows of %d, the grid has %d' % (cores, rows, width, grid[1]))
    return (width, rows)


def program_config(operations, grid, rows, columns):
    """The drafter's 1D multicast matmul program (draft_mlp_branch.project) on `grid`: only the grid differs between the served and the wide one."""
    return operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
        in0_block_w=IN0_BLOCK_W, out_subblock_h=1, out_subblock_w=1, per_core_M=rows // 32, per_core_N=columns,
        fuse_batch=True, fused_activation=None, mcast_in0=True)


def _matmul(operations, value, weight, grid, rows, columns, kernel):
    return operations.matmul(value, weight, dtype=operations.float32, program_config=program_config(operations, grid, rows, columns),
                             compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG)


def _audit(operations, mesh, value, weight, served_grid, wide, rows, columns, kernel, site):
    """Both grids on the same operands, every chip's bytes compared; the temporaries are freed here. Eager: only after
    draft_fusion_tp.audit_begin said yes (outside any capture, the device synchronized)."""
    held = []
    try:
        served = _matmul(operations, value, weight, served_grid, rows, columns, kernel)
        held.append(served)
        wider = _matmul(operations, value, weight, wide, rows, columns, kernel)
        held.append(wider)
        operations.synchronize_device(mesh)
        bad = fusion.differing_chips(operations, wider, served)
    finally:
        for tensor in reversed(held):
            try:
                operations.deallocate(tensor)
            except Exception:  # noqa: BLE001 - a temporary of an audit: never mask the verdict
                pass
    fusion.audit_result(fusion.MM_GRID, fusion.MM_GRID_AUDIT_LINE, fusion.MM_GRID_MISMATCH, site, rows, bad,
                        fusion.audited_total(fusion.MM_GRID), served='%dx%d' % tuple(served_grid), wide='%dx%d' % tuple(wide))


def grid_for(operations, mesh, value, weight, served_grid, columns, rows, kernel, site=None):
    """The program grid the branch's `project(value, weight, grid, columns)` should use: the wide one, or `served_grid` itself (with one logged line
    saying why) when the weight or the device grid do not allow it. Under the audit flag it also runs the matmul on both grids once per call, up to
    draft_fusion_tp.AUDIT_CALLS per site and row count."""
    shape = tuple(weight.shape)
    site = site or 'k%dn%d' % (shape[-2] if len(shape) > 1 else 0, shape[-1])
    try:
        wide = wide_grid(mesh, cores_for(shape, columns))
    except ValueError as failure:
        fusion.fell_back(fusion.MM_GRID_FALLBACK, site, str(failure))
        return served_grid
    if fusion.audit_enabled(fusion.MM_GRID_AUDIT) and fusion.audit_begin(operations, mesh, fusion.MM_GRID, site, rows):
        _audit(operations, mesh, value, weight, served_grid, wide, rows, columns, kernel, site)
    fusion.STATS['mmgrid'] += 1
    fusion.note(fusion.MM_GRID_ENGAGED, 'site=%s served=%dx%d wide=%dx%d cores=%d per_core_N=%d in0_block_w=%d' % (
        site, served_grid[0], served_grid[1], wide[0], wide[1], cores_for(shape, columns), columns, IN0_BLOCK_W))
    return wide
