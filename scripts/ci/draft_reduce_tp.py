"""The drafter's local reduce at four cards: ONE launch for the four slices and three adds that follow the gather (QWEN_FAST_DRAFT_REDUCE,
default off; tp4/fx-wp6, F-F1 "DR").

WHY. Every drafter projection that ends in the dim-0 all-gather (the attention output projection and the MLP down projection of each of the
five layers: ten chains a pass, plus the feature projection's) is followed by the same glue: one slice per chip's partial (7 us each), three
fp32 adds (10.4 us each), then the typecast the caller does. Measured on the v676 quad (traces 363 and 365): 80 launches, 0.648 ms a quad,
of which 0.324 ms are the slices and fp32 adds this module replaces by ten launches (docs/tp4-fusion-wp6.md).

WHAT. The all-gather is untouched (same call, same semaphore cycling, same links and topology: its order is exactness-locked). The gathered
(4, 1, rows, 5120) fp32 TILE tensor is then read once, tile by tile, by draft_reduce_tp_io.cpp, summed by draft_reduce_tp_compute.cpp in the
served order - ((p0 + p1) + p2) + p3 with the SFPU fp32 add that binary_ng uses for two fp32 tensors - and written by draft_fuse_out.cpp as the
(1, 1, rows, 5120) fp32 result: the tensor the caller would have got from the last add. The launch spreads its tiles (160 a tile row: 160 at
32 rows, 320 at 64) over min(tiles, grid, WORKER_CAP) cores of the grid the device reports. The typecast to bf16 that follows stays the
caller's own op (a fused typecast would be a second output and one more rounding point to prove; its 5.4 us is left in).

EXACTNESS. Bit-identical, by construction: the served path adds fp32 tiles with the SFPU, in this order, writing each sum to DRAM as fp32 and
reading it back (lossless), so a running sum held in the destination register is the same value; no op between rounds anything. The slices
are tile-aligned copies (rows 0..rows of each chip's block), so the tiles this kernel reads are the bytes the slices copy. CPU: the
launch plan is held against the slice/add composition with the kernel transliterated (test_draft_reduce_tp), with a negative control (a
different add order differs). Card: QWEN_FAST_DRAFT_REDUCE_AUDIT=1 runs the served slices and adds on the same gathered tensor beside the
launch and compares every chip's bytes (draft_fusion_tp's eager audit), then the draft-singles audit and the position-keyed prefix compare.

WHO CALLS IT. feature_collective_tp.gather_add_projection (the single, pair and feature-projection chains: 1, 8 and 32 rows) dispatches here
when the flag is on; draft_mlp_branch asks choose() for the MLP branch's chain at the quad's 64 rows (quad.gather_add_projection is the pinned
quad_draft function). The attention branch (draft_attention_branch.py, WP7's) needs the same one-line choose() at its gather; until it has it
that half of the chains stays served - the engaged line's chains=<n> counts what actually ran. A call this module cannot take (another shape,
dtype, layout or placement, a grid that cannot be read, a launch that raises) is a logged fall-back to the served function.

Stdlib only at import, py 3.7. The flag is strict 0 or 1 (draft_fusion_tp) and refused at the pair.
"""

from pathlib import Path

import draft_fusion_tp as fusion
import tp_shapes

IO_KERNEL = 'draft_reduce_tp_io.cpp'
COMPUTE_KERNEL = 'draft_reduce_tp_compute.cpp'
OUT_KERNEL = 'draft_fuse_out.cpp'
HIDDEN = 5120
COLUMN_TILES = HIDDEN // 32                 # 160 tiles across a hidden row
TILE_BYTES = 4096                           # one fp32 tile
INPUT_PAGES_PER_SLOT = 4                    # a chip's partial each, per output tile
SLOTS = 2                                   # CB 0 holds two sets so the next tile's reads overlap this tile's adds
OUTPUT_PAGES = 2
SERVED_ROWS = (1, 8, 32)                    # feature_collective's rows
QUAD_ROWS = 64                              # quad_draft.ROWS
MILESTONES = (1, 5, 10)                     # chains=<n> is logged when this many chains have run (one layer set, a whole quad pass)

RUNTIME_FILES = ('draft_reduce_tp.py', IO_KERNEL, COMPUTE_KERNEL, OUT_KERNEL, 'draft_fusion_tp.py')


def enabled(environ=None):
    return fusion.enabled(fusion.REDUCE, environ)


def tile_rows(rows):
    """Tile rows of a `rows`-row block (a row count below 32 pads to one tile row)."""
    return (rows + 31) // 32


def plan(rows, grid, chips=4):
    """The launch's geometry for a (1, 1, rows, 5120) block gathered over `chips`: tiles per chip block (the stride), the output tiles, the
    workers, and the runs [(first tile, count)]. Raises ValueError for a row count or a chip count the kernels do not take."""
    if type(rows) is not int or not 1 <= rows <= 64:
        raise ValueError('rows must be 1 to 64, got %r' % (rows,))
    if chips not in (2, 3, 4):
        raise ValueError('the fp32 destination holds %d chips\' tiles at most, got %r' % (4, chips))
    stride = tile_rows(rows) * COLUMN_TILES
    workers = fusion.worker_count(stride, grid)
    return dict(rows=rows, chips=chips, stride=stride, tiles=stride, workers=workers, runs=fusion.plan_runs(stride, workers))


def ineligible(operations, mesh, value, chips):
    """Why this call cannot take the launch (a short reason), or None."""
    shape = tuple(value.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != HIDDEN:
        return 'shape %r is not (1, 1, rows, %d)' % (shape, HIDDEN)
    if shape[2] not in SERVED_ROWS and shape[2] != QUAD_ROWS:
        return 'rows %r is not one of %s' % (shape[2], list(SERVED_ROWS) + [QUAD_ROWS])
    if value.dtype != operations.float32:
        return 'the projection is not float32'
    if getattr(value, 'layout', operations.TILE_LAYOUT) != operations.TILE_LAYOUT:
        return 'the projection is not TILE layout'
    try:
        if value.memory_config() != operations.DRAM_MEMORY_CONFIG:
            return 'the projection is not interleaved DRAM'
    except Exception:  # noqa: BLE001 - a diagnostic read
        return 'the projection placement is unreadable'
    if tp_shapes.mesh_width(mesh) != chips:
        return 'the mesh is not (1, %d)' % chips
    if fusion.grid_of(mesh) is None:
        return 'the compute grid cannot be read'
    return None


def served_sum(operations, gathered, rows, width, retain):
    """The served chain after the gather, on an already gathered tensor: one slice per chip, then the adds in chip order
    (feature_collective_tp.gather_add_projection / quad_draft.gather_add_projection, op for op). The audit's reference."""
    pieces = [retain(operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, rows, HIDDEN))) for chip in range(width)]
    total = retain(operations.add(pieces[0], pieces[1], dtype=operations.float32, memory_config=operations.DRAM_MEMORY_CONFIG))
    for chip in range(2, width):
        total = retain(operations.add(total, pieces[chip], dtype=operations.float32, memory_config=operations.DRAM_MEMORY_CONFIG))
    return total


def reduce_launch(operations, mesh, gathered, rows, grid, chips):
    """The one launch: gathered (chips, 1, rows, 5120) fp32 -> a new (1, 1, rows, 5120) fp32 tensor. Returns the output (the caller owns it)."""
    found = plan(rows, grid, chips)
    output = operations.empty((1, 1, rows, HIDDEN), dtype=operations.float32, layout=operations.TILE_LAYOUT, device=mesh,
                              memory_config=operations.DRAM_MEMORY_CONFIG)
    try:
        cores = fusion.core_ranges(operations, grid, found['workers'])
        coordinates = fusion.coordinates(grid, found['workers'])
        buffers = [fusion.circular_buffer(operations, cores, 0, INPUT_PAGES_PER_SLOT * SLOTS, operations.float32, TILE_BYTES),
                   fusion.circular_buffer(operations, cores, 16, OUTPUT_PAGES, operations.float32, TILE_BYTES)]
        config = fusion.fp32_compute_config(operations, (0,))
        in_shards = operations.get_device_tensors(gathered)
        out_shards = operations.get_device_tensors(output)
        if len(in_shards) != chips or len(out_shards) != chips:
            raise ValueError('one shard per chip required')
        program = operations.MeshProgramDescriptor()
        for chip in range(chips):
            local_in, local_out = in_shards[chip], out_shards[chip]
            if local_in.buffer_address() == local_out.buffer_address():
                raise ValueError('The reduce output must not alias the gathered tensor')
            reading, writing, computing = operations.RuntimeArgs(), operations.RuntimeArgs(), operations.RuntimeArgs()
            for (x, y), (first, count) in zip(coordinates, found['runs']):
                reading[x][y] = [local_in.buffer_address(), first, count, found['stride']]
                writing[x][y] = [local_out.buffer_address(), first, count]
                computing[x][y] = [count]
            reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(IO_KERNEL)), core_ranges=cores,
                compile_time_args=list(operations.TensorAccessorArgs(local_in).get_compile_time_args()) + [chips],
                runtime_args=reading, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            writer = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(OUT_KERNEL)), core_ranges=cores,
                compile_time_args=list(operations.TensorAccessorArgs(local_out).get_compile_time_args()) + [TILE_BYTES],
                runtime_args=writing, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default))
            compute = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(COMPUTE_KERNEL)), core_ranges=cores,
                compile_time_args=[chips], runtime_args=computing, config=config)
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, writer, compute], cbs=buffers)
        operations.generic_op([gathered, output], program)
    except BaseException:
        operations.deallocate(output)
        raise
    return output


def gather_add(operations, mesh, collectives, value, *, retain_temporaries=None, observe=None, served, site, quad=False):
    """feature_collective_tp.gather_add_projection / quad_draft.gather_add_projection with the slices and adds as one launch.

    `served` is the function this one stands in for (same arguments); `site` names the caller in the markers ('feature', 'mlp', 'attention');
    `quad` is True where the caller is a quad pass (a trace-owned 64-row block, whose served function requires a lifetime owner). The
    gather is the served call; the tensors it leaves, the order they are retained and freed, and the returned output are the served ones'."""
    if not enabled():
        return served(operations, mesh, collectives, value, retain_temporaries=retain_temporaries, observe=observe)
    shape = tuple(value.shape)
    rows = shape[2] if len(shape) == 4 else -1
    chips = tp_shapes.chip_count()
    reason = ineligible(operations, mesh, value, chips)
    if reason is None and quad and not callable(retain_temporaries):
        reason = 'a quad chain needs a trace lifetime owner'
    if reason is None and not quad and rows == QUAD_ROWS:
        reason = '64 rows outside a quad pass'
    if reason is not None:
        fusion.fell_back(fusion.REDUCE_FALLBACK, site, reason)
        return served(operations, mesh, collectives, value, retain_temporaries=retain_temporaries, observe=observe)
    grid = fusion.grid_of(mesh)
    from mesh_link_policy import fast_ccl_topology, projection_links

    links = projection_links()
    temporaries = []
    output = None

    def retain(tensor):
        temporaries.append(tensor)
        if retain_temporaries is not None:
            retain_temporaries(tensor)
        return tensor

    try:
        gathered = operations.experimental.all_gather_async(value,
            persistent_output_buffer=None, dim=0,
            multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=links,
            memory_config=operations.DRAM_MEMORY_CONFIG, topology=fast_ccl_topology(operations),
            chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        retain(gathered)
        if observe is not None:
            # QWEN_FAST_PROPOSAL_AUDIT (dflash_device.ProposalAudit): both partials as this chip received them, before the add.
            observe('gathered', gathered)
        try:
            output = reduce_launch(operations, mesh, gathered, rows, grid, chips)
        except Exception as failure:  # noqa: BLE001 - the served adds can still run on the gathered tensor
            fusion.fell_back(fusion.REDUCE_FALLBACK, site, 'the launch failed: %s: %s' % (type(failure).__name__, str(failure)[:120]))
            pieces = [retain(operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, rows, HIDDEN))) for chip in range(chips)]
            output = operations.add(pieces[0], pieces[1], dtype=operations.float32, memory_config=operations.DRAM_MEMORY_CONFIG)
            for chip in range(2, chips):
                partial = output
                output = operations.add(partial, pieces[chip], dtype=operations.float32, memory_config=operations.DRAM_MEMORY_CONFIG)
                retain(partial)
        else:
            if fusion.audit_enabled(fusion.REDUCE_AUDIT) and fusion.audit_begin(operations, mesh, fusion.REDUCE, site, rows):
                _audit(operations, mesh, gathered, output, rows, chips, site)
            fusion.STATS['reduce'] += 1
            if fusion.STATS['reduce'] in MILESTONES:
                fusion.note(fusion.REDUCE_ENGAGED, 'chains=%d site=%s rows=%d workers=%d tiles=%d' % (
                    fusion.STATS['reduce'], site, rows, plan(rows, grid, chips)['workers'], plan(rows, grid, chips)['tiles']))
        if retain_temporaries is None:
            operations.synchronize_device(mesh)
        return output
    except BaseException:
        if output is not None:
            operations.deallocate(output)
        raise
    finally:
        if retain_temporaries is None:
            for tensor in reversed(temporaries):
                operations.deallocate(tensor)


def _audit(operations, mesh, gathered, output, rows, chips, site):
    """The served slices and adds on the same gathered tensor, compared with the launch's output on every chip. Eager: only after
    draft_fusion_tp.audit_begin said yes (outside any capture, the device synchronized)."""
    held = []

    def keep(tensor):
        held.append(tensor)
        return tensor

    try:
        reference = served_sum(operations, gathered, rows, chips, keep)
        operations.synchronize_device(mesh)
        bad = fusion.differing_chips(operations, output, reference)
    finally:
        for tensor in reversed(held):
            operations.deallocate(tensor)
    fusion.audit_result(fusion.REDUCE, fusion.REDUCE_AUDIT_LINE, fusion.REDUCE_MISMATCH, site, rows, bad, fusion.audited_total(fusion.REDUCE))


def choose(served, quad, site):
    """The gather-add function a branch should call: `served` itself (the identical function object) with the lever off or outside a quad
    pass, else a function with its signature that runs the launch and falls back to `served`. `quad` is the branch's quad pass or None. A
    branch without a quad pass already calls feature_collective_tp.gather_add_projection (the tp_addresses rebinding of the pinned name), which
    dispatches on the flag itself, so only the quad's 64-row trace-owned chains (quad.gather_add_projection, the pinned quad_draft function)
    need this. The one-line hook of draft_mlp_branch and, for the attention branch, of WP7's file."""
    if quad is None or not enabled():
        return served

    def chain(operations, mesh, collectives, value, *, retain_temporaries=None, observe=None):
        return gather_add(operations, mesh, collectives, value, retain_temporaries=retain_temporaries, observe=observe,
                          served=served, site=site, quad=True)
    return chain
