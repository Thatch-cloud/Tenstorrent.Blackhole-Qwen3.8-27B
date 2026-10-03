"""The drafter's fused convolution with a rewritten I/O stage at four cards (QWEN_FAST_TP4_DRAFT_CONV, default off; D2a).

WHY. The device profile of the production recipe shows the drafter's fused convolution at 20 launches of 177 us per pair pass
(3.54 ms of a 12.38 ms pass; about 265 us each at the quad's 64 rows). The cost is not the 80-core grid (160 pages over 80 workers
is two pages each, and 110 workers still leaves two or three) but the I/O kernel's per-element scalar loop and its wait for every
page's output before it reads the next page (draft_convolution_fused_io.cpp, quad_conv_io.cpp).

WHAT. draft_conv_io_fast.cpp builds the same seven-tile input image per page with 32-bit word copies and fills, into a CB 0 that
holds two seven-tile slots; draft_conv_out.cpp writes each result tile from the other data-movement core; the compute kernel is the
served draft_convolution_fused_compute.cpp, unchanged, with the same per-core page counts. Pages, workers, cores and runtime
semantics are the served ones (pair: 80 workers x 2 pages on the 8 x 10 grid; quad: E1b's 110 workers x 3 pages or E1's 80 x 4).
The compute kernel receives byte-identical tiles for every page, so the output and the drafter's proposals are byte-identical:
tau cannot change. test_tp4_draft_conv transliterates the served loop and the word-copy image and holds them byte-equal.

FLAG OFF. tp4_draft_convolution twins call the served function (draft_convolution_fused_tp.served_fused_convolution,
quad_draft_tp.served_quad_fused_convolution) exactly as before; this module is not imported. A call this module cannot take
(another dtype, layout, mesh, rows) is handed to the served function with a 'fell back' line, which raises its own refusal if the
call is truly invalid.

AUDIT (QWEN_FAST_TP4_DRAFT_CONV_AUDIT=1, needs the lever). Each engaged call also runs the served call on the same operands, and the
engaged output is cloned; both are held, outside the owners' lifetimes, as a pair. compare_pending() byte-compares every held pair on
every chip (int16 bit patterns: -0 and +0 differ) and logs '[PINDIAG] tp4 draft conv audit exact=True convs=N'; any difference
logs the mismatch marker and raises. packed_verifier calls it at the top of each round's readback (everything the drafter traces
enqueued has completed by then) for the first AUDIT_ROUNDS rounds. release_audit() frees what the audit holds.

Stdlib only at import, py 3.7.
"""

from pathlib import Path

import tp4_sampdraft
import tp_shapes

IO_KERNEL = 'draft_conv_io_fast.cpp'
OUT_KERNEL = 'draft_conv_out.cpp'
COMPUTE_KERNEL = 'draft_convolution_fused_compute.cpp'
TILE_PAGES = 160          # pages per tile row of the 5,120-wide hidden block
TILE_BYTES = 2048
SLOT_TILES = 7            # one page's input image: hidden, shift, base0, dynamic0, base1, dynamic1, zero
SLOTS = 2                 # CB 0 holds two images so the next page is prepared while the compute kernel runs this one
AUDIT_ROUNDS = 4          # compare_pending compares on this many calls, then stops reading back

# Held (engaged output clone, served output) pairs of the audit, and how many compare_pending calls ran.
_HELD = []
_STATE = {'rounds': 0, 'convs': 0}


def pages_for(workers, rows):
    """{worker: [pages]}: page = worker, worker + workers, ... below 160 per tile row (the served plan, quad_draft.conv_pages)."""
    total = TILE_PAGES * ((rows + 31) // 32)
    return {worker: list(range(worker, total, workers)) for worker in range(workers)}


def group_workers(pages):
    """{pages per worker: [workers]}: the compute kernel is one compile argument per group of workers taking the same count."""
    groups = {}
    for worker, owned in pages.items():
        groups.setdefault(len(owned), []).append(worker)
    return groups


def ineligible(operations, mesh, tensors, rows):
    """Why this call cannot take the fast path (a short reason), or None."""
    chips = tp_shapes.chip_count()
    if type(rows) is not int or not 1 <= rows <= 64:
        return 'rows %r is not 1 to 64' % (rows,)
    try:
        if list(mesh.shape) != [1, chips]:
            return 'mesh %r is not (1, %d)' % (list(mesh.shape), chips)
    except Exception:  # noqa: BLE001 - a diagnostic read; the served call decides
        return 'mesh shape unreadable'
    for tensor in tensors:
        if (tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT
                or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG):
            return 'an operand is not interleaved DRAM bfloat16 tiled'
    return None


def convolution(operations, mesh, hidden, dynamic, base, *, rows, seams_low, seams_high, workers, coordinates, core_set,
                served, label):
    """The fused convolution on the fast I/O kernels, or the served call when this call cannot take them.

    `coordinates[w]` is worker w's (x, y); `core_set(coordinates)` the CoreRangeSet of a list of coordinates; `served()` the served
    call (no arguments) returning its output; `label` names the site in the marker ('pair' or 'quad')."""
    tensors = [hidden, *dynamic, *base]
    reason = ineligible(operations, mesh, tensors, rows)
    if reason is not None:
        tp4_sampdraft.note(tp4_sampdraft.CONV_FALLBACK, 'site=%s reason=%s' % (label, reason))
        return served()
    chips = tp_shapes.chip_count()
    parts = [operations.get_device_tensors(value) for value in tensors]
    if any(len(shards) != chips for shards in parts):
        tp4_sampdraft.note(tp4_sampdraft.CONV_FALLBACK, 'site=%s reason=operand shards are not one per chip' % label)
        return served()
    pages = pages_for(workers, rows)
    groups = group_workers(pages)
    output = operations.empty(tuple(hidden.shape), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                              device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    audit = tp4_sampdraft.audit_enabled(tp4_sampdraft.DRAFT_CONV_AUDIT)
    parts.append(operations.get_device_tensors(output))
    cores = core_set(coordinates)
    buffers = [operations.CBDescriptor(total_size=TILE_BYTES * count, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
            page_size=TILE_BYTES, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
        for index, count in ((0, SLOT_TILES * SLOTS), (1, 2), (16, 1))]
    computes = [operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(COMPUTE_KERNEL)),
        core_ranges=core_set([coordinates[worker] for worker in group]), compile_time_args=[count],
        config=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                  math_approx_mode=False))
        for count, group in sorted(groups.items())]
    program = operations.MeshProgramDescriptor()
    try:
        for chip in range(chips):
            local = [shards[chip] for shards in parts]
            if local[-1].buffer_address() in {value.buffer_address() for value in local[:-1]}:
                raise ValueError('Convolution output must not alias borrowed inputs')
            reading, writing = operations.RuntimeArgs(), operations.RuntimeArgs()
            for worker, (x, y) in enumerate(coordinates):
                reading[x][y] = [value.buffer_address() for value in local[:-1]] + [rows, worker, workers, seams_low, seams_high]
                writing[x][y] = [local[-1].buffer_address(), rows, worker, workers]
            reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(IO_KERNEL)), core_ranges=cores,
                compile_time_args=[argument for value in local[:-1]
                                   for argument in operations.TensorAccessorArgs(value).get_compile_time_args()],
                runtime_args=reading, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            writer = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(OUT_KERNEL)), core_ranges=cores,
                compile_time_args=list(operations.TensorAccessorArgs(local[-1]).get_compile_time_args()),
                runtime_args=writing, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default))
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, writer, *computes], cbs=buffers)
        operations.generic_op([*tensors, output], program)
    except BaseException:
        operations.deallocate(output)
        raise
    tp4_sampdraft.note(tp4_sampdraft.CONV_ENGAGED, 'site=%s rows=%d pages=%d workers=%d' % (
        label, rows, TILE_PAGES * ((rows + 31) // 32), workers))
    if audit:
        reference = served()
        try:
            _HELD.append((operations.clone(output, memory_config=operations.DRAM_MEMORY_CONFIG), reference))
        except BaseException:
            operations.deallocate(reference)
            raise
    return output


def compare_pending(operations):
    """Byte-compare every held (engaged, served) pair on every chip, for the first AUDIT_ROUNDS calls. Returns the number of
    pairs compared (0 when nothing is held or the rounds are spent). Logs the audit marker; a difference logs the mismatch
    marker and raises AssertionError."""
    if not _HELD or _STATE['rounds'] >= AUDIT_ROUNDS:
        return 0
    import torch

    _STATE['rounds'] += 1
    bad = []
    for index, (mine, served) in enumerate(_HELD):
        for chip, (left, right) in enumerate(zip(operations.get_device_tensors(mine), operations.get_device_tensors(served))):
            a = operations.to_torch(left).contiguous().view(torch.int16)
            b = operations.to_torch(right).contiguous().view(torch.int16)
            if a.shape != b.shape or not torch.equal(a, b):
                bad.append((index, chip))
    count = len(_HELD)
    _STATE['convs'] += count
    if bad:
        message = '%s round=%d pairs=%d differing=%s' % (tp4_sampdraft.CONV_MISMATCH, _STATE['rounds'], count, bad[:8])
        tp4_sampdraft.log_line(message)
        raise AssertionError(message)
    tp4_sampdraft.log_line('%s exact=True convs=%d round=%d' % (tp4_sampdraft.CONV_AUDIT, count, _STATE['rounds']))
    return count


def release_audit(operations):
    """Free every tensor the audit holds."""
    held = [tensor for pair in _HELD for tensor in pair]
    del _HELD[:]
    for tensor in held:
        try:
            operations.deallocate(tensor)
        except BaseException:
            pass
