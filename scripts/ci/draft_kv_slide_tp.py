"""The drafter's K/V sliding-history transport at any served width (the four-card twin of draft_kv_slide.prepare).

WHY. DraftKVHistory.prepare publishes one accepted prefix into the spare bank with a six-op eager chain
(slice, slice, concat, slice, pad, copy) per layer and per k / v. While the history is still filling (history_rows < 2048:
every prompt under 2,048 tokens, for its whole answer) history_rows grows by the accepted prefix each round, so three or
four of those ops meet a shape they have never seen and build a program each: measured on the first four-card run (v140)
at about 310 ms per ramp user per round, 85 s of a 145 s packed phase. At the pair the same commit is one generic_op per
(layer, k / v) whose history_rows, prefix, drop and rows are RUNTIME arguments (draft_kv_slide.cpp), so no binary is
built per round: 5.1 ms per ramp commit measured on the pair.

WHAT. The same kernel, driven from a host transport that reads the served width instead of the pair's literals:
  - chips: tp_shapes.chip_count() shards (the pair's transport hard-codes two, and range(2) in its program loop);
  - K/V heads per chip: tp_shapes.active().draft_kv_heads (2 at four cards; the banks are (1, 2, 2048, 128) and the delta
    (1, 2, 32, 128)), where the pair's are (1, 4, ...);
  - workers: one per (head, column tile) = 4 per head (8 at four cards, 16 at the pair). The kernel takes head = worker / 4
    and column = worker % 4 and addresses page (head * 64 + tile) * 4 + column of the 64 x 4 tile grid, so it is generic in the
    head count: draft_kv_slide.cpp is used AS IT IS, never a four-card copy. The image carries the bundle's (the qualified
    direct-DMA kernel, sha256 1679bbd7; nothing copies the checkout's scalar one, bc45d472) and both are generic in the head count;
    test_draft_kv_slide_tp holds the checkout's file to one of the two qualified digests.
draft_kv_slide.py stays what the frozen bundle carries (test_tp2_pins); this module imports nothing from it (the pair's
module is off at four cards and the closure test keeps it out of the served set), so the small geometry below is its own and
test_draft_kv_slide_tp holds it equal to the pair's.

SELECTION. draft_kv_history_tp.DraftKVHistory.prepare calls prepare() here under QWEN_FAST_TP_KV_SLIDE=1 and keeps the
eager chain as the flag-off reference (the exactness tests compare the two, and the hardware A/B does). Nothing at the pair
imports this module. Not qualified on hardware: the kernel has run at four KV heads only (the pair's one-card harness); the
first run at two heads is the speed window of scripts/ci/references/tp4-speed-jobs (K0-E4; a quad job cannot run the one-card
cardm step). Its exactness is judged on the drafter's PROPOSALS, not the texts: speed_window_compare.py compares the solo
accepted-prefix sequences of two arms.
"""

from pathlib import Path

import tp_shapes

FLAG = 'QWEN_FAST_TP_KV_SLIDE'
HISTORY_CAPACITY = 2048
DELTA_ROWS = 32
COLUMN_TILES = 4
HEAD_DIM = 128
KERNEL = Path(__file__).with_name('draft_kv_slide.cpp')


def enabled(environ=None):
    """QWEN_FAST_TP_KV_SLIDE=1 (unset or anything else is the eager chain). Read at every call, like the image's flags."""
    import os

    return (os.environ if environ is None else environ).get(FLAG) == '1'


def geometry(history_rows, prefix):
    """The pair's draft_kv_slide.geometry: rows kept, and rows dropped from the front once the window is full."""
    if (type(history_rows) is not int or not 1 <= history_rows <= HISTORY_CAPACITY
            or type(prefix) is not int or not 1 <= prefix <= DELTA_ROWS):
        raise ValueError('One committed history and bounded accepted prefix required')
    rows = min(HISTORY_CAPACITY, history_rows + prefix)
    return dict(history_rows=history_rows, prefix=prefix, rows=rows, drop=history_rows + prefix - rows)


def bank_shape():
    return (1, tp_shapes.active().draft_kv_heads, HISTORY_CAPACITY, HEAD_DIM)


def delta_shape():
    return (1, tp_shapes.active().draft_kv_heads, DELTA_ROWS, HEAD_DIM)


def worker_count():
    """Head/column transport workers per chip: four column tiles for each of the width's KV heads."""
    return tp_shapes.active().draft_kv_heads * COLUMN_TILES


def prepare(mesh, active, delta, spare, *, history_rows, prefix):
    """The one-generic_op sliding publication of `active` + `delta` into `spare` for this (history_rows, prefix): returns
    the callable that runs it (draft_kv_slide.prepare's contract, so the caller and the tests treat both alike)."""
    import ttnn

    shape = geometry(history_rows, prefix)
    chips = tp_shapes.chip_count()
    tensors = [active, delta, spare]
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != chips for parts in shards):
        raise ValueError('%s independent device shards required' % tp_shapes.count_word().lower())
    for parts, expected in zip(shards, (bank_shape(), delta_shape(), bank_shape())):
        for value in parts:
            if (tuple(value.shape) != expected or value.dtype != ttnn.bfloat16
                    or value.layout != ttnn.TILE_LAYOUT or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or tuple(value.tile.tile_shape) != (32, 32)
                    or value.tile.transpose_of_faces or value.tile.transpose_within_face):
                raise ValueError('Exact non-transposed interleaved BF16 cache tiles at %s required' % (expected,))
    workers = worker_count()
    grid = mesh.compute_with_storage_grid_size()
    if grid.x * grid.y < workers:
        raise ValueError('%d head/column transport workers required' % workers)
    coordinates = [ttnn.CoreCoord(worker % grid.x, worker // grid.x) for worker in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffer = ttnn.CBDescriptor(total_size=8192, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 3:
            raise ValueError('Active, delta and spare storage must not alias')
        kernel = ttnn.KernelDescriptor(kernel_source=str(KERNEL), core_ranges=cores,
            compile_time_args=[argument for value in local
                for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker, core in enumerate(coordinates):
            runtime[core.x][core.y] = addresses + [history_rows, prefix, shape['drop'], shape['rows'], worker]
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    return lambda: ttnn.generic_op(tensors, program)
