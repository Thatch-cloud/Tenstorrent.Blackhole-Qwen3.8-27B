"""WP4 read-bandwidth probe (host side): how fast can the cores pull an interleaved bfloat4_b / bfloat8_b weight out of DRAM when only the request pattern changes?

WHY. The card-M sweep (C1) found the MLP gate flat at about 240 GB/s from 34 to 68 cores and the up (same bytes, no SiLU) at 284, against 381 for the bfloat8_b down, and a fused
gate|up launch of this branch (which reads tile by tile like the stock reader) at 241: three different kernels, one plateau, so it is not per-core compute or the SiLU alone.
The hypothesis this probe tests: a DRAM request has a fixed cost that a 576-byte bfloat4_b tile page does not amortize and a 1,088-byte bfloat8_b page half does, and the stock
1D-mcast reader issues one request per tile page. Fitting the sweep's up (576 B pages, 2.03 ns per tile) and down (1,088 B, 2.85 ns) rows gives about 1.1 ns of fixed cost per
request and 0.0016 ns per byte, which would make a 3-tile request about 36 percent faster per tile and a 4-tile one 41 percent. If that holds, the lever is a weight reader that
issues bank-contiguous multi-tile requests (below), not a different matmul.

THE LAYOUT FACT IT USES. An interleaved DRAM tensor puts page p in bank p % banks at bank offset (p / banks) * page_bytes. With the tile columns a multiple of the bank count
(4,352 / 32 = 136 and 5,120 / 32 = 160 are multiples of 8), the tiles (k, c), (k, c + banks), (k, c + 2 banks) ... of one tile row are CONSECUTIVE PAGES OF ONE BANK, adjacent in
that bank's memory: one NoC read of n pages from the address of page (k, c) returns all n. The stock reader hands a core `per_core_N` neighbouring columns (neighbouring pages are
in different banks, so one request per tile); a reader that hands a core the columns c, c + banks, ... of one bank can read them n at a time. No weight is copied or permuted.

THE PROBE. One data-movement kernel (readprobe_reader.cpp), no compute. A worker owns `run` tiles of every tile row and issues them as `ceil(run / chunk)` requests of up to `chunk`
tiles; mode 'bank' gives it bank-strided columns (chunk > 1 allowed), mode 'stock' gives it neighbouring columns read one tile per request (the stock pattern, as the baseline).
Timing runs read only (no writes); a correctness run copies every tile it read to the same page of a second tensor and the host compares the two tensors, so the layout fact
above is checked on the card, not assumed. Everything is `generic_op` on a 1x1 mesh; nothing here changes a served path.

Stdlib only at import; ttnn is the `operations` handle.
"""

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = 'readprobe_reader.cpp'
BLOCK_ROWS = 8
MAX_RUN = 8
PAGE_BYTES = {'bfp4': 576, 'bfp8': 1088}
# the (run, chunk) pairs the probe times by default: run = tiles per tile row per worker, chunk = tiles per request
BANK_POINTS = ((2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 2), (4, 4), (6, 1), (6, 3), (6, 6))
STOCK_POINTS = ((2, 1), (3, 1), (4, 1))


def worker_plan(mode, run, chunk, tile_rows, tile_columns, banks, grid, width=None):
    """The probe's workers: [{'core': (x, y), 'requests': [(column, tiles)], 'tiles': n}] covering every tile of every row exactly once.

    'bank': worker (bank b, group g) owns the columns b + banks * j for j in [g * run, (g + 1) * run) that exist, read in requests of up to `chunk` consecutive j (one
    contiguous read each); 'stock': worker w owns the neighbouring columns [w * run, (w + 1) * run), one tile per request. Cores fill a rectangle `width` wide row-major."""
    if mode not in ('bank', 'stock'):
        raise ValueError('mode is bank or stock, got %r' % (mode,))
    if not (1 <= chunk <= run <= MAX_RUN):
        raise ValueError('need 1 <= chunk <= run <= %d, got run %r chunk %r' % (MAX_RUN, run, chunk))
    if mode == 'stock' and chunk != 1:
        raise ValueError('the stock pattern is one tile per request')
    if tile_rows % BLOCK_ROWS:
        raise ValueError('%d tile rows are not whole blocks of %d' % (tile_rows, BLOCK_ROWS))
    entries = []
    if mode == 'stock':
        for first in range(0, tile_columns, run):
            columns = list(range(first, min(first + run, tile_columns)))
            entries.append([(column, 1) for column in columns])
    else:
        for bank in range(banks):
            owned = [bank + banks * j for j in range(tile_columns) if bank + banks * j < tile_columns]
            for first in range(0, len(owned), run):
                group = owned[first:first + run]
                entries.append([(group[i], min(chunk, len(group) - i)) for i in range(0, len(group), chunk)])
    grid_x, grid_y = int(grid[0]), int(grid[1])
    width = grid_x if width is None else int(width)
    if not 1 <= width <= grid_x:
        raise ValueError('width %d is outside the device grid width %d' % (width, grid_x))
    cols = min(width, len(entries))
    rows = int(math.ceil(len(entries) / float(cols)))
    if rows > grid_y:
        raise ValueError('%d workers %d wide need %d rows; the device has %d' % (len(entries), cols, rows, grid_y))
    return [dict(core=(index % cols, index // cols), requests=requests, tiles=sum(count for unused, count in requests))
            for index, requests in enumerate(entries)]


def covered_pages(plan, tile_rows, tile_columns, banks):
    """Every page id the plan reads, with multiplicity, in the order the kernel issues them (the host check that each page is read exactly once)."""
    pages = []
    for worker in plan:
        for k in range(tile_rows):
            for column, tiles in worker['requests']:
                pages.extend(k * tile_columns + column + banks * m for m in range(tiles))
    return pages


def request_bytes(plan, page_bytes):
    """(the largest, the mean) request in bytes, and the request count per tile row summed over workers."""
    sizes = [tiles * page_bytes for worker in plan for unused, tiles in worker['requests']]
    return max(sizes), sum(sizes) / float(len(sizes)), len(sizes)


def landing_bytes(run, page_bytes):
    return BLOCK_ROWS * run * page_bytes


class ReadProbe(object):
    """One launch: read all of `source` (an interleaved TILE tensor, bfloat4_b or bfloat8_b, tile columns a multiple of the bank count) with `plan`, optionally copying every tile
    to the same page of `destination` (write_back). __call__() launches; nothing is returned (the timing arm frees nothing)."""

    def __init__(self, operations, mesh, source, destination, dtype, tile_rows, tile_columns, banks, mode, run, chunk, grid, write_back=False, width=None):
        if dtype not in PAGE_BYTES:
            raise ValueError('dtype is bfp4 or bfp8, got %r' % (dtype,))
        if tile_columns % banks:
            raise ValueError('%d tile columns are not a multiple of the %d banks: the tiles of one bank are not a regular stride' % (tile_columns, banks))
        self.operations, self.mesh, self.source, self.destination = operations, mesh, source, destination
        self.dtype, self.page_bytes = dtype, PAGE_BYTES[dtype]
        self.tile_rows, self.tile_columns, self.banks = tile_rows, tile_columns, banks
        self.mode, self.run, self.chunk, self.write_back = mode, run, chunk, bool(write_back)
        self.plan = worker_plan(mode, run, chunk, tile_rows, tile_columns, banks, grid, width)
        self.largest, self.mean, self.requests_per_row = request_bytes(self.plan, self.page_bytes)
        self.calls = 0

    def descriptor(self, local_source, local_destination, chip=0):
        ttnn = self.operations
        cores = [worker['core'] for worker in self.plan]
        ranges = []
        for x, y in cores:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
        workers = ttnn.CoreRangeSet(ranges)
        size = int(math.ceil(landing_bytes(self.run, self.page_bytes) / 2048.0)) * 2048
        landing = ttnn.CBDescriptor(total_size=size, core_ranges=workers,
                                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048,
                                                                                tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(HERE / KERNEL), core_ranges=workers,
            compile_time_args=(ttnn.TensorAccessorArgs(local_source).get_compile_time_args()
                               + ttnn.TensorAccessorArgs(local_destination).get_compile_time_args()),
            named_compile_time_args=[('rows', self.tile_rows), ('row_pages', self.tile_columns), ('page_bytes', self.page_bytes), ('block_rows', BLOCK_ROWS),
                                     ('run', self.run), ('banks', self.banks), ('write_back', int(self.write_back))],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
        arguments = ttnn.RuntimeArgs()
        widest = max(len(worker['requests']) for worker in self.plan)
        for worker in self.plan:
            flat = []
            for column, tiles in worker['requests']:
                flat += [column, tiles]
            flat += [0, 0] * (widest - len(worker['requests']))           # constant argument length on every core; the kernel reads only `requests` pairs
            arguments[worker['core'][0]][worker['core'][1]] = [local_source.buffer_address(), local_destination.buffer_address(), len(worker['requests'])] + flat
        kernel.runtime_args = arguments
        return ttnn.ProgramDescriptor(kernels=[kernel], cbs=[landing], semaphores=[])

    def __call__(self):
        ttnn = self.operations
        program = ttnn.MeshProgramDescriptor()
        for chip, (local_source, local_destination) in enumerate(zip(ttnn.get_device_tensors(self.source), ttnn.get_device_tensors(self.destination))):
            coordinate = ttnn.MeshCoordinate(0, chip)
            program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = self.descriptor(local_source, local_destination, chip)
        ttnn.generic_op([self.source, self.destination], program)
        self.calls += 1
