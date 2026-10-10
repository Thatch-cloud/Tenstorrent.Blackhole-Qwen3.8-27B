"""WP4 F-D3: the fused gate|up launch with a BANK-STRIDED multi-tile weight reader (QWEN_FAST_MLP_GATEUP_BANK=1).

WHY. The card-M read probe (C5: optimisation/ttnn-op/mlp_gateup/readprobe*) found the bfloat4_b gate shape reads at 402.7 GB/s (78.7 percent of the 512 peak) when each worker issues
ONE bank-contiguous request for `chunk` = 4 tiles, against 293.6 GB/s for the stock pattern (one 576-byte request per tile page): ratio 1.37, every probe arm bit-exact by copy-back.
The served gate streams at 240 GB/s, the up at 284, and the first fused op (tp4_mlp_fused, exact but slower) reads tile by tile like the stock reader. The DRAM request has a fixed cost that a
576-byte page does not amortize; this launch changes only the request pattern.

THE LAYOUT FACT. Interleaved page p lives in bank p % banks at offset (p / banks) * page_bytes. With 136 tile columns (a multiple of the 8 banks) the tiles (k, c), (k, c + 8), (k, c + 16) ...
of one tile row are adjacent pages of one bank: a worker that owns the columns first, first + 8, ... (`chunk` of them) reads all of one row's tiles in one request. Worker (bank b, group g)
owns the columns b + 8 * j for j in [g * chunk, (g + 1) * chunk) that exist (17 per bank, so 5 groups of 4 with the last one column wide: 40 workers at chunk 4, 48 at 3, 72 at 2).
Nothing is copied or permuted: the served w1 and w3 tensors are read in place.

WHICH OP, AND WHY. Two designs were possible:
  (i)  the fused launch of tp4_mlp_fused with a different weight layout and a paired epilogue, on the native compute kernel;
  (ii) a reader-only replacement under the STOCK matmul (the stock compute kernel and program, only the in1 reader replaced).
(ii) is refused. In the pinned tree (the tt-metal build the image pins) the stock 1D-mcast in1 reader and the OUTPUT WRITER are ONE kernel, matmul/device/kernels/dataflow/
reader_bmm_tile_layout_in1_sender_writer_padding.cpp: its runtime arguments (lines 22 to 48) carry the weight address and start tile and the output address and start tile together,
and it reads the weight block one tile per NoC read with a compile-time tile stride between them (lines 410 to 436), so a stride gives it strided columns but not bigger requests. A
reader-only replacement therefore does not exist: the writer goes with it, the contiguous per_core_N column range the factory hands every core (and the output addressing the stock compute
and writer share) has to change with it, and the replacement must be fed through the factory's own argument lists, which only the program factory builds. The weight block layout itself
would permit it (a bank run of `chunk` tiles lands exactly as a [K tile][chunk columns] block), but it would still be two launches and the 6 us multiply. (i) keeps the proven native K loop and
changes only data movement and the epilogue pairing: the activation kernel is the first fused op's unchanged, the weights kernel (tp4_mlp_bank_weights.cpp) only decides which pages land in
which circular-buffer slot, and the compute kernel's two edits are the first fused op's two edits (the pack of the last K block and the product loop after the K loop), re-pointed at two
input subblocks.

EXACT BY CONSTRUCTION (the arguments; a card still has to say so, and the card-M arm and the audit compare every bit):
  * the K schedule of every output tile is the native kernel's loop over 20 blocks of in0_block_w = 8 (compile argument 0, unchanged), partials in fp32 with packer L1 accumulation and the
    last block reloaded (lines 337 to 349 and 444 to 472 of the pinned build's bmm_large_block_zm_fused_bias_activation.cpp), exactly the served matmuls' schedule. What a subblock width
    changes is only how many output tiles share one pass of the matmul_block call (lines 358 to 376: each of the ct_dim output tiles accumulates its own column of in1 against the same in0
    row), never the order in which one tile accumulates; the served gate (subblock 1x2), up (1x3) and the sweep's other partitions (to 1x4) are bit-equal to each other for exactly this reason
    (the T1 #11 rule, C1: every candidate exact);
  * the compute kernel's in1 indexing (line 439: in1_index_subblock_offset += out_subblock_w per subblock, line 374: += in1_block_w per K tile) reads the block as [K tile][subblock 0
    columns][subblock 1 columns]: subblock 0 = the gate run, subblock 1 = the up run, each `chunk` tiles, and the same column index in the two is the same output column.
    out_subblock_w = chunk <= 4 is the fp32 destination capacity of a half-synced DST;
  * the epilogue is the first fused op's, which the C3 run found bit-equal to the served gate, up and multiply: SiLU applied at the pack of the gate tiles only (native line 385; the anchor
    `if (last_out) {` at line 380 and the `} else {` at line 415 bound the text the patch replaces; apply_activation_from_pack(n) applies the activation to dst tiles 0 to n-1, header lines 108 to 122),
    both operands packed as bf16 to an intermediate circular buffer, then a bf16 SFPU product of gate tile i with up tile i (BF16_PRODUCT, the first op's) in the order the output writer expects;
  * the padding of a worker with fewer than `chunk` valid columns is zero tiles (zeroed once before the K loop), which multiply to zero and are never written.

WHAT C6 MEASURED, AND THE THREE KNOBS (depth, sub, ablate). The probe's read-only rates at the points the op uses (run = chunk) are 292 (2), 331 (3) and 403 (4) GB/s a tensor, i.e. 85.9, 75.8 and
62.3 us for the two tensors; the fused op took 97.8, 94.8 and 109.0 us (C6). The excess over the read-only time is 11.9, 19.0 and 46.8 us: the exposed epilogue is about 5.6, 8.4 and 11.2 us (the
SiLU is 1.1 us a gate tile, 2 chunk tiles a worker, serialized after the last K block), so c2 and c3 sit within 6 to 11 us of read floor + epilogue, and c4 is about 35 us off it. The op has
three stages per K block that must all keep up (the weight reads, the lock-stepped single-sender activation multicast, the compute kernel), each with a 2-block buffer, and the time does not
follow the read rate (the stock-reader fused op p4a took 112.8 us at 85 us of read-only, this one 109.0 at 62). Two exact knobs address that, and two timing-only cuts attribute it:
  depth   the activation and weight circular buffers hold 2 (default), 3 or 4 K blocks. The reader (and so the DRAM queue) and the activation multicast run further ahead of the compute kernel,
          which decouples the lock-stepped stages; nothing a kernel does per block changes, only how many blocks may be in flight. The activation handshake is safe at any depth: a receiver
          raises `ready` for block b + 1 only after it has received block b, and the sender zeroes `ready` before it multicasts block b.
  sub     the output subblock is `sub` tiles wide (a divisor of the chunk) with 2 chunk / sub input subblocks: the request size and the DST granularity are decoupled. The circular-buffer
          layout is unchanged; the SiLU applies to the first chunk / sub subblocks. Exact for the reason chunk is: a subblock width never changes how one tile accumulates.
  ablate  skip_compute (the native SKIP_COMPUTE define: the matmul math is left out, every copy, pack, SiLU, product and write stays) and no_silu: timing only, reachable from the card-M
          harness's `bankopt` arm and from no flag of the lever, never exact, never part of a verdict.

Stdlib only at import (py 3.7); ttnn is the `operations` handle.
"""

import hashlib
import math
from collections import namedtuple
from pathlib import Path

import tp4_mlp_fused as fused
import tp4_mlp_gateup as lever

HERE = Path(__file__).resolve().parent
KERNELS = (fused.KERNELS[0], 'tp4_mlp_bank_weights.cpp')       # the activation multicast is the first fused op's, unchanged

TILE = fused.TILE
BLOCK_TILES = fused.BLOCK_TILES
K_TILES = fused.K_TILES
PAIR_COLUMNS = fused.PAIR_COLUMNS
ROW_TILES = fused.ROW_TILES
L1_BUDGET = fused.L1_BUDGET
BANKS = 8                       # the p150a's DRAM channels (the device is asked; this is the default when it cannot say)
CHUNKS = lever.BANK_CHUNKS      # tiles per request = columns per worker = the output subblock width (at most the fp32 destination capacity, 4)
DEFAULT_CHUNK = lever.DEFAULT_BANK_CHUNK    # the probe's best point (402.7 GB/s) at 4 tiles a worker; (3, 3) read 331 and (2, 2) 292
DEPTHS = lever.BANK_DEPTHS      # K blocks the activation and weight circular buffers hold (2: the reader fills one block while the compute kernel consumes another)
DEFAULT_DEPTH = lever.DEFAULT_BANK_DEPTH
ABLATIONS = ('skip_compute', 'no_silu')     # timing-only variants of the card-M tuning arm: never reachable from the lever's flags, never exact

Worker = namedtuple('Worker', ('x', 'y', 'bank', 'first', 'valid'))


def dram_banks(mesh):
    """The DRAM bank count of the device behind `mesh` (dram_grid_size().x), BANKS when it cannot say."""
    try:
        return int(mesh.dram_grid_size().x)
    except (AttributeError, TypeError, ValueError):
        return BANKS


def check_chunk(chunk):
    if chunk not in CHUNKS:
        raise ValueError('chunk must be one of %s (tiles per request, at most the fp32 destination capacity), got %r' % (', '.join(map(str, CHUNKS)), chunk))


def check_depth(depth):
    if depth not in DEPTHS:
        raise ValueError('depth must be one of %s (K blocks the circular buffers hold), got %r' % (', '.join(map(str, DEPTHS)), depth))


def resolve_sub(chunk, sub):
    """The output subblock width: `chunk` unless a divisor of it is named (the DST holds four fp32 tiles, so the width is at most 4 either way)."""
    sub = chunk if sub is None else sub
    if type(sub) is not int or not 1 <= sub <= chunk or chunk % sub:
        raise ValueError('sub must be a divisor of chunk %d (the output subblock width), got %r' % (chunk, sub))
    return sub


def check_ablate(ablate):
    if ablate is not None and ablate not in ABLATIONS:
        raise ValueError('ablate must be None or one of %s, got %r' % (', '.join(ABLATIONS), ablate))


def bank_epilogue(chunk, sub=None, ablate=None):
    """The replacement for the native kernel's last-K-block pack: the gate subblocks (the first chunk / sub of the 2 chunk / sub input subblocks) get the SiLU on all their tiles, the up
    subblocks none; every subblock is packed as bf16 to circular buffer 30. With sub = chunk (the default) there are two subblocks and the text is the one the card-M job C6 ran. The
    static_asserts fail the JIT compile (never the result) if the compile arguments and this text ever disagree. `ablate` = no_silu (timing only, never exact) leaves the SiLU out."""
    sub = resolve_sub(chunk, sub)
    check_ablate(ablate)
    if ablate == 'no_silu':
        activation = """                                tile_regs_wait();
"""
    elif sub == chunk:
        activation = """                                if (in1_subblock == 0) {
                                    apply_activation_from_pack<KernelActivation::SILU>(%(s)d);
                                } else {
                                    tile_regs_wait();
                                }
""" % dict(s=sub)
    else:
        activation = """                                if (in1_subblock < %(g)d) {
                                    apply_activation_from_pack<KernelActivation::SILU>(%(s)d);
                                } else {
                                    tile_regs_wait();
                                }
""" % dict(g=chunk // sub, s=sub)
    return """                            if (last_out) {
                                static_assert(out_subblock_num_tiles == %(s)d && out_subblock_h == 1 && in1_num_subblocks == %(n)d);
                                constexpr uint32_t rounded_cb = 30;
                                tile_regs_commit();
                                cb_reserve_back(rounded_cb, %(s)d);
""" % dict(s=sub, n=2 * chunk // sub) + activation + """                                PACK((pack_reconfig_data_format(rounded_cb)));
                                PACK((llk_pack_reconfig_l1_acc(0)));
                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, rounded_cb, %(s)d);
                                tile_regs_release();
                                cb_push_back(rounded_cb, %(s)d);
""" % dict(s=sub)


def bank_tail(chunk, row_tiles):
    """The product loop after the K loop: per row tile, wait for its gate run and up run (2 * chunk tiles), then the bf16 product of gate tile i with up tile chunk + i, one output tile
    at a time in the order the weights kernel's writer takes them (row tile, then column i)."""
    return """
    constexpr uint32_t rounded_cb = 30;
    static_assert(in0_num_subblocks == %(r)d);
    reconfig_data_format_srca(rounded_cb);
    copy_tile_to_dst_init_short(rounded_cb);
    mul_binary_tile_init();
    for (uint32_t row = 0; row < %(r)d; ++row) {
        cb_wait_front(rounded_cb, %(w)d);
        for (uint32_t pair = 0; pair < %(c)d; ++pair) {
            tile_regs_acquire();
            copy_tile(rounded_cb, pair, 0);
            copy_tile(rounded_cb, %(c)d + pair, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(out_dfb_id, 1);
            tile_regs_wait();
            PACK((pack_reconfig_data_format(out_dfb_id)));
            pack_tile(0, out_dfb_id);
            tile_regs_release();
            cb_push_back(out_dfb_id, 1);
        }
        cb_pop_front(rounded_cb, %(w)d);
    }
""" % dict(c=chunk, r=row_tiles, w=2 * chunk)


def bank_compute(source, chunk, row_tiles=ROW_TILES, sub=None, ablate=None):
    """The native compute kernel with its last K block's pack replaced by bank_epilogue and bank_tail added after the K loop (the anchors and the other two edits are the first fused op's:
    the sfpu binary include, BF16_PRODUCT for the product, the header path). ValueError when the native source no longer has the anchors (fail closed)."""
    check_chunk(chunk)
    if row_tiles not in (1, 2):
        raise ValueError('row_tiles must be 1 or 2, got %r' % (row_tiles,))
    if source.count(fused.LAST_BLOCK) != 1:
        raise ValueError('Native final-pack anchor changed')
    start = source.index(fused.LAST_BLOCK)
    end = source.index(fused.LAST_BLOCK_END, start)
    result = source[:start] + bank_epilogue(chunk, sub, ablate) + source[end:]
    finish = result.rfind('}')
    result = result[:finish] + bank_tail(chunk, row_tiles) + result[finish:]
    result = '#include "api/compute/eltwise_binary_sfpu.h"\n' + result
    result = result.replace('mul_binary_tile(0, 1, 0);', fused.BF16_PRODUCT)
    return result.replace('"bmm_fused_activation.hpp"', fused.HPP)


def compute_arguments(chunk, row_tiles=ROW_TILES, k_blocks=K_TILES // BLOCK_TILES, sub=None):
    """The native kernel's compile-time arguments for one worker: in0_block_w, in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles, in1_num_subblocks (2 with the default
    subblock: the gate run and the up run; 2 chunk / sub otherwise), in1_block_num_tiles, in1_block_w, num_blocks, out blocks x, out blocks y, out_subblock_h (1), out_subblock_w (sub,
    chunk by default), out_subblock_num_tiles (sub), batch, out_block_num_tiles, then three zeros (no untilize, no batch from the reader, no in0 transpose). The circular-buffer layout
    [K tile][gate run][up run] is the same for every sub: input subblock s covers the columns s * sub .. s * sub + sub - 1 of the 2 chunk a K tile row holds."""
    check_chunk(chunk)
    sub = resolve_sub(chunk, sub)
    return [BLOCK_TILES, row_tiles, BLOCK_TILES * row_tiles, BLOCK_TILES, 2 * chunk // sub, 2 * BLOCK_TILES * chunk, 2 * chunk, k_blocks, 1, 1, 1, sub, sub, 1,
            2 * chunk * row_tiles, 0, 0, 0]


def groups_of(chunk, per_bank):
    """[(first j, valid)] of one bank's `per_bank` columns in groups of `chunk`: the last group is the short one."""
    return [(first, min(chunk, per_bank - first)) for first in range(0, per_bank, chunk)]


def plan(chunk, grid, width=None, row_tiles=ROW_TILES, k_tiles=K_TILES, pair_columns=PAIR_COLUMNS, banks=BANKS, depth=DEFAULT_DEPTH, sub=None):
    """The host plan of one chip's program: the workers (core, bank, first tile column, valid columns) bank by bank in row-major order on a rectangle `width` wide (the probe's placement,
    the one its 402.7 GB/s was measured at), the rectangle, the circular buffer sizes and the compile-time arguments. ValueError when the tile columns are not a multiple of the bank
    count (the tiles of a bank would not be a regular stride), the grid cannot hold the workers, or the buffers do not fit L1."""
    check_chunk(chunk)
    check_depth(depth)
    sub = resolve_sub(chunk, sub)
    if pair_columns % banks:
        raise ValueError('%d tile columns are not a multiple of the %d DRAM banks: the tiles of one bank are not a regular stride' % (pair_columns, banks))
    grid_x, grid_y = int(grid[0]), int(grid[1])
    width = grid_x if width is None else int(width)
    if not 1 <= width <= grid_x:
        raise ValueError('width %d is outside the device grid width %d' % (width, grid_x))
    per_bank = pair_columns // banks
    entries = [(bank, bank + banks * first, valid) for bank in range(banks) for first, valid in groups_of(chunk, per_bank)]
    workers = len(entries)
    cols = min(width, workers)
    rows = int(math.ceil(workers / float(cols)))
    if rows > grid_y:
        raise ValueError('%d workers %d wide need %d rows; the device has %d' % (workers, cols, rows, grid_y))
    mapping = [Worker(index % cols, index // cols, bank, first, valid) for index, (bank, first, valid) in enumerate(entries)]
    k_blocks = k_tiles // BLOCK_TILES
    buffers = dict(in0=dict(index=0, bytes=depth * BLOCK_TILES * row_tiles * 2048, cores='all'),
                   in1=dict(index=1, bytes=depth * 16 * chunk * 576, cores='workers'),
                   out=dict(index=4, bytes=row_tiles * chunk * 2048, cores='workers'),
                   partial=dict(index=5, bytes=row_tiles * 2 * chunk * 4096, cores='workers'),
                   rounded=dict(index=30, bytes=row_tiles * 2 * chunk * 2048, cores='workers'))
    worker_bytes = sum(item['bytes'] for item in buffers.values())
    if worker_bytes > L1_BUDGET:
        raise ValueError('%d bytes of circular buffers a worker exceed the %d budget' % (worker_bytes, L1_BUDGET))
    return dict(chunk=chunk, depth=depth, sub=sub, banks=banks, per_bank=per_bank, workers=workers, cols=cols, rows=rows, cells=cols * rows, mapping=mapping, k_blocks=k_blocks,
                row_tiles=row_tiles, pair_columns=pair_columns, buffers=buffers, worker_bytes=worker_bytes,
                request_bytes=chunk * 576, compile_arguments=compute_arguments(chunk, row_tiles, k_blocks, sub))


def owned_columns(worker, banks=BANKS):
    """The tile columns a worker owns: first, first + banks, ..."""
    return [worker.first + banks * i for i in range(worker.valid)]


def kernel_sources():
    """{file name: sha256 of the kernel source this module ships}."""
    return dict((name, hashlib.sha256((HERE / name).read_bytes()).hexdigest()) for name in KERNELS)


_COMPUTE_CACHE = {}


class BankGateUp(object):
    """One layer's bank-strided fused gate|up + SwiGLU launch. `operations` is ttnn; `mesh` the mesh device the tensors live on; `w1` and `w3` the served gate and up weights (bfloat4_b,
    interleaved DRAM, K = 5,120 by N = 4,352 a chip). __call__(x) takes the (1, 1, 64, 5120) bf16 TILE input in L1 interleaved and returns the (1, 1, 64, 4352) bf16 TILE product
    silu(x w1) * (x w3) in L1 interleaved. It does not free x. Same call contract as tp4_mlp_fused.FusedGateUp."""

    def __init__(self, operations, mesh, w1, w3, chunk=DEFAULT_CHUNK, grid=(13, 10), width=None, math_approx_mode=None, source_root='/opt/tt-metal',
                 rows=lever.ROWS, banks=None, depth=DEFAULT_DEPTH, sub=None, ablate=None):
        if type(math_approx_mode) is not bool:
            raise ValueError('Explicit boolean math approximation mode required (the served compute config\'s)')
        if rows not in (lever.TILE, lever.ROWS):
            raise ValueError('rows must be 32 or 64, got %r' % (rows,))
        check_chunk(chunk)
        check_depth(depth)
        sub = resolve_sub(chunk, sub)
        check_ablate(ablate)
        self.operations, self.mesh, self.w1, self.w3 = operations, mesh, w1, w3
        self.row_tiles = rows // lever.TILE
        self.math_approx_mode = math_approx_mode
        self.chunk, self.depth, self.sub, self.ablate = chunk, depth, sub, ablate
        self.banks = dram_banks(mesh) if banks is None else int(banks)
        self.plan = plan(chunk, grid, width, self.row_tiles, banks=self.banks, depth=depth, sub=sub)
        key = (str(source_root), chunk, self.row_tiles, sub, ablate)
        if key not in _COMPUTE_CACHE:
            original = fused.native_source(source_root)
            _COMPUTE_CACHE[key] = (bank_compute(original, chunk, self.row_tiles, sub, ablate), hashlib.sha256(original.encode()).hexdigest())
        self.compute, native_sha = _COMPUTE_CACHE[key]
        self.manifest = dict(native_compute_sha256=native_sha, bank_compute_sha256=hashlib.sha256(self.compute.encode()).hexdigest(), kernels=kernel_sources(),
                             workers=self.plan['workers'], rectangle=[self.plan['cols'], self.plan['rows']], chunk=chunk, depth=depth, sub=sub, ablate=ablate, banks=self.banks,
                             request_bytes=self.plan['request_bytes'], k_block=BLOCK_TILES, row_tiles=self.row_tiles, math_approx_mode=math_approx_mode,
                             epilogue='BF16(silu(gate)), BF16(up), then the BF16 product by the SFPU')
        self.calls = 0

    # -- checks -----------------------------------------------------------------------------------------------------------
    def check(self, x):
        ttnn = self.operations
        if list(x.shape) != [1, 1, self.row_tiles * lever.TILE, K_TILES * TILE] or x.dtype != ttnn.bfloat16:
            raise ValueError('The bank gate|up input must be (1, 1, %d, %d) bfloat16; got %r %r' % (
                self.row_tiles * lever.TILE, K_TILES * TILE, list(x.shape), x.dtype))
        for name, weight in (('w1', self.w1), ('w3', self.w3)):
            if weight.dtype != ttnn.bfloat4_b:
                raise ValueError('The bank gate|up weight %s must be bfloat4_b; got %r' % (name, weight.dtype))

    # -- the launch ---------------------------------------------------------------------------------------------------------
    def __call__(self, x):
        ttnn = self.operations
        self.check(x)
        plan = self.plan
        chunk, row_tiles = plan['chunk'], plan['row_tiles']
        output = ttnn.empty((1, 1, row_tiles * lever.TILE, PAIR_COLUMNS * TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(plan['cols'] - 1, plan['rows'] - 1))])
        full_rows, remainder = divmod(plan['workers'], plan['cols'])
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(plan['cols'] - 1, full_rows - 1)))
        if remainder:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
        workers = ttnn.CoreRangeSet(ranges)

        def cb(index, dtype, page, count, cores):
            return ttnn.CBDescriptor(total_size=page * count, core_ranges=cores,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page,
                                                                                 tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])

        depth = plan['depth']
        buffers = [cb(0, ttnn.bfloat16, 2048, depth * BLOCK_TILES * row_tiles, all_cores),
                   cb(1, ttnn.bfloat4_b, 576, depth * 16 * chunk, workers),
                   cb(4, ttnn.bfloat16, 2048, row_tiles * chunk, workers),
                   cb(5, ttnn.float32, 4096, row_tiles * 2 * chunk, workers),
                   cb(30, ttnn.bfloat16, 2048, row_tiles * 2 * chunk, workers)]
        named = [('k_blocks', plan['k_blocks']), ('block_tiles', BLOCK_TILES), ('row_tiles', row_tiles)]
        mesh_program = ttnn.MeshProgramDescriptor()
        shards = zip(ttnn.get_device_tensors(x), ttnn.get_device_tensors(self.w1), ttnn.get_device_tensors(self.w3), ttnn.get_device_tensors(output))
        for chip, (local_x, local_gate, local_up, local_output) in enumerate(shards):
            device = local_x.device()
            first = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            last = device.worker_core_from_logical_core(ttnn.CoreCoord(plan['cols'] - 1, plan['rows'] - 1))
            input_kernel = ttnn.KernelDescriptor(
                kernel_source=str(HERE / KERNELS[0]), core_ranges=all_cores,
                compile_time_args=ttnn.TensorAccessorArgs(local_x).get_compile_time_args(),
                named_compile_time_args=named + [('row_stride', K_TILES)],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default))
            input_args = ttnn.RuntimeArgs()
            for index in range(plan['cells']):
                input_args[index % plan['cols']][index // plan['cols']] = [local_x.buffer_address(), index, first.x, first.y, last.x, last.y,
                                                                           plan['workers'], plan['cells']]
            input_kernel.runtime_args = input_args
            writer = ttnn.KernelDescriptor(
                kernel_source=str(HERE / KERNELS[1]), core_ranges=workers,
                compile_time_args=(ttnn.TensorAccessorArgs(local_gate).get_compile_time_args()
                                   + ttnn.TensorAccessorArgs(local_up).get_compile_time_args()
                                   + ttnn.TensorAccessorArgs(local_output).get_compile_time_args()),
                named_compile_time_args=named + [('chunk', chunk), ('pair_columns', PAIR_COLUMNS), ('banks', plan['banks']), ('depth', depth)],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
            writer_args = ttnn.RuntimeArgs()
            for worker in plan['mapping']:
                writer_args[worker.x][worker.y] = [local_gate.buffer_address(), local_up.buffer_address(), local_output.buffer_address(), worker.first,
                                                   worker.valid]
            writer.runtime_args = writer_args
            compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True,
                                                          math_approx_mode=self.math_approx_mode)
            unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
            unpack_modes[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
            compute_config.unpack_to_dest_mode.extend(unpack_modes)
            compute = ttnn.KernelDescriptor(
                kernel_source=self.compute, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=workers,
                compile_time_args=list(plan['compile_arguments']),
                named_compile_time_args=[('cb_in0', 0), ('cb_in1', 1), ('cb_out', 4), ('cb_intermed0', 5), ('cb_in0_transposed', 10),
                                         ('activation_type', 4), ('activation_param0', 0), ('activation_param1', 0), ('activation_param2', 0)],
                defines=[('FP32_DEST_ACC_EN', '1'), ('PACKER_L1_ACC', '1'), ('SFPU_ACTIVATION', '1')] + ([('SKIP_COMPUTE', '1')] if self.ablate == 'skip_compute' else []),
                config=compute_config)
            program = ttnn.ProgramDescriptor(kernels=[input_kernel, writer, compute], cbs=buffers,
                                             semaphores=[ttnn.SemaphoreDescriptor(id=index, core_ranges=all_cores, initial_value=0) for index in (0, 1)])
            coordinate = ttnn.MeshCoordinate(0, chip)
            mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = program
        ttnn.generic_op([x, self.w1, self.w3, output], mesh_program)
        self.calls += 1
        return output
