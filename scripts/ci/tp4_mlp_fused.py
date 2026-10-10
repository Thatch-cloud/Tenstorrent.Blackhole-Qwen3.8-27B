"""WP4 F-D1: the MLP gate and up as ONE launch with the SwiGLU epilogue, at the TP4 per-chip shape and 64 rows (QWEN_FAST_MLP_GATEUP=1).

A port of the TP2 fused_1d.FusedProjection (which stays pinned and untouched) to what the packed 64-row verify block needs:

  * four chips, K = 5,120 and N = 4,352 per chip (136 tile pairs, not the pair's 272), 13x10 or 11x10 workers read from the device;
  * two row tiles (64 rows) instead of one: the native compute kernel runs two input subblocks per output block;
  * the served w1 and w3 tensors read as two accessors; there is no pair-packed copy (a packed copy is 1.6 GB a chip for 64 layers, and the
    Lever N ship profile does not build it).

WHAT RUNS. One program per chip on `workers` cores (ceil(136 / pairs_per_worker)), each owning `pairs_per_worker` adjacent gate/up column pairs:
  activation  tp4_mlp_fused_input.cpp   worker 0 reads one K block (8 K tiles of each of the 2 row tiles) from the interleaved L1 input and multicasts it
                                        to the whole core rectangle; cores with no columns drain it
  weights     tp4_mlp_fused_weights.cpp each worker reads its gate tiles from w1 and its up tiles from w3 into the block layout [K tile][g0 u0 g1 u1 ...]
                                        and writes the products to the interleaved L1 output
  compute     the image's own bmm_large_block_zm_fused_bias_activation.cpp, patched exactly as fused_1d.fused_compute patches it (the native K loop:
              in0_block_w = 8, LoFi, fp32 destination accumulation, packer L1 accumulation; only the last K block's pack is replaced): SiLU applied at
              the pack of each gate tile, the gate and up tiles packed as bf16, then a bf16 product of each pair by the SFPU. That is what the served
              pair of matmuls (gate with SiLU fused, up) followed by ttnn.multiply compute and round, in the same places.

EXACTNESS. Not claimed until a card says so: the host plan, the kernels' data movement and the patch are checked on CPU (test_tp4_mlp_fused), the
arithmetic is the native kernel's. The two open points are the ones the TP2 port left open: the multiply unit (U3: BinaryNg's bf16 product) and the SFPU
approximation mode of the SiLU, which here follows the served compute config (`math_approx_mode` is an explicit argument, never a default). The card-M
harness (optimisation/ttnn-op/mlp_gateup, --arms fused) compares the product with the served multiply's bit for bit and times each pairs-per-worker;
the audit (QWEN_FAST_MLP_GATEUP_AUDIT) does the same in the trace on every chip.

Stdlib only at import (py 3.7); ttnn is the `operations` handle.
"""

import hashlib
import math
from pathlib import Path

import tp4_mlp_gateup as lever

COMPUTE = 'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp'
HERE = Path(__file__).resolve().parent
KERNELS = ('tp4_mlp_fused_input.cpp', 'tp4_mlp_fused_weights.cpp')

TILE = 32
BLOCK_TILES = 8                      # in0_block_w, the K loop's block: the served matmuls' 8
K_TILES = 160                        # K = 5,120
PAIR_COLUMNS = 136                   # N = 4,352 per chip, per tensor
ROW_TILES = 2                        # 64 rows
L1_BUDGET = 1200000                  # the circular buffers of one core, the sweep's budget (the allocatable L1 is about 1.43 MB)

# The anchors of the native kernel's last K block, and the replacement epilogue: the strings of fused_1d.fused_compute (test_tp4_mlp_fused holds that this
# module's output equals the original's for the shapes the original supports).
LAST_BLOCK = '                            if (last_out) {'
LAST_BLOCK_END = '                            } else {\n                                tile_regs_commit();'
BF16_PRODUCT = """MATH((SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, 8, false), 0, 1, 0, VectorMode::RC)));"""
EPILOGUE = """                            if (last_out) {
                                static_assert(out_subblock_num_tiles == 2);
                                constexpr uint32_t rounded_cb = 30;
                                tile_regs_commit();
                                cb_reserve_back(rounded_cb, 2);
                                apply_activation_from_pack<KernelActivation::SILU>(1);
                                PACK((pack_reconfig_data_format(rounded_cb)));
                                PACK((llk_pack_reconfig_l1_acc(0)));
                                uint32_t start_dst_index = 0;
                                pack_block(start_dst_index, rounded_cb, 2);
                                tile_regs_release();
                                cb_push_back(rounded_cb, 2);
"""
TAIL = """
    constexpr uint32_t rounded_cb = 30;
    cb_wait_front(rounded_cb, 14);
    reconfig_data_format_srca(rounded_cb);
    copy_tile_to_dst_init_short(rounded_cb);
    mul_binary_tile_init();
    for (uint32_t pair = 0; pair < 7; ++pair) {
        cb_wait_front(rounded_cb, 2);
        tile_regs_acquire();
        copy_tile(rounded_cb, 0, 0);
        copy_tile(rounded_cb, 1, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(out_dfb_id, 1);
        tile_regs_wait();
        PACK((pack_reconfig_data_format(out_dfb_id)));
        pack_tile(0, out_dfb_id);
        tile_regs_release();
        cb_push_back(out_dfb_id, 1);
        cb_pop_front(rounded_cb, 2);
    }
"""
HPP = '"ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"'


def fused_compute(source, pairs_per_worker, row_tiles=ROW_TILES):
    """The native compute kernel with its last K block's pack replaced by the SiLU-and-round pack and a bf16 product loop after the K loop, for
    `pairs_per_worker` pairs and `row_tiles` row tiles per worker (the loop consumes one rounded gate and up tile pair per output pair). ValueError
    when the native source no longer has the anchors (fail closed)."""
    if pairs_per_worker not in lever.PAIRS_PER_WORKER:
        raise ValueError('pairs_per_worker must be one of %s, got %r' % (lever.PAIRS_PER_WORKER, pairs_per_worker))
    if row_tiles not in (1, 2):
        raise ValueError('row_tiles must be 1 or 2, got %r' % (row_tiles,))
    if source.count(LAST_BLOCK) != 1:
        raise ValueError('Native final-pack anchor changed')
    start = source.index(LAST_BLOCK)
    end = source.index(LAST_BLOCK_END, start)
    result = source[:start] + EPILOGUE + source[end:]
    finish = result.rfind('}')
    result = result[:finish] + TAIL + result[finish:]
    result = '#include "api/compute/eltwise_binary_sfpu.h"\n' + result
    result = result.replace('mul_binary_tile(0, 1, 0);', BF16_PRODUCT)
    pairs = pairs_per_worker * row_tiles
    result = result.replace('cb_wait_front(rounded_cb, 14);', 'cb_wait_front(rounded_cb, %d);' % (2 * pairs))
    result = result.replace('pair < 7;', 'pair < %d;' % pairs)
    return result.replace('"bmm_fused_activation.hpp"', HPP)


def compute_arguments(pairs_per_worker, row_tiles=ROW_TILES, k_blocks=K_TILES // BLOCK_TILES):
    """The native kernel's compile-time arguments for one worker: in0_block_w, in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles,
    in1_num_subblocks, in1_block_num_tiles, in1_per_core_w, num_blocks, out blocks x, out blocks y, out_subblock_h, out_subblock_w,
    out_subblock_num_tiles, batch, out_block_num_tiles, then three zeros. One output subblock is one gate and up pair (1 x 2 tiles)."""
    p = pairs_per_worker
    return [BLOCK_TILES, row_tiles, BLOCK_TILES * row_tiles, BLOCK_TILES, p, 2 * BLOCK_TILES * p, 2 * p, k_blocks, 1, 1, 1, 2, 2, 1,
            2 * p * row_tiles, 0, 0, 0]


def plan(pairs_per_worker, grid, width=None, row_tiles=ROW_TILES, k_tiles=K_TILES, pair_columns=PAIR_COLUMNS):
    """The host plan of one chip's program: the workers (core, first pair, valid pairs) in row-major order on a rectangle `width` wide, the
    rectangle, the circular buffer sizes and the compile-time arguments. ValueError when the grid cannot hold it or the buffers do not fit L1."""
    if pairs_per_worker not in lever.PAIRS_PER_WORKER:
        raise ValueError('pairs_per_worker must be one of %s, got %r' % (lever.PAIRS_PER_WORKER, pairs_per_worker))
    grid_x, grid_y = int(grid[0]), int(grid[1])
    width = grid_x if width is None else int(width)
    if not 1 <= width <= grid_x:
        raise ValueError('width %d is outside the device grid width %d' % (width, grid_x))
    workers = int(math.ceil(pair_columns / float(pairs_per_worker)))
    cols = min(width, workers)
    rows = int(math.ceil(workers / float(cols)))
    if rows > grid_y:
        raise ValueError('%d workers %d wide need %d rows; the device has %d' % (workers, cols, rows, grid_y))
    mapping = [(index % cols, index // cols, index * pairs_per_worker, min(pairs_per_worker, pair_columns - index * pairs_per_worker))
               for index in range(workers)]
    k_blocks = k_tiles // BLOCK_TILES
    p = pairs_per_worker
    buffers = dict(in0=dict(index=0, bytes=2 * BLOCK_TILES * row_tiles * 2048, cores='all'),
                   in1=dict(index=1, bytes=32 * p * 576, cores='workers'),
                   out=dict(index=4, bytes=row_tiles * p * 2048, cores='workers'),
                   partial=dict(index=5, bytes=row_tiles * 2 * p * 4096, cores='workers'),
                   rounded=dict(index=30, bytes=row_tiles * 2 * p * 2048, cores='workers'))
    worker_bytes = sum(item['bytes'] for item in buffers.values())
    if worker_bytes > L1_BUDGET:
        raise ValueError('%d bytes of circular buffers a worker exceed the %d budget' % (worker_bytes, L1_BUDGET))
    return dict(pairs_per_worker=p, workers=workers, cols=cols, rows=rows, cells=cols * rows, mapping=mapping, k_blocks=k_blocks,
                row_tiles=row_tiles, pair_columns=pair_columns, buffers=buffers, worker_bytes=worker_bytes,
                compile_arguments=compute_arguments(p, row_tiles, k_blocks))


_COMPUTE_CACHE = {}


def native_source(source_root):
    return (Path(source_root) / COMPUTE).read_text()


def kernel_sources():
    """{file name: sha256 of the kernel source this module ships}."""
    return dict((name, hashlib.sha256((HERE / name).read_bytes()).hexdigest()) for name in KERNELS)


class FusedGateUp(object):
    """One layer's fused gate|up + SwiGLU launch. `operations` is ttnn; `mesh` the mesh device the tensors live on; `w1` and `w3` the served gate and
    up weights (bfloat4_b, interleaved DRAM, K = 5,120 by N = 4,352 a chip). __call__(x) takes the (1, 1, 64, 5120) bf16 TILE input in L1 interleaved
    and returns the (1, 1, 64, 4352) bf16 TILE product silu(x w1) * (x w3) in L1 interleaved. It does not free x."""

    def __init__(self, operations, mesh, w1, w3, pairs_per_worker=lever.DEFAULT_PAIRS, grid=(13, 10), width=None, math_approx_mode=None,
                 source_root='/opt/tt-metal', rows=lever.ROWS):
        if type(math_approx_mode) is not bool:
            raise ValueError('Explicit boolean math approximation mode required (the served compute config\'s)')
        if rows not in (lever.TILE, lever.ROWS):
            raise ValueError('rows must be 32 or 64, got %r' % (rows,))
        self.operations, self.mesh, self.w1, self.w3 = operations, mesh, w1, w3
        self.row_tiles = rows // lever.TILE
        self.math_approx_mode = math_approx_mode
        self.plan = plan(pairs_per_worker, grid, width, self.row_tiles)
        self.pairs_per_worker = pairs_per_worker
        key = (str(source_root), pairs_per_worker, self.row_tiles)
        if key not in _COMPUTE_CACHE:
            original = native_source(source_root)
            _COMPUTE_CACHE[key] = (fused_compute(original, pairs_per_worker, self.row_tiles), hashlib.sha256(original.encode()).hexdigest())
        self.compute, native_sha = _COMPUTE_CACHE[key]
        self.manifest = dict(native_compute_sha256=native_sha, fused_compute_sha256=hashlib.sha256(self.compute.encode()).hexdigest(),
                             kernels=kernel_sources(), workers=self.plan['workers'], rectangle=[self.plan['cols'], self.plan['rows']],
                             pairs_per_worker=pairs_per_worker, k_block=BLOCK_TILES, row_tiles=self.row_tiles,
                             math_approx_mode=math_approx_mode,
                             epilogue='BF16(silu(gate)), BF16(up), then the BF16 product by the SFPU')
        self.calls = 0

    # -- checks -----------------------------------------------------------------------------------------------------------
    def check(self, x):
        ttnn = self.operations
        if list(x.shape) != [1, 1, self.row_tiles * lever.TILE, K_TILES * TILE] or x.dtype != ttnn.bfloat16:
            raise ValueError('The fused gate|up input must be (1, 1, %d, %d) bfloat16; got %r %r' % (
                self.row_tiles * lever.TILE, K_TILES * TILE, list(x.shape), x.dtype))
        for name, weight in (('w1', self.w1), ('w3', self.w3)):
            if weight.dtype != ttnn.bfloat4_b:
                raise ValueError('The fused gate|up weight %s must be bfloat4_b; got %r' % (name, weight.dtype))

    # -- the launch ---------------------------------------------------------------------------------------------------------
    def __call__(self, x):
        ttnn = self.operations
        self.check(x)
        plan = self.plan
        p, row_tiles = plan['pairs_per_worker'], plan['row_tiles']
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

        buffers = [cb(0, ttnn.bfloat16, 2048, 2 * BLOCK_TILES * row_tiles, all_cores),
                   cb(1, ttnn.bfloat4_b, 576, 32 * p, workers),
                   cb(4, ttnn.bfloat16, 2048, row_tiles * p, workers),
                   cb(5, ttnn.float32, 4096, row_tiles * 2 * p, workers),
                   cb(30, ttnn.bfloat16, 2048, row_tiles * 2 * p, workers)]
        named = [('k_blocks', plan['k_blocks']), ('block_tiles', BLOCK_TILES), ('row_tiles', row_tiles)]
        mesh_program = ttnn.MeshProgramDescriptor()
        shards = zip(ttnn.get_device_tensors(x), ttnn.get_device_tensors(self.w1), ttnn.get_device_tensors(self.w3),
                     ttnn.get_device_tensors(output))
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
                named_compile_time_args=named + [('pairs_per_worker', p), ('pair_columns', PAIR_COLUMNS)],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
            writer_args = ttnn.RuntimeArgs()
            for core_x, core_y, begin, count in plan['mapping']:
                writer_args[core_x][core_y] = [local_gate.buffer_address(), local_up.buffer_address(), local_output.buffer_address(), begin, count]
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
                defines=[('FP32_DEST_ACC_EN', '1'), ('PACKER_L1_ACC', '1'), ('SFPU_ACTIVATION', '1')], config=compute_config)
            program = ttnn.ProgramDescriptor(kernels=[input_kernel, writer, compute], cbs=buffers,
                                             semaphores=[ttnn.SemaphoreDescriptor(id=index, core_ranges=all_cores, initial_value=0)
                                                         for index in (0, 1)])
            coordinate = ttnn.MeshCoordinate(0, chip)
            mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = program
        ttnn.generic_op([x, self.w1, self.w3, output], mesh_program)
        self.calls += 1
        return output
