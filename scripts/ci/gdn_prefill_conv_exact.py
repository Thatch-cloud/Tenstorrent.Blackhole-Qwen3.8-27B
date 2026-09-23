"""gdn_prefill_conv_exact: one generic_op launch for the GDN prefill causal conv (lever #2).

It replaces, bit for bit, what gdn/tp.py forward_prefill runs today on its valid_len path:

    _causal_conv1d_fir(qkv, None, None, 4, mesh, memory_config=L1, conv_state=carry,
                       weight_taps=taps, bias_dev=None, valid_len=valid_len)
    q, k, v = three ttnn.slice of the conv output

That is 22 device ops per layer-chunk (concat, a host one-hot plus matmul for the carry,
three untilize/slice/tilize windows, one multiply, three addcmul, SiLU, three slices;
2.28 ms at T=2048, gate12b). Here it is three kernels:

  reader  (RISCV_1)  reads each qkv tile once, builds the three row-shifted copies of it
                     (x[t-3], x[t-2], x[t-1]) from the tile and the one above it (or the
                     carry / zero tile at the top of a column) with 32-byte face-row copies,
                     and builds new_state rows x[vl-3..vl-1] in the column's one ht_v tile;
  compute (TRISC)    per output tile, in 16-bit DST and in the served tap order and
                     operand slots: mul_binary_tile (tap 0), three
                     addcmul_tile<Float16_b> (taps 1-3), silu_tile - the four LLK calls the
                     served binary_ng / ternary / unary kernels make, on the same bf16 bytes;
  writer  (RISCV_0)  routes each output tile to q, k or v and writes the state tile.

Exactness (spec section 1): every SFPU op above stores a value that is already exactly bf16
and bf16 pack/unpack is the identity, so keeping the accumulator in DST between taps does not
change a value. MIRROR_PACK replays the served pack/unpack literally to remove even the
denormal edge. The FIR's new_state comes out canonical on both state paths: -0 -> +0 and
denormals -> +0 (card M, pcx-20260923T063828: the one-hot matmul on the valid_len path, the
x_padded untilize / concat / tilize round trip on the static slice). The reader reproduces it
(canon, CANON_DENORM; STATIC_CANON for the static slice). Separately, if the round trip flushed
denormals inside x itself, PCX_FLUSH_X_DENORM (FLUSH_X_DENORM_DEFAULT) flushes every x / carry
tile the reader loads, on both state paths, so the card-M run can select it. These are
edge-value knobs: real activations are never denormal, and an exact -0 state row is rare.

This module imports no ttnn at top level: the planners below are pure python so the CPU suite
can pin the index maths (test_gdn_prefill_conv_exact.py). In the container it is mounted at
models/demos/blackhole/qwen36/tt/gdn/gdn_prefill_conv_exact.py beside the grafted gdn/tp.py,
with its three kernels next to it (lever_n_m3native_patch.PREFILL_CONV_FILES is the one table).
"""

import hashlib
from pathlib import Path

TILE = 32
PAGE = 2048            # bytes in one bf16 32x32 tile
FACE_BYTES = 512       # one 16x16 bf16 face
ROW_BYTES = 32         # one face row: 16 bf16
KERNEL_SIZE = 4
STATE_ROWS = KERNEL_SIZE - 1
ONE_F32_BITS = 0x3F800000   # the addcmul `value` the served ternary passes (bit_cast<uint32_t>(1.0f))

READER = 'gdn_prefill_conv_exact_reader.cpp'
COMPUTE = 'gdn_prefill_conv_exact_compute.cpp'
WRITER = 'gdn_prefill_conv_exact_writer.cpp'
KERNEL_FILES = (READER, COMPUTE, WRITER)
# Everything the op needs at run time, by basename. The graft mounts exactly these next to
# gdn/tp.py; lever_n_m3native_patch derives its mount table from this tuple.
RUNTIME_FILES = (Path(__file__).name,) + KERNEL_FILES

# Circular buffers (mirrored by namespace pcx in the kernels): index -> pages.
CB_X3, CB_X2, CB_X1, CB_X0 = 0, 1, 2, 3      # x shifted by 3 / 2 / 1 / 0 (taps 0 / 1 / 2 / 3)
CB_TAPS, CB_BCAST, CB_OUT = 4, 5, 6
CB_RING, CB_STATE, CB_ACC, CB_HALO = 7, 8, 9, 10
CB_PAGES = {CB_X3: 2, CB_X2: 2, CB_X1: 2, CB_X0: 2, CB_TAPS: 4, CB_BCAST: 4, CB_OUT: 4,
            CB_RING: 3, CB_STATE: 1, CB_HALO: 2}
CB_ACC_PAGES = 2       # only with MIRROR_PACK

# The one common-runtime-arg layout all three kernels read (get_common_arg_val).
COMMON_ARGS = ('qkv', 'carry', 'tap0', 'tap1', 'tap2', 'tap3', 'q', 'k', 'v', 'state', 'valid_len', 'canon')
ACCESSORS = COMMON_ARGS[:10]       # the tensors, in TensorAccessorArgs order after the 6 compile args
COMPILE_ARGS = ('Ht', 'Ct', 'R', 'strips', 'q_tiles', 'has_carry')

# Whether a denormal in a canonicalised new_state is flushed to +0 like -0 is (the CANON_DENORM
# define). Card M pcx-20260923T063828: only True matches the FIR's one-hot state (every special
# carry and denormal case). The card-M test re-checks it and fails while this default disagrees.
# The model path always runs this default.
CANON_DENORM_DEFAULT = True

# Whether the FIR's byte-movement round trip (untilize / concat / tilize of x_padded, and the
# second untilize / slice / tilize of taps 1-3 and of the static-slice state) flushes a bf16
# denormal: 0 = no (the identity), 1 = to +0, 2 = to a zero of its own sign. The reader applies it
# to every x and carry tile it loads (the PCX_FLUSH_X_DENORM define), so it reaches q/k/v and
# new_state on both the valid_len None and the one-hot paths. The card-M run tries all three on
# the denormal cases, records which match per path, and fails while this default disagrees.
FLUSH_X_DENORM_DEFAULT = 0
FLUSH_X_DENORM_MODES = (0, 1, 2)

# Whether the static-slice state (valid_len None) is canonicalised like the one-hot state (the
# reader's canon runtime arg; a runtime arg, so no recompile). Card M pcx-20260923T063828: the
# FIR's static state came out -0 -> +0 (data=zeros) and denormal -> +0 (data=denormal) with q/k/v
# exact, so its round trip canonicalises exactly as the one-hot matmul does. The model path never
# reaches the static path (the graft engages only when valid_len is given); the card-M test
# re-checks it and fails while this default disagrees.
STATIC_CANON_DEFAULT = True

NEGATIVE_CONTROLS = {'fp32': 'NEG_FP32', 'shift': 'NEG_SHIFT', 'stale': 'NEG_STALE', 'tapswap': 'NEG_TAPSWAP'}


class Unsupported(ValueError):
    """Raised before any device work when an input is outside what the op implements; the model
    side takes the FIR instead."""


# ---------------------------------------------------------------------------------------------
# Pure planners (CPU-tested).
# ---------------------------------------------------------------------------------------------

def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def ceil_div(a, b):
    return -(-a // b)


def plan(T, C, cores):
    """Work split. A unit is R consecutive tile rows of one tile column; R divides Ht. R minimises
    the reads of the busiest core - its tiles plus one halo tile per unit - then its unit count.
    (Tiles alone would pick R=2 at T=2048: 94 tiles but 47 units, each a halo read and a
    loop setup; R=32 is 96 tiles and 3 units.) Units are dealt out in order: core w takes
    [start_w, start_w + count_w).

    Returns dict(Ht, Ct, R, strips, units, ranges=[(start, count) per core]).
    At T=2048, C=5120 on 110 cores: R=32, 320 units, 100 cores x 3 + 10 x 2, 96 tiles max."""
    if T % TILE or C % TILE or T <= 0 or C <= 0 or cores <= 0:
        raise Unsupported('T and C must be positive multiples of 32 and cores positive')
    Ht, Ct = T // TILE, C // TILE

    def cost(r):
        per_core = ceil_div(Ct * (Ht // r), cores)
        return per_core * r + per_core, per_core

    R = min(divisors(Ht), key=cost)
    strips = Ht // R
    units = Ct * strips
    base, extra = divmod(units, cores)
    ranges, start = [], 0
    for worker in range(cores):
        count = base + (1 if worker < extra else 0)
        ranges.append((start, count))
        start += count
    return dict(Ht=Ht, Ct=Ct, R=R, strips=strips, units=units, ranges=ranges)


def unit_tiles(unit, strips, R):
    """(ct, [ht...]) of one unit: unit u -> column u // strips, rows (u % strips) * R .. + R - 1."""
    ct, ht0 = unit // strips, (unit % strips) * R
    return ct, list(range(ht0, ht0 + R))


def face_offset(face_row, half):
    """Byte offset of face (face_row, half) in a 32x32 bf16 tile: faces are stored 0,1 / 2,3."""
    return (2 * face_row + half) * FACE_BYTES


def shift_copies(s, prev_is_x):
    """The 8 face-row copies that build `x shifted down by s rows` (s in 1..3) in a fresh tile.

    Output row r of the tile at tile-row ht is x[32*ht + r - s]: rows s..31 come from the
    current tile, rows 0..s-1 from the tile above - the last s rows of the previous x tile
    (prev_is_x), or rows 3-s..2 of the carry / zero tile at the top of a column (the carry
    holds x[-3..-1] in its rows 0..2).

    Returns [(source, source_offset, length, destination_offset)], source in {'cur', 'prev'};
    every offset and length is a multiple of 32 bytes."""
    if s not in (1, 2, 3):
        raise ValueError('shift must be 1, 2 or 3')
    out = []
    for half in (0, 1):
        upper, lower = face_offset(0, half), face_offset(1, half)
        if prev_is_x:
            halo = lower + ROW_BYTES * (16 - s)
        else:
            halo = upper + ROW_BYTES * (STATE_ROWS - s)
        out += [('cur', upper, (16 - s) * ROW_BYTES, upper + ROW_BYTES * s),
                ('prev', halo, ROW_BYTES * s, upper),
                ('cur', lower, (16 - s) * ROW_BYTES, lower + ROW_BYTES * s),
                ('cur', upper + ROW_BYTES * (16 - s), ROW_BYTES * s, lower)]
    return out


def state_tile_row(valid_len):
    """ht_v: the tile row that holds x[valid_len - 1], where the column's state is built."""
    return (valid_len - 1) // TILE


def state_rows(valid_len):
    """new_state row j <- x[valid_len - 3 + j] (the FIR's one-hot selects x_padded[vl + j]).

    Returns three (where, row): 'cur' is the ht_v tile, 'prev' the x tile above it, 'carry' the
    carry tile (or the zero tile when there is none: x_padded's zero pad)."""
    if valid_len < 1:
        raise ValueError('valid_len must be at least 1')
    ht_v = state_tile_row(valid_len)
    rows = []
    for j in range(STATE_ROWS):
        source = valid_len - STATE_ROWS + j
        if source >= TILE * ht_v:
            rows.append(('cur', source - TILE * ht_v))
        elif source >= 0:
            rows.append(('prev', source - TILE * (ht_v - 1)))
        else:
            rows.append(('carry', STATE_ROWS + source))
    return rows


def route(ct, q_tiles, C_tiles):
    """Output tensor and column of conv tile column ct: q | k | v, v being the C - 2kd rest."""
    if ct < q_tiles:
        return 'q', ct, q_tiles
    if ct < 2 * q_tiles:
        return 'k', ct - q_tiles, q_tiles
    return 'v', ct - 2 * q_tiles, C_tiles - 2 * q_tiles


def output_page(ct, ht, q_tiles, C_tiles):
    which, column, width = route(ct, q_tiles, C_tiles)
    return which, ht * width + column


def core_coordinates(grid_x, grid_y, count):
    """Worker w -> (w % grid_x, w // grid_x), row-major over the compute grid."""
    if grid_x * grid_y < count:
        raise ValueError('grid too small')
    return [(worker % grid_x, worker // grid_x) for worker in range(count)]


def cb_plan(mirror_pack=False):
    plan_ = dict(CB_PAGES)
    if mirror_pack:
        plan_[CB_ACC] = CB_ACC_PAGES
    return plan_


def source_sha(directory=None):
    """First 8 hex of sha256 over the three kernels, passed as a define so a revised kernel at the
    same path can never reuse a stale JIT binary (defines are hashed; file contents may not be)."""
    directory = Path(directory) if directory is not None else Path(__file__).parent
    digest = hashlib.sha256()
    for name in KERNEL_FILES:
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()[:8]


def kernel_defines(sha, *, mirror_pack=False, canon_denorm=False, shift_words=False, negative=None,
                   shift_only=False, flush_x_denorm=0):
    defines = {'PCX_SRC_SHA': '0x' + sha}
    if mirror_pack:
        defines['MIRROR_PACK'] = '1'
    if canon_denorm:
        defines['CANON_DENORM'] = '1'
    if flush_x_denorm not in FLUSH_X_DENORM_MODES:
        raise ValueError('flush_x_denorm must be one of %s' % (FLUSH_X_DENORM_MODES,))
    if flush_x_denorm:
        defines['PCX_FLUSH_X_DENORM'] = str(flush_x_denorm)
    if shift_words:
        defines['PCX_SHIFT_WORDS'] = '1'
    if shift_only:
        defines['PCX_MB_SHIFT_ONLY'] = '1'
    if negative is not None:
        if negative not in NEGATIVE_CONTROLS:
            raise ValueError('unknown negative control %r' % (negative,))
        defines[NEGATIVE_CONTROLS[negative]] = '1'
    return tuple(sorted(defines.items()))


def compile_args(layout, key_dim_tp, has_carry):
    return [layout['Ht'], layout['Ct'], layout['R'], layout['strips'], key_dim_tp // TILE, 1 if has_carry else 0]


def common_args(addresses, valid_len, T, static_canon=False):
    """[10 buffer addresses in ACCESSORS order, valid_len, canon]. A given valid_len is the FIR's
    one-hot matmul (canon 1: -0 -> +0, and denormals under CANON_DENORM); valid_len None is its
    static slice of x_padded's last rows, canonicalised the same way only under static_canon."""
    if len(addresses) != len(ACCESSORS):
        raise ValueError('expected %d addresses' % len(ACCESSORS))
    canon = 1 if valid_len is not None or static_canon else 0
    return list(addresses) + [T if valid_len is None else int(valid_len), canon]


def mesh_coordinates(shape):
    rows, cols = shape
    return [(index // cols, index % cols) for index in range(rows * cols)]


# ---------------------------------------------------------------------------------------------
# Validation (no device work).
# ---------------------------------------------------------------------------------------------

def _interleaved(operations, tensor):
    return tensor.memory_config() in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG)


def _bf16_tile(operations, tensor):
    return tensor.dtype == operations.bfloat16 and tensor.layout == operations.TILE_LAYOUT


def unsupported(operations, mesh, qkv, carry, taps, valid_len, key_dim_tp, *, flat=True, kernel_size=KERNEL_SIZE):
    """None when the op can run these inputs, else the reason (the model logs it and takes the FIR)."""
    if not flat:
        return 'q/k/v not flat'
    if kernel_size != KERNEL_SIZE:
        return 'kernel size %s != 4' % (kernel_size,)
    shape = tuple(qkv.shape)
    if len(shape) != 3 or shape[0] != 1:
        return 'qkv shape %s is not [1, T, C]' % (shape,)
    _, T, C = shape
    if T % TILE or C % TILE or key_dim_tp % TILE or not 0 < 2 * key_dim_tp < C:
        return 'T=%d C=%d kd=%d not tile multiples with 2*kd < C' % (T, C, key_dim_tp)
    if valid_len is not None and not 1 <= int(valid_len) <= T:
        return 'valid_len %s outside 1..%d' % (valid_len, T)
    try:
        mesh_shape = tuple(mesh.shape)
    except (AttributeError, TypeError):
        return 'no mesh shape'
    if mesh_shape not in ((1, 1), (1, 2)):
        return 'mesh %s is not [1, 1] or [1, 2]' % (mesh_shape,)
    if not _bf16_tile(operations, qkv) or not _interleaved(operations, qkv):
        return 'qkv not bf16 TILE interleaved'
    if len(taps) != KERNEL_SIZE:
        return '%d taps, expected 4' % len(taps)
    for tap in taps:
        if tuple(tap.shape) != (1, 1, C) or not _bf16_tile(operations, tap) or not _interleaved(operations, tap):
            return 'tap %s not a bf16 TILE interleaved [1, 1, %d]' % (tuple(tap.shape), C)
    if carry is not None:
        if tuple(carry.shape) != (1, STATE_ROWS, C) or not _bf16_tile(operations, carry) \
                or not _interleaved(operations, carry):
            return 'carry %s not a bf16 TILE interleaved [1, 3, %d]' % (tuple(carry.shape), C)
    return None


# ---------------------------------------------------------------------------------------------
# Device wrapper.
# ---------------------------------------------------------------------------------------------

_DESCRIPTORS = {}
_SOURCE_SHA = {}


def clear_cache():
    _DESCRIPTORS.clear()


def cache_size():
    return len(_DESCRIPTORS)


def _buffer_kind(operations, tensor):
    return 'l1' if tensor.memory_config() == operations.L1_MEMORY_CONFIG else 'dram'


def _sha(directory):
    key = str(directory)
    if key not in _SOURCE_SHA:
        _SOURCE_SHA[key] = source_sha(directory)
    return _SOURCE_SHA[key]


class _Prepared:
    """Built once per key: the core set, the CBs and, per chip, the three KernelDescriptors with
    their compile args and constant per-core [unit_start, unit_count]. Per call only the common
    runtime args (addresses, valid_len, canon) change - generic_op hashes their COUNT, not their
    values, so every call after the first is a program-cache hit."""

    def __init__(self, operations, mesh, chips, layout, compile_head, accessor_args, defines, directory,
                 fp32_dest, shift_only, mirror_pack):
        grid = mesh.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        self.coordinates = core_coordinates(grid.x, grid.y, cores)
        core_set = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0),
                                                                 operations.CoreCoord(grid.x - 1, grid.y - 1))])
        self.cbs = [operations.CBDescriptor(
            total_size=pages * PAGE, core_ranges=core_set,
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
                                                              page_size=PAGE,
                                                              tile=operations.TileDescriptor(operations.Tile([TILE, TILE])))])
            for index, pages in sorted(cb_plan(mirror_pack).items())]
        per_core = operations.RuntimeArgs()
        for (x, y), (start, count) in zip(self.coordinates, layout['ranges']):
            per_core[x][y] = [start, count]
        self.chips = []
        for chip in range(chips):
            dm_args = list(compile_head) + list(accessor_args[chip])
            reader = operations.KernelDescriptor(
                kernel_source=str(directory / READER), core_ranges=core_set, compile_time_args=dm_args,
                defines=list(defines), runtime_args=per_core,
                config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1,
                                                               noc=operations.NOC.RISCV_1_default))
            writer = operations.KernelDescriptor(
                kernel_source=str(directory / WRITER), core_ranges=core_set, compile_time_args=dm_args,
                defines=list(defines), runtime_args=per_core,
                config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                               noc=operations.NOC.RISCV_0_default))
            kernels = [reader, writer]
            if not shift_only:
                kernels.append(operations.KernelDescriptor(
                    kernel_source=str(directory / COMPUTE), core_ranges=core_set,
                    compile_time_args=list(compile_head), defines=list(defines), runtime_args=per_core,
                    config=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
                                                              fp32_dest_acc_en=fp32_dest, math_approx_mode=False)))
            self.chips.append(kernels)

    def program(self, operations, coordinates, commons):
        program = operations.MeshProgramDescriptor()
        for (row, col), kernels, common in zip(coordinates, self.chips, commons):
            for kernel in kernels:
                kernel.common_runtime_args = common
            coordinate = operations.MeshCoordinate(row, col)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=kernels, cbs=self.cbs)
        return program


def gdn_prefill_conv_exact(mesh, qkv, carry, taps, *, valid_len=None, key_dim_tp, mirror_pack=False,
                           canon_denorm=None, shift_words=False, negative=None, shift_only=False,
                           flush_x_denorm=None, static_canon=None, operations=None, directory=None):
    """(q [1,T,kd], k [1,T,kd], v [1,T,C-2kd], new_state [1,3,C]): bf16 TILE, DRAM interleaved.

    qkv [1,T,C] bf16 TILE interleaved (L1 or DRAM), carry None (zeros: the FIR's zero pad) or
    [1,3,C], taps 4 x [1,1,C]. valid_len None means T with the FIR's byte-copy state; a given
    valid_len reproduces its one-hot state. shift_only is the section 7.1 movement microbench:
    no compute kernel, and the only output is x shifted by 3 as one [1,T,C] tensor.
    canon_denorm / flush_x_denorm / static_canon None take CANON_DENORM_DEFAULT /
    FLUSH_X_DENORM_DEFAULT / STATIC_CANON_DEFAULT (the model path never passes them)."""
    if operations is None:
        import ttnn as operations
    reason = unsupported(operations, mesh, qkv, carry, taps, valid_len, key_dim_tp)
    if reason is not None:
        raise Unsupported(reason)
    directory = Path(directory) if directory is not None else Path(__file__).parent
    if canon_denorm is None:
        canon_denorm = CANON_DENORM_DEFAULT
    if flush_x_denorm is None:
        flush_x_denorm = FLUSH_X_DENORM_DEFAULT
    if static_canon is None:
        static_canon = STATIC_CANON_DEFAULT
    _, T, C = tuple(qkv.shape)
    shape = tuple(mesh.shape)
    coordinates = mesh_coordinates(shape)
    has_carry = carry is not None and negative != 'stale'
    defines = kernel_defines(_sha(directory), mirror_pack=mirror_pack, canon_denorm=canon_denorm,
                             shift_words=shift_words, negative=negative, shift_only=shift_only,
                             flush_x_denorm=flush_x_denorm)
    grid = mesh.compute_with_storage_grid_size()
    inputs = [qkv, carry if carry is not None else qkv] + list(taps)
    kinds = tuple(_buffer_kind(operations, tensor) for tensor in inputs)
    key = (id(mesh), shape, grid.x, grid.y, T, C, key_dim_tp, has_carry, kinds, defines, shift_only, mirror_pack)

    def empty(dims):
        return operations.empty(dims, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh,
                                memory_config=operations.DRAM_MEMORY_CONFIG)

    outputs = []
    try:
        if shift_only:
            shifted = empty((1, T, C))
            outputs = [shifted, shifted, shifted, shifted]
        else:
            kd = key_dim_tp
            outputs = [empty((1, T, kd)), empty((1, T, kd)), empty((1, T, C - 2 * kd)), empty((1, STATE_ROWS, C))]
        tensors = inputs + outputs
        shards = [operations.get_device_tensors(tensor) for tensor in tensors]
        if any(len(parts) != len(coordinates) for parts in shards):
            raise ValueError('every tensor must have one shard per mesh device')
        commons = []
        for chip in range(len(coordinates)):
            addresses = [parts[chip].buffer_address() for parts in shards]
            borrowed, owned = set(addresses[:len(inputs)]), addresses[len(inputs):]
            if borrowed & set(owned):
                raise ValueError('outputs must not alias borrowed inputs')
            commons.append(common_args(addresses, valid_len, T, static_canon))
        prepared = _DESCRIPTORS.get(key)
        if prepared is None:
            layout = plan(T, C, grid.x * grid.y)
            head = compile_args(layout, key_dim_tp, has_carry)
            accessor_args = [[argument for parts in shards
                              for argument in operations.TensorAccessorArgs(parts[chip]).get_compile_time_args()]
                             for chip in range(len(coordinates))]
            prepared = _Prepared(operations, mesh, len(coordinates), layout, head, accessor_args, defines, directory,
                                 fp32_dest=(negative == 'fp32'), shift_only=shift_only, mirror_pack=mirror_pack)
            _DESCRIPTORS[key] = prepared
        unique = []
        for tensor in tensors:
            if not any(tensor is seen for seen in unique):
                unique.append(tensor)
        operations.generic_op(unique, prepared.program(operations, coordinates, commons))
    except BaseException:
        for tensor in {id(t): t for t in outputs}.values():
            operations.deallocate(tensor)
        raise
    if shift_only:
        return outputs[0]
    return tuple(outputs)


def audit_against_fir(operations, fir, mesh, qkv, carry, taps, valid_len, key_dim_tp, outputs, *,
                      kernel_size=KERNEL_SIZE):
    """Run the served FIR the way gdn/tp.py calls it, slice q/k/v the way it does, and compare every
    chip's bytes with the op's (int16 views, so -0 and NaN payloads count). Returns a report; the
    caller decides whether a mismatch raises. qkv and carry must still hold this chunk's inputs."""
    import torch

    conv, state = fir(qkv, None, None, kernel_size, mesh, memory_config=operations.L1_MEMORY_CONFIG,
                      conv_state=carry, weight_taps=taps, bias_dev=None, valid_len=valid_len)
    _, T, C = tuple(qkv.shape)
    kd = key_dim_tp
    references = [operations.slice(conv, (0, 0, 0), (1, T, kd)),
                  operations.slice(conv, (0, 0, kd), (1, T, 2 * kd)),
                  operations.slice(conv, (0, 0, 2 * kd), (1, T, C)),
                  state]
    report = dict(T=T, valid_len=valid_len, carry=carry is not None, exact=True, mismatches={})
    try:
        for name, candidate, reference in zip(('q', 'k', 'v', 'new_state'), outputs, references):
            for chip, (mine, theirs) in enumerate(zip(operations.get_device_tensors(candidate),
                                                      operations.get_device_tensors(reference))):
                a = operations.to_torch(mine).contiguous().view(torch.int16)
                b = operations.to_torch(theirs).contiguous().view(torch.int16)
                if a.shape != b.shape or not torch.equal(a, b):
                    report['exact'] = False
                    report['mismatches']['%s/chip%d' % (name, chip)] = (
                        int((a != b).sum()) if a.shape == b.shape else 'shape %s vs %s' % (tuple(a.shape), tuple(b.shape)))
    finally:
        for tensor in references:
            operations.deallocate(tensor)
        operations.deallocate(conv)
    return report
