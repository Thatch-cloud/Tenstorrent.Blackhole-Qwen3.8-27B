"""mlp_c1e_pack: C1e - the served prefill gate/up, bit for bit, without the persistent packed copy.

Under QWEN_FAST_SINGLE_GATEUP=1 the model builds no w_gate_up (3.209 GB per chip at TP2), and the
prefill MLP falls back to C1/C1c: separate w1 (SiLU fused) and w3 matmuls, each packed to bf16,
then a bf16 multiply. That can never be byte-identical to the served fused path, whose
all_gather_minimal_matmul_async(fuse_swiglu=True) forms silu(gate) * up from the fp32 destination
accumulators and rounds ONCE: with the SAME fp32 gate and up, rounding them to bf16 first changes
~36% of the outputs under round-to-nearest-even (and ~81% under truncation) - e.g. gate
0xbfb582d0, up 0x3fad03bc: fused 0xbebf (-0.37305), separate 0xbec0 (-0.375). It also changes the
matmul program, the K schedule and the compute kernel config (C1c runs the decode config,
packer_l1_acc=True).

C1e keeps the served op and only replaces where its weight comes from: before each layer's
prefill MLP, this module rewrites ONE shared scratch tensor, with the per-chip spec of the served
w_gate_up ([K, 2N] bfloat4_b, TILE, DRAM interleaved), from that layer's w1 and w3, page for page
(packed page r*2C + 2c + g = (w1, w3)[g] page r*C + c; the inverse of packed_weight_check.cpp's
map). The unchanged tpc.all_gather_swiglu_prefill then runs on bytes identical to the served
weight, with the served config, compute kernel config, gather and epilogue - so its output is the
served output by construction, whatever the kernel does inside. The cost is the copy: per layer
per chip, 2 x 25.07 MB read and 50.14 MB written at TP2, and one 50.14 MB scratch per chip instead
of 64 of them.

This module imports no ttnn at top level: the planners are pure python so the CPU suite pins the
index maths (test_mlp_c1e_pack.py). In the container it sits beside mlp.py
(models/demos/blackhole/qwen36/tt/mlp_c1e_pack.py) with its kernels next to it (RUNTIME_FILES;
lever_n_m3native_patch.C1E_FILES is the one mount table the graft, the arm and the card-B harness
optimisation/ttnn-op/c1e_gateup read).
"""

import hashlib
from pathlib import Path

TILE = 32
PAGE = 576            # one bfloat4_b 32x32 tile: 1024 x 4-bit mantissas (512 B) + 64 shared exponents
BATCH = 32            # pairs per batch per RISC: 64 pages, 36,864 B of L1 staging
ALIGN_SLACK = 64      # the kernel aligns its staging base up to 64 B
CB_PAGE = 2048        # the reservation CB's page (a plain L1 reservation; no CB handshake is used)
PROCESSORS = 2        # RISCV_0 and RISCV_1 on every worker core

KERNEL = 'mlp_c1e_pack.cpp'
CHECK_KERNEL = 'packed_weight_check.cpp'
KERNEL_FILES = (KERNEL, CHECK_KERNEL)
# Everything the op needs at run time, by basename (the check kernel serves the optional audit).
RUNTIME_FILES = (Path(__file__).name,) + KERNEL_FILES

CHECK_SENTINEL = 0x514B5631   # packed_weight_check.cpp's coverage word
CHECK_MAX_WORKERS = 64


class Unsupported(ValueError):
    """Raised before any device work when a tensor is outside what the copy implements."""


# ---------------------------------------------------------------------------------------------
# Pure planners (CPU-tested).
# ---------------------------------------------------------------------------------------------

def _rank2(shape):
    shape = tuple(int(value) for value in shape)
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    return shape


def geometry(separate_shape, packed_shape=None):
    """(k_tiles, columns, pairs) for one projection shard [K, N] and its packed [K, 2N]."""
    separate = _rank2(separate_shape)
    if len(separate) != 2 or any(value <= 0 or value % TILE for value in separate):
        raise ValueError('a tile-aligned [K, N] projection shard is required, got %r' % (tuple(separate_shape),))
    if packed_shape is not None and _rank2(packed_shape) != (separate[0], 2 * separate[1]):
        raise ValueError('packed %r is not [K, 2N] of %r' % (tuple(packed_shape), tuple(separate_shape)))
    k_tiles, columns = separate[0] // TILE, separate[1] // TILE
    return k_tiles, columns, k_tiles * columns


def packed_page(pair, columns, offset):
    """The packed page that separate page `pair` of w1 (offset 0) or w3 (offset 1) lands on."""
    if offset not in (0, 1) or type(offset) is not int:
        raise ValueError('offset 0 (gate / w1) or 1 (up / w3)')
    return (pair // columns) * columns * 2 + (pair % columns) * 2 + offset


def separate_page(page, columns):
    """(offset, pair): which w1 / w3 page a packed page comes from (the inverse of packed_page)."""
    row, within = divmod(page, 2 * columns)
    return within % 2, row * columns + within // 2


def split(pairs, workers):
    """Balanced contiguous ranges [(start, count)] covering 0..pairs-1 once; trailing workers may be empty."""
    if workers <= 0 or pairs < 0:
        raise ValueError('at least one worker and a non-negative pair count')
    return [((pairs * w) // workers, (pairs * (w + 1)) // workers - (pairs * w) // workers) for w in range(workers)]


def core_coordinates(grid_x, grid_y):
    """Every worker core, row-major over the compute grid: core i -> (i % grid_x, i // grid_x)."""
    return [(core % grid_x, core // grid_x) for core in range(grid_x * grid_y)]


def worker_layout(grid_x, grid_y, pairs):
    """{processor: [((x, y), (start, count))]}: worker 2*core + processor owns split()'s range."""
    cores = core_coordinates(grid_x, grid_y)
    ranges = split(pairs, len(cores) * PROCESSORS)
    return {processor: [(core, ranges[PROCESSORS * index + processor]) for index, core in enumerate(cores)]
            for processor in range(PROCESSORS)}


def cb_bytes(batch=BATCH):
    """One RISC's staging reservation: 2*batch pages plus the alignment slack, in whole CB pages."""
    need = 2 * batch * PAGE + ALIGN_SLACK
    return -(-need // CB_PAGE) * CB_PAGE


def traffic_bytes(separate_shape):
    """(read, written) DRAM bytes of one copy: both projections read once, the packed tensor written once."""
    _, _, pairs = geometry(separate_shape)
    return 2 * pairs * PAGE, 2 * pairs * PAGE


def source_sha(directory=None):
    """First 8 hex of sha256 over the copy kernel, passed as a define (C1E_SRC_SHA) so a revised
    kernel at the same path never reuses a stale JIT binary."""
    directory = Path(directory) if directory is not None else Path(__file__).parent
    return hashlib.sha256((directory / KERNEL).read_bytes()).hexdigest()[:8]


def check_geometry(pages):
    """(workers, pages) of the packed_weight_check comparison: min(64, pages) workers, strided."""
    if pages <= 0:
        raise ValueError('at least one page to compare')
    return min(CHECK_MAX_WORKERS, pages), pages


def mesh_coordinates(shape):
    rows, cols = shape
    return [(index // cols, index % cols) for index in range(rows * cols)]


def mesh_shape(mesh):
    """(rows, cols) of a MeshDevice; a device without a shape is one chip."""
    try:
        return tuple(int(value) for value in mesh.shape)
    except (AttributeError, TypeError):
        return (1, 1)


# ---------------------------------------------------------------------------------------------
# Validation (no device work).
# ---------------------------------------------------------------------------------------------

def _plain_dram_bf4(operations, tensor):
    return (tensor.dtype == operations.bfloat4_b and tensor.layout == operations.TILE_LAYOUT
            and tensor.memory_config() == operations.DRAM_MEMORY_CONFIG)


def unsupported(operations, w1, w3, scratch):
    """None when the copy can run these tensors, else the reason."""
    for name, tensor in (('w1', w1), ('w3', w3), ('scratch', scratch)):
        if tensor is None:
            return '%s is None' % name
        if not _plain_dram_bf4(operations, tensor):
            return '%s is not bfloat4_b TILE DRAM-interleaved' % name
    if _rank2(w1.shape) != _rank2(w3.shape):
        return 'w1 %s and w3 %s differ' % (tuple(w1.shape), tuple(w3.shape))
    try:
        geometry(w1.shape, scratch.shape)
    except ValueError as error:
        return str(error)
    return None


def spec(tensor):
    """The per-chip tensor spec the fused op sees (for the scratch-vs-served equality check)."""
    padded = getattr(tensor, 'padded_shape', None)
    tile = getattr(tensor, 'tile', None)
    return dict(shape=tuple(tensor.shape), padded_shape=tuple(padded) if padded is not None else None,
                dtype=str(tensor.dtype), layout=str(tensor.layout), memory_config=str(tensor.memory_config()),
                tile=tuple(tile.tile_shape) if tile is not None else None)


# ---------------------------------------------------------------------------------------------
# Device wrappers.
# ---------------------------------------------------------------------------------------------

_PREPARED = {}
_SHA = {}
_SCRATCH = {}


def clear_cache():
    _PREPARED.clear()


def cache_size():
    return len(_PREPARED)


def _sha(directory):
    key = str(directory)
    if key not in _SHA:
        _SHA[key] = source_sha(directory)
    return _SHA[key]


def _scratch_key(mesh, like):
    """(mesh, K, N): `like` is a w1 shard or its per-chip [K, N] shape."""
    shape = like if isinstance(like, (tuple, list)) else like.shape
    k, n = geometry(shape)[:2]
    return id(mesh), k * TILE, n * TILE


def has_scratch(mesh, like):
    return _scratch_key(mesh, like) in _SCRATCH


def allocate_scratch(operations, mesh, like, *, served_topology=True):
    """ONE scratch per mesh and projection shape, [K, 2N] per chip of `like` (a w1 shard, or its
    per-chip [K, N] shape). Allocate it while the weights load (before the KV cache), so it never
    splits the KV region; every later call returns the same tensor.

    served_topology (the model's choice): made the way mlp.py _build_gate_up makes w_gate_up - a
    host [K, 2N x chips] bf16 tensor (zeros here) converted to bfloat4_b, TILE, DRAM and split by
    ShardTensorToMesh(dim=-1) - so the fused op sees the served tensor's spec AND mesh placement,
    not merely its per-chip shape. Otherwise a plain ttnn.empty of the per-chip shape."""
    key = _scratch_key(mesh, like)
    if key not in _SCRATCH:
        _, k, n = key
        if served_topology:
            import torch

            chips = len(mesh_coordinates(mesh_shape(mesh)))
            _SCRATCH[key] = operations.from_torch(
                torch.zeros((k, 2 * n * chips), dtype=torch.bfloat16), dtype=operations.bfloat4_b,
                layout=operations.TILE_LAYOUT, device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG,
                mesh_mapper=operations.ShardTensorToMesh(mesh, dim=-1))
        else:
            _SCRATCH[key] = operations.empty((k, 2 * n), dtype=operations.bfloat4_b, layout=operations.TILE_LAYOUT,
                                             device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    return _SCRATCH[key]


def release_scratch(operations, mesh):
    """Deallocate and forget every scratch of this mesh (tests and harness teardown)."""
    for key in [key for key in _SCRATCH if key[0] == id(mesh)]:
        operations.deallocate(_SCRATCH.pop(key))


class _Prepared:
    """Built once per (mesh, grid, geometry, accessor args): the core set, the two CBs and, per chip,
    the two KernelDescriptors with their constant per-core [start, count]. Per call only the common
    runtime args (three addresses) change: generic_op hashes their count, not their values, so
    every call after the first is a program-cache hit."""

    def __init__(self, operations, mesh, chips, columns, pairs, accessor_args, defines, directory):
        grid = mesh.compute_with_storage_grid_size()
        core_set = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0),
                                                                 operations.CoreCoord(grid.x - 1, grid.y - 1))])
        self.cbs = [operations.CBDescriptor(
            total_size=cb_bytes(), core_ranges=core_set,
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
                                                              page_size=CB_PAGE,
                                                              tile=operations.TileDescriptor(operations.Tile([TILE, TILE])))])
            for index in range(PROCESSORS)]
        layout = worker_layout(grid.x, grid.y, pairs)
        processors = (operations.DataMovementProcessor.RISCV_0, operations.DataMovementProcessor.RISCV_1)
        nocs = (operations.NOC.RISCV_0_default, operations.NOC.RISCV_1_default)
        self.chips = []
        for chip in range(chips):
            kernels = []
            for processor in range(PROCESSORS):
                per_core = operations.RuntimeArgs()
                for (x, y), (start, count) in layout[processor]:
                    per_core[x][y] = [start, count]
                kernels.append(operations.KernelDescriptor(
                    kernel_source=str(directory / KERNEL), core_ranges=core_set,
                    compile_time_args=[PAGE, columns, BATCH, processor] + list(accessor_args[chip]),
                    defines=list(defines), runtime_args=per_core,
                    config=operations.DataMovementConfigDescriptor(processor=processors[processor],
                                                                   noc=nocs[processor])))
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


def pack_gate_up(mesh, w1, w3, scratch, *, operations=None, directory=None):
    """Rewrite `scratch` as the served tile-pair-interleaved [w1 | w3] packing, on every chip; returns
    `scratch`. One generic_op; nothing is allocated (trace-safe once compiled)."""
    if operations is None:
        import ttnn as operations
    reason = unsupported(operations, w1, w3, scratch)
    if reason is not None:
        raise Unsupported(reason)
    directory = Path(directory) if directory is not None else Path(__file__).parent
    _, columns, pairs = geometry(w1.shape, scratch.shape)
    coordinates = mesh_coordinates(mesh_shape(mesh))
    shards = [operations.get_device_tensors(tensor) for tensor in (w1, w3, scratch)]
    if any(len(parts) != len(coordinates) for parts in shards):
        raise Unsupported('every tensor needs one shard per mesh device')
    accessor_args, commons = [], []
    for chip in range(len(coordinates)):
        local = [parts[chip] for parts in shards]
        addresses = [tensor.buffer_address() for tensor in local]
        if addresses[2] in addresses[:2]:
            raise Unsupported('the scratch must not alias w1 or w3')
        accessor_args.append(tuple(argument for tensor in local
                                   for argument in operations.TensorAccessorArgs(tensor).get_compile_time_args()))
        commons.append(addresses)
    defines = (('C1E_SRC_SHA', '0x' + _sha(directory)),)
    grid = mesh.compute_with_storage_grid_size()
    key = (id(mesh), mesh_shape(mesh), grid.x, grid.y, columns, pairs, tuple(accessor_args), defines, str(directory))
    prepared = _PREPARED.get(key)
    if prepared is None:
        prepared = _PREPARED[key] = _Prepared(operations, mesh, len(coordinates), columns, pairs, accessor_args,
                                              defines, directory)
    operations.generic_op([w1, w3, scratch], prepared.program(operations, coordinates, commons))
    return scratch


def check_pairs(mesh, packed, separate, offset, owned, *, operations=None, directory=None):
    """packed_weight_check.cpp on any mesh: packed page packed_page(p, C, offset) vs separate page p,
    word for word, on every chip. Returns the [workers*32, 32] uint32 result (read_check reads it)."""
    if operations is None:
        import ttnn as operations
    directory = Path(directory) if directory is not None else Path(__file__).parent
    for tensor in (packed, separate):
        if not _plain_dram_bf4(operations, tensor):
            raise Unsupported('interleaved tiled bfloat4_b weights required')
    _, columns, pages = geometry(separate.shape, packed.shape)
    workers, _ = check_geometry(pages)
    grid = mesh.compute_with_storage_grid_size()
    width = min(grid.x, workers)
    if grid.y < -(-workers // width):
        raise Unsupported('the comparison workers exceed the grid')
    cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(w % width, w // width),
                                                          operations.CoreCoord(w % width, w // width))
                                     for w in range(workers)])
    output = operations.empty((1, 1, workers * TILE, TILE), dtype=operations.uint32, layout=operations.TILE_LAYOUT,
                              device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    owned.append(output)
    buffer = operations.CBDescriptor(total_size=8192, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=0, data_format=operations.uint32,
            page_size=4096, tile=operations.TileDescriptor(operations.Tile([TILE, TILE])))])
    coordinates = mesh_coordinates(mesh_shape(mesh))
    shards = [operations.get_device_tensors(tensor) for tensor in (packed, separate, output)]
    if any(len(parts) != len(coordinates) for parts in shards):
        raise Unsupported('every tensor needs one shard per mesh device')
    program = operations.MeshProgramDescriptor()
    for chip, (row, col) in enumerate(coordinates):
        local = [parts[chip] for parts in shards]
        runtime = operations.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % width][worker // width] = ([tensor.buffer_address() for tensor in local]
                                                         + [worker, workers, columns, pages, offset])
        reader = operations.KernelDescriptor(kernel_source=str(directory / CHECK_KERNEL), core_ranges=cores,
            runtime_args=runtime,
            compile_time_args=[argument for tensor in local
                               for argument in operations.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                           noc=operations.NOC.RISCV_0_default))
        coordinate = operations.MeshCoordinate(row, col)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=[reader], cbs=[buffer])
    operations.generic_op([packed, separate, output], program)
    return output


def read_check(operations, result, pages):
    """[{chip, pages, mismatched_words, first_page, first_word, exact}] from check_pairs' result, with
    packed_weight_check's coverage, integrity and canary checks (AssertionError when one fails)."""
    import torch

    workers, _ = check_geometry(pages)
    checks = []
    for chip, shard in enumerate(operations.get_device_tensors(result)):
        values = operations.to_torch(shard).long().reshape(workers, TILE, TILE) & 0xFFFFFFFF
        counts = values[:, 0, :6]
        for worker in range(workers):
            if int(counts[worker, 1]) != len(range(worker, pages, workers)) or int(counts[worker, 4]) != CHECK_SENTINEL:
                raise AssertionError('incomplete device comparison coverage (chip %d worker %d)' % (chip, worker))
            if int(counts[worker, 5]) != (int(counts[worker, 0]) ^ 0xFFFFFFFF):
                raise AssertionError('comparison counter readback integrity failed (chip %d worker %d)' % (chip, worker))
        padding = values.clone()
        padding[:, 0, :6] = 0
        if torch.count_nonzero(padding):
            raise AssertionError('comparison output canary changed (chip %d)' % chip)
        mismatches = int(counts[:, 0].sum())
        firsts = [(int(counts[w, 2]) - 1, int(counts[w, 3]) - 1) for w in range(workers) if int(counts[w, 2])]
        first = min(firsts) if firsts else (None, None)
        checks.append(dict(chip=chip, pages=int(counts[:, 1].sum()), mismatched_words=mismatches,
                           first_page=first[0], first_word=first[1], exact=mismatches == 0))
    return checks


def audit_pairs(operations, mesh, packed, w1, w3, *, directory=None):
    """Is `packed` the served tile-pair interleave of (w1, w3) on every chip, byte for byte? Two
    packed_weight_check launches (gate pages vs w1, up pages vs w3); the result tensors are freed
    here. {exact, pages, gate: read_check(...), up: read_check(...), mismatched_words}.

    Allocates and reads back, so the model calls it only on its audit path (QWEN_FAST_C1_EXACT_AUDIT:
    at load, and for the first packs of the first, eager, forward) - never inside a trace capture."""
    _, _, pages = geometry(w1.shape, packed.shape)
    report = dict(pages=pages)
    for name, separate, offset in (('gate', w1, 0), ('up', w3, 1)):
        owned = []
        try:
            result = check_pairs(mesh, packed, separate, offset, owned, operations=operations, directory=directory)
            report[name] = read_check(operations, result, pages)
        finally:
            for tensor in owned:
                operations.deallocate(tensor)
    checks = report['gate'] + report['up']
    report['mismatched_words'] = sum(item['mismatched_words'] for item in checks)
    report['exact'] = bool(checks) and all(item['exact'] for item in checks)
    return report
