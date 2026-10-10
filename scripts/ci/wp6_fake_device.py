"""A small FUNCTIONAL fake of the ttnn surface the tp4/fx-wp6 drafter levers use (test support only: not imported by any served module, not in the
image copy lists).

Tensors carry real values: a device tensor is a logical shape, a dtype name ('fp32', 'bf16'), a layout and one torch tensor per chip. Tile pages
(page = (batch * tile rows + tile row) * tile columns + tile column, tiles padded up to 32 x 32 with zeros) are what the kernels address, so a
launch built by the levers can be EXECUTED by the transliterations in the tests, which read the launch's own runtime arguments, circular buffers
and compile-time arguments exactly as the .cpp files do. Ops execute on the host with torch: fp32 adds and multiplies are IEEE float32, a
typecast to bf16 is torch's round-to-nearest-even, silu is torch's float32 silu (the one primitive CPU cannot hold bit for bit against the
SFPU; both sides of a comparison share it). The matmul reduces K in `in0_block_w` tile blocks, each block summed exactly (float64) and
accumulated into an fp32 accumulator in block order, so a column's value depends on its own K loop only.
"""

from types import SimpleNamespace

import torch

TILE = 32
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}


def pad_tiles(data):
    """`data` (batch, 1, rows, width) padded with zeros to whole 32 x 32 tiles."""
    batch, one, rows, width = data.shape
    rows_p, width_p = -(-rows // TILE) * TILE, -(-width // TILE) * TILE
    if (rows_p, width_p) == (rows, width):
        return data
    out = torch.zeros(batch, one, rows_p, width_p, dtype=data.dtype)
    out[:, :, :rows, :width] = data
    return out


class Shard:
    """One chip's tensor."""

    counter = [0]

    def __init__(self, data, dtype, layout, owner=None):
        Shard.counter[0] += 1
        self.data = data
        self.dtype = dtype
        self.layout = layout
        self.address = 4096 * Shard.counter[0]
        self.shape = tuple(data.shape)
        self.owner = owner
        self.freed = False

    def buffer_address(self):
        return self.address

    def memory_config(self):
        return 'dram'

    def pages(self):
        """The tile pages of this chip's data: [batch][tile row][tile column] -> a (32, 32) tensor, as a flat list in page order."""
        padded = pad_tiles(self.data if self.data.dim() == 4 else self.data.reshape(1, 1, *self.data.shape))
        batch, _, rows, width = padded.shape
        return [padded[b, 0, 32 * r:32 * r + 32, 32 * c:32 * c + 32] for b in range(batch)
                for r in range(rows // TILE) for c in range(width // TILE)]


class Tensor:
    def __init__(self, shape, dtype, layout, shards, name='t'):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.shards = shards
        self.name = name
        self.freed = False
        for shard in shards:
            shard.owner = self

    def memory_config(self):
        return 'dram'

    def __repr__(self):
        return '<%s %s %s>' % (self.name, self.shape, self.dtype)


class RuntimeArgs(dict):
    def __getitem__(self, key):
        return self.setdefault(key, {})


class FakeOperations:
    float32, bfloat16, bfloat8_b, uint32 = 'fp32', 'bf16', 'bf8', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'tile', 'row_major'
    DRAM_MEMORY_CONFIG = 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    NOC = SimpleNamespace(RISCV_0_default=0, RISCV_1_default=1)
    DataMovementProcessor = SimpleNamespace(RISCV_0=0, RISCV_1=1)
    UnpackToDestMode = SimpleNamespace(Default='default', UnpackToDestFp32='fp32')
    Topology = SimpleNamespace(Linear='linear', Ring='ring')
    RuntimeArgs = RuntimeArgs
    MeshProgramDescriptor = dict
    CoreCoord = staticmethod(lambda x, y: (x, y))
    CoreRange = staticmethod(lambda a, b: (a, b))
    CoreRangeSet = staticmethod(lambda ranges: list(ranges))
    Tile = staticmethod(lambda shape: tuple(shape))
    TileDescriptor = staticmethod(lambda shape: shape)
    CBDescriptor = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
    CBFormatDescriptor = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
    KernelDescriptor = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
    DataMovementConfigDescriptor = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
    ProgramDescriptor = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))
    MeshCoordinate = staticmethod(lambda *args: tuple(args))
    MeshCoordinateRange = staticmethod(lambda a, b: (a, b))

    def __init__(self, chips=4):
        self.chips = chips
        self.log = []
        self.tensors = []
        self.launches = []
        self.emulators = []                      # callables (operations, tensors, program) -> True when they executed the program
        self.experimental = SimpleNamespace(all_gather_async=self.all_gather_async)
        self.sync_count = 0
        self.programs = []                       # the program config of every matmul, in order
        self.grid_noise = None                   # a callable (grid, result data) -> data: a 'bad hardware' model for the grid audit

    # -- descriptors --------------------------------------------------------------------------------------------------
    @staticmethod
    def ComputeConfigDescriptor(**kwargs):
        return SimpleNamespace(unpack_to_dest_mode=[], **kwargs)

    @staticmethod
    def TensorAccessorArgs(tensor):
        return SimpleNamespace(get_compile_time_args=lambda: [1, 0])

    @staticmethod
    def MatmulMultiCoreReuseMultiCast1DProgramConfig(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def WormholeComputeKernelConfig(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def ShardTensorToMesh(mesh, dim):
        return ('shard', dim)

    @staticmethod
    def ReplicateTensorToMesh(mesh):
        return ('replicate',)

    # -- tensors ------------------------------------------------------------------------------------------------------
    def _make(self, datas, dtype, layout, name):
        tensor = Tensor(datas[0].shape, dtype, layout, [Shard(data, dtype, layout) for data in datas], name)
        self.tensors.append(tensor)
        return tensor

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        kind = mesh_mapper[0] if mesh_mapper else 'replicate'
        if kind == 'shard':
            parts = list(value.chunk(self.chips, dim=mesh_mapper[1]))
        else:
            parts = [value] * self.chips
        target = DTYPES.get(dtype, torch.bfloat16)                 # 'bf8' weights are bf16 here: both sides quantise alike
        datas = [part.to(target).clone() for part in parts]
        return self._make(datas, dtype, layout, 'from_torch')

    def from_chips(self, datas, dtype='fp32', layout='tile', name='in'):
        """A tensor from one torch tensor per chip (test setup)."""
        torch_type = DTYPES[dtype]
        return self._make([data.to(torch_type).clone() for data in datas], dtype, layout, name)

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        shape = tuple(shape)
        self.log.append(('empty', shape, dtype))
        return self._make([torch.zeros(shape, dtype=DTYPES.get(dtype, torch.float32)) for _ in range(self.chips)], dtype, layout, 'empty')

    def get_device_tensors(self, tensor):
        return tensor.shards

    def to_torch(self, value, **options):
        if isinstance(value, Shard):
            return value.data.clone()
        return torch.stack([shard.data for shard in value.shards])

    def deallocate(self, tensor):
        if tensor.freed:
            raise AssertionError('deallocate of an already freed tensor %r' % (tensor,))
        tensor.freed = True
        self.log.append(('deallocate', tensor.name))

    def synchronize_device(self, mesh):
        self.sync_count += 1

    # -- ops (per-chip torch) ---------------------------------------------------------------------------------------------
    def _map(self, name, tensor, fn, dtype, **extra):
        self.log.append((name, tensor.shape))
        return self._make([fn(shard.data) for shard in tensor.shards], dtype, tensor.layout, name)

    def slice(self, tensor, start, end, **options):
        self.log.append(('slice', tuple(start), tuple(end)))
        index = tuple(slice(a, b) for a, b in zip(start, end))
        return self._make([shard.data[index].clone() for shard in tensor.shards], tensor.dtype, tensor.layout, 'slice')

    def typecast(self, tensor, dtype, **options):
        self.log.append(('typecast', tensor.shape, tensor.dtype, dtype))
        return self._make([shard.data.to(DTYPES[dtype]) for shard in tensor.shards], dtype, tensor.layout, 'typecast')

    def add(self, left, right, dtype=None, memory_config=None, **options):
        self.log.append(('add', left.shape, left.dtype, right.dtype))
        if left.dtype == 'fp32' and right.dtype == 'fp32':
            out = [a.data + b.data for a, b in zip(left.shards, right.shards)]
            return self._make(out, 'fp32', left.layout, 'add')
        out = [(a.data.float() + b.data.float()) for a, b in zip(left.shards, right.shards)]
        return self._make([o.to(DTYPES[dtype]) for o in out], dtype, left.layout, 'add')

    def multiply(self, left, right, dtype=None, memory_config=None, **options):
        self.log.append(('multiply', left.shape))
        return self._make([a.data * b.data for a, b in zip(left.shards, right.shards)], dtype or left.dtype, left.layout, 'multiply')

    def silu(self, tensor, memory_config=None, **options):
        self.log.append(('silu', tensor.shape))
        return self._make([torch.nn.functional.silu(shard.data) for shard in tensor.shards], tensor.dtype, tensor.layout, 'silu')

    def rms_norm(self, tensor, epsilon=1e-6, weight=None, compute_kernel_config=None, memory_config=None, **options):
        self.log.append(('rms_norm', tensor.shape))
        out = []
        for shard in tensor.shards:
            x = shard.data.float()
            out.append((x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)).to(shard.data.dtype))
        return self._make(out, tensor.dtype, tensor.layout, 'rms_norm')

    def matmul(self, left, right, dtype=None, program_config=None, compute_kernel_config=None, memory_config=None, **options):
        self.log.append(('matmul', left.shape, right.shape, getattr(program_config, 'per_core_N', None)))
        self.programs.append(program_config)
        block = TILE * getattr(program_config, 'in0_block_w', 4)
        out = []
        for a, b in zip(left.shards, right.shards):
            x = a.data.reshape(-1, a.data.shape[-1])
            w = b.data
            accumulator = torch.zeros(x.shape[0], w.shape[1], dtype=torch.float32)
            for start in range(0, x.shape[1], block):
                partial = (x[:, start:start + block].double() @ w[start:start + block].double()).float()
                accumulator = accumulator + partial
            result = accumulator.reshape(*a.data.shape[:-1], w.shape[1])
            if self.grid_noise is not None:
                result = self.grid_noise(getattr(program_config, 'compute_with_storage_grid_size', None), result)
            out.append(result)
        return self._make(out, dtype or 'fp32', left.layout, 'matmul')

    def all_gather_async(self, tensor, persistent_output_buffer=None, dim=0, multi_device_global_semaphore=None,
                         barrier_semaphore=None, num_links=1, memory_config=None, topology=None, chunks_per_sync=10,
                         num_workers_per_link=2, num_buffers_per_channel=2, **options):
        self.log.append(('all_gather', tensor.shape, dim))
        gathered = torch.cat([shard.data for shard in tensor.shards], dim=dim)
        return self._make([gathered.clone() for _ in range(self.chips)], tensor.dtype, tensor.layout, 'gathered')

    # -- launches -----------------------------------------------------------------------------------------------------
    def generic_op(self, tensors, program):
        self.log.append(('generic_op', len(program)))
        self.launches.append((tensors, program))
        for emulator in self.emulators:
            if emulator(self, tensors, program):
                return
        raise AssertionError('no emulator took this launch: %s' % [kernel.kernel_source for p in program.values() for kernel in p.kernels])

    def names(self, *kinds):
        return [entry[0] for entry in self.log if not kinds or entry[0] in kinds]


def chip_of(key):
    """The chip index of a MeshCoordinateRange key ((0, chip), (0, chip))."""
    return key[0][1]


def shard_by_address(tensors, chip, address):
    """The chip's shard among `tensors` whose buffer address is `address`."""
    for tensor in tensors:
        shard = tensor.shards[chip]
        if shard.address == address:
            return shard
    raise AssertionError('no tensor of the launch lives at %#x on chip %d' % (address, chip))


def kernels_by_name(program_for_chip):
    """{kernel file basename: kernel descriptor} of one chip's ProgramDescriptor."""
    found = {}
    for kernel in program_for_chip.kernels:
        found[kernel.kernel_source.rsplit('/', 1)[-1]] = kernel
    return found


def cores_of(kernel):
    """[(x, y)] with runtime arguments, in (x, y) order."""
    return [(x, y) for x in sorted(kernel.runtime_args) for y in sorted(kernel.runtime_args[x])]


def write_page(shard, page, tile):
    """Write a (32, 32) tile into page `page` of the shard's (batch, 1, rows, width) data (the logical area only)."""
    data = shard.data
    batch, _, rows, width = data.shape
    tile_rows, tile_columns = -(-rows // TILE), -(-width // TILE)
    b, rest = divmod(page, tile_rows * tile_columns)
    r, c = divmod(rest, tile_columns)
    top, left = 32 * r, 32 * c
    view = data[b, 0, top:top + 32, left:left + 32]
    view.copy_(tile[:view.shape[0], :view.shape[1]])


def read_page(shard, page, cache={}):
    key = (shard.address, id(shard.data), shard.data._version)
    if key not in cache:
        cache.clear()
        cache[key] = shard.pages()
    return cache[key][page]


# ---------------------------------------------------------------------------------------------------------------------
# Transliterations of the kernels. Each reads the launch's own descriptors the way the .cpp reads its arguments (the
# argument indices, the page arithmetic, the compile-time argument order) and does the .cpp's arithmetic tile by tile.
# ---------------------------------------------------------------------------------------------------------------------

def round_bf16(x):
    """typecast_tile<Float32, Float16_b> on an fp32 tile held in a destination register: round to nearest even to the top 16 bits and keep
    the result as an fp32 lane (no dtype conversion anywhere - integer arithmetic on the bit pattern). Finite values only (an overflow
    rounds to infinity as the hardware's does); NaN and infinity keep their top bits."""
    bits = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) & 0xFFFF0000)
    special = ((bits >> 23) & 0xFF) == 0xFF
    out = torch.where(special, bits & 0xFFFF0000, rounded)
    return _int_to_float(out)


def _int_to_float(values):
    """int64 values in [0, 2^32) as the fp32 lanes with those bit patterns."""
    signed = torch.where(values >= (1 << 31), values - (1 << 32), values).to(torch.int32)
    return signed.view(torch.float32)


def check_buffers(program_for_chip, expect):
    """The circular buffers of the launch: {index: (pages, page bytes)} must be exactly `expect` (a CB is total_size = pages * page_size)."""
    found = {}
    for buffer in program_for_chip.cbs:
        for descriptor in buffer.format_descriptors:
            assert buffer.total_size % descriptor.page_size == 0
            found[descriptor.buffer_index] = (buffer.total_size // descriptor.page_size, descriptor.page_size)
    assert found == expect, (found, expect)


def emulate_reduce(operations, tensors, program, grid=(11, 10)):
    """draft_reduce_tp_io.cpp + draft_reduce_tp_compute.cpp + draft_fuse_out.cpp."""
    touched = 0
    for key, chip_program in program.items():
        kernels = kernels_by_name(chip_program)
        if 'draft_reduce_tp_io.cpp' not in kernels:
            return False
        chip = chip_of(key)
        reader, writer, compute = kernels['draft_reduce_tp_io.cpp'], kernels['draft_fuse_out.cpp'], kernels['draft_reduce_tp_compute.cpp']
        chips = reader.compile_time_args[-1]
        assert compute.compile_time_args == [chips] and chips in (2, 3, 4)
        check_buffers(chip_program, {0: (4 * 2, 4096), 16: (2, 4096)})
        assert cores_of(reader) == cores_of(writer) == cores_of(compute)
        for x, y in cores_of(reader):
            assert x < grid[0] and y < grid[1]
            in_address, first, count, stride = reader.runtime_args[x][y]
            out_address, out_first, out_count = writer.runtime_args[x][y]
            (compute_count,) = compute.runtime_args[x][y]
            assert (out_first, out_count) == (first, count) and compute_count == count
            source = shard_by_address(tensors, chip, in_address)
            target = shard_by_address(tensors, chip, out_address)
            for tile in range(first, first + count):
                pages = [read_page(source, c * stride + tile).clone() for c in range(chips)]       # CB 0: four pages
                accumulator = pages[0]                                                             # destination tile 0
                for c in range(1, chips):
                    accumulator = accumulator + pages[c]                                           # add_binary_tile(0, c, 0)
                write_page(target, tile, accumulator)                                              # CB 16 -> writer
                touched += 1
    emulate_reduce.touched = touched
    return True


def emulate_tail(operations, tensors, program, grid=(11, 10)):
    """draft_tail_tp_io.cpp + draft_tail_tp_{swiglu,residual}_compute.cpp + draft_fuse_out.cpp."""
    touched = 0
    for key, chip_program in program.items():
        kernels = kernels_by_name(chip_program)
        if 'draft_tail_tp_io.cpp' not in kernels:
            return False
        chip = chip_of(key)
        reader, writer = kernels['draft_tail_tp_io.cpp'], kernels['draft_fuse_out.cpp']
        swiglu = 'draft_tail_tp_swiglu_compute.cpp' in kernels
        compute = kernels['draft_tail_tp_swiglu_compute.cpp' if swiglu else 'draft_tail_tp_residual_compute.cpp']
        a_bytes, b_bytes = reader.compile_time_args[-2:]
        assert (a_bytes, b_bytes) == ((4096, 4096) if swiglu else (2048, 2048))
        check_buffers(chip_program, {0: (2, a_bytes), 1: (2, b_bytes), 16: (2, 2048)})
        assert cores_of(reader) == cores_of(writer) == cores_of(compute)
        for x, y in cores_of(reader):
            assert x < grid[0] and y < grid[1]
            a_address, b_address, first, count, a_stride, b_stride, b_col, columns = reader.runtime_args[x][y]
            out_address, out_first, out_count = writer.runtime_args[x][y]
            (compute_count,) = compute.runtime_args[x][y]
            assert (out_first, out_count) == (first, count) and compute_count == count
            a_shard = shard_by_address(tensors, chip, a_address)
            b_shard = shard_by_address(tensors, chip, b_address)
            target = shard_by_address(tensors, chip, out_address)
            row, column = divmod(first, columns)
            for task in range(count):
                a = read_page(a_shard, row * a_stride + column).clone()
                b = read_page(b_shard, row * b_stride + b_col + column).clone()
                if swiglu:
                    gate, up = a.float(), b.float()                       # CB 0, CB 1 unpacked to destination tiles 0 and 1
                    gate = round_bf16(gate)                                # typecast_tile(0)
                    gate = torch.nn.functional.silu(gate)                  # silu_tile(0)
                    gate = round_bf16(gate)                                # typecast_tile(0)
                    up = round_bf16(up)                                    # typecast_tile(1)
                    result = round_bf16(gate * up)                         # mul_binary_tile(0, 1, 0); typecast_tile(0)
                else:
                    result = round_bf16(a.float() + b.float())             # add_binary_tile(0, 1, 0); typecast_tile(0)
                write_page(target, row * columns + column, result.to(torch.bfloat16))     # pack as bf16 (already bf16-exact)
                touched += 1
                column += 1
                if column == columns:
                    column, row = 0, row + 1
    emulate_tail.touched = touched
    return True


def install_emulators(operations, grid=(11, 10)):
    operations.emulators.append(lambda ops, tensors, program: emulate_reduce(ops, tensors, program, grid))
    operations.emulators.append(lambda ops, tensors, program: emulate_tail(ops, tensors, program, grid))
    return operations
