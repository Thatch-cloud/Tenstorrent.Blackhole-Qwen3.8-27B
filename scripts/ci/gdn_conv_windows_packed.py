"""gdn_conv_windows_packed: every packed user's four causal-conv windows in ONE generic_op.

Verify-trace T2 cut #1 (verify_trace_t2.py, QWEN_FAST_VERIFY_T2). It replaces, byte for byte,
the per-user launch the batched packed block runs once per user and GDN layer
(gdn_user_batch_conv.run_user_batched_projected):

    windows = gdn_conv_windows.build_windows(mesh, piece, history)     # x 4 users

Each served launch runs gdn_conv_windows.cpp on the audited 8x6 grid (48 workers) over the 160
channel pages; per page it reads the piece tile and the four history tiles, then builds each
window slot s in a zeroed scratch page (row r of face f <- history r + s row 0 while r + s < 4,
else piece row r + s - 4), writes it and waits on a write barrier - four times. 72.3 us each,
bound by latency. Here the 4 x 160 (user, page) tasks go to every core of the grid (110 on a
p150a) in one launch, reading the SAME tensors at the same page and byte offsets and writing
the same 16 per-user window tensors, (1, 16, 5120) bf16 TILE DRAM, so the conv gates op and the
commit DMA (packed_conv_states) see nothing new.

Exactness (class B). What each output page holds, for user u, slot s, page p (2048 bytes, four
512-byte faces, row r of face f at byte f * 512 + r * 32):
  faces 2-3            all 0x00 (the served scratch is zeroed and no token >= 16 exists);
  face f, row r < 16   32 bytes: history H[u][r + s] face f row 0 while r + s < 4, else piece
                       P[u] face f row r + s - 4 - one contiguous block of (12 + s) * 32 bytes
                       from piece row 0 on (window_copies).
The op moves those bytes with RISC-V uint32 word copies (the served primitive) or local NOC
copies, and NOC tile reads and writes. No compute kernel runs, so nothing is packed, unpacked,
rounded or canonicalised: -0, denormals and NaN payloads travel as bits. The pieces are read as
the served path cut them (the pieces of users 1 and 3 come through the served untilize / slice /
tilize round trip, upstream of both paths), so nothing is added or removed there either.

Trims (all exact, every one checked on card M against the served kernel re-driven):
  - reader and writer on the two data-movement RISCs, through a double-buffered CB;
  - faces 2-3 of the scratch pages zeroed ONCE per core - legal only while rows == 16, which
    the host refuses otherwise and the kernel static_asserts;
  - one write barrier per task (nbuf 1) or none until the end (nbuf 2: a flush before a scratch
    set is reused);
  - hist_row: only the history bytes the windows use (face 0 row 0 and face 1 row 0, read as two
    64-byte spans per history page) instead of the whole 2 KB page.
VTW_PORT (settings port=True) is the served kernel transcribed literally onto (user, page)
tasks over the whole grid: the timing anchor and the first card-M target.

This module imports no ttnn at top level; the planners are pure python (CPU-tested,
test_gdn_conv_windows_packed.py). Its kernel is the SIBLING .cpp (Path(__file__).with_suffix),
so the pair travels together (test_serving_image_copy_closure).
"""

import hashlib
from pathlib import Path

PAGES = 160            # 5120 channels / 32: one tile row of every window, piece and history
ROWS = 16              # the only width the trimmed kernel serves (faces 2-3 zeroed once)
SLOTS = 4
HISTORY = 4
PAGE = 2048            # one bf16 32x32 tile
FACE_BYTES = 512
ROW_BYTES = 32
CHANNELS = 5120
TILES_IN = 1 + HISTORY  # the piece tile and four history tiles per task
NBUF_IN = 2            # CB_IN depth in tasks
CB_IN, CB_SCRATCH = 0, 1
SOURCE_NAME = 'gdn_conv_windows_packed.cpp'
RUNTIME_FILES = ('gdn_conv_windows_packed.py', SOURCE_NAME)
# What the model path runs. copy_noc is the served primitive (word copies) until card M
# measures the NOC copies faster and exact; the card-M settings verdict fails the run while
# DEFAULTS is not within 2 us of the fastest exact combination.
DEFAULTS = dict(port=False, hist_row=True, copy_noc=False, nbuf=2)
NEGATIVE_CONTROLS = {'slot': 'VTW_NEG_SLOT', 'user': 'VTW_NEG_USER', 'hist': 'VTW_NEG_HIST', 'pad': 'VTW_NEG_PAD'}
ROLES = ('VTW_ROLE_READER', 'VTW_ROLE_WRITER', 'VTW_PORT')


class Unsupported(ValueError):
    """Raised before any kernel runs when an input is outside what the op implements; the model
    side takes the served per-user path (and counts windows_fallback)."""


# ---------------------------------------------------------------------------------------------
# Pure planners (CPU-tested).
# ---------------------------------------------------------------------------------------------

def plan(users, cores):
    """Tasks t in [0, users * 160): user t // 160, page t % 160. Worker w takes the contiguous
    range [start_w, start_w + count_w), count_w = tasks // cores plus one for the first
    tasks % cores workers. 4 users on 110 cores: [6] * 90 + [5] * 20."""
    if type(users) is not int or users < 1 or type(cores) is not int or cores < 1:
        raise ValueError('positive users and cores required')
    tasks = users * PAGES
    base, extra = divmod(tasks, cores)
    ranges, start = [], 0
    for worker in range(cores):
        count = base + (1 if worker < extra else 0)
        ranges.append((start, count))
        start += count
    return dict(tasks=tasks, ranges=ranges)


def task(index):
    """(user, page) of task `index`."""
    return divmod(index, PAGES)


def window_copies(slot):
    """The copies that build window `slot` in each face (f in 0..1) of a scratch page whose
    faces 2-3 are already zero, at rows == 16: [(kind, source, destination_row, length_bytes)].

    'hist': 32 bytes of history `source` (its row 0 of the face); 'piece': length_bytes from
    piece row `source` on. Destination rows 0..3-s take history rows s..3, rows 4-s..15 take
    piece rows 0..11+s in one block - the served kernel's token loop (gdn_conv_windows.cpp:12-22)
    with its per-row copies merged where the rows are contiguous."""
    if slot not in range(SLOTS):
        raise ValueError('slot must be 0..3')
    copies = [('hist', row + slot, row, ROW_BYTES) for row in range(HISTORY - slot)]
    copies.append(('piece', 0, HISTORY - slot, (ROWS - HISTORY + slot) * ROW_BYTES))
    return copies


def served_row_map(slot, rows=ROWS):
    """The served kernel's own loop, literally: (token, history, source tile, source row,
    destination row) per token, tile 0 the piece and tile h + 1 history h."""
    out = []
    for token in range(rows):
        history = token + slot
        source_row = 0 if history < HISTORY else history - HISTORY
        tile = history + 1 if history < HISTORY else 0
        out.append((token, history, tile, source_row, token))
    return out


def common_args(pieces, histories, outputs):
    """The one common-runtime-arg layout every kernel reads (get_common_arg_val), user-major:
    P[u] (users words), then H[u][h] at users + 4u + h, then O[u][s] at 5 users + 4u + s."""
    users = len(pieces)
    if users < 1 or len(histories) != users or len(outputs) != users \
            or any(len(row) != HISTORY for row in histories) or any(len(row) != SLOTS for row in outputs):
        raise ValueError('one piece, four histories and four outputs per user required')
    return [int(value) for value in pieces] + [int(value) for row in histories for value in row] \
        + [int(value) for row in outputs for value in row]


def common_index(users, kind, user, index=0):
    """Where common_args puts one address: kind 'piece', 'history' or 'output'."""
    if kind == 'piece':
        return user
    if kind == 'history':
        return users + HISTORY * user + index
    if kind == 'output':
        return users + HISTORY * users + SLOTS * user + index
    raise ValueError(kind)


def resolve_settings(settings=None):
    """DEFAULTS overlaid with `settings`, validated. The port ignores the trims, so it is
    normalised to one canonical setting (one descriptor-cache key)."""
    resolved = dict(DEFAULTS)
    if settings:
        unknown = sorted(set(settings) - set(DEFAULTS))
        if unknown:
            raise ValueError('unknown settings %s' % unknown)
        resolved.update(settings)
    for name in ('port', 'hist_row', 'copy_noc'):
        if type(resolved[name]) is not bool:
            raise ValueError('setting %s must be a bool' % name)
    if resolved['nbuf'] not in (1, 2) or type(resolved['nbuf']) is not int:
        raise ValueError('nbuf must be 1 or 2')
    if resolved['port']:
        return dict(port=True, hist_row=False, copy_noc=False, nbuf=1)
    return resolved


def settings_matrix():
    """Every setting card M must find exact: the port, then hist_row x copy_noc x nbuf."""
    combos = [dict(port=True, hist_row=False, copy_noc=False, nbuf=1)]
    for hist_row in (False, True):
        for copy_noc in (False, True):
            for nbuf in (1, 2):
                combos.append(dict(port=False, hist_row=hist_row, copy_noc=copy_noc, nbuf=nbuf))
    return combos


def settings_name(settings):
    settings = resolve_settings(settings)
    if settings['port']:
        return 'port'
    return 'hist%d_noc%d_nbuf%d' % (int(settings['hist_row']), int(settings['copy_noc']), settings['nbuf'])


def cb_plan(settings):
    """CB index -> pages (2048 bytes each). The port keeps the served layout (5 staging tiles
    and one scratch page in c_0); the trimmed pair takes CB_IN 5 x 2 and 4 x nbuf scratch."""
    settings = resolve_settings(settings)
    if settings['port']:
        return {CB_IN: TILES_IN + 1}
    return {CB_IN: TILES_IN * NBUF_IN, CB_SCRATCH: SLOTS * settings['nbuf']}


def cb_bytes(settings):
    return sum(pages * PAGE for pages in cb_plan(settings).values())


def compile_args(users, settings, accessor_args):
    """[USERS, PAGES, ROWS, NBUF, TASKS] + the three class representatives' TensorAccessorArgs
    (piece, history, output), identical for every kernel."""
    settings = resolve_settings(settings)
    return [users, PAGES, ROWS, settings['nbuf'], users * PAGES] + [int(value) for value in accessor_args]


def source_path(directory=None):
    if directory is not None:
        return Path(directory) / SOURCE_NAME
    return Path(__file__).with_suffix('.cpp')


def source_sha(directory=None):
    """First 8 hex of sha256 over the kernel, passed as a define: defines are hashed into the
    JIT key, so a revised kernel at the same path can never reuse a stale binary."""
    return hashlib.sha256(source_path(directory).read_bytes()).hexdigest()[:8]


def kernel_defines(sha, role, settings, negative=None):
    """The sorted (name, value) defines of one kernel role."""
    if role not in ROLES:
        raise ValueError('unknown role %r' % (role,))
    settings = resolve_settings(settings)
    if (role == 'VTW_PORT') != settings['port']:
        raise ValueError('the port runs alone; the trimmed pair runs as reader and writer')
    defines = {'VTW_SRC_SHA': '0x' + sha, role: '1'}
    if settings['hist_row']:
        defines['VTW_HIST_ROW'] = '1'
    if settings['copy_noc']:
        defines['VTW_COPY_NOC'] = '1'
    if negative is not None:
        if negative not in NEGATIVE_CONTROLS:
            raise ValueError('unknown negative control %r' % (negative,))
        defines[NEGATIVE_CONTROLS[negative]] = '1'
    return tuple(sorted(defines.items()))


def roles(settings):
    return ('VTW_PORT',) if resolve_settings(settings)['port'] else ('VTW_ROLE_READER', 'VTW_ROLE_WRITER')


def core_coordinates(grid_x, grid_y, count):
    """Worker w -> (w % grid_x, w // grid_x), row-major over the compute grid."""
    if grid_x * grid_y < count:
        raise ValueError('grid too small')
    return [(worker % grid_x, worker // grid_x) for worker in range(count)]


def mesh_coordinates(shape):
    rows, cols = shape
    return [(index // cols, index % cols) for index in range(rows * cols)]


def reference_windows(piece_pages, history_pages, rows=ROWS):
    """The served kernel's output, in torch: pages as int16 (pages, 1024) views of the raw
    2048-byte tiles (face-ordered), so -0, denormals, NaN payloads and padding all count.
    piece_pages: (pages, 1024); history_pages: four (pages, 1024). Returns four (pages, 1024),
    one per slot - the served loop (gdn_conv_windows.cpp:4-27) transcribed, zeroed scratch and
    all, for any rows <= 32."""
    import torch

    tiles = [piece_pages] + list(history_pages)
    if len(tiles) != TILES_IN or any(tuple(tile.shape) != tuple(piece_pages.shape) for tile in tiles):
        raise ValueError('one piece and four histories of one page count required')
    out = []
    for slot in range(SLOTS):
        window = torch.zeros_like(piece_pages)
        for token in range(rows):
            history = token + slot
            source_row = 0 if history < HISTORY else history - HISTORY
            source = (source_row // 16) * 256 + (source_row % 16) * 16
            destination = (token // 16) * 256 + (token % 16) * 16
            tile = tiles[history + 1 if history < HISTORY else 0]
            for face in range(2):
                window[:, destination + face * 256:destination + face * 256 + 16] = \
                    tile[:, source + face * 256:source + face * 256 + 16]
        out.append(window)
    return out


def apply_copies(piece_pages, history_pages):
    """window_copies applied in torch on zeroed pages: the trimmed kernel's map (rows == 16)."""
    import torch

    out = []
    for slot in range(SLOTS):
        window = torch.zeros_like(piece_pages)
        for face in range(2):
            base = face * 256
            for kind, source, row, length in window_copies(slot):
                words = length // 2
                if kind == 'hist':
                    window[:, base + row * 16:base + row * 16 + words] = history_pages[source][:, base:base + words]
                else:
                    start = base + source * 16
                    window[:, base + row * 16:base + row * 16 + words] = piece_pages[:, start:start + words]
        out.append(window)
    return out


# ---------------------------------------------------------------------------------------------
# Validation (no device work).
# ---------------------------------------------------------------------------------------------

def _interleaved(operations, tensor):
    return tensor.memory_config() in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG)


def _kind(operations, tensor):
    return 'l1' if tensor.memory_config() == operations.L1_MEMORY_CONFIG else 'dram'


def unsupported(operations, mesh, users):
    """None when the op can run these (piece, history4) users, else the reason."""
    from gdn_multitoken_conv import validate_projected
    import gdn_user_batch

    users = list(users)
    if not 1 <= len(users) <= gdn_user_batch.MAX_USERS:
        return '%d users outside 1..%d' % (len(users), gdn_user_batch.MAX_USERS)
    for index, user in enumerate(users):
        try:
            piece, history = user
            history = list(history)
            rows = validate_projected(tuple(piece.shape), history)
        except (TypeError, ValueError) as error:
            return 'user %d: %s' % (index, error)
        if rows != ROWS:
            return 'user %d rows %d != %d' % (index, rows, ROWS)
        for tensor in [piece] + history:
            if tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT:
                return 'user %d: not bf16 TILE' % index
            if not _interleaved(operations, tensor):
                return 'user %d: not DRAM or L1 interleaved' % index
    try:
        shape = tuple(mesh.shape)
    except (AttributeError, TypeError):
        return 'no mesh shape'
    if shape not in ((1, 1), (1, 2)):
        return 'mesh %s is not [1, 1] or [1, 2]' % (shape,)
    return None


def class_args(operations, shards, chips):
    """Per chip, the one TensorAccessorArgs every member of a class yields, or raise Unsupported."""
    out = []
    for chip in range(chips):
        found = {tuple(operations.TensorAccessorArgs(parts[chip]).get_compile_time_args()) for parts in shards}
        if len(found) != 1:
            raise Unsupported('class accessor args differ on chip %d' % chip)
        out.append(found.pop())
    return out


def check_aliases(pieces, histories, outputs):
    """Per chip: inside each user the 9 addresses are distinct (the served rule, the served
    text); across users the 16 outputs are distinct and disjoint from every input. Inputs of
    different users may coincide: they are only read."""
    message = 'Immutable input and mutable windows must not alias'
    inputs = set(pieces) | {value for row in histories for value in row}
    flat = [value for row in outputs for value in row]
    for piece, history, output in zip(pieces, histories, outputs):
        own = [piece, *history, *output]
        if len(set(own)) != len(own):
            raise ValueError(message)
    if len(set(flat)) != len(flat) or set(flat) & inputs:
        raise ValueError(message)


# ---------------------------------------------------------------------------------------------
# Device wrapper.
# ---------------------------------------------------------------------------------------------

_DESCRIPTORS = {}
_SOURCE_SHA = {}


def clear_cache():
    _DESCRIPTORS.clear()


def cache_size():
    return len(_DESCRIPTORS)


def _sha(directory):
    key = str(directory)
    if key not in _SOURCE_SHA:
        _SOURCE_SHA[key] = source_sha(directory)
    return _SOURCE_SHA[key]


class _Prepared:
    """Built once per key: the core set, the CBs and, per chip, the kernel descriptors with their
    compile args and constant per-core [task_start, task_count]. Per call only the common
    runtime args (36 addresses at four users) change - generic_op hashes their COUNT, not their
    values, so every call after the first is a program-cache hit (the gdn_prefill_conv_exact
    pattern, card M proven)."""

    def __init__(self, operations, mesh, chips, users, settings, accessor_args, directory, defines):
        grid = mesh.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        layout = plan(users, cores)
        self.coordinates = core_coordinates(grid.x, grid.y, cores)
        core_set = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0),
                                                                 operations.CoreCoord(grid.x - 1, grid.y - 1))])
        self.cbs = [operations.CBDescriptor(
            total_size=pages * PAGE, core_ranges=core_set,
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
                                                              page_size=PAGE,
                                                              tile=operations.TileDescriptor(operations.Tile([32, 32])))])
            for index, pages in sorted(cb_plan(settings).items())]
        per_core = operations.RuntimeArgs()
        for (x, y), (start, count) in zip(self.coordinates, layout['ranges']):
            per_core[x][y] = [start, count]
        source = str(source_path(directory))
        self.chips = []
        for chip in range(chips):
            arguments = compile_args(users, settings, accessor_args[chip])
            kernels = []
            for role in roles(settings):
                if role == 'VTW_ROLE_READER':
                    config = operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1,
                                                                     noc=operations.NOC.RISCV_1_default)
                else:
                    config = operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                                     noc=operations.NOC.RISCV_0_default)
                kernels.append(operations.KernelDescriptor(
                    kernel_source=source, core_ranges=core_set, compile_time_args=list(arguments),
                    defines=list(defines[role]), runtime_args=per_core, config=config))
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


def build_windows_packed(mesh, users, *, operations=None, directory=None, settings=None, negative=None):
    """users: [(piece, history4)] in segment order (the pieces resident_piece made, the
    histories `pending` carries). Returns [[O[u][0..3]] per user]: (1, 16, 5120) bf16 TILE DRAM,
    the served signature. Raises Unsupported, before any kernel runs, for anything outside the
    op; every output is freed on any failure."""
    if operations is None:
        import ttnn as operations
    users = [(piece, list(history)) for piece, history in users]
    reason = unsupported(operations, mesh, users)
    if reason is not None:
        raise Unsupported(reason)
    settings = resolve_settings(settings)
    if negative is not None and negative not in NEGATIVE_CONTROLS:
        raise ValueError('unknown negative control %r' % (negative,))
    if negative == 'user' and len(users) < 2:
        raise ValueError('the user negative control needs two users')
    directory = Path(directory) if directory is not None else Path(__file__).parent
    count = len(users)
    rows, width = users[0][0].shape[1], users[0][0].shape[2]
    shape = tuple(mesh.shape)
    coordinates = mesh_coordinates(shape)
    chips = len(coordinates)
    grid = mesh.compute_with_storage_grid_size()
    pieces = [piece for piece, history in users]
    histories = [history for piece, history in users]
    piece_shards = [operations.get_device_tensors(value) for value in pieces]
    history_shards = [operations.get_device_tensors(value) for row in histories for value in row]
    if any(len(parts) != chips for parts in piece_shards + history_shards):
        raise Unsupported('a tensor lacks a shard on some chip')
    piece_kinds = {_kind(operations, value) for value in pieces}
    history_kinds = {_kind(operations, value) for row in histories for value in row}
    if len(piece_kinds) != 1 or len(history_kinds) != 1:
        raise Unsupported('mixed DRAM / L1 placement within a class')
    piece_args = class_args(operations, piece_shards, chips)
    history_args = class_args(operations, history_shards, chips)
    outputs, flat = [], []
    try:
        for user in range(count):
            own = []
            for slot in range(SLOTS):
                own.append(operations.empty((1, rows, CHANNELS), device=mesh, dtype=operations.bfloat16,
                                            layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG))
                flat.append(own[-1])
            outputs.append(own)
        output_shards = [operations.get_device_tensors(value) for value in flat]
        if any(len(parts) != chips for parts in output_shards):
            raise Unsupported('an output lacks a shard on some chip')
        output_args = class_args(operations, output_shards, chips)
        commons = []
        for chip in range(chips):
            piece_addresses = [parts[chip].buffer_address() for parts in piece_shards]
            history_addresses = [[history_shards[HISTORY * user + index][chip].buffer_address() for index in range(HISTORY)]
                                 for user in range(count)]
            output_addresses = [[output_shards[SLOTS * user + slot][chip].buffer_address() for slot in range(SLOTS)]
                                for user in range(count)]
            check_aliases(piece_addresses, history_addresses, output_addresses)
            commons.append(common_args(piece_addresses, history_addresses, output_addresses))
        accessor_args = [list(piece_args[chip]) + list(history_args[chip]) + list(output_args[chip])
                         for chip in range(chips)]
        sha = _sha(directory)
        defines = {role: kernel_defines(sha, role, settings, negative) for role in roles(settings)}
        key = (id(mesh), shape, grid.x, grid.y, count, rows, width, str(directory),
               (piece_kinds.pop(), history_kinds.pop()), tuple(tuple(args) for args in accessor_args),
               tuple(sorted(settings.items())), tuple(sorted(defines.items())))
        prepared = _DESCRIPTORS.get(key)
        if prepared is None:
            prepared = _Prepared(operations, mesh, chips, count, settings, accessor_args, directory, defines)
            _DESCRIPTORS[key] = prepared
        unique = []
        for tensor in pieces + [value for row in histories for value in row] + flat:
            if not any(tensor is seen for seen in unique):
                unique.append(tensor)
        operations.generic_op(unique, prepared.program(operations, coordinates, commons))
    except BaseException:
        for tensor in flat:
            operations.deallocate(tensor)
        raise
    return outputs
