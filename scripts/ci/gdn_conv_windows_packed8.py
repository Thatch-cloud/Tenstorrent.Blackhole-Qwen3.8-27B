"""gdn_conv_windows_packed8: every octo user's four causal-conv windows in ONE generic_op, EIGHT-row users (QWEN_FAST_OCTO_GLUE8, T2 cut #1 at the octo block).

gdn_conv_windows_packed (verify_trace_t2 cut #1) builds the four sixteen-row users' windows in one launch and refuses any other width: its trimmed kernel zeroes faces 2-3 of the
scratch pages once, which is exact only while every window row lives in faces 0-1 and all sixteen rows of those faces are rewritten for every window. The octo block's users are
EIGHT rows, so the T2 windows cut declined there (gdn_user_batch_conv: windows_fallback, EIGHT_ROW_SERVED) and eight users x four windows = 32 served launches of 72 us ran per GDN
layer. This twin is the same launch for eight-row pieces: the sibling gdn_conv_windows_packed8.cpp, the same three roles, tasks and byte moves, and the geometry of an eight-row
window:

    window slot s, page p, face f in 0..1, row r in 0..7 (byte f * 512 + r * 32 of the 2048-byte tile):
        r + s < 4   history H[r + s] face f row 0                     (32 bytes)
        r + s >= 4  piece P face f row r + s - 4                       one contiguous block of (4 + s) * 32 bytes from piece row 0 on
        rows 8-15 of faces 0-1, and faces 2-3: all 0x00 (the served kernel zeroes its scratch page before each window and writes tokens 0 .. rows - 1 only).

The trimmed kernel zeroes each scratch page WHOLE once and rewrites rows 0-7 of both left faces for every window, which is the served page byte for byte. Nothing is computed, packed,
unpacked or canonicalised: -0, denormals and NaN payloads travel as bits (the pieces are read as the split left them, upstream of both paths).

The pinned module's helpers that do not depend on the row count (the plan over (user, page) tasks, the common-arg layout, the settings, the roles, the aliasing rule, the
coordinates) are imported from it, so the two cannot drift; what depends on it (the copies, the validation, the compile args, the source) is here. gdn_conv_windows_packed.py and its
kernel are untouched and the M3 path never imports this module.

This module imports no ttnn at top level; the planners are pure python (CPU-tested, test_octo_glue8_windows.py). Its kernel is the SIBLING .cpp (Path(__file__).with_suffix),
so the pair travels together (test_serving_image_copy_closure).
"""

import hashlib
from pathlib import Path

import gdn_conv_windows_packed as pinned
from gdn_conv_windows_packed import (HISTORY, NEGATIVE_CONTROLS, PAGE, SLOTS, Unsupported, cb_plan, channels, check_aliases, class_args, common_args,
                                     core_coordinates, kernel_defines, mesh_coordinates, pages, plan, resolve_settings, roles)

ROWS = 8               # the only width this kernel serves (whole scratch pages zeroed once, rows 0-7 rewritten)
ROW_BYTES = pinned.ROW_BYTES
SOURCE_NAME = 'gdn_conv_windows_packed8.cpp'
RUNTIME_FILES = ('gdn_conv_windows_packed8.py', SOURCE_NAME)
DEFAULTS = pinned.DEFAULTS


def window_copies(slot):
    """The copies that build window `slot` in each left face (f in 0..1) of a scratch page that is already all zero, at rows == 8: [(kind, source, destination_row, length_bytes)].

    'hist': 32 bytes of history `source` (its row 0 of the face); 'piece': length_bytes from piece row `source` on. Destination rows 0..3-s take history rows s..3, rows
    4-s..7 take piece rows 0..3+s in one block - the served kernel's token loop (gdn_conv_windows.cpp:12-22) at rows == 8, with its per-row copies merged where contiguous."""
    if slot not in range(SLOTS):
        raise ValueError('slot must be 0..3')
    copies = [('hist', row + slot, row, ROW_BYTES) for row in range(HISTORY - slot)]
    copies.append(('piece', 0, HISTORY - slot, (ROWS - HISTORY + slot) * ROW_BYTES))
    return copies


def apply_copies(piece_pages, history_pages):
    """window_copies applied in torch on zeroed pages: the trimmed kernel's map (rows == 8). Pages are int16 (pages, 1024) views of the raw 2048-byte tiles (face-ordered)."""
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


def reference_windows(piece_pages, history_pages):
    """The served kernel's output at rows == 8 (gdn_conv_windows.cpp's loop transcribed, zeroed scratch and all): the pinned reference at eight rows."""
    return pinned.reference_windows(piece_pages, history_pages, rows=ROWS)


def compile_args(users, settings, accessor_args):
    """[USERS, PAGES, ROWS, NBUF, TASKS] + the three class representatives' TensorAccessorArgs (piece, history, output), identical for every kernel."""
    settings = resolve_settings(settings)
    return [users, pages(), ROWS, settings['nbuf'], users * pages()] + [int(value) for value in accessor_args]


def source_path(directory=None):
    if directory is not None:
        return Path(directory) / SOURCE_NAME
    return Path(__file__).with_suffix('.cpp')


def source_sha(directory=None):
    """First 8 hex of sha256 over the kernel, passed as a define: defines are hashed into the JIT key, so a revised kernel at the same path can never reuse a stale binary."""
    return hashlib.sha256(source_path(directory).read_bytes()).hexdigest()[:8]


def unsupported(operations, mesh, users):
    """None when the op can run these (piece, history4) users, else the reason."""
    from gdn_multitoken_conv import validate_projected
    import gdn_seq_block

    # The user cap of the batched GDN launch THIS process serves with: gdn_seq_block.batch is the width-aware module (gdn_user_batch_tp, 8 users, at four cards), where the
    # pinned name `gdn_user_batch` is the pair's (4) - which is what refused the octo block's eight users in the pinned op ('8 users outside 1..4', card log).
    limit = gdn_seq_block.batch.MAX_USERS
    users = list(users)
    if not 1 <= len(users) <= limit:
        return '%d users outside 1..%d' % (len(users), limit)
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
            if not pinned._interleaved(operations, tensor):
                return 'user %d: not DRAM or L1 interleaved' % index
    try:
        shape = tuple(mesh.shape)
    except (AttributeError, TypeError):
        return 'no mesh shape'
    import tp_shapes

    if shape not in ((1, 1), (1, tp_shapes.chip_count())):
        return 'mesh %s is not [1, 1] or [1, %d]' % (shape, tp_shapes.chip_count())
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


def _sha(directory):
    key = str(directory)
    if key not in _SOURCE_SHA:
        _SOURCE_SHA[key] = source_sha(directory)
    return _SOURCE_SHA[key]


class _Prepared:
    """pinned._Prepared for this kernel: the core set, the CBs and, per chip, the kernel descriptors with their compile args and constant per-core [task_start, task_count]. Per call
    only the common runtime args (addresses) change - generic_op hashes their COUNT, not their values, so every call after the first is a program-cache hit."""

    def __init__(self, operations, mesh, chips, users, settings, accessor_args, directory, defines):
        grid = mesh.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        layout = plan(users, cores)
        self.coordinates = core_coordinates(grid.x, grid.y, cores)
        core_set = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0),
                                                                 operations.CoreCoord(grid.x - 1, grid.y - 1))])
        self.cbs = [operations.CBDescriptor(
            total_size=count * PAGE, core_ranges=core_set,
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
                                                              page_size=PAGE,
                                                              tile=operations.TileDescriptor(operations.Tile([32, 32])))])
            for index, count in sorted(cb_plan(settings).items())]
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


def build_windows_packed8(mesh, users, *, operations=None, directory=None, settings=None, negative=None):
    """users: [(piece, history4)] in segment order (the (1, 8, W) pieces the split made, the histories `pending` carries). Returns [[O[u][0..3]] per user]: (1, 8, C) bf16 TILE DRAM,
    the served signature. Raises Unsupported, before any kernel runs, for anything outside the op; every output is freed on any failure."""
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
    piece_kinds = {pinned._kind(operations, value) for value in pieces}
    history_kinds = {pinned._kind(operations, value) for row in histories for value in row}
    if len(piece_kinds) != 1 or len(history_kinds) != 1:
        raise Unsupported('mixed DRAM / L1 placement within a class')
    piece_args = class_args(operations, piece_shards, chips)
    history_args = class_args(operations, history_shards, chips)
    outputs, flat = [], []
    try:
        for user in range(count):
            own = []
            for slot in range(SLOTS):
                own.append(operations.empty((1, rows, channels()), device=mesh, dtype=operations.bfloat16,
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
