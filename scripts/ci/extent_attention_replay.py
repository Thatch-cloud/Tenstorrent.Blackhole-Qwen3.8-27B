"""S2 extent replay readers: one SDPA program for every 256-key family, beside the pinned readers.

WHY. The M3 block serves a round only when every live frontier sits in the one captured
native chunk family: the pinned validate_ticket (attention_mask_replay.py) admits a start only
inside [capacity - 256, capacity), and the replay SDPA takes its extent from the compile-time
capacity (attention_parallel.py passes no cur_pos tensor). K64j's flag 0x20
(optimisation/ttnn-op/k64j) lets the reader, compute and writer kernels take each entry's extent
from a cur_pos tensor instead, so one captured program serves a user at any position: that user's
own family E = (start // 256 + 1) * 256, with cur_pos = E - 1 (design s2-design.md section 2).

WHAT. ExtentSegmentReader is one packed user's reader and PackedExtentReplayReader is the block's
adapter over one per segment, with the duck API of the pinned ReplayAttentionReader and of
pooled_attention_replay.PackedReplayAttentionReader, so model_batch, packed_verifier and
verify_prestage drive them the same way. Per segment, per bundle, the reader keeps:
  - a pool-lent full-width page table (B, page_width) and a pool-lent (B,) int32 cur_pos
    (serving_buffer_pool.PackedExtentStorage): state kept across steps, allocated before any trace;
  - its own eight-word positions word, word0 = start & 255 (never the absolute start);
  - its own narrow (B, 1, 12 * rows, 256) tail mask, written in-trace by the PINNED mask kernel
    (attention_mask_replay.cpp) run at capacity 256 with that relative word. At capacity 256 the
    kernel's predicate and page index are exactly the last eight column tiles of the pinned wide
    mask at capacity E (design 2.3 item 2), so no new kernel exists and none is compiled.
The word and cur_pos come from one helper, extent_values(start), so they cannot describe
different families. Construction stages word, cur_pos and tables for the reader's start, fenced
and address-checked (design A6): without it the pool's zeroed cur_pos would describe E = 256
while `start` says otherwise, and model_batch skips the initial stage exactly when the start
already matches the capture start.

WHAT IT NEVER DOES. attention_replay.py, attention_mask_replay.py/.cpp, attention_parallel.py and
attention_fold_dma.* are frozen-recipe evidence (target_t16_attention_gate.SOURCES) and keep their
bytes. Nothing here subclasses the pinned reader or calls its validate_ticket or prepare:
prepare_narrow and execute_extent are line-for-line copies beside them with the one change each
names, and the mask kernel is the pinned .cpp, resolved from attention_mask_replay.__file__.

QUALIFIED GEOMETRY ONLY. K64j's card-B pass (CB1) qualified 0x27 at G8B2 (eight-row groups, one
two-entry bundle per sixteen-row user): tail 0x1 | share 0x2 | slice 0x4 | extent 0x20. A reader
whose bundles are not all two eight-row groups, or whose flags come out other than 0x27, refuses
itself. Nothing here runs unless the S2 block builds these readers (QWEN_FAST_EXTENT_REPLAY=1);
with the flag off every byte path is the pinned or pooled one, unchanged.
"""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path

import attention_mask_replay
from attention_fold_dma import device_layout_dma
from attention_head_fold import parallel_groups
from gdn_multitoken_conv import addresses, release_owned
from pooled_attention_replay import QWEN_SDPA_EXTENT_MARKER, apply_sdpa_modes, sdpa_modes, validate_segments


# One native k-chunk: the width of every narrow mask, and the family step.
K = 256
# Rows below 128 have 128-key native chunks (attention_head_fold.chunk_groups: p = 127 gives 4
# tiles, p = 128 gives 5 -> 8), so a live ticket starting there would not bundle like its family.
MIN_LIVE_START = 128
# The one geometry K64j's CB1 qualified: 0x1 | 0x2 | 0x4 | 0x20 on bundles of two eight-row groups.
EXTENT_FLAGS = 0x27
EXTENT_GROUP_ROWS = 8
EXTENT_BUNDLE_ENTRIES = 2
TREE_SCRATCH_ENV = 'QWEN_SDPA_TREE_SCRATCH_ROUNDS'
# The K64j factory's F22 literal: apply_sdpa_modes requires it in the mapped _ttnncpp.so for 'extent'.
F22_MARKER = QWEN_SDPA_EXTENT_MARKER
ENGAGED_MARKER = '[PINDIAG] extent replay engaged'
MASK_HEAD_ROWS = 12      # folded query rows per token (attention_head_fold.fold_query)
BF16_NEG_INF = 0xff80    # the pinned kernel's masked word (attention_mask_replay.cpp:28)


def _pindiag(text):
    """One server-log line: loguru where the engine has it (as pooled_attention_replay), print otherwise."""
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.info('{}', text)


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('%s must be an integer >= %d, got %r' % (name, minimum, value))
    return value


def extent(start):
    """The 256-key family a row at `start` reads: [0, E), E = (start // 256 + 1) * 256."""
    _integer('start', start)
    return (start // K + 1) * K


def extent_values(start):
    """(word0, cur_pos) for a segment at `start`, from one place: the mask kernel's relative start
    (start & 255, the kernel runs at capacity 256) and K64j's per-entry position (E - 1)."""
    family = extent(start)
    return start & (K - 1), family - 1


def accept_limit(start, rows):
    """How many of a ticket's rows may commit: those at positions < E. Rows past E see all of
    [0, E) and never their own key (tail mode masks only the final chunk), so they are wrong by
    design and are capped like rejected drafts (design 2.4)."""
    _integer('rows', rows, 1)
    return min(rows, extent(start) - start)


def admits(start, rows, capacity):
    """A live ticket the extent path serves: an integer start with 128 <= start and start + rows <= capacity."""
    return (type(start) is int and type(rows) is int and type(capacity) is int and rows >= 1
            and MIN_LIVE_START <= start and start + rows <= capacity)


def LAYOUT(rows, group_rows):
    """The extent readers' bundles: the native chunk grouping at position 256. Every non-crossing
    ticket at a start >= 128 shares it (all its rows have the chunk signature (256, E)), so the
    pinned reader captured at that start would bundle identically (design 2.3 item 1)."""
    return parallel_groups(K, rows, max_group_rows=group_rows)


def mask_tiles(word, rows, batches, offset, capacity):
    """attention_mask_replay.cpp:18-33 on the host, one entry per kernel task in task order.

    Returns (pages, tiles): the page each task writes (:32) and its 32x32 tile of bf16 bit
    patterns, 0 or 0xff80 (:28), as int16. The tile is in (row, column) order: the kernel's
    `index` (:26) is the TILE layout's face order of that row and column, so this is what a
    readback of the tile shows at (row, column)."""
    import torch

    for name, value, minimum in (('word', word, 0), ('rows', rows, 1), ('batches', batches, 1),
                                 ('offset', offset, 0), ('capacity', capacity, K)):
        _integer(name, value, minimum)
    if word >= 1 << 32 or rows > 8 or batches > 3 or capacity % K:
        raise ValueError('Bounded mask geometry required: word < 2**32, rows <= 8, batches <= 3, capacity % 256 == 0')
    head_tiles = (rows * MASK_HEAD_ROWS + 31) // 32
    task = torch.arange(batches * head_tiles * 8, dtype=torch.int64)
    batch = task // (head_tiles * 8)
    head_tile = (task // 8) % head_tiles
    column_tile = task % 8
    lane = torch.arange(32, dtype=torch.int64)
    head = head_tile[:, None] * 32 + lane[None, :]
    position = word + offset + batch[:, None] * rows + (head % (rows * 6)) // 6
    cache_position = capacity - K + column_tile[:, None] * 32 + lane[None, :]
    masked = (head[:, :, None] >= rows * MASK_HEAD_ROWS) | (cache_position[:, None, :] > position[:, :, None])
    tiles = torch.where(masked, torch.tensor(BF16_NEG_INF - (1 << 16), dtype=torch.int16),
                        torch.tensor(0, dtype=torch.int16))
    pages = (batch * head_tiles + head_tile) * (capacity // 32) + capacity // 32 - 8 + column_tile
    return pages, tiles


def replay_mask_host(word, rows, batches, offset, capacity):
    """The folded mask as the device holds it after one pinned-kernel refresh of a zero upload:
    (batches, 1, 12 * rows, capacity) bf16, +0.0 outside the kernel's last eight column tiles."""
    import torch

    pages, tiles = mask_tiles(word, rows, batches, offset, capacity)
    head_tiles = (rows * MASK_HEAD_ROWS + 31) // 32
    grid = torch.zeros(batches * head_tiles * (capacity // 32), 32, 32, dtype=torch.int16)
    grid[pages] = tiles
    dense = grid.reshape(batches, head_tiles, capacity // 32, 32, 32).permute(0, 1, 3, 2, 4)
    dense = dense.reshape(batches, head_tiles * 32, capacity)[:, :rows * MASK_HEAD_ROWS].contiguous()
    return dense.view(torch.bfloat16).reshape(batches, 1, rows * MASK_HEAD_ROWS, capacity)


def narrow_mask_host(word, rows, batches, offset):
    """The narrow tail mask the extent reader's refresh writes: the pinned kernel at capacity 256
    with word = start & 255. The extent audit's and the tests' mirror."""
    return replay_mask_host(word, rows, batches, offset, K)


def prepare_narrow(mesh, positions, mask, *, rows, batches, offset):
    """attention_mask_replay.prepare (:44-84), line for line, at capacity 256.

    Two changes and no others: the family check (:48-49) is left out, because the pinned
    validate_ticket refuses capacity 256 (:18, :26) and the extent reader validates its own
    starts; and the mask width (:52) and the capacity runtime argument (:79-80) are 256. The
    kernel is the exact .cpp the pinned prepare compiles (:73)."""
    import ttnn

    if any(type(value) is not int for value in (rows, batches, offset)):
        raise ValueError('Integer mask geometry required')
    if not 1 <= rows <= 8 or not 1 <= batches <= 3 or offset < 0 or offset + rows * batches > 32:
        raise ValueError('At most three bounded contiguous query groups required')
    if tuple(positions.shape) != (8,) or positions.dtype != ttnn.int32 or positions.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError('Eight-word position input required; first word is block start')
    if tuple(mask.shape) != (batches, 1, rows * 12, K) or mask.dtype != ttnn.bfloat16 or mask.layout != ttnn.TILE_LAYOUT:
        raise ValueError('Fixed-shape BF16 folded narrow attention mask required')
    if any(tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in (positions, mask)):
        raise ValueError('Interleaved DRAM metadata required')
    head_tiles = (rows * 12 + 31) // 32
    tasks = batches * head_tiles * 8
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < tasks // 8:
        raise ValueError('Bounded eight-column mask worker grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, tasks // 8 - 1))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    parts = [ttnn.get_device_tensors(tensor) for tensor in (positions, mask)]
    if any(len(values) != 2 for values in parts):
        raise ValueError('Two chip-local metadata buffers required')
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [values[chip] for values in parts]
        if local[0].buffer_address() == local[1].buffer_address():
            raise ValueError('Mask and position input must not alias')
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(attention_mask_replay.__file__).with_suffix('.cpp')),
            core_ranges=cores,
            compile_time_args=[argument for tensor in local for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for task in range(tasks):
            runtime[task % 8][task // 8] = [local[0].buffer_address(), local[1].buffer_address(),
                rows, K, offset, task]
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    return program


def execute_extent(mesh, operations, query, keys, values, metadata, cur_pos, owned, *, scale, memory_config):
    """attention_parallel.execute (:6-30), line for line, plus cur_pos_tensor=cur_pos[k] in the SDPA call
    (:17-19): bundle k takes its extent from its own pool-lent cur_pos tensor (K64j flag 0x20)."""
    chunks = []
    if len(cur_pos) != len(metadata):
        raise ValueError('One cur_pos tensor per bundle required')
    for (bundle, pages, mask, config), positions in zip(metadata, cur_pos, strict=True):
        if not 1 <= len(bundle) <= 3 or not 1 <= bundle[0]['rows'] <= 8:
            raise ValueError('At most three groups of up to eight queries required')
        count = bundle[0]['rows']
        if any(group['rows'] != count or group['signature'] != bundle[0]['signature'] for group in bundle):
            raise ValueError('Parallel groups must share shape and native chunk workload')
        packed = [device_layout_dma(mesh, query, count, owned, offset=group['offset']) for group in bundle]
        stacked = operations.concat(packed, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else packed[0]
        owned.append(stacked)
        result = operations.transformer.paged_scaled_dot_product_attention_decode(stacked, keys, values,
            page_table_tensor=pages, cur_pos_tensor=positions, is_causal=False, attn_mask=mask, scale=scale,
            program_config=config, memory_config=memory_config)
        owned.append(result)
        for index in range(len(bundle)):
            selected = operations.slice(result, (0, index, 0, 0), (1, index + 1, count * 12, 256),
                memory_config=operations.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else result
            owned.append(selected)
            chunks.append(device_layout_dma(mesh, selected, count, owned, inverse=True))
    if not chunks:
        raise ValueError('Complete nonempty group metadata required')
    output = operations.concat(chunks, dim=1, memory_config=memory_config)
    owned.append(output)
    return output


def _padded_last(tensor):
    shape = getattr(tensor, 'padded_shape', None)
    return tuple(tensor.shape)[-1] if shape is None else tuple(shape)[-1]


def _memory_config(operations, tensor):
    config = getattr(tensor, 'memory_config', None)
    return config() if callable(config) else operations.DRAM_MEMORY_CONFIG


def independent(operations, tensors, what):
    """Refuse two tensors sharing a chip-local buffer: a lent table, cur_pos, word or mask is written
    by the host and read in-trace, so an alias would stage one user's state into another's."""
    seen = [addresses(operations, tensor) for tensor in tensors]
    for chip in range(2):
        if len({pair[chip] for pair in seen}) != len(seen):
            raise ValueError('%s must own independent chip storage; two share a buffer on chip %d' % (what, chip))


def validate_extent_storage(operations, storage, bundles, page_width):
    """Pool-lent (table, cur_pos) per bundle, in bundle order, of exactly the geometry the trace bakes and
    K64j's F20 takes: a (B, page_width) and a (B,) row-major int32 each, interleaved in DRAM, and no two
    sharing a buffer (serving_buffer_pool.PackedExtentStorage)."""
    if storage is None:
        raise ValueError('The extent reader needs the pool-lent (table, cur_pos) per bundle; it uploads neither')
    storage = [tuple(pair) for pair in storage]
    if len(storage) != len(bundles) or any(len(pair) != 2 for pair in storage):
        raise ValueError('Extent storage must be one (table, cur_pos) pair per reader bundle: %d for %d'
                         % (len(storage), len(bundles)))
    for (table, positions), bundle in zip(storage, bundles, strict=True):
        batches = len(bundle)
        if (tuple(table.shape) != (batches, page_width) or table.dtype != operations.int32
                or table.layout != operations.ROW_MAJOR_LAYOUT
                or _memory_config(operations, table) != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Extent page table must be a row-major int32 (%d, %d) in interleaved DRAM'
                             % (batches, page_width))
        if (tuple(positions.shape) != (batches,) or _padded_last(positions) != batches
                or positions.dtype != operations.int32 or positions.layout != operations.ROW_MAJOR_LAYOUT
                or _memory_config(operations, positions) != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Extent cur_pos must be a row-major int32 (%d,) in interleaved DRAM (K64j F20: '
                             'padded_shape[-1] == B)' % batches)
    independent(operations, [tensor for pair in storage for tensor in pair], 'Extent storage')
    return storage


def _upload(operations, mesh, value, dtype):
    """model_batch's upload_replay: replicated, interleaved DRAM, row-major integers and tiled BF16."""
    return operations.from_torch(value, device=mesh, dtype=dtype,
        layout=operations.ROW_MAJOR_LAYOUT if dtype == operations.int32 else operations.TILE_LAYOUT,
        memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(mesh))


class ExtentSegmentReader:
    """One packed user's extent reader: the pinned reader's contracts over a pool-lent full-width table
    and cur_pos per bundle, a reader-owned positions word holding start & 255, and reader-owned narrow
    masks the pinned kernel writes at capacity 256."""

    runtime_extent = True
    short_context = False

    def __init__(self, operations, mesh, rows, page_width, pages_host, *, storage, max_group_rows, start):
        import torch

        # Every refusal before anything is allocated.
        if type(rows) is not int or rows not in (8, 16, 32):
            raise ValueError('Extent reader requires an explicit T8/T16/T32 segment')
        if type(max_group_rows) is not int or max_group_rows != EXTENT_GROUP_ROWS:
            raise ValueError('Extent replay is qualified at eight-row groups only (K64j CB1, G8B2 0x27), got %r'
                             % (max_group_rows,))
        if os.environ.get(TREE_SCRATCH_ENV) != '1':
            raise ValueError('Eight-row replay requires process-fixed compact native scratch (%s=1), '
                             'the pinned reader\'s G8 precondition' % TREE_SCRATCH_ENV)
        modes = sdpa_modes()
        if 'tail' not in modes:
            raise ValueError('Extent replay needs QWEN_FAST_SDPA_MODES to include tail (K64j refuses 0x20 without 0x1)')
        if type(page_width) is not int or page_width < 4 or page_width % (K // 64):
            raise ValueError('Extent replay needs an integer page-table width of whole 256-key families, got %r'
                             % (page_width,))
        if getattr(pages_host, 'ndim', None) != 2 or pages_host.shape[0] != 1 or pages_host.shape[1] < page_width:
            raise ValueError('One complete native cache page table required')
        bundles = LAYOUT(rows, max_group_rows)
        if any(len(bundle) != EXTENT_BUNDLE_ENTRIES or any(group['rows'] != EXTENT_GROUP_ROWS for group in bundle)
               for bundle in bundles):
            raise ValueError('Extent replay is qualified at G8B2 only (0x27, K64j CB1): a %d-row segment bundles as %r'
                             % (rows, [[group['rows'] for group in bundle] for bundle in bundles]))
        self.operations, self.mesh = operations, mesh
        self.rows, self.page_width, self.capacity = rows, page_width, page_width * 64
        self.max_group_rows = max_group_rows
        self.closed = self.failed = False
        self.start = None
        self.validate(start)
        lent = validate_extent_storage(operations, storage, bundles, page_width)
        self.borrowed = [tensor for pair in lent for tensor in pair]
        self.cur_pos = [positions for table, positions in lent]
        self.pages_host = pages_host[:, :page_width].clone()
        self.owned, self.metadata, self.programs = [], [], []
        self.calls, self.refresh_calls = 0, 0
        self.mask_scope = None
        grid = mesh.compute_with_storage_grid_size()
        try:
            self.positions = _upload(operations, mesh, torch.zeros(8, dtype=torch.int32), operations.int32)
            self.owned.append(self.positions)
            for bundle, (table, positions) in zip(bundles, lent, strict=True):
                count = bundle[0]['rows']
                mask = _upload(operations, mesh, torch.zeros(len(bundle), 1, count * 12, K, dtype=torch.bfloat16),
                               operations.bfloat16)
                self.owned.append(mask)
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
                self.metadata.append((bundle, table, mask, config))
                self.programs.append(prepare_narrow(mesh, self.positions, mask, rows=count, batches=len(bundle),
                                                    offset=bundle[0]['offset']))
            independent(operations, [*self.borrowed, *self.owned], 'Extent reader buffers')
            # Design A6: the word, cur_pos and tables for `start`, before any forward reads them.
            self.stage(start)
            apply_sdpa_modes(self, modes | {'extent'})
            flags = tuple(getattr(self, 'sdpa_modes_applied', None) or ())
            if len(flags) != len(self.metadata) or any(value != EXTENT_FLAGS for value in flags):
                raise ValueError('Extent replay is qualified at 0x27 only (tail, share, slice, extent at G8B2); '
                                 'QWEN_FAST_SDPA_MODES=%s gives %s' % (','.join(sorted(modes)),
                                                                      ['0x%x' % value for value in flags]))
        except BaseException:
            self.close()
            raise

    @property
    def audit(self):
        return None

    @audit.setter
    def audit(self, value):
        if value is not None:
            raise ValueError('The extent replay reader takes no attention audit: AttentionReplayAudit compares '
                             'against one capture family\'s wide causal mask (the extent audit is the block\'s)')

    def validate(self, start):
        """Host only: open, unpoisoned, an integer start whose rows fit the full-width table. No floor:
        idle segments sit at start 0 or 32 (the block's admits holds live tickets to >= 128)."""
        if self.closed:
            raise RuntimeError('Extent replay reader is closed')
        if self.failed:
            raise RuntimeError('Extent replay reader is poisoned after a failed operation')
        if type(start) is not int or start < 0 or start + self.rows > self.capacity:
            raise ValueError('Extent ticket must be an integer start with 0 <= start and start + %d <= %d, got %r'
                             % (self.rows, self.capacity, start))

    def stage_values(self, start, table):
        """(destination, host value, dtype, layout) for `start`, in order: the word [start & 255, 0 x 7],
        then per bundle its cur_pos [E - 1] * B and its table (1, >= page_width) repeated B times."""
        import torch

        self.validate(start)
        if getattr(table, 'ndim', None) != 2 or table.shape[0] != 1 or table.shape[1] < self.page_width:
            raise ValueError('One complete (1, >= %d) host page table required' % self.page_width)
        operations = self.operations
        relative, position = extent_values(start)
        words = torch.zeros(8, dtype=torch.int32)
        words[0] = relative
        values = [(self.positions, words, operations.int32, operations.ROW_MAJOR_LAYOUT)]
        for (bundle, pages, mask, config), cur_pos in zip(self.metadata, self.cur_pos, strict=True):
            values.append((cur_pos, torch.full((len(bundle),), position, dtype=torch.int32),
                           operations.int32, operations.ROW_MAJOR_LAYOUT))
            values.append((pages, table[:, :self.page_width].repeat(len(bundle), 1).contiguous(),
                           operations.int32, operations.ROW_MAJOR_LAYOUT))
        return values

    def stage(self, start, table=None):
        """Write exactly stage_values(start, the current table), fenced, addresses checked unchanged; a
        failed copy poisons the reader. `table` replaces the current host table once written."""
        values = self.stage_values(start, self.pages_host if table is None else table)
        if self.mask_scope is not None:
            raise RuntimeError('Cannot stage positions during a shared-mask forward')
        operations = self.operations
        destinations = [destination for destination, value, dtype, layout in values]
        before = [addresses(operations, destination) for destination in destinations]
        try:
            for destination, value, dtype, layout in values:
                source = operations.from_torch(value, dtype=dtype, layout=layout,
                    mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
                operations.copy_host_to_device_tensor(source, destination)
            operations.synchronize_device(self.mesh)
            if [addresses(operations, destination) for destination in destinations] != before:
                raise AssertionError('Staging replaced a captured extent buffer')
        except BaseException:
            self.failed = True
            raise
        self.start = start
        if table is not None:
            self.pages_host = table[:, :self.page_width].clone()

    def refresh(self):
        self.validate(self.start)
        for entry, program in zip(self.metadata, self.programs, strict=True):
            attention_mask_replay.execute(self.positions, entry[2], program)
            self.refresh_calls += 1

    @contextmanager
    def shared_masks(self, expected_calls):
        self.validate(self.start)
        if type(expected_calls) is not int or not 1 <= expected_calls <= 16:
            raise ValueError('Explicit one-to-sixteen attention call budget required')
        if self.mask_scope is not None:
            raise RuntimeError('Shared-mask forwards cannot nest')
        self.mask_scope = self.calls + expected_calls
        try:
            self.refresh()
            yield
            if self.calls != self.mask_scope:
                raise AssertionError('Shared-mask forward did not consume its exact attention call budget')
        except BaseException:
            self.failed = True
            raise
        finally:
            self.mask_scope = None

    def __call__(self, query, keys, values, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        self.validate(self.start)
        if tuple(query.shape) != (1, self.rows, 12, 256):
            raise ValueError('Replay query geometry changed')
        owned = []
        protected = {addresses(self.operations, value) for value in (query, keys, values)}
        try:
            if self.mask_scope is None:
                self.refresh()
            elif self.calls >= self.mask_scope:
                raise AssertionError('Shared-mask forward exceeded its attention call budget')
            result = execute_extent(self.mesh, self.operations, query, keys, values, self.metadata, self.cur_pos,
                                    owned, scale=kwargs['scale'], memory_config=kwargs['memory_config'])
            protected.add(addresses(self.operations, result))
            self.calls += 1
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            release_owned(self.operations, [value for value in owned if addresses(self.operations, value) not in protected])

    def close(self):
        if getattr(self, 'closed', True):
            return
        if self.mask_scope is not None:
            raise RuntimeError('Cannot close a reader during a shared-mask forward')
        # The word and the masks only: the tables and cur_pos are the pool's.
        release_owned(self.operations, getattr(self, 'owned', []))
        self.owned = []
        self.closed = True


class PackedExtentReplayReader:
    """The packed block's attention: one ExtentSegmentReader per segment, each at its own start, word,
    cur_pos and family, a (1, block_rows, 12, 256) query dispatched a segment at a time (a DRAM slice
    of its rows, as pooled_attention_replay.PackedReplayAttentionReader) and reassembled in row order."""

    runtime_extent = True
    short_context = False

    def __init__(self, operations, mesh, segments, page_width, tables_host, *, storage, max_group_rows, starts):
        self.segments = validate_segments(segments)
        tables_host, starts = list(tables_host), tuple(starts)
        lent = None if storage is None else [list(pairs) for pairs in storage]
        if (len(tables_host) != len(self.segments) or len(starts) != len(self.segments)
                or lent is None or len(lent) != len(self.segments)):
            raise ValueError('One host page table, one start and one lent (table, cur_pos) set per packed segment required')
        if type(max_group_rows) is not int or type(page_width) is not int:
            raise ValueError('Integer group width and page-table width required')
        # Every segment's lent storage against the bundles its reader will take, and no buffer shared
        # between segments, before any reader is built: a wrong set refuses the block with nothing staged.
        for (first, last), pairs in zip(self.segments, lent, strict=True):
            validate_extent_storage(operations, pairs, LAYOUT(last - first, max_group_rows), page_width)
        independent(operations, [tensor for pairs in lent for pair in pairs for tensor in pair], 'Extent storage')
        self.operations, self.mesh = operations, mesh
        self.page_width, self.capacity = page_width, page_width * 64
        self.max_group_rows = max_group_rows
        self.rows = self.segments[-1][1]
        self.readers = []
        self.calls = 0
        self.closed = False
        try:
            for (first, last), pages_host, pairs, start in zip(self.segments, tables_host, lent, starts, strict=True):
                self.readers.append(ExtentSegmentReader(operations, mesh, last - first, page_width, pages_host,
                    storage=pairs, max_group_rows=max_group_rows, start=start))
        except BaseException:
            self.close()
            raise
        _pindiag('%s segments=%d flags=%s mask=narrow capacity=%d' % (
            ENGAGED_MARKER, len(self.readers),
            ','.join('0x%x' % value for reader in self.readers for value in reader.sdpa_modes_applied), self.capacity))

    @property
    def audit(self):
        return None

    @audit.setter
    def audit(self, value):
        if value is not None:
            raise ValueError('The extent replay reader takes no attention audit: AttentionReplayAudit compares '
                             'against one capture family\'s wide causal mask (the extent audit is the block\'s)')

    @property
    def borrowed(self):
        """Every pool-lent table and cur_pos of every reader, in segment then bundle order."""
        return [tensor for reader in self.readers for tensor in reader.borrowed]

    @property
    def metadata(self):
        return [entry for reader in self.readers for entry in reader.metadata]

    @property
    def starts(self):
        return tuple(reader.start for reader in self.readers)

    @property
    def refresh_calls(self):
        return sum(reader.refresh_calls for reader in self.readers)

    @property
    def failed(self):
        return any(reader.failed for reader in self.readers)

    @failed.setter
    def failed(self, value):
        for reader in self.readers:
            reader.failed = value

    def check_open(self):
        if self.closed:
            raise RuntimeError('Packed extent replay reader is closed')

    def validate(self, starts):
        """Every segment's start against its own reader, host only, before any copy."""
        self.check_open()
        starts = tuple(starts)
        if len(starts) != len(self.readers):
            raise ValueError('One start per packed segment required')
        for reader, start in zip(self.readers, starts, strict=True):
            reader.validate(start)

    def stage(self, starts):
        """Each segment's word, cur_pos and tables, one fenced write per segment."""
        starts = tuple(starts)
        self.validate(starts)
        for reader, start in zip(self.readers, starts, strict=True):
            reader.stage(start)

    def refresh(self):
        self.check_open()
        for reader in self.readers:
            reader.refresh()

    @contextmanager
    def shared_masks(self, expected_calls):
        self.check_open()
        with ExitStack() as scopes:
            for reader in self.readers:
                scopes.enter_context(reader.shared_masks(expected_calls))
            yield

    def __call__(self, query, keys, values, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        self.check_open()
        if tuple(query.shape) != (1, self.rows, 12, 256):
            raise ValueError('Packed extent query geometry changed')
        operations = self.operations
        rows, outputs = [], []
        try:
            for reader, (first, last) in zip(self.readers, self.segments, strict=True):
                selected = operations.slice(query, (0, first, 0, 0), (1, last, 12, 256),
                                            memory_config=operations.DRAM_MEMORY_CONFIG)
                rows.append(selected)
                outputs.append(reader(selected, keys, values, page_table_tensor=page_table_tensor,
                                      cur_pos_tensor=cur_pos_tensor, **kwargs))
            if len(outputs) == 1:
                result, outputs = outputs[0], []
            else:
                result = operations.concat(outputs, dim=1, memory_config=kwargs['memory_config'])
            self.calls += 1
            return result
        finally:
            for value in (*outputs, *rows):
                operations.deallocate(value)

    def close(self):
        if getattr(self, 'closed', True):
            return
        for reader in self.readers:
            reader.close()
        self.closed = True
