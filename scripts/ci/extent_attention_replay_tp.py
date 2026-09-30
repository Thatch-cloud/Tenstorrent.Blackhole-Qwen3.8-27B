"""The S2 extent replay readers at four cards (NKV = 1): the pinned readers' twin, beside them.

extent_attention_replay.py is frozen evidence (packed_any_evidence pins its sha256) and is written for the pair: 12 folded
query rows per token (two KV heads of six), K64j flags 0x27 (tail | share | slice | extent), two chips. At four cards a
chip holds six query heads on ONE KV head. This module is that file with exactly those numbers changed, everything else
line for line:
  - the folded rows per token are tp_shapes' attn_fold_rows (6), so a token's query is (1, T, 6, 256), a group's folded
    query is (1, 1, rows * 6, 256) and its narrow tail mask is (B, 1, rows * 6, 256);
  - the qualified flags are 0x23 (tail 0x1 | share 0x2 | extent 0x20): the q-slice (0x4) is illegal at one KV head
    (optimisation/ttnn-op/sdpa_decode_slice, F15), and pooled_attention_replay.q_slice_saves is False there;
  - the chip loops run over the mesh's devices (tp_shapes.chip_count()), and the mask and fold kernels are the _tp
    siblings whose head-row count is a define (attention_mask_replay_tp.cpp, attention_fold_dma_tp.cpp).
The geometry-free helpers (extent, extent_values, accept_limit, check_start, admits, LAYOUT, the constants) are the pinned
module's own objects, imported, so the two cannot describe different families.

Installed in place of the pinned module at QWEN_FAST_TP=4 only (tp_addresses.install(): model_batch and packed_verifier
import extent_attention_replay lazily, so they get this module; the pair keeps the pinned one). NOT qualified: the K64j
program at NKV = 1 has not run on a card (S2T-01 / CB1-TP4), so serving it is gate-only until packed_any_evidence_tp4
holds a qualifying pass.
"""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path

import attention_block_fold_tp
import attention_mask_replay
from attention_fold_dma_tp import device_layout_dma
from extent_attention_replay import (BF16_NEG_INF, ENGAGED_MARKER, EXTENT_BUNDLE_ENTRIES, EXTENT_GROUP_ROWS, F22_MARKER,
                                     K, LAYOUT, MIN_LIVE_START, TREE_SCRATCH_ENV, _integer, _pindiag, accept_limit,
                                     admits, check_start, extent, extent_values)
import pooled_attention_replay as pooled
from pooled_attention_replay import apply_sdpa_modes, sdpa_modes, validate_segments
import tp4_vglue
import tp_kernels
import tp_shapes
from tp_addresses import addresses, release_owned


# The one geometry the four-card extent replay is written for: 0x1 | 0x2 | 0x20 on bundles of two eight-row groups (the
# slice, 0x4, needs a second KV head to slice between).
EXTENT_FLAGS = pooled.QWEN_MASK_TAIL | pooled.QWEN_KV_SHARE | pooled.QWEN_RUNTIME_EXTENT


def head_rows():
    """Folded query rows per token at the width this process serves at (tp_shapes attn_fold_rows): 6 at four cards."""
    return tp_shapes.active().attn_fold_rows


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
    head_tiles = (rows * head_rows() + 31) // 32
    task = torch.arange(batches * head_tiles * 8, dtype=torch.int64)
    batch = task // (head_tiles * 8)
    head_tile = (task // 8) % head_tiles
    column_tile = task % 8
    lane = torch.arange(32, dtype=torch.int64)
    head = head_tile[:, None] * 32 + lane[None, :]
    position = word + offset + batch[:, None] * rows + (head % (rows * 6)) // 6
    cache_position = capacity - K + column_tile[:, None] * 32 + lane[None, :]
    masked = (head[:, :, None] >= rows * head_rows()) | (cache_position[:, None, :] > position[:, :, None])
    tiles = torch.where(masked, torch.tensor(BF16_NEG_INF - (1 << 16), dtype=torch.int16),
                        torch.tensor(0, dtype=torch.int16))
    pages = (batch * head_tiles + head_tile) * (capacity // 32) + capacity // 32 - 8 + column_tile
    return pages, tiles


def replay_mask_host(word, rows, batches, offset, capacity):
    """The folded mask as the device holds it after one pinned-kernel refresh of a zero upload:
    (batches, 1, 12 * rows, capacity) bf16, +0.0 outside the kernel's last eight column tiles."""
    import torch

    pages, tiles = mask_tiles(word, rows, batches, offset, capacity)
    head_tiles = (rows * head_rows() + 31) // 32
    grid = torch.zeros(batches * head_tiles * (capacity // 32), 32, 32, dtype=torch.int16)
    grid[pages] = tiles
    dense = grid.reshape(batches, head_tiles, capacity // 32, 32, 32).permute(0, 1, 3, 2, 4)
    dense = dense.reshape(batches, head_tiles * 32, capacity)[:, :rows * head_rows()].contiguous()
    return dense.view(torch.bfloat16).reshape(batches, 1, rows * head_rows(), capacity)


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
    if tuple(mask.shape) != (batches, 1, rows * head_rows(), K) or mask.dtype != ttnn.bfloat16 or mask.layout != ttnn.TILE_LAYOUT:
        raise ValueError('Fixed-shape BF16 folded narrow attention mask required')
    if any(tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in (positions, mask)):
        raise ValueError('Interleaved DRAM metadata required')
    head_tiles = (rows * head_rows() + 31) // 32
    tasks = batches * head_tiles * 8
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < tasks // 8:
        raise ValueError('Bounded eight-column mask worker grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, tasks // 8 - 1))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    parts = [ttnn.get_device_tensors(tensor) for tensor in (positions, mask)]
    chips = tp_shapes.chip_count()
    if any(len(values) != chips for values in parts):
        raise ValueError('%s chip-local metadata buffers required' % tp_shapes.count_word())
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [values[chip] for values in parts]
        if local[0].buffer_address() == local[1].buffer_address():
            raise ValueError('Mask and position input must not alias')
        descriptor = ttnn.KernelDescriptor(kernel_source=tp_kernels.source(Path(attention_mask_replay.__file__).with_suffix('.cpp')),
            core_ranges=cores, defines=tp_kernels.fold_defines(),
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
            selected = operations.slice(result, (0, index, 0, 0), (1, index + 1, count * head_rows(), 256),
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


def independent(operations, tensors, what):
    """Refuse two tensors sharing a chip-local buffer: a lent table, cur_pos, word or mask is written
    by the host and read in-trace, so an alias would stage one user's state into another's."""
    seen = [addresses(operations, tensor) for tensor in tensors]
    for chip in range(tp_shapes.chip_count()):
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
                or table.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Extent page table must be a row-major int32 (%d, %d) in interleaved DRAM'
                             % (batches, page_width))
        if (tuple(positions.shape) != (batches,) or _padded_last(positions) != batches
                or positions.dtype != operations.int32 or positions.layout != operations.ROW_MAJOR_LAYOUT
                or positions.memory_config() != operations.DRAM_MEMORY_CONFIG):
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
            raise ValueError('Extent replay is qualified at eight-row groups only (K64j G8B2, 0x23 at one KV head), got %r'
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
            raise ValueError('Extent replay is qualified at G8B2 only (0x23 at one KV head): a %d-row segment bundles as %r'
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
        self.owned, self.metadata, self.programs = [], [], []
        self.calls, self.refresh_calls = 0, 0
        self.mask_scope = None
        grid = mesh.compute_with_storage_grid_size()
        try:
            self.positions = _upload(operations, mesh, torch.zeros(8, dtype=torch.int32), operations.int32)
            self.owned.append(self.positions)
            for bundle, (table, positions) in zip(bundles, lent, strict=True):
                count = bundle[0]['rows']
                mask = _upload(operations, mesh, torch.zeros(len(bundle), 1, count * head_rows(), K, dtype=torch.bfloat16),
                               operations.bfloat16)
                self.owned.append(mask)
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
                self.metadata.append((bundle, table, mask, config))
                self.programs.append(prepare_narrow(mesh, self.positions, mask, rows=count, batches=len(bundle),
                                                    offset=bundle[0]['offset']))
            independent(operations, [*self.borrowed, *self.owned], 'Extent reader buffers')
            # Design A6: the word, cur_pos and tables for `start`, before any forward reads them.
            self.stage(start, table=pages_host)
            apply_sdpa_modes(self, modes | {'extent'})
            flags = tuple(getattr(self, 'sdpa_modes_applied', None) or ())
            if len(flags) != len(self.metadata) or any(value != EXTENT_FLAGS for value in flags):
                raise ValueError('Four-card extent replay is qualified at 0x23 only (tail, share, extent at G8B2); '
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
        check_start(start, self.rows, self.capacity)

    def stage_values(self, start, table):
        """(destination, host value, dtype, layout) for `start`, in order: the word [start & 255, 0 x 7],
        then per bundle its cur_pos [E - 1] * B and, when `table` is a host table, that table
        (1, >= page_width) repeated B times. `table=None` gives only what the start determines, the
        word and cur_pos; the argument is required, so a caller that means to write tables says so."""
        import torch

        self.validate(start)
        if table is not None and (getattr(table, 'ndim', None) != 2 or table.shape[0] != 1
                                  or table.shape[1] < self.page_width):
            raise ValueError('One complete (1, >= %d) host page table required' % self.page_width)
        operations = self.operations
        relative, position = extent_values(start)
        words = torch.zeros(8, dtype=torch.int32)
        words[0] = relative
        values = [(self.positions, words, operations.int32, operations.ROW_MAJOR_LAYOUT)]
        for (bundle, pages, mask, config), cur_pos in zip(self.metadata, self.cur_pos, strict=True):
            values.append((cur_pos, torch.full((len(bundle),), position, dtype=torch.int32),
                           operations.int32, operations.ROW_MAJOR_LAYOUT))
            if table is not None:
                values.append((pages, table[:, :self.page_width].repeat(len(bundle), 1).contiguous(),
                               operations.int32, operations.ROW_MAJOR_LAYOUT))
        return values

    def stage(self, start, table=None):
        """Write exactly stage_values(start, table), fenced, addresses checked unchanged; a failed copy
        poisons the reader. Without a table that is the word and cur_pos only: the lent tables are the
        block's to keep current (packed_values -> write_packed, every round), and a restage from a
        table held since capture would silently point every attention layer at old pages."""
        values = self.stage_values(start, table)
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
        if tuple(query.shape) != (1, self.rows, head_rows(), 256):
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
        # Every start on the host before any reader is built: a bad start for a later segment
        # refuses the block with nothing staged into any segment's lent storage.
        for (first, last), start in zip(self.segments, starts, strict=True):
            check_start(start, last - first, page_width * 64)
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
        """Each segment's word and cur_pos, one fenced write per segment. The tables are not written:
        construction staged them and the block's packed_values keeps them current."""
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
        if tuple(query.shape) != (1, self.rows, head_rows(), 256):
            raise ValueError('Packed extent query geometry changed')
        operations = self.operations
        if tp4_vglue.enabled(tp4_vglue.ATTN_FOLD):
            chunks = self.fold_chunks()
            reason = attention_block_fold_tp.problem(query, chunks, kwargs['memory_config'], self.operations)
            if reason is None:
                return self.call_block_folded(query, keys, values, chunks, **kwargs)
            self.note_fold_fallback(reason)
        rows, outputs = [], []
        try:
            for reader, (first, last) in zip(self.readers, self.segments, strict=True):
                selected = operations.slice(query, (0, first, 0, 0), (1, last, head_rows(), 256),
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

    def fold_chunks(self):
        """QWEN_FAST_TP4_ATTN_FOLD: the block's SDPA bundles in dispatch order (attention_block_fold_tp.Chunk)."""
        return attention_block_fold_tp.chunks_of(
            self.segments, [[bundle for bundle, pages, mask, config in reader.metadata] for reader in self.readers])

    fold_fallbacks = set()

    def note_fold_fallback(self, reason):
        if reason not in self.fold_fallbacks:
            self.fold_fallbacks.add(reason)
            tp4_vglue.log_line('%s site=attention reason=%s' % (tp4_vglue.FALLBACK, reason))
        tp4_vglue.note('attn_fold_fallback')

    def call_block_folded(self, query, keys, values, chunks, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        """__call__ with the per-segment query slices, fold DMAs, stacking concats, result slices, inverse folds and
        concats replaced by attention_block_fold_tp's two launches (QWEN_FAST_TP4_ATTN_FOLD). Each segment reader keeps
        its bookkeeping exactly as ExtentSegmentReader.__call__ does it (validate, mask refresh or shared-mask budget,
        calls, failed), and every SDPA call is the served one: the same stacked query bytes, keys, values, page table,
        cur_pos word, mask, scale, program config and output memory config."""
        operations = self.operations
        scale, memory_config = kwargs['scale'], kwargs['memory_config']
        owned, results = [], []
        protected = {addresses(operations, value) for value in (query, keys, values)}
        try:
            for reader in self.readers:
                reader.validate(reader.start)
                if reader.mask_scope is None:
                    reader.refresh()
                elif reader.calls >= reader.mask_scope:
                    raise AssertionError('Shared-mask forward exceeded its attention call budget')
            stacked = attention_block_fold_tp.fold_in(self.mesh, query, chunks, owned)
            entries = [(entry, positions) for reader in self.readers
                       for entry, positions in zip(reader.metadata, reader.cur_pos, strict=True)]
            for stack, ((bundle, pages, mask, config), positions) in zip(stacked, entries, strict=True):
                result = operations.transformer.paged_scaled_dot_product_attention_decode(stack, keys, values,
                    page_table_tensor=pages, cur_pos_tensor=positions, is_causal=False, attn_mask=mask, scale=scale,
                    program_config=config, memory_config=memory_config)
                owned.append(result)
                results.append(result)
            output = attention_block_fold_tp.fold_out(self.mesh, results, chunks, memory_config, owned)
            protected.add(addresses(operations, output))
            for reader in self.readers:
                reader.calls += 1
            self.calls += 1
            tp4_vglue.note('attn_fold')
            return output
        except BaseException:
            for reader in self.readers:
                reader.failed = True
            raise
        finally:
            release_owned(operations, [value for value in owned if addresses(operations, value) not in protected])

    def close(self):
        if getattr(self, 'closed', True):
            return
        for reader in self.readers:
            reader.close()
        self.closed = True
