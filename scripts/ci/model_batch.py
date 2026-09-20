"""Static-fixture full-model batching; no installed or class-global patches."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

from attention_batch import OrderedCacheWriter, SerialAttentionReader, SerialCacheWriter, serial_tail
from gdn_prefix import decode_projected, gated_decode, prepare_token_rows, validate_reused_input
from packed_cache_writer import TILE_ROWS, SegmentedOrderedCacheWriter, tile as cache_tile, tile_rows


@contextmanager
def instance_overrides(bindings):
    previous = []
    try:
        for instance, name, value in bindings:
            previous.append((instance, name, name in instance.__dict__, instance.__dict__.get(name)))
            setattr(instance, name, value)
        yield
    finally:
        for instance, name, existed, value in reversed(previous):
            if existed:
                setattr(instance, name, value)
            else:
                delattr(instance, name)


def validate_pack(pack, gdn_layers=48):
    """Per-user rows for a packed verify block, or None for one sequence.

    Each user brings its own frontier, its own page table, its own accepted prefix,
    and - because the row axis is TIME for the 48 GDN layers - its own per-layer
    checkpoint and carried recurrent state. Without that last pair, user B would
    continue user A's recurrence: wrong for every row of B, and A left advanced by
    the whole block.

    The weight-heavy work is untouched by packing. Each GDN layer still runs one
    input projection across all rows and one output projection over the whole
    block; only the elementwise recurrence runs once per segment.
    """
    if pack is None:
        return None
    from target_packed_pages import packed_rows

    users = tuple(pack)
    packed = packed_rows([dict(start=user['start'], rows=user['rows'], pages=user['pages'])
                          for user in users])
    for user, (first, last) in zip(users, packed['segments']):
        width = last - first
        if (type(user.get('prefix')) is not int or not 0 <= user['prefix'] <= width
                or len(user.get('checkpoints') or ()) != gdn_layers
                or len(user.get('slots') or ()) != gdn_layers):
            raise ValueError('Each packed user needs an accepted prefix within its own segment '
                             'and one GDN checkpoint and carried state per linear layer')
    packed['prefixes'] = tuple(user['prefix'] for user in users)
    packed['checkpoints'] = tuple(tuple(user['checkpoints']) for user in users)
    packed['slots'] = tuple(tuple(user['slots']) for user in users)
    # Each user's own (1, blocks) table, for that user's replay reader (M1b).
    packed['tables'] = tuple(user['pages'] for user in users)
    return packed


def packed_replay_family(start, pack, *, short_context=False):
    """The native chunk family of a packed replay block, and every segment's ticket in it.

    The family is the capture position's, `(start // 256 + 1) * 256`, exactly as for one
    request; but the block's attention is one pinned-geometry reader PER USER
    (pooled_attention_replay.PackedReplayAttentionReader), so what must fit the family is
    each user's segment at that user's own start - not the block as one ticket.
    """
    from attention_mask_replay import validate_ticket

    if type(short_context) is not bool or short_context:
        raise ValueError('Packed replay attention is long-context only')
    capacity = (start // 256 + 1) * 256
    for first, last in pack['segments']:
        validate_ticket(int(pack['positions'][first]), last - first, capacity, short_context=False)
    return capacity


BATCH_INPUTS = ('tokens', 'positions', 'pages', 'singleton_pages', 'cos', 'sin')


def validate_storage(ttnn, storage, rows, page_width):
    """Pooled fixture inputs (serving_buffer_pool.BucketSlot.batch) for exactly this
    width and page-table width, or None to upload the fixture's own.

    The trace bakes every input's address and shape, so a bucket borrows buffers of
    its exact geometry: rows are matched by the pool's bucket, and the page table must
    be as wide as the request's - both come from serving_runtime's one page_width. A
    wider pooled table would report a larger capacity to verifier_inputs.stage_inputs
    than the request was admitted with, so it is refused rather than sliced.
    """
    if storage is None:
        return None
    integers = dict(tokens=((rows, 1), ttnn.uint32), positions=((rows,), ttnn.int32),
                    pages=((rows, page_width), ttnn.int32), singleton_pages=((1, page_width), ttnn.int32))
    for name, (shape, dtype) in integers.items():
        value = getattr(storage, name, None)
        if value is None or tuple(value.shape) != shape or value.dtype != dtype or value.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError('Pooled fixture input %r must be a row-major %s of shape %r; the pooled page-table '
                             'width must equal the request\'s %d' % (name, dtype, shape, page_width))
    singletons = list(getattr(storage, 'singleton_positions', None) or ())
    if len(singletons) != rows or any(tuple(value.shape) != (1,) or value.dtype != ttnn.int32
                                      or value.layout != ttnn.ROW_MAJOR_LAYOUT for value in singletons):
        raise ValueError('Pooled fixture needs one row-major int32 singleton position per row')
    for name in ('cos', 'sin'):
        value = getattr(storage, name, None)
        if value is None or len(value.shape) != 4 or value.shape[1] != rows or value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT:
            raise ValueError('Pooled fixture input %r must be a tiled BF16 rotary table for %d rows' % (name, rows))
    return storage


def replay_storage(storage, capacity):
    """The pooled replay page tables for this capture's native chunk family, or None
    when the fixture uploads its own.

    The trace bakes each bundle's page table (attention_replay.py) and the request keeps
    it across steps, so pooled fixtures must borrow it: a pool that lends the other
    fixture inputs but not these would leave the one persistent per-request allocation
    that runs 35492676194 and 35493208438 found overwritten. The family is the capture
    position's, `(start // 256 + 1) * 256`, so the pool holds one set per family it can
    serve (serving_buffer_pool.py, pooled_attention_replay.family_capacities) and the
    fixture picks its own; a family the pool lacks is refused here, before any upload.
    """
    if storage is None:
        return None
    tables = getattr(storage, 'replay_pages', None)
    if not isinstance(tables, dict) or capacity not in tables:
        raise ValueError('Pooled fixture holds replay page tables for families %r; this capture is in family %d'
                         % (sorted(tables) if isinstance(tables, dict) else None, capacity))
    return list(tables[capacity])


def prepare_inputs(ttnn, model, rows, tokens, positions, page_rows, pages, *, storage=None, packed=False):
    """The fixture's device inputs, uploaded (owned) or staged into pooled buffers (borrowed).

    The unpooled order is the one the fixture always had: tokens, positions, pages,
    the singleton page table, the packed per-row tables, the singleton positions, then
    the rotary tables. Pooled, the same host values go through copy_host_to_device_tensor
    into the lent buffers - the path verifier_inputs.stage_inputs takes before every
    verify - and the rotary tables are copied from the native construction, so the
    values are the same and only the addresses differ: pre-trace, pooled ones.
    """
    import torch
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

    storage = validate_storage(ttnn, storage, rows, pages.shape[1])
    if storage is not None and packed:
        raise ValueError('Pooled fixture inputs describe one request; a packed block cannot borrow them')
    result = SimpleNamespace(owned=[], borrowed=[], staged=[])

    def upload(value, dtype):
        uploaded = ttnn.from_torch(value, device=model.mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device))
        result.owned.append(uploaded)
        return uploaded

    def stage(value, dtype, destination):
        if tuple(destination.shape) != tuple(value.shape):
            raise ValueError('Pooled fixture input shape %r does not fit %r' % (tuple(destination.shape), tuple(value.shape)))
        host = ttnn.from_torch(value, device=None, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device))
        ttnn.copy_host_to_device_tensor(host, destination)
        result.staged.append(host)
        result.borrowed.append(destination)
        return destination

    def place(value, dtype, name):
        return upload(value, dtype) if storage is None else stage(value, dtype, getattr(storage, name))

    result.tokens = place(torch.tensor(tokens, dtype=torch.int32).reshape(rows, 1), ttnn.uint32, 'tokens')
    result.positions = place(positions, ttnn.int32, 'positions')
    result.pages = place(page_rows, ttnn.int32, 'pages')
    result.singleton_pages = place(pages, ttnn.int32, 'singleton_pages')
    # Per row, so a packed row reads its own user's blocks rather than the first
    # user's table repeated. Unpacked, every row table IS the singleton page table -
    # pooled with it, so no per-request row table outlives a trace.
    result.row_tables = ([upload(page_rows[index:index + 1], ttnn.int32) for index in range(rows)]
                         if packed else [result.singleton_pages] * rows)
    result.singleton_positions = [upload(position.reshape(1), ttnn.int32) if storage is None
                                  else stage(position.reshape(1), ttnn.int32, storage.singleton_positions[index])
                                  for index, position in enumerate(positions)]
    # A block wider than the ordered cache kernel's 32-row tile (the 64-row M3 block)
    # writes its K/V one tile at a time (packed_cache_writer.SegmentedOrderedCacheWriter),
    # each tile over its own positions word and page-table rows: uploaded here, owned,
    # and restaged with the rest of the packed inputs before every verify. Pooled inputs
    # describe one request within a tile, so a pooled fixture never has any.
    result.cache_tiles = [cache_tile((first, last), upload(positions[first:last], ttnn.int32),
                                     upload(page_rows[first:last], ttnn.int32))
                          for first, last in (tile_rows(rows) if rows > TILE_ROWS else ())]
    cos, sin = rot_mats_decode(model.mesh_device, model.args.rope_head_dim,
                               model.args.max_seq_len, model.args.rope_theta, positions)
    if storage is None:
        result.cos, result.sin = cos, sin
        result.owned.extend([cos, sin])
        return result
    try:
        for value, destination in ((cos, storage.cos), (sin, storage.sin)):
            if (tuple(destination.shape) != tuple(value.shape) or destination.dtype != value.dtype
                    or destination.layout != value.layout):
                raise ValueError('Pooled rotary table %r does not match the native %r' % (tuple(destination.shape), tuple(value.shape)))
            ttnn.copy(value, destination)
            result.borrowed.append(destination)
        # Before the host sources and the native tables go: every copy above reads them.
        before = [tuple(part.buffer_address() for part in ttnn.get_device_tensors(value)) for value in result.borrowed]
        ttnn.synchronize_device(model.mesh_device)
        if [tuple(part.buffer_address() for part in ttnn.get_device_tensors(value)) for value in result.borrowed] != before:
            raise AssertionError('Staging the fixture inputs replaced a pooled buffer')
    finally:
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        result.staged.clear()
    result.cos, result.sin = storage.cos, storage.sin
    return result


# 64: the M3 packed block, four T16 users (packed_shapes.m3_shape); the GDN decode runs
# it as four 16-row segments and the K/V write as two 32-row tiles.
BLOCK_WIDTHS = (1, 2, 4, 8, 16, 32, 64)


def validate_checkpoint(rows, prefix):
    if type(rows) is not int or rows not in BLOCK_WIDTHS:
        raise ValueError("Expected T=1/2/4/8/16/32/64")
    if type(prefix) is not int or not 0 <= prefix <= rows:
        raise ValueError("Checkpoint must be in [0, T]")


def recurrence_rows(rows, pack, norm_batch):
    """The rows one GDN recurrence runs over, which is where the norm-batch decision is
    made (gdn_batched_conv.norm_batch_enabled): the whole block unpacked, or packed each
    user's own segment (gdn_device_loop_state._decode_packed runs the batched recurrence
    per segment, and DeviceLoopState checks the decision at the segment's rows)."""
    from gdn_batched_conv import norm_batch_enabled

    if pack is None:
        return rows
    widths = sorted({last - first for first, last in pack['segments']})
    if len({norm_batch_enabled(width, norm_batch) for width in widths}) != 1:
        raise ValueError('Packed segments must all take the same norm-batch decision')
    return widths[0]


def cache_writer(ttnn, mesh, kernels, *, ordered_cache, cache_tiles, serial):
    """One full-attention layer's K/V writer. The serial per-row writer
    (attention_batch.SerialCacheWriter, pinned frozen-recipe evidence, 32 rows at most) is
    built only where it serves; the ordered writer replaces it, and beyond one 32-row tile
    the ordered write runs tile by tile over the fixture's cache tiles
    (packed_cache_writer.SegmentedOrderedCacheWriter)."""
    if not ordered_cache:
        return serial()
    if cache_tiles:
        return SegmentedOrderedCacheWriter(mesh, ttnn, kernels, cache_tiles)
    return OrderedCacheWriter(mesh, ttnn, kernels)


def attention_reader(*, replay_reader, serial_sdpa, grouped_attention, serial, grouped):
    """One full-attention layer's reader: the fixture's replay reader when it has one, the
    grouped reader when selected, else the serial per-row reader
    (attention_batch.SerialAttentionReader, pinned, 32 rows at most) - built only where it
    serves - or none."""
    if replay_reader is not None:
        return replay_reader
    if grouped_attention:
        return grouped()
    return serial() if serial_sdpa else None


def two_tile_bindings(rows, model, operations):
    """The wide block's two-tile binders (two_tile_norm.py, two_tile_decode.py), or none
    within one tile. Each carries `bindings`, the (instance, name, value) triples applied
    through instance_overrides around the block's forward, and `calls` against
    `expected_calls`, which run() checks per forward.

    Beyond one 32-row tile the model's own decode path assumes one tile in three places
    (run 35502452429 refused the 64-row block at layer 0's attention norm gather): the
    decode norm configs from `args.get_norm_config`, the attention's fused prep path (gated
    at one tile, with a one-tile fused QKV config beneath it) and the MLP's first arm (the
    prefill all-gather fusion, taken above one tile). The block supplies each from its own
    side. Within one tile there is nothing: the 32-row block's calls stay exactly as the M1
    gate ran them, and the modules are not imported.

    Two more of the model's own decode-mode linear projections take a slow prefill arm
    above one tile: the attention output projection (`_wo_proj`, looked up by
    `_decode_from_prep`) and the GDN output projection (`_row_proj`, looked up by
    gdn_multitoken_conv.finish_output, FROZEN). Both are bound two-tile the same way as
    the head concat and the MLP forward - `_wo_proj` alongside the attention binder,
    `_row_proj` as its own GDN output binder."""
    validate_checkpoint(rows, rows)
    if rows <= TILE_ROWS:
        return ()
    from two_tile_norm import bind_two_tile_norms
    from two_tile_decode import bind_two_tile_attention, bind_two_tile_gdn_output, bind_two_tile_mlp

    return (bind_two_tile_norms(model, rows, operations), bind_two_tile_attention(model, rows, operations),
            bind_two_tile_mlp(model, rows, operations), bind_two_tile_gdn_output(model, rows, operations))


def compact_gdn_enabled(rows, requested, serial_sdpa, profiler):
    validate_checkpoint(rows, rows)
    if requested and (not serial_sdpa or profiler is not None):
        raise ValueError("Compact GDN requires the unprofiled B1-SDPA correctness path")
    return bool(requested and rows > 1)


def device_loop_enabled(rows, requested, compact_gdn, hoist_row_layout, compact_prologue=False, packed_checkpoints=False):
    validate_checkpoint(rows, rows)
    if requested and not (compact_gdn and hoist_row_layout):
        raise ValueError('Device loop requires the exact compact row-layout control')
    return bool(requested and rows >= (8 if compact_prologue and not packed_checkpoints else 2))


class ModelBatch:
    def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, serial_sdpa=False, profiler=None,
                 compact_gdn=False, reuse_gdn_input=False, skip_row_clones=False, hoist_row_layout=False,
                 device_loop_gdn=False, compact_prologue=False, batch_conv=False, packed_checkpoints=False,
                 retain_records=False, ordered_cache=False, norm_batch=False, grouped_attention=False, attention_dma=False,
                 attention_parallel=False, attention_replay=False, attention_tree=False, attention_mask_once=False,
                 replay_group_rows=4, prefix_zero_reuse=False, defer_conv_publication=False, short_context=False,
                 attention_audit=False, commit_only_gdn=False, pack=None, storage=None, packed_replay_pages=None):
        import torch
        import ttnn

        self.rows = len(tokens)
        self.pack = validate_pack(pack)
        if self.pack is not None and self.pack['rows'] != self.rows:
            raise ValueError('Packed segments must cover exactly the block rows')
        validate_checkpoint(self.rows, prefix)
        if type(commit_only_gdn) is not bool or (commit_only_gdn and self.rows > 1 and not (
                device_loop_gdn and packed_checkpoints and batch_conv and retain_records)):
            raise ValueError('Commit-only GDN requires a retained packed-history verifier and an explicit decision')
        self.commit_only_gdn = commit_only_gdn and self.rows > 1
        # A packed retained block is decided per user after the verify readback, through
        # the block's commit_user; the decode must therefore defer, which is the
        # commit-only device loop with one entry per user.
        if self.pack is not None and retain_records and not self.commit_only_gdn:
            raise ValueError('A packed retained block commits per user after the readback: commit-only GDN required')
        if type(short_context) is not bool or (short_context and (not attention_replay or replay_group_rows != 4)):
            raise ValueError('Short-context attention requires explicit four-row replay groups')
        self.short_context = short_context
        if type(attention_audit) is not bool or (attention_audit and not short_context):
            raise ValueError('Diagnostic shadow attention requires explicit short-context replay')
        if type(attention_tree) is not bool or (attention_tree and not attention_parallel):
            raise ValueError('Eight-row attention requires explicit parallel selection')
        self.attention_tree = bool(attention_tree and self.rows >= 8)
        if attention_replay and (not ordered_cache or not serial_sdpa or not norm_batch or profiler is not None or grouped_attention):
            raise ValueError('Replay attention requires standalone ordered-cache norm-batch verification')
        self.attention_replay = bool(attention_replay and self.rows >= 8)
        # The packed block's lent per-user replay tables (serving_buffer_pool.PackedReplayTables)
        # describe a packed replay fixture and nothing else.
        if packed_replay_pages is not None and (self.pack is None or not self.attention_replay):
            raise ValueError('Packed replay page tables need a packed fixture with replay attention')
        if type(replay_group_rows) is not int or replay_group_rows not in (4, 8) or (replay_group_rows == 8 and not attention_replay):
            raise ValueError('Eight-row replay grouping requires explicit replay attention')
        self.replay_group_rows = replay_group_rows
        if type(attention_mask_once) is not bool or (attention_mask_once and not attention_replay):
            raise ValueError('Shared attention masks require explicit replay attention')
        self.attention_mask_once = attention_mask_once and self.attention_replay
        self.replay_reader = None
        if self.attention_replay and self.pack is None:
            from attention_mask_replay import validate_ticket
            self.replay_capacity = (start // 256 + 1) * 256
            validate_ticket(start, self.rows, self.replay_capacity, short_context=short_context)
        elif self.attention_replay:
            # Packed: one family for the block, keyed on the capture position as unpacked,
            # and each user's segment validated at that user's own start (its own reader).
            self.replay_capacity = packed_replay_family(start, self.pack, short_context=short_context)
        if attention_parallel and not attention_dma:
            raise ValueError('Parallel attention requires DMA layout')
        self.attention_parallel = bool(attention_parallel and self.rows >= 8)
        if attention_dma and not grouped_attention:
            raise ValueError('Attention DMA requires grouped attention')
        self.attention_dma = bool(attention_dma and self.rows >= 8)
        if grouped_attention and (not ordered_cache or not serial_sdpa or profiler is not None or retain_records):
            raise ValueError('Grouped attention requires static ordered-cache verification without retained replay')
        self.grouped_attention = bool(grouped_attention and self.rows >= 8)
        if ordered_cache and (not serial_sdpa or profiler is not None):
            raise ValueError('Ordered cache requires the unprofiled exact B1 SDPA path')
        self.ordered_cache = ordered_cache and self.rows > 1
        # Beyond one 32-row tile (the 64-row M3 block) the per-row serial adapters of the
        # pinned attention_batch.py stop: the K/V write goes tile by tile through the
        # ordered writer and the attention through the per-user replay readers, or the
        # block is refused here, before any upload.
        if self.rows > TILE_ROWS and not (self.ordered_cache and self.attention_replay):
            raise ValueError('A block wider than %d rows needs the ordered cache writer and replay attention'
                             % TILE_ROWS)
        cache_kernels = None
        if self.ordered_cache:
            import os
            from ordered_cache import load_kernels
            cache_kernels = load_kernels(os.environ['TT_METAL_HOME'])
        self.compact_gdn = compact_gdn_enabled(self.rows, compact_gdn, serial_sdpa, profiler)
        if reuse_gdn_input and not compact_gdn:
            raise ValueError("Input reuse requires the exact compact GDN control")
        self.reuse_gdn_input = reuse_gdn_input and self.rows > 1
        if skip_row_clones and not reuse_gdn_input:
            raise ValueError("Clone removal requires the exact reused-input control")
        self.skip_row_clones = skip_row_clones and self.rows > 1
        if hoist_row_layout and not skip_row_clones:
            raise ValueError("Layout hoisting requires the selective-clone control")
        self.hoist_row_layout = hoist_row_layout and self.rows > 1
        self.device_loop_gdn = device_loop_enabled(self.rows, device_loop_gdn, compact_gdn, hoist_row_layout,
                                                   compact_prologue, packed_checkpoints)
        if compact_prologue and not device_loop_gdn:
            raise ValueError('Compact prologue requires device-loop GDN')
        self.compact_prologue = compact_prologue and self.device_loop_gdn
        if batch_conv and not compact_prologue:
            raise ValueError('Batched convolution requires compact-prologue control')
        self.batch_conv = batch_conv and self.device_loop_gdn
        if packed_checkpoints and not batch_conv:
            raise ValueError('Packed checkpoints require batched convolution')
        self.packed_checkpoints = packed_checkpoints and self.device_loop_gdn
        if type(defer_conv_publication) is not bool or (defer_conv_publication and not packed_checkpoints):
            raise ValueError('Deferred publication requires explicit bool and packed checkpoints')
        self.defer_conv_publication = defer_conv_publication and self.packed_checkpoints
        if type(prefix_zero_reuse) is not bool or (prefix_zero_reuse and not packed_checkpoints):
            raise ValueError('Prefix zero reuse requires explicit bool and packed checkpoints')
        self.prefix_zero_reuse = prefix_zero_reuse and self.packed_checkpoints
        from gdn_batched_conv import norm_batch_enabled
        if norm_batch and not packed_checkpoints:
            raise ValueError('Norm batching requires packed checkpoints')
        self.norm_batch = norm_batch_enabled(recurrence_rows(self.rows, self.pack, norm_batch), norm_batch)
        self.norm_batch_calls = 0
        if retain_records and not self.packed_checkpoints:
            raise ValueError('Retained records require active packed checkpoints')
        from gdn_records import RetainedGDNBlock
        self.retained = RetainedGDNBlock(self.rows, ttnn) if retain_records else None
        self.working_states = []
        if len(helpers) != 48 or len(checkpoints) != 48 or len(model.layers) != 64:
            raise ValueError("Expected all 64 model layers and 48 GDN checkpoints")
        self.model = model
        self.operations = ttnn
        self.prefix = prefix
        self.profiler = profiler
        self.buffers = []
        # Pooled inputs (serving_buffer_pool.BucketSlot.batch): read and written here,
        # never freed here - close() returns nothing the pool lent.
        self.borrowed = []
        self.bindings = []
        self.writers = []
        self.readers = []
        self.grouped_readers = []
        self.gdn_calls = 0

        # Packed, the rows belong to different users: each brings its own frontier
        # and its own blocks, so neither a single arange nor one repeated page table
        # describes the block any more.
        positions = (self.pack['positions'] if self.pack is not None
                     else torch.arange(start, start + self.rows, dtype=torch.int32))
        page_rows = self.pack['pages'] if self.pack is not None else pages.repeat(self.rows, 1)
        if tuple(positions.shape) != (self.rows,) or page_rows.shape[0] != self.rows:
            raise ValueError('One position and one page-table row per query row required')
        inputs = prepare_inputs(ttnn, model, self.rows, tokens, positions, page_rows, pages,
                                storage=storage, packed=self.pack is not None)
        self.buffers.extend(inputs.owned)
        self.borrowed.extend(inputs.borrowed)
        self.tokens, self.positions, self.pages = inputs.tokens, inputs.positions, inputs.pages
        singleton_pages = self.singleton_pages = inputs.singleton_pages
        self.row_pages = inputs.row_tables
        self.cache_tiles = inputs.cache_tiles
        singleton_positions = self.singleton_positions = inputs.singleton_positions
        self.cos, self.sin = inputs.cos, inputs.sin
        if self.attention_replay:
            from attention_replay import ReplayAttentionReader

            def upload_replay(value, dtype=ttnn.bfloat16):
                return ttnn.from_torch(value, device=model.mesh_device, dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device))

            # Pooled, the per-bundle page tables are the slot's for this capture's family
            # and the reader is the pooled one (pooled_attention_replay.py), which stages
            # the request's table into them and frees none of them. attention_replay.py
            # is frozen-recipe evidence, pinned byte for byte, so the unpooled call is the
            # pinned reader exactly as before. Packed (M1b), the rows belong to several
            # users and the pinned reader takes one start and one table: one reader per
            # user, over the block's lent table sets (packed_replay_pages,
            # serving_buffer_pool.PackedReplayTables) or its own uploads when unpooled.
            tables = None if self.pack is not None else replay_storage(storage, self.replay_capacity)
            if self.pack is not None:
                from pooled_attention_replay import PackedReplayAttentionReader

                self.replay_reader = PackedReplayAttentionReader(ttnn, model.mesh_device, self.pack['segments'],
                    self.replay_capacity, self.pack['tables'], upload_replay, storage=packed_replay_pages,
                    max_group_rows=self.replay_group_rows)
                self.borrowed.extend(self.replay_reader.borrowed)
            elif tables is None:
                self.replay_reader = ReplayAttentionReader(ttnn, model.mesh_device, self.rows, self.replay_capacity, pages,
                    upload_replay, max_group_rows=self.replay_group_rows, short_context=self.short_context)
            else:
                from pooled_attention_replay import PooledReplayAttentionReader

                self.replay_reader = PooledReplayAttentionReader(ttnn, model.mesh_device, self.rows, self.replay_capacity,
                    pages, upload_replay, storage=tables, max_group_rows=self.replay_group_rows,
                    short_context=self.short_context)
                self.borrowed.extend(self.replay_reader.borrowed)
            self.grouped_readers.append(self.replay_reader)
            if attention_audit:
                from attention_replay_audit import AttentionReplayAudit
                self.replay_reader.audit = AttentionReplayAudit(ttnn,
                    SerialAttentionReader(ttnn, singleton_positions, self.row_pages),
                    pages=pages[:, :self.replay_capacity // 64],
                    output_directory='/experiment/results/attention-mismatch', masks=self.replay_reader.metadata)
            if self.pack is not None:
                starts = tuple(int(positions[first]) for first, last in self.pack['segments'])
                if self.replay_reader.starts != starts:
                    self.replay_reader.stage(starts)
            elif self.replay_reader.start != start:
                self.replay_reader.stage(start)
        gdn_index = 0
        for layer in model.layers:
            attention = layer.attention
            if layer.is_full_attention:
                def grouped(attention=attention):
                    from attention_grouped import GroupedAttentionReader

                    def upload_group(value, dtype=ttnn.bfloat16):
                        return ttnn.from_torch(value, device=model.mesh_device, dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device))

                    reader = GroupedAttentionReader(ttnn, model.mesh_device, start, self.rows, pages,
                        singleton_positions, singleton_pages, upload_group, dma_layout=self.attention_dma, parallel=self.attention_parallel,
                        max_group_rows=8 if self.attention_tree else 4)
                    self.grouped_readers.append(reader)
                    return reader

                writer = cache_writer(ttnn, model.mesh_device, cache_kernels, ordered_cache=self.ordered_cache,
                    cache_tiles=self.cache_tiles,
                    serial=lambda attention=attention: SerialCacheWriter(ttnn, singleton_positions, self.row_pages,
                                                                         attention._kv_shard_cfg(1)))
                self.writers.append(writer)
                reader = attention_reader(replay_reader=self.replay_reader, serial_sdpa=serial_sdpa,
                    grouped_attention=self.grouped_attention, grouped=grouped,
                    serial=lambda: SerialAttentionReader(ttnn, singleton_positions, self.row_pages))
                if reader is not None:
                    self.readers.append(reader)
                write = profiler.wrap("attention.kv_write", writer) if profiler else writer
                read = profiler.wrap("attention.sdpa_and_row_packing", reader) if profiler and reader else reader
                self.bindings.append((attention, "_decode_from_prep", serial_tail(attention, write, ttnn, read)))
                if profiler:
                    for name, category in (("forward_decode", "attention.block"),
                                           ("_qkv_raw_decode", "attention.input_projection"),
                                           ("_wo_proj", "attention.output_projection")):
                        self.bindings.append((attention, name, profiler.wrap(category, getattr(attention, name))))
            else:
                if helpers[gdn_index].gdn is not attention:
                    raise ValueError("GDN checkpoint layer order mismatch")
                forward = self.gdn_forward(attention, helpers[gdn_index], checkpoints[gdn_index], gdn_index)
                if profiler:
                    forward = profiler.wrap("gdn.block", forward)
                    for name, category in (("_project_qkvzab_raw", "gdn.input_projection"),
                                           ("_row_proj", "gdn.output_projection"),
                                           ("_slice_along", "gdn.active_state_slice"),
                                           ("_write_recurrent_state_prefix", "gdn.active_state_write")):
                        self.bindings.append((attention, name, profiler.wrap(category, getattr(attention, name))))
                self.bindings.append((attention, "forward_decode", forward))
                gdn_index += 1
        # Beyond one tile, the model's decode norms, the attention's fused prep path and the
        # MLP's first arm all assume one tile; the block binds its own two-tile forms of
        # each (two_tile_bindings), after the per-layer adapters so a full-attention
        # forward finds the block's writer and readers already bound. Nothing within one tile.
        self.two_tile = two_tile_bindings(self.rows, model, ttnn)
        for binder in self.two_tile:
            self.bindings.extend(binder.bindings)
        if profiler:
            from stage_profile import decoder_bindings
            for index, layer in enumerate(model.layers):
                self.bindings.extend(decoder_bindings(layer, index, profiler))
            for name, category in (("embd", "embedding"), ("_final_norm_decode", "final_norm"), ("_lm_head", "lm_head")):
                self.bindings.append((model, name, profiler.wrap(category, getattr(model, name))))

    def gdn_forward(self, layer, helper, checkpoint, gdn_slot=0):
        operations = self.operations
        if self.device_loop_gdn:
            from pathlib import Path
            from gdn_device_loop_state import DeviceLoopState
            from gdn_multitoken import load_kernels
            from gdn_multitoken_conv import finish_output, release_owned
            # Commit-only and packed: the decode defers every user's decision, so each
            # user needs its own block-start entry for its later commit.
            deferred = self.pack is not None and self.commit_only_gdn
            state = DeviceLoopState(helper, operations, load_kernels(Path('/opt/tt-metal'), True),
                                    self.compact_prologue, self.batch_conv, self.batch_conv, self.packed_checkpoints,
                                    norm_batch=self.norm_batch, prefix_zero_reuse=self.prefix_zero_reuse,
                                    defer_conv_publication=self.defer_conv_publication,
                                    **(dict(commit_only=True) if self.commit_only_gdn else {}),
                                    **(dict(users=len(self.pack['segments'])) if deferred else {}))
            self.working_states.append(state)

            def device_forward(value):
                from models.tt_transformers.tt.ccl import tt_all_reduce
                if tuple(value.shape) != (1, 1, self.rows, 5120):
                    raise ValueError('Unexpected full-model GDN input geometry')
                packed = operations.reshape(value, (1, self.rows, 5120))
                if self.pack is None:
                    result = state.decode(packed, checkpoint, self.prefix)
                else:
                    # One recurrence per user, each from its own carried state, over
                    # its own slice of the single input projection. Deferred, the
                    # pack's prefixes are placeholders: the readback decides them.
                    result = state.decode(packed,
                        [user[gdn_slot] for user in self.pack['checkpoints']],
                        list(self.pack['prefixes']),
                        segments=self.pack['segments'],
                        slots=[user[gdn_slot] for user in self.pack['slots']],
                        deferred=deferred)
                if result.get('commit_only_gdn', False) != self.commit_only_gdn:
                    raise AssertionError('Commit-only GDN must engage in every selected layer')
                finish_output(layer, result, operations, tt_all_reduce)
                output = result['layer_output']
                if self.retained is not None:
                    from gdn_records import retain_checkpoint_histories
                    retain_checkpoint_histories(operations, result, output)
                    # Packed, each user's decision is committed into that user's carry,
                    # the state the next block's decode restores that user from.
                    self.retained.append(state, result, checkpoint if self.pack is None
                                         else tuple(user[gdn_slot] for user in self.pack['slots']))
                else:
                    release_owned(operations, [value for value in result['owned'] if value is not output])
                self.gdn_calls += 1
                self.norm_batch_calls += int(result.get('norm_batch', False))
                return output

            return device_forward
        if self.reuse_gdn_input:
            import inspect
            validate_reused_input(inspect.getsource(type(layer).forward_decode))
        native_gated = gated_decode(layer, profiler=self.profiler)
        snapshot = helper.save
        working = None
        if self.compact_gdn:
            from gdn_working_state import WorkingState
            working = WorkingState(helper, operations, compact_dma=True, skip_row_clones=self.skip_row_clones,
                                   hoist_row_layout=self.hoist_row_layout)
            self.working_states.append(working)
            snapshot = working.save
        if self.profiler:
            native_gated = self.profiler.wrap("gdn.native_row", native_gated)
            snapshot = self.profiler.wrap("gdn.checkpoint", snapshot)

        def forward(value):
            from models.tt_transformers.tt.ccl import tt_all_reduce

            if tuple(value.shape) != (1, 1, self.rows, 5120):
                raise ValueError("Unexpected full-model GDN input geometry")
            packed = operations.reshape(value, (1, self.rows, 5120))
            tokens, owned_tokens = prepare_token_rows(operations, packed, reuse=self.reuse_gdn_input)

            def save(prefix):
                if prefix == self.prefix:
                    snapshot(checkpoint)

            if working:
                outputs = working.decode(packed, tokens, save)
            else:
                save(0)
                outputs = decode_projected(layer, packed, tokens, save, operations, forward=native_gated,
                                           profiler=self.profiler)
            gated = outputs[0] if self.rows == 1 else operations.concat(outputs, dim=1)
            partial = layer._row_proj(gated, layer.tw["out"])
            if self.rows != 1:
                operations.deallocate(gated)
            for tensor in outputs + owned_tokens:
                operations.deallocate(tensor)
            partial = operations.reshape(partial, (1, 1, self.rows, partial.shape[-1]))
            result = tt_all_reduce(partial, self.model.mesh_device, layer.tt_ccl, cluster_axis=0, dim=3,
                                   topology=self.model.args.ccl_topology(), memory_config=operations.DRAM_MEMORY_CONFIG)
            self.gdn_calls += 1
            return result

        return forward

    def run(self, *, sharded_logits=False):
        if self.retained is not None and (self.retained.closed or self.retained.records):
            raise ValueError('A retained fixture owns exactly one captured or eager block')
        before_gdn = self.gdn_calls
        before_norm_batch = self.norm_batch_calls
        before_compact = [(state.calls, state.checkpoint_calls) for state in self.working_states]
        before_clones = [state.skipped_clones for state in self.working_states]
        before_writes = [writer.calls for writer in self.writers]
        before_reads = [reader.calls for reader in self.readers]
        before_mask_refresh = self.replay_reader.refresh_calls if self.attention_mask_once else 0
        two_tile = tuple(getattr(self, 'two_tile', ()))
        before_two_tile = [binder.calls for binder in two_tile]
        mask_scope = self.replay_reader.shared_masks(16) if self.attention_mask_once else nullcontext()
        with instance_overrides(self.bindings), mask_scope:
            result = self.model._forward_decode(self.tokens, self.cos, self.sin, self.positions, self.pages,
                **({'sharded_lm_head': True} if sharded_logits else {}))
        if self.attention_mask_once and self.replay_reader.refresh_calls - before_mask_refresh != len(self.replay_reader.metadata):
            raise AssertionError('Shared masks must refresh exactly once per model forward')
        for binder, before in zip(two_tile, before_two_tile, strict=True):
            if binder.calls - before != binder.expected_calls:
                raise AssertionError('Every %s of the wide block must take its two-tile form: %d engaged, %d expected'
                                     % (binder.label, binder.calls - before, binder.expected_calls))
        if self.gdn_calls - before_gdn != 48 or any(
            writer.calls - before != 2 for writer, before in zip(self.writers, before_writes, strict=True)
        ):
            raise AssertionError("All 48 GDN and 16 attention adapters must engage")
        if self.norm_batch_calls - before_norm_batch != (48 if self.norm_batch else 0):
            raise AssertionError('Selected row-parallel norm must engage in all48 GDN layers')
        if any(reader.calls - before != (16 if self.attention_replay else 1)
               for reader, before in zip(self.readers, before_reads, strict=True)):
            raise AssertionError("Every selected B1 SDPA adapter must engage")
        # One decision per block, or packed one per user: each user's segment is its
        # own checkpoint, deferred or not.
        if len(self.working_states) != (48 if self.compact_gdn else 0) or any(
            state.calls - before[0] != (1 if self.device_loop_gdn else self.rows)
            or state.checkpoint_calls - before[1] != (1 if self.pack is None else len(self.pack['segments']))
            for state, before in zip(self.working_states, before_compact, strict=True)
        ):
            raise AssertionError("Every compact GDN layer must update in place and checkpoint exactly once per user")
        if any(state.skipped_clones - before != (self.rows - 1 if self.skip_row_clones and not self.device_loop_gdn else 0)
               for state, before in zip(self.working_states, before_clones, strict=True)):
            raise AssertionError("Projected-row clone removal did not engage exactly")
        return result

    def close(self):
        if self.retained is not None:
            self.retained.close()
        for state in self.working_states:
            state.close()
        self.working_states.clear()
        for reader in self.grouped_readers:
            reader.close()
        self.grouped_readers.clear()
        for value in self.buffers:
            self.operations.deallocate(value)
        self.buffers.clear()
        if getattr(self, 'borrowed', None):
            self.borrowed.clear()
