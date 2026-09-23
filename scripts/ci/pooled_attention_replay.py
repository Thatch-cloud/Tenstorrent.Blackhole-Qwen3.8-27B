"""The replay attention reader over pooled page tables, beside the pinned reader.

WHY A SEPARATE MODULE. attention_replay.py is frozen-recipe evidence: the T16 attention
gates (target_t16_attention_gate.SOURCES) and the combined runtime's admission check
hash every source their reports name, and a changed attention_replay.py refused run
35495227738 (image v42) at its first admission - 'Combined runtime component source
differs: attention_replay.py'. That evidence is not regenerated cheaply, so the pinned
reader keeps its bytes at 8c102b20 and the pooled behaviour lives here, in a module the
image overrides the way it overrides model_batch.py.

WHAT THE POOL CLOSES. The per-bundle page tables are the one thing the replay reader
keeps across steps: (batches, capacity // 64) each, uploaded once at capture, read by all
sixteen attention layers inside the trace, rewritten only when the scheduler's blocks
change (serving_page_binding.VerifierPageBinding.refresh). The mask is recomputed inside
every replay from the positions word and that word is restaged before every verify
(verifier_inputs.stage_inputs), so those two are safe; a page table that an EARLIER
request's verify replay writes over - allocated after that trace was captured, into a
hole the trace baked for an intermediate it freed (serving_buffer_pool.py) - stays wrong
from the next verify on, sending every attention layer to another user's pages. Runs
35492676194 and 35493208438: the second-admitted user's text was right for its first
verify and wrong from its second, after the first user's first replay.

HOW. PooledReplayAttentionReader is the pinned reader with its page-table upload
intercepted. The base asks its `upload` callable for, in order, the 1-D int32 positions
word, then per bundle the 2-D int32 page table and the BF16 mask; the interception hands
back the lent table for every 2-D int32 request and forwards the rest, so metadata,
positions, masks, programs, refresh, shared_masks and __call__ are the base's, and
serving_page_binding.py and verifier_engine.py see the same reader. The tables come
through `storage=`, allocated at attach before any trace; the request's page table is
staged into them at construction the way refresh rewrites them, they are kept in
`borrowed`, and close() frees only what the base uploaded.

PER USER, FOR THE PACKED BLOCK (M1b). The pinned reader is one user's: one start word,
one page table repeated per bundle, bundles of four rows over ITS rows. A packed verify
block (packed_verifier.py) holds several users with different positions and page
tables, so it cannot be served by one reader - and without a replay reader it ran the
per-row SerialAttentionReader: 32 serial SDPA launches per full-attention layer at 32K
context, ~65 ms of the 150.6 ms packed verify against 9.9 ms for the bundled reader
(docs/batch-spec-tasks-2026-09-19.md, packed-round cost model). PackedReplayAttentionReader
is the segment-granularity version of the serial reader's own mechanism: one pooled
reader per user over that user's own start word and page tables, and a (1, block_rows,
12, 256) query dispatched a segment at a time - slice the user's rows, that user's
bundles, concatenate the results in row order. The masks are recomputed in-trace from
each reader's positions word, so per-user masks cost nothing extra; the block restages
every user's word and tables before each verify (packed_verifier.stage_packed).
"""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path

from attention_head_fold import parallel_groups
from attention_mask_replay import validate_ticket
from attention_replay import ReplayAttentionReader
from gdn_multitoken_conv import addresses


# QWEN-SDPA DECODE MODES (optimisation/ttnn-op/sdpa_decode_qwen, docs: the one-pass spec,
# stages 1 and 3). The grafted sdpa_decode factory reads a per-call sentinel from
# SDPAProgramConfig.q_chunk_size - a field the decode path otherwise never reads.
#   'tail' (flag 0x1, stage 1, K64e onward): the provided mask is read and added on each
#     head's FINAL k-chunk only. The replay mask is +0.0 everywhere else by construction
#     (the pinned refresh kernel writes only the last eight column tiles of a zero-initialised
#     mask), so the skipped adds were adds of +0.0; card-M byte comparison is what proves it.
#   'share' (flag 0x2, stage 3, K64f onward): in a bundle of more than one entry every entry
#     reads the SAME page-table row (the reader repeats one user's row per bundle), so entry 0
#     reads each K/V chunk from DRAM and multicasts it to the others. Exact by construction;
#     a bundle of one entry gets no 0x2. At eight-row groups each T16 user is one batch-2
#     bundle, so share halves the K/V bytes again on top of the eight-row fold.
# The pinned reader's bytes are untouched: only its per-bundle config entries are replaced,
# before any trace is captured. Unset, nothing here runs and every config is the pinned one.
SDPA_MODES_ENV = 'QWEN_FAST_SDPA_MODES'
QWEN_DECODE_MAGIC = 0x51DEC000                    # factory F1; the low byte holds the flags
QWEN_MASK_TAIL, QWEN_KV_SHARE = 0x1, 0x2
SDPA_MODE_NAMES = ('tail', 'share')               # what this build serves
# Named by the spec, not in this build: refused by name rather than as unknown.
SDPA_MODES_LATER = {'narrow': 'stage 1b (the narrow (b,1,48,256) tail mask)'}
QWEN_SDPA_BINARY_MARKER = b'[QWEN-SDPA] flags='   # factory F4's format literal, only in a graft .so
QWEN_SDPA_SHARE_MARKER = b'[QWEN-SDPA] KV-share twin bands'   # factory F9's, only in the stage-3 .so
SDPA_MODES_MARKER = '[PINDIAG] sdpa qwen-modes'
_binary_checked = []


def _pindiag(text):
    """One server-log line: loguru where the engine has it (as dflash_device.pindiag), print otherwise."""
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.info('{}', text)


def sdpa_modes(environ=None):
    """The requested decode modes from QWEN_FAST_SDPA_MODES: a comma list, empty when unset."""
    value = (os.environ if environ is None else environ).get(SDPA_MODES_ENV, '')
    modes = frozenset(name.strip() for name in value.split(',') if name.strip())
    later = sorted(modes.intersection(SDPA_MODES_LATER))
    if later:
        raise ValueError('%s=%s: %s not in this build (it serves tail and share): %s'
                         % (SDPA_MODES_ENV, value, ','.join(later), '; '.join(SDPA_MODES_LATER[name] for name in later)))
    unknown = modes.difference(SDPA_MODE_NAMES)
    if unknown:
        raise ValueError('Unknown %s entries: %s' % (SDPA_MODES_ENV, ','.join(sorted(unknown))))
    return modes


def mode_flags(modes, batches):
    return (QWEN_MASK_TAIL if 'tail' in modes else 0) | (QWEN_KV_SHARE if 'share' in modes and batches > 1 else 0)


def required_binary_markers(modes):
    """The factory format literals the loaded binary must carry for these modes: the
    [QWEN-SDPA] branch always, and the stage-3 KV-share branch for 'share' (a stage-1 .so
    such as K64e would refuse flag 0x2 by TT_FATAL at the first capture)."""
    return (QWEN_SDPA_BINARY_MARKER,) + ((QWEN_SDPA_SHARE_MARKER,) if 'share' in modes else ())


def loaded_binary_has_modes(markers=(QWEN_SDPA_BINARY_MARKER,), maps='/proc/self/maps'):
    """An old binary ignores q_chunk_size and silently runs the legacy op, so check the
    _ttnncpp.so this process actually mapped for every marker the modes need."""
    import mmap

    paths = sorted({line.split()[-1] for line in Path(maps).read_text().splitlines()
                    if line.rstrip().endswith('_ttnncpp.so')})
    if len(paths) != 1:
        raise RuntimeError('Expected exactly one mapped _ttnncpp.so, found %r' % (paths,))
    with open(paths[0], 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return paths[0], all(view.find(marker) >= 0 for marker in markers)


def apply_sdpa_modes(reader, modes, *, log=None, binary_check=None):
    """Rewrite a built replay reader's per-bundle SDPA config in place, before any trace
    capture. Only entries of its metadata list are replaced - the bundle, page table and
    mask objects are the same ones - and the pinned reader's bytes are not touched.
    Idempotent: a second call (the packed reader over a pooled one) is a no-op."""
    if not modes:
        return None
    if getattr(reader, 'sdpa_modes_applied', None) is not None:
        return reader.sdpa_modes_applied
    log = _pindiag if log is None else log
    binary_check = loaded_binary_has_modes if binary_check is None else binary_check
    modes = frozenset(modes)
    if not modes.issubset(SDPA_MODE_NAMES):
        raise ValueError('Qwen sdpa modes %s are not in this build' % ','.join(sorted(modes.difference(SDPA_MODE_NAMES))))
    if reader.short_context:
        raise ValueError('Qwen sdpa modes are long-context only')
    markers = required_binary_markers(modes)
    if not any(checked == markers for _path, checked in _binary_checked):
        path, present = binary_check(markers)
        if not present:
            if 'share' in modes:
                raise RuntimeError('%s=%s needs the stage-3 [QWEN-SDPA] KV-share factory branch (K64f onward); '
                                   '%s lacks it' % (SDPA_MODES_ENV, ','.join(sorted(modes)), path))
            raise RuntimeError('%s is set but %s lacks the [QWEN-SDPA] factory branch' % (SDPA_MODES_ENV, path))
        _binary_checked.append((path, markers))
        log('%s binary %s carries the [QWEN-SDPA] branch%s'
            % (SDPA_MODES_MARKER, path, ' with KV share' if 'share' in modes else ''))
    operations = reader.operations
    grid = reader.mesh.compute_with_storage_grid_size()
    replaced, applied = [], []
    for bundle, pages, mask, config in reader.metadata:
        flags = mode_flags(modes, len(bundle))
        if flags:
            # The pinned reader's own construction (attention_replay.py), with the sentinel.
            config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                exp_approx_mode=False, q_chunk_size=QWEN_DECODE_MAGIC | flags, k_chunk_size=256)
        replaced.append((bundle, pages, mask, config))
        applied.append(flags)
    reader.metadata[:] = replaced
    reader.sdpa_modes_applied = tuple(applied)
    log('%s modes=%s rows=%d capacity=%d bundles=%s flags=%s mask=wide'
        % (SDPA_MODES_MARKER, ','.join(sorted(modes)), reader.rows, reader.capacity,
           [len(entry[0]) for entry in reader.metadata], ['0x%x' % value for value in applied]))
    return reader.sdpa_modes_applied


# The largest native chunk family any validate_ticket regime admits (the simulator's
# context ladder tops out at 262144 + 256); family_capacities probes up to it.
MAX_FAMILY_CAPACITY = 262400


def family_start(capacity, *, short_context=False):
    """The first position of a native chunk family, exactly as the reader computes it."""
    return max(128, capacity - 256) if short_context else capacity - 256


def family_capacities(*, short_context=False, page_width=None):
    """Every native chunk family a replay reader can be captured in, in order: the
    capacities validate_ticket admits in the current regime, cut to the ones a
    `page_width`-column page table can hold (the reader slices its `capacity // 64`
    columns from the request's table). The serving pool allocates one page-table set
    per family because the family is fixed by the request's position at capture, which
    the pool cannot know at attach."""
    if page_width is not None and (type(page_width) is not int or page_width < 1):
        raise ValueError('Positive integer page-table width required')
    capacities = []
    for capacity in range(256, MAX_FAMILY_CAPACITY + 1, 256):
        if page_width is not None and capacity // 64 > page_width:
            break
        # Probed with the narrowest replay width every regime admits (short context is T8 only).
        try:
            validate_ticket(family_start(capacity, short_context=short_context), 8, capacity,
                            short_context=short_context)
        except ValueError:
            continue
        capacities.append(capacity)
    return tuple(capacities)


def bundle_batches(rows, capacity, *, max_group_rows=4, short_context=False):
    """How many groups each bundle of a `rows`-row reader in this family packs - the
    first dimension of its page table - in bundle order. Computed from the family's
    first position exactly as the reader does, so the pool's tables fit it."""
    first = family_start(capacity, short_context=short_context)
    return tuple(len(bundle) for bundle in parallel_groups(first, rows, max_group_rows=max_group_rows))


def validate_storage(operations, storage, bundles, capacity):
    """Pooled page tables, one per bundle in order, of exactly the geometry the trace
    will bake: (batches, capacity // 64), row-major int32. Or None to upload."""
    if storage is None:
        return None
    storage = list(storage)
    if len(storage) != len(bundles):
        raise ValueError('Pooled replay page tables must number the reader bundles: %d for %d'
                         % (len(storage), len(bundles)))
    for table, bundle in zip(storage, bundles, strict=True):
        shape = (len(bundle), capacity // 64)
        if (tuple(table.shape) != shape or table.dtype != operations.int32
                or table.layout != operations.ROW_MAJOR_LAYOUT):
            raise ValueError('Pooled replay page table must be a row-major int32 of shape %r' % (shape,))
    return storage


class PooledReplayAttentionReader(ReplayAttentionReader):
    def __init__(self, operations, mesh, rows, capacity, pages_host, upload, *, storage, max_group_rows=4,
                 short_context=False):
        # The base's geometry checks, first, so the lent tables can be checked against
        # the bundles they must fit before anything is uploaded.
        if (type(rows) is not int or rows not in (8, 16, 32) or type(capacity) is not int
                or type(max_group_rows) is not int or max_group_rows not in (4, 8) or type(short_context) is not bool):
            raise ValueError('Pooled replay reader requires an explicit T8/T16/T32 bucket, an integer capacity '
                             'and four- or eight-row groups')
        first = family_start(capacity, short_context=short_context)
        validate_ticket(first, rows, capacity, short_context=short_context)
        bundles = parallel_groups(first, rows, max_group_rows=max_group_rows)
        tables = validate_storage(operations, storage, bundles, capacity)
        if tables is None:
            raise ValueError('Pooled replay reader needs the lent page tables; ReplayAttentionReader uploads its own')
        self.borrowed = []

        def intercept(value, dtype=None):
            # The base uploads, in order: the 1-D positions word (int32), then per bundle
            # the 2-D page table (int32) and the BF16 mask. Only the tables are lent.
            if dtype == operations.int32 and value.ndim == 2:
                if len(self.borrowed) >= len(tables):
                    raise AssertionError('The pinned reader asked for more page tables than it has bundles')
                table = tables[len(self.borrowed)]
                if tuple(table.shape) != tuple(value.shape):
                    raise AssertionError('The pinned reader asked for a page table of shape %r; the pool lent %r'
                                         % (tuple(value.shape), tuple(table.shape)))
                self.borrowed.append(table)
                return table
            return upload(value) if dtype is None else upload(value, dtype)

        super().__init__(operations, mesh, rows, capacity, pages_host, intercept, max_group_rows=max_group_rows,
                         short_context=short_context)
        self._disown_lent()
        try:
            if len(self.borrowed) != len(tables):
                raise AssertionError('The pinned reader took %d of the %d lent page tables' % (len(self.borrowed), len(tables)))
            self.stage_pages(pages_host)
            apply_sdpa_modes(self, sdpa_modes())
        except BaseException:
            self.close()
            raise

    def _disown_lent(self):
        """The base appends every table its upload hands back to `owned`; they are the pool's."""
        lent = {id(table) for table in self.borrowed}
        self.owned[:] = [value for value in self.owned if id(value) not in lent]

    def stage_pages(self, pages_host):
        """The request's page table into the lent tables, the way VerifierPageBinding.refresh
        rewrites them on a block change: in place, fenced, addresses checked unchanged."""
        operations = self.operations
        before = [addresses(operations, entry[1]) for entry in self.metadata]
        try:
            for bundle, pages, mask, config in self.metadata:
                host = pages_host[:, :self.capacity // 64].repeat(len(bundle), 1).contiguous()
                source = operations.from_torch(host, dtype=operations.int32, layout=operations.ROW_MAJOR_LAYOUT,
                    mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
                operations.copy_host_to_device_tensor(source, pages)
            operations.synchronize_device(self.mesh)
            if [addresses(operations, entry[1]) for entry in self.metadata] != before:
                raise AssertionError('Staging replaced a pooled replay page table')
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if getattr(self, 'closed', True):
            return
        # Also on the base constructor's own failure path, which calls close() with the
        # tables it was handed still in `owned`.
        self._disown_lent()
        super().close()


PACKED_QUERY_HEADS, PACKED_QUERY_WIDTH = 12, 256


# 64: the M3 block, four T16 users (packed_shapes.m3_shape), four 16-row readers.
PACKED_BLOCK_ROWS = 64


def validate_segments(segments, block_rows_limit=PACKED_BLOCK_ROWS):
    """The packed block's row spans (target_packed_pages.segments): contiguous from row 0,
    each one a width the pinned reader takes (8, 16 or 32 rows), within the widest block."""
    segments = tuple(tuple(span) for span in segments)
    cursor = 0
    for span in segments:
        if (len(span) != 2 or any(type(value) is not int for value in span) or span[0] != cursor
                or span[1] - span[0] not in (8, 16, 32)):
            raise ValueError('Packed replay segments must tile the block from row 0 in T8/T16/T32 spans')
        cursor = span[1]
    if not segments or cursor > block_rows_limit:
        raise ValueError('Packed replay segments must fill at most a %d-row block' % block_rows_limit)
    return segments


class PackedReplayAttentionReader:
    """One replay reader per packed user, each over that user's own positions word and
    page tables, serving the block's full-attention layers as one adapter.

    `segments` are the pack's row spans in pack order; `tables_host` one (1, >= capacity // 64)
    host page table per segment; `storage` None (each reader uploads its own tables through
    `upload`, exactly as the pinned reader does) or one lent table list per segment, in
    which case each reader is the pooled one and frees none of them. Every reader is
    captured in the same native chunk family (`capacity`), the block's; a user whose
    ticket leaves it is refused by that reader's own validate.
    """

    def __init__(self, operations, mesh, segments, capacity, tables_host, upload, *, storage=None, max_group_rows=4,
                 short_context=False):
        self.segments = validate_segments(segments)
        tables_host = list(tables_host)
        lent = None if storage is None else [list(tables) for tables in storage]
        if len(tables_host) != len(self.segments) or (lent is not None and len(lent) != len(self.segments)):
            raise ValueError('One page table per packed segment required, and one lent table set per segment when pooled')
        if type(short_context) is not bool or short_context:
            raise ValueError('Packed replay attention is long-context only')
        if lent is not None:
            # Every segment's lent tables against the bundles its reader will take, before
            # any reader is built: a wrong set refuses the block with nothing staged.
            if type(capacity) is not int or type(max_group_rows) is not int:
                raise ValueError('Integer capacity and group width required')
            first = family_start(capacity, short_context=False)
            for (begin, end), tables in zip(self.segments, lent, strict=True):
                validate_ticket(first, end - begin, capacity, short_context=False)
                validate_storage(operations, tables, parallel_groups(first, end - begin, max_group_rows=max_group_rows), capacity)
        self.operations, self.mesh, self.capacity = operations, mesh, capacity
        self.rows = self.segments[-1][1]
        self.readers = []
        self.calls = 0
        self.closed = False
        self.audit = None
        # Read once, before any reader is built: a bad value refuses the block with nothing staged.
        modes = sdpa_modes()
        try:
            for index, ((first, last), pages_host) in enumerate(zip(self.segments, tables_host, strict=True)):
                if lent is None:
                    reader = ReplayAttentionReader(operations, mesh, last - first, capacity, pages_host, upload,
                                                   max_group_rows=max_group_rows, short_context=False)
                else:
                    reader = PooledReplayAttentionReader(operations, mesh, last - first, capacity, pages_host, upload,
                                                         storage=lent[index], max_group_rows=max_group_rows,
                                                         short_context=False)
                self.readers.append(reader)
                # Unpooled readers are the pinned class; a pooled one applied it at construction (no-op).
                apply_sdpa_modes(reader, modes)
        except BaseException:
            self.close()
            raise

    @property
    def borrowed(self):
        """Every pool-lent table of every reader, in segment then bundle order."""
        return [table for reader in self.readers for table in getattr(reader, 'borrowed', ())]

    @property
    def metadata(self):
        """Every reader's (bundle, pages, mask, config) entries, in segment then bundle order."""
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
            raise RuntimeError('Packed replay reader is closed')

    def validate(self, starts):
        """Every segment's start against its own reader, host only, before any copy."""
        self.check_open()
        starts = tuple(starts)
        if len(starts) != len(self.readers):
            raise ValueError('One start per packed segment required')
        for reader, start in zip(self.readers, starts, strict=True):
            reader.validate(start)

    def stage(self, starts):
        """Each reader's positions word, one fenced copy per reader (the pinned stage)."""
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
        if tuple(query.shape) != (1, self.rows, PACKED_QUERY_HEADS, PACKED_QUERY_WIDTH):
            raise ValueError('Packed replay query geometry changed')
        operations = self.operations
        rows, outputs = [], []
        try:
            for reader, (first, last) in zip(self.readers, self.segments, strict=True):
                # This user's rows, the way SerialAttentionReader takes one row: a DRAM slice on
                # the row axis, so the reader sees exactly its (1, rows, 12, 256) query.
                selected = operations.slice(query, (0, first, 0, 0), (1, last, PACKED_QUERY_HEADS, PACKED_QUERY_WIDTH),
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
