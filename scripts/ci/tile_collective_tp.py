"""The four-card block all-reduce, one 32-row tile at a time: the packed verify block's reduction order made the sequential engine's.

WHY (S3a at four cards, docs/tp4-exact-ring-parity.md). At the pair every cross-chip sum has two addends, which
commute, so the block's second 32-row tile can never differ from a one-tile call. At four cards the model's
tt_all_reduce (ccl.py: at (1, N) the reduce_scatter_minimal_async alone, each chip keeping its slice) runs RING collectives, and the ring
reduce-scatter sends even chunks forward and odd chunks backward, so the two directions add the four partials in
different orders. The parity of a chunk is fixed by the flat tile index inside the per-chip slice
(reduce_scatter_common::chunk_ring_parity: (tiles_read / tile_granularity) % 2), and the slice of a (1, 1, 64, 5120)
input at four chips is two rows of 40 tiles: with a chunk of 8 tiles the second row starts at chunk 5, an odd one.
Every chunk of rows 32..63 therefore has the opposite direction, hence a different association, from the same row in
a one-tile call (the sequential engine's four rows are one tile). The sums differ in the last bit only, which is
enough to flip a near-tie: users 2 and 3 of the S3a matrix sit in rows 32..63, users 0 and 1 in rows 0..31.

WHAT. TileSplitAllReduce wraps the model's own tt_all_reduce. Inside block_scope(rows) it runs a `rows`-row all-reduce
(rows a whole number of tiles beyond one) as one call per 32-row tile - the very call, with the very arguments, the
sequential engine makes on its one tile - and joins the results on the row axis. Everything else (prefill, the
sequential engine, the drafter, any other shape) passes straight through. It is installed by tp_addresses.install(),
which the process runs at QWEN_FAST_TP != 2 only, so the pair never sees it. The same install wraps
model_batch.ModelBatch.run (a class attribute; the pinned file is not edited) so that the block's forward runs inside the
scope, and the scope refuses the round unless every all-reduce the block issues was split.

The wrapper consumes its input, as tt_all_reduce does (ccl.py deallocates it after the reduce-scatter).

U1 (QWEN_FAST_TP4_RS_UNIT_MAJOR=1, default off, byte-identical off). The per-tile split costs two slices, two reduce-scatters and a
concat per all-reduce. The stock reduce_scatter_minimal_async called ONCE on the unit-major view (1, R/32, 32, W) restarts the
ring's chunk parity for every 32-row unit (the reader resets tiles_read per channel), so each unit is reduced in the very order
of its one-tile call: bit-exact to the split, measured by the X1 spike on four cards (rows 64 and 128, FABRIC_1D, Ring, two
links: 0 of 163,840 elements differing in 6 of 6 comparisons; docs/tp4-exact-ring-parity.md). The reshape both ways is a view.
The lever engages for exactly the evidenced shapes (CENSUS_ROWS x CENSUS_WIDTH, bfloat16, tile layout, interleaved, four chips,
Ring, two links, the keyword arguments the layers pass) and refuses every other call by name, logging the reason and serving
the per-tile split. QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT=1 (gate arms only) also runs the split on the same input for the first
QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT_CALLS (default 32) calls of each shape in every forward, serves the split's result, and
compares both on every chip, element for element as int16 bit patterns, after the replay (packed_verifier calls audit_claim,
audit_round and audit_release); both results are cloned into DRAM held outside the model's buffers, so the compare is of what
each path produced, not of a buffer the model has reused. Every buffer the path uses (the view, the reduce-scatter's output and
intermediate, the semaphores of the model's TT_CCL, the audit's clones) is made inside the forward the capture records.

WP5 (QWEN_FAST_CCL_OPTIONS, scripts/ci/ccl_options_tp.py): the unit-major call may carry a named set of per-call options (links, workers, chunks per
sync, buffers per channel) that cannot change a bit (the ring's chunk parity is a function of the tile's index in its channel and of nothing the options
move); default off, and with it off every statement below runs as before. Under QWEN_FAST_CCL_OPTIONS_AUDIT the first N unit-major calls of a forward also
run with the values the model passes today, and both results are held and compared after the replay like the U1 audit's (audit_claim, audit_replayed,
audit_round and audit_release serve both). The WP5 modules (ccl_options_tp, distributed_norm_gather_tp) are imported BY NAME and only when one of the lever's
flags is in the environment: with the lever off they are not even loaded (and no served module imports them statically, which the image-list closure tests would
otherwise ask to be listed twice; the overlay manifest carries them).

Stdlib only; ttnn is imported on first use.
"""

from contextlib import contextmanager, nullcontext
import functools
import importlib
import os
import sys

TILE = 32
CCL_MODULE = 'models.tt_transformers.tt.ccl'
# WP5: the ccl options lever's modules and flags, by name (see the module docstring).
CCL_OPTIONS_MODULE = 'ccl_options_tp'
CCL_GATHER_MODULE = 'distributed_norm_gather_tp'
CCL_OPTIONS_FLAGS = ('QWEN_FAST_CCL_OPTIONS', 'QWEN_FAST_CCL_OPTIONS_AUDIT', 'QWEN_FAST_CCL_OPTIONS_AUDIT_CALLS')
NAME = 'tt_all_reduce'

UNIT_MAJOR_FLAG = 'QWEN_FAST_TP4_RS_UNIT_MAJOR'
UNIT_MAJOR_AUDIT_FLAG = 'QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT'
AUDIT_CALLS_FLAG = 'QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT_CALLS'
DEFAULT_AUDIT_CALLS = 32
# What the X1 spike (run v157) measured exact, and what the A1 audit re-proves on the serving image: the 64-row block. 128 rows has
# one X1 seed behind it and no audit line in the smoke rule, so it falls back to the split (named) until it is re-proven.
CENSUS_ROWS = (64,)
CENSUS_MESH_SHAPE = (1, 4)
CENSUS_WIDTH = 5120
CENSUS_CHIPS = 4
CENSUS_LINKS = 2
# The keyword arguments the model's layers pass (attention/tp.py, mlp.py); anything else is not in the census.
CENSUS_KEYWORDS = ('cluster_axis', 'dim', 'topology', 'memory_config')
ENGAGED_MARKER = '[PINDIAG] tp4 u1 engaged'
FALLBACK_MARKER = '[PINDIAG] tp4 u1 fell back'
AUDIT_MARKER = '[PINDIAG] tp4 u1 audit'
AUDIT_MISMATCH_MARKER = '[PINDIAG] tp4 u1 audit mismatch'

# The one scope in force (the block's forward runs on one thread): the row count being split, and how many all-reduces
# this process has split so far. unit_major / audit_calls: the U1 settings of the scope; engaged and fallbacks count its calls;
# audited counts the audited calls per shape in this scope.
_STATE = {'rows': None, 'splits': 0, 'tiles': 0, 'unit_major': False, 'audit_calls': 0, 'engaged': 0, 'fallbacks': 0,
          'audited': {}, 'reasons': set(), 'ccl': None}
# Audited pairs: {'owner': object or None (not yet claimed), 'shape': (rows, width), 'mine': clone, 'served': clone, 'layout': problem}
_HELD = []


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def ccl_options_asked(environ=None):
    """Whether one of the ccl options lever's flags is in the environment (any value: the lever's own strict parser judges it)."""
    source = os.environ if environ is None else environ
    return any(name in source for name in CCL_OPTIONS_FLAGS)


def ccl_options_settings(environ=None):
    """ccl_options_tp.settings() when a flag of the lever is in the environment (the module is imported by name then), else None."""
    if not ccl_options_asked(environ):
        return None
    return importlib.import_module(CCL_OPTIONS_MODULE).settings(environ)


def current_ccl_plan():
    """The ccl options plan of the block forward in progress, None outside one or with the lever off."""
    module = _STATE['ccl']
    return module.current() if module is not None else None


def unit_major_settings(environ=None):
    """(unit_major, audit_calls) for this process: audit_calls is 0 when the audit is off. Strict; every flag needs four cards
    (QWEN_FAST_TP=4: the lever exists for the ring at four chips only), the audit needs the lever, and the call count is a
    positive integer."""
    source = os.environ if environ is None else environ
    on = _flag(UNIT_MAJOR_FLAG, source)
    audit = _flag(UNIT_MAJOR_AUDIT_FLAG, source)
    calls_text = source.get(AUDIT_CALLS_FLAG)
    if not (on or audit or calls_text is not None):
        return False, 0
    import tp_shapes

    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s are TP4 levers: they need QWEN_FAST_TP=4, this process serves the pair'
                         % ', '.join(name for name in (UNIT_MAJOR_FLAG, UNIT_MAJOR_AUDIT_FLAG, AUDIT_CALLS_FLAG)
                                     if source.get(name) not in (None, '0')))
    if audit and not on:
        raise ValueError('%s needs %s=1' % (UNIT_MAJOR_AUDIT_FLAG, UNIT_MAJOR_FLAG))
    calls = DEFAULT_AUDIT_CALLS
    if calls_text is not None:
        if not audit:
            raise ValueError('%s needs %s=1' % (AUDIT_CALLS_FLAG, UNIT_MAJOR_AUDIT_FLAG))
        if not calls_text.isdigit() or int(calls_text) < 1 or str(int(calls_text)) != calls_text:
            raise ValueError('%s must be a positive integer, got %r' % (AUDIT_CALLS_FLAG, calls_text))
        calls = int(calls_text)
    return on, (calls if audit else 0)


def _log(message):
    import tp4_vglue

    tp4_vglue.log_line(message)


def _same_buffer(left, right):
    """Whether two tensors are views of one device buffer: their addresses agree. When an address cannot be read, they are
    taken as the same (a leak is a smaller fault than freeing a shared buffer twice)."""
    try:
        return left.buffer_address() == right.buffer_address()
    except Exception:
        return True


def _view_of(tensor):
    """What a consumer of the tensor sees: shape, dtype, layout, memory config."""
    return dict(shape=tuple(tensor.shape), dtype=getattr(tensor, 'dtype', None), layout=getattr(tensor, 'layout', None),
                memory=tensor.memory_config() if callable(getattr(tensor, 'memory_config', None)) else None)


def _differences(left, right):
    """'<field> <left> against <right>' for each field of two _view_of dictionaries that differ, '' when none."""
    return '; '.join('%s %r against %r' % (key, left[key], right[key]) for key in sorted(left) if left[key] != right[key])


def validate_rows(rows):
    """The tile count of a block wider than one tile, or ValueError."""
    if type(rows) is not int or rows <= TILE or rows % TILE:
        raise ValueError('A split all-reduce serves blocks of whole %d-row tiles beyond one tile; %r rows given'
                         % (TILE, rows))
    return rows // TILE


def tile_spans(rows):
    """The (first, last) row of each 32-row tile of a block of `rows` rows."""
    return tuple((first, first + TILE) for first in range(0, TILE * validate_rows(rows), TILE))


def splits():
    """How many all-reduces this process has split (a counter for the guard and the tests)."""
    return _STATE['splits']


class TileSplitAllReduce:
    """`original` (the model's tt_all_reduce) that, inside a block_scope, runs a scoped-width input tile by tile."""

    def __init__(self, original, operations=None):
        if isinstance(original, TileSplitAllReduce):
            raise ValueError('The all-reduce is already the tile-split wrapper')
        if not callable(original):
            raise ValueError('The all-reduce to wrap must be callable')
        self.original = original
        self._operations = operations

    @property
    def operations(self):
        if self._operations is None:
            import ttnn
            self._operations = ttnn
        return self._operations

    def __call__(self, tensor, *args, **kwargs):
        rows = _STATE['rows']
        shape = tuple(tensor.shape) if rows is not None else ()
        if rows is None or len(shape) != 4 or shape[2] != rows:
            return self.original(tensor, *args, **kwargs)
        if _STATE['unit_major']:
            return self.unit_major(tensor, shape, args, kwargs)
        return self.split(tensor, shape, args, kwargs)

    # --- U1: one reduce-scatter on the unit-major view ------------------------------------------------------------

    def refusal(self, tensor, shape, args, kwargs):
        """Why this call is not in the X1 census (a string naming it), or None when it is."""
        operations = self.operations
        if shape[0] != 1 or shape[1] != 1:
            return 'shape %r: leading dimensions are not (1, 1)' % (tuple(shape),)
        if shape[2] not in CENSUS_ROWS:
            return 'rows %d are not a measured block height %r' % (shape[2], CENSUS_ROWS)
        if shape[3] != CENSUS_WIDTH:
            return 'width %d is not the measured %d' % (shape[3], CENSUS_WIDTH)
        if len(args) != 2:
            return '%d positional arguments after the tensor (the layers pass the mesh and the collective only)' % len(args)
        extra = sorted(set(kwargs) - set(CENSUS_KEYWORDS))
        if extra:
            return 'keyword %s is not in the census %r' % (','.join(extra), CENSUS_KEYWORDS)
        if kwargs.get('cluster_axis', 0) != 0:
            return 'cluster_axis %r is not 0' % (kwargs.get('cluster_axis'),)
        # tt_all_reduce defaults dim to 0, so an omitted dim is not the census's dim 3.
        if kwargs.get('dim') != 3:
            return 'dim %r is not 3' % (kwargs.get('dim'),)
        if 'topology' not in kwargs or kwargs['topology'] != operations.Topology.Ring:
            return 'topology %r is not Ring' % (kwargs.get('topology'),)
        if tensor.dtype != operations.bfloat16:
            return 'dtype %r is not bfloat16' % (tensor.dtype,)
        if tensor.layout != operations.TILE_LAYOUT:
            return 'layout %r is not tile' % (tensor.layout,)
        memories = [tensor.memory_config()]
        if kwargs.get('memory_config') is not None:
            memories.append(kwargs['memory_config'])
        for memory in memories:
            is_sharded = getattr(memory, 'is_sharded', None)
            if callable(is_sharded) and is_sharded():
                return 'a sharded memory config'
        mesh, collective = args
        chips = getattr(mesh, 'get_num_devices', None)
        if not callable(chips) or chips() != CENSUS_CHIPS:
            return 'the mesh is not %d chips' % CENSUS_CHIPS
        try:
            mesh_shape = tuple(mesh.shape)
        except Exception:
            mesh_shape = None
        if mesh_shape != CENSUS_MESH_SHAPE:
            return 'the mesh shape %r is not %r' % (mesh_shape, CENSUS_MESH_SHAPE)
        links = getattr(collective, 'get_num_links', None)
        try:
            said = links(kwargs.get('cluster_axis', 0)) if callable(links) else None
        except Exception as error:
            return 'the collective could not say its links (%s)' % type(error).__name__
        if said != CENSUS_LINKS:
            return 'the collective does not say %d links' % CENSUS_LINKS
        return None

    def unit_major(self, tensor, shape, args, kwargs):
        plan = current_ccl_plan()
        reason = self.refusal(tensor, shape, args, kwargs)
        if reason is not None:
            _STATE['fallbacks'] += 1
            if reason not in _STATE['reasons']:
                _STATE['reasons'].add(reason)
                _log('%s rows=%d reason=%s' % (FALLBACK_MARKER, shape[2], reason))
            if plan is not None and plan.selection.rs:
                _STATE['ccl'].note_fallback(plan, 'rs', reason)
            return self.split(tensor, shape, args, kwargs)
        key = (shape[2], shape[3])
        audited = _STATE['audited']
        if plan is not None:
            plan.rs_engaged += 1
        if _STATE['audit_calls'] and audited.get(key, 0) < _STATE['audit_calls']:
            audited[key] = audited.get(key, 0) + 1
            return self.audited(tensor, shape, args, kwargs)
        if plan is not None and plan.audit_open('rs'):
            return self.ccl_audited(tensor, shape, args, kwargs)
        output = self.reduce_unit_major(tensor, shape, args, kwargs, consume=True)
        _STATE['splits'] += 1
        _STATE['tiles'] += shape[2] // TILE
        _STATE['engaged'] += 1
        return output

    def reduce_unit_major(self, tensor, shape, args, kwargs, consume, options=None):
        """The one reduce-scatter on the (1, R/32, 32, W) view, as the X1 spike called it, and its result viewed back as
        (1, 1, R, W/chips). `consume` frees the input as tt_all_reduce does (never when the split runs on it next). `options`: the
        keyword overrides of a QWEN_FAST_CCL_OPTIONS set; None takes the block scope's (none when the lever is off), {} the model's own values."""
        operations = self.operations
        mesh, collective = args
        axis = kwargs.get('cluster_axis', 0)
        output_memory = kwargs.get('memory_config') or tensor.memory_config()
        view = operations.reshape(tensor, (1, shape[2] // TILE, TILE, shape[3]))
        call = dict(persistent_output_buffers=None, dim=3,
                    multi_device_global_semaphore=collective.get_and_cycle_rs_semaphore_handles(),
                    barrier_semaphore=collective.get_and_cycle_barrier_semaphore_handle(),
                    num_links=collective.get_num_links(axis), memory_config=output_memory,
                    intermediate_memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Ring,
                    chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
        if options is None:
            module = _STATE['ccl']
            options = module.rs_overrides() if module is not None else {}
        call.update(options)
        scattered = operations.experimental.reduce_scatter_minimal_async(view, **call)
        output = operations.reshape(scattered, (1, 1, shape[2], scattered.shape[3]))
        # A view shares its buffer with its source: free a copy only if the reshape made one.
        if view is not tensor and not _same_buffer(view, tensor):
            operations.deallocate(view)
        if consume:
            operations.deallocate(tensor)
        if output is not scattered and not _same_buffer(output, scattered):
            operations.deallocate(scattered)
        return output

    def audited(self, tensor, shape, args, kwargs):
        """U1 under the audit: the unit-major result and the split's, each cloned into DRAM and held for audit_round; the
        split's result is the one served (it consumes the input, which the unit-major call left alone)."""
        operations = self.operations
        mine = self.reduce_unit_major(tensor, shape, args, kwargs, consume=False)
        _STATE['engaged'] += 1
        mine_view = _view_of(mine)
        copy_mine = operations.clone(mine, memory_config=operations.DRAM_MEMORY_CONFIG)
        operations.deallocate(mine)
        try:
            served = self.split(tensor, shape, args, kwargs)
        except BaseException:
            operations.deallocate(copy_mine)
            raise
        problems = [name + ' ' + text for name, text in
                    (('layout', _differences(mine_view, _view_of(served))),) if text]
        pair = dict(owner=None, shape=(shape[2], shape[3]), mine=copy_mine, served=None,
                    layout='; '.join(problems) or None)
        try:
            pair['served'] = operations.clone(served, memory_config=operations.DRAM_MEMORY_CONFIG)
        except BaseException:
            operations.deallocate(copy_mine)
            raise
        _HELD.append(pair)
        return served

    def ccl_audited(self, tensor, shape, args, kwargs):
        """QWEN_FAST_CCL_OPTIONS_AUDIT: the unit-major call with the set's options and the same call with the values the model passes today, each
        cloned into DRAM and held for audit_round; the model's values' result is the one served (it consumes the input, which the first call left
        alone). One reduce-scatter more than an unaudited call: the plan's count keeps the forward's total even."""
        operations = self.operations
        mine = self.reduce_unit_major(tensor, shape, args, kwargs, consume=False)
        mine_view = _view_of(mine)
        copy_mine = operations.clone(mine, memory_config=operations.DRAM_MEMORY_CONFIG)
        operations.deallocate(mine)
        try:
            served = self.reduce_unit_major(tensor, shape, args, kwargs, consume=True, options={})
        except BaseException:
            operations.deallocate(copy_mine)
            raise
        module = _STATE['ccl']
        hold_pair(operations, copy_mine, served, (shape[2], shape[3]), mine_view, 'rs', (module.AUDIT_MARKER, module.AUDIT_MISMATCH_MARKER))
        _STATE['splits'] += 1
        _STATE['tiles'] += shape[2] // TILE
        _STATE['engaged'] += 1
        return served

    def split(self, tensor, shape, args, kwargs):
        operations = self.operations
        spans = tile_spans(shape[2])
        input_memory = tensor.memory_config() if callable(getattr(tensor, 'memory_config', None)) else None
        output_memory = kwargs.get('memory_config') or input_memory
        pieces, outputs = [], []
        consumed = False
        try:
            for first, last in spans:
                pieces.append(operations.slice(tensor, (0, 0, first, 0), (shape[0], shape[1], last, shape[3]),
                                               **(dict(memory_config=input_memory) if input_memory is not None else {})))
            # tt_all_reduce consumes its input; the tiles now stand for it.
            operations.deallocate(tensor)
            consumed = True
            for index, piece in enumerate(pieces):
                try:
                    outputs.append(self.original(piece, *args, **kwargs))
                except BaseException:
                    for later in pieces[index + 1:]:
                        operations.deallocate(later)
                    raise
            joined = operations.concat(outputs, dim=2,
                                       **(dict(memory_config=output_memory) if output_memory is not None else {}))
        except BaseException:
            if not consumed:
                for piece in pieces:
                    operations.deallocate(piece)
            raise
        finally:
            for value in outputs:
                operations.deallocate(value)
        _STATE['splits'] += 1
        _STATE['tiles'] += len(spans)
        return joined


@contextmanager
def block_scope(rows, expected=None, log=None, unit_major=False, audit_calls=0, ccl=None, layers=None):
    """Reduce every `rows`-row all-reduce issued inside in the sequential engine's order: split per 32-row tile, or (unit_major)
    one reduce-scatter on the unit-major view for the calls the X1 census covers (the rest split, the reason logged), the first
    `audit_calls` calls of each shape also split and held for audit_round. `ccl` (ccl_options_tp.settings(), None when the lever is off; its module is found through the settings' own class): the
    named per-call options of the unit-major reduce-scatters and of the norms' gathers over `layers` layers, with their own audit quota. When it
    exits without an error, `expected` (when given) must equal the number reduced: a scope that engaged nothing means the wrapper was never bound
    where the block's layers look it up, and the round would run the unsplit, inexact reduction silently."""
    validate_rows(rows)
    if _STATE['rows'] is not None:
        raise ValueError('A block scope is already open (%d rows)' % _STATE['rows'])
    if audit_calls and not unit_major:
        raise ValueError('The unit-major audit needs the unit-major lever')
    if ccl is not None and ccl.selection.rs and not unit_major:
        raise ValueError('The ccl options for the reduce-scatter ride the unit-major lever')
    before = _STATE['splits']
    _STATE.update(rows=rows, unit_major=bool(unit_major), audit_calls=int(audit_calls), engaged=0, fallbacks=0, audited={})
    module = sys.modules[type(ccl).__module__] if ccl is not None else None
    plan = module.begin(ccl, rows, layers, log) if module is not None else None
    _STATE['ccl'] = module
    try:
        yield
    finally:
        _STATE['rows'] = None
        _STATE['unit_major'] = False
        _STATE['audit_calls'] = 0
        _STATE['ccl'] = None
        if module is not None:
            module.close(plan)
    engaged = _STATE['splits'] - before
    engaged_unit_major, fallbacks = _STATE['engaged'], _STATE['fallbacks']
    audited = sum((_STATE['audited'] or {}).values())
    if expected is not None and engaged != expected:
        raise AssertionError('The tile-split all-reduce engaged %d times in this %d-row forward, %d expected: the model '
                             'layers do not all reach the wrapper (tp_addresses.install binds it at QWEN_FAST_TP != 2)'
                             % (engaged, rows, expected))
    if unit_major and engaged_unit_major + fallbacks != engaged:
        raise AssertionError('The unit-major all-reduce counted %d engaged and %d fallbacks against %d reductions'
                             % (engaged_unit_major, fallbacks, engaged))
    if unit_major and expected is not None:
        # The ring's semaphore parity is a hang factor (request_width_warm.py): the forward's reduce-scatter count must stay even.
        # A unit-major call is one, an audited call three (one, then the split's two), a fallback rows/32, a ccl-audited call two.
        scatters = (engaged_unit_major - audited) + 3 * audited + fallbacks * (rows // TILE) + (plan.audited['rs'] if plan is not None else 0)
        if scatters % 2:
            raise AssertionError('The unit-major forward issued %d reduce-scatters (%d engaged, %d audited, %d fallbacks): an odd '
                                 'count flips the ring semaphore parity' % (scatters, engaged_unit_major, audited, fallbacks))
    if log is not None:
        log('[PINDIAG] tile-split all-reduce: {} of {} rows in {} tiles each', engaged, rows, rows // TILE)
        if unit_major:
            log('{}', '%s rows=%d calls=%d unit_major=%d fallbacks=%d audited=%d'
                % (ENGAGED_MARKER, rows, engaged, engaged_unit_major, fallbacks, audited))
    if module is not None:
        module.finish(plan)


# --- the U1 audit: compared after the replay -----------------------------------------------------------------------------


def audit_claim(owner, kind='audit'):
    """Give every audited pair not yet owned to `owner` (the fixture whose forward just ran: the warm forward or the capture).
    The pairs claimed together share a label, kind + a process-wide ordinal ('capture3'), so the log tells one block's audit
    from the other's."""
    claimed = 0
    label = None
    for pair in _HELD:
        if pair['owner'] is None:
            if label is None:
                _STATE['owners'] = _STATE.get('owners', 0) + 1
                label = '%s%d' % (kind, _STATE['owners'])
            pair['owner'] = owner
            pair['label'] = label
            claimed += 1
    return claimed


def audit_replayed(owner):
    """Note that `owner`'s trace replayed just now (a no-op while nothing is held). The audited clones are persistent buffers
    allocated while the second block's capture ran, after the first block's traces existed, so they may sit in holes the first
    block's replay overwrites: they are only valid to read right after their own block's replay, and audit_round refuses
    otherwise (a false mismatch would stop the window)."""
    if _HELD:
        _STATE['replayed'] = owner


def audit_pairs(owner):
    return [pair for pair in _HELD if pair['owner'] is owner]


def audit_round(operations, owner, round_number, log=None):
    """Compare every pair `owner` holds, on every chip, as int16 bit patterns (-0 and +0 differ). Returns the pairs compared (0
    when the owner holds none: the audit off). One AUDIT_MARKER line per shape (calls, chips, elements) on rounds 0 to 3 and
    every 50th after; a layout difference, a shape difference or any differing element logs AUDIT_MISMATCH_MARKER and raises. The pairs of the
    ccl options audit (QWEN_FAST_CCL_OPTIONS_AUDIT) carry their own markers and an op= field."""
    pairs = audit_pairs(owner)
    if not pairs:
        return 0
    if round_number > 0 and _STATE.get('replayed') is not owner:
        raise AssertionError('%s round=%d: the audit was read after another block replayed, which may have overwritten these '
                             'clones (they are valid only right after their own replay)' % (AUDIT_MISMATCH_MARKER, round_number))
    import torch

    log = log or _log
    mismatches, tallies = [], {}
    markers = []

    def fail(pair, text):
        mismatches.append(text)
        markers.append(pair.get('mismatch', AUDIT_MISMATCH_MARKER))

    for index, pair in enumerate(pairs):
        label = 'shape=%dx%d call=%d' % (pair['shape'][0], pair['shape'][1], index)
        if pair.get('op'):
            label += ' op=%s' % pair['op']
        if pair['layout']:
            fail(pair, '%s layout %s' % (label, pair['layout']))
        lefts = operations.get_device_tensors(pair['mine'])
        rights = operations.get_device_tensors(pair['served'])
        if len(lefts) != len(rights):
            fail(pair, '%s chips %d against %d' % (label, len(lefts), len(rights)))
            continue
        tally = tallies.setdefault((pair.get('marker', AUDIT_MARKER), pair.get('op'), pair.get('label'), pair['shape']),
                                   dict(calls=0, chips=len(lefts), elements=0))
        tally['calls'] += 1
        for chip, (left, right) in enumerate(zip(lefts, rights)):
            a = operations.to_torch(left).contiguous().view(torch.int16)
            b = operations.to_torch(right).contiguous().view(torch.int16)
            if a.shape != b.shape:
                fail(pair, '%s chip %d: shape %r against %r' % (label, chip, tuple(a.shape), tuple(b.shape)))
            elif not torch.equal(a, b):
                fail(pair, '%s chip %d: %d of %d elements differ' % (label, chip, int((a != b).sum()), a.numel()))
            else:
                tally['elements'] += a.numel()
    if mismatches:
        message = '%s round=%d %s' % (markers[0], round_number, '; '.join(mismatches[:4]))
        log(message)
        raise AssertionError(message)
    if round_number <= 3 or round_number % 50 == 0:
        for key in sorted(tallies, key=lambda key: (key[0], str(key[1]), str(key[2]), key[3])):
            marker, op, label, shape = key
            tally = tallies[key]
            log('%s shape=%dx%d owner=%s round=%d%s calls=%d chips=%d elements=%d exact=True'
                % (marker, shape[0], shape[1], label, round_number, (' op=%s' % op) if op else '', tally['calls'], tally['chips'],
                   tally['elements']))
    return len(pairs)


def hold_pair(operations, copy_mine, served, shape, mine_view, op, markers):
    """Hold a ccl options audit pair for audit_round: `copy_mine` is the options result already cloned into DRAM (the caller freed the original),
    `served` the result the model's own values produced, which is cloned here and stays the caller's to serve; `markers` (the audit line's marker and the
    mismatch line's) are ccl_options_tp's. A layout difference between the two is recorded and raised by audit_round."""
    problems = [name + ' ' + text for name, text in (('layout', _differences(mine_view, _view_of(served))),) if text]
    pair = dict(owner=None, shape=shape, mine=copy_mine, served=None, layout='; '.join(problems) or None,
                marker=markers[0], mismatch=markers[1], op=op)
    try:
        pair['served'] = dram_copy(operations, served)
    except BaseException:
        operations.deallocate(copy_mine)
        raise
    _HELD.append(pair)
    return pair


def dram_copy(operations, tensor):
    """A copy of `tensor` in interleaved DRAM, held outside the model's buffers: a sharded tensor (the norms' gathered input) is converted, an
    interleaved one cloned."""
    memory = tensor.memory_config() if callable(getattr(tensor, 'memory_config', None)) else None
    is_sharded = getattr(memory, 'is_sharded', None)
    if callable(is_sharded) and is_sharded():
        return operations.to_memory_config(tensor, operations.DRAM_MEMORY_CONFIG)
    return operations.clone(tensor, memory_config=operations.DRAM_MEMORY_CONFIG)


def audit_release(operations, owner):
    """Free every clone `owner` holds (before its fixture closes). Returns how many pairs."""
    pairs = audit_pairs(owner)
    if _STATE.get('replayed') is owner:
        _STATE['replayed'] = None
    for pair in pairs:
        _HELD.remove(pair)
        for name in ('mine', 'served'):
            if pair.get(name) is not None:
                operations.deallocate(pair[name])
    return len(pairs)


class ClassAttributes:
    """A class's attributes through the item interface tp_addresses.uninstall restores namespaces by."""

    def __init__(self, cls):
        self.cls = cls

    def __setitem__(self, name, value):
        setattr(self.cls, name, value)


def scope_for(batch):
    """The block scope of a ModelBatch's forward, or nothing within one tile. Inside it the model's tt_all_reduce (this
    module's wrapper) splits every block-wide call, and the scope refuses the round unless all of them were split: the
    mixer's (the 16 wo projections and the 48 GDN output projections) and, when the MLP is native at the block's rows, the
    64 w2 reductions (the non-native MLP already runs two 32-row calls)."""
    rows = getattr(batch, 'rows', None)
    if type(rows) is not int or rows <= TILE:
        return nullcontext()
    from dflash_device import pindiag

    layers = len(batch.model.layers)
    unit_major, audit_calls = unit_major_settings()
    return block_scope(rows, expected=layers * (2 if batch.native_m3 else 1), log=pindiag, unit_major=unit_major,
                       audit_calls=audit_calls, ccl=ccl_options_settings(), layers=layers)


def scoped_run(original):
    """ModelBatch.run inside the batch's block scope; `original` is the pinned method, called unchanged."""

    @functools.wraps(original)
    def run(self, *args, **kwargs):
        with scope_for(self):
            return original(self, *args, **kwargs)

    run.tile_scope_of = original
    return run


def install_scope():
    """Wrap model_batch.ModelBatch.run so every block forward runs in its scope. -> [(namespace, name, original)], [] when
    already wrapped."""
    import model_batch

    current = model_batch.ModelBatch.run
    if getattr(current, 'tile_scope_of', None) is not None:
        return []
    model_batch.ModelBatch.run = scoped_run(current)
    return [(ClassAttributes(model_batch.ModelBatch), 'run', current)]


def install_gather_options():
    """WP5: wrap DistributedNorm.forward in the gather options' shim when QWEN_FAST_CCL_OPTIONS is set (distributed_norm_gather_tp.install; [] when it is
    unset or the model tree is not importable). Bound here, with the scope it runs in, so tp_addresses records and puts back both together."""
    if not ccl_options_asked():
        return []
    return importlib.import_module(CCL_GATHER_MODULE).install()


def install(module=None, scope=True):
    """Bind the wrapper in place of the model's tt_all_reduce: on the ccl module and on every loaded module whose
    global of that name IS the original function (`from ccl import tt_all_reduce` binds a copy at import), and (scope)
    wrap ModelBatch.run in the block scope. -> [(namespace, name, original)] for each binding changed, so tp_addresses can
    put them back; [] when the ccl module is not importable here (no model tree: the CPU suite, where nothing is
    wrapped either) or already carries the wrapper."""
    if module is None:
        try:
            module = importlib.import_module(CCL_MODULE)
        except ImportError as error:
            if getattr(error, 'name', None) in (CCL_MODULE.split('.')[0], CCL_MODULE, 'models.tt_transformers',
                                                'models.tt_transformers.tt'):
                return []
            raise
    original = getattr(module, NAME, None)
    if original is None or isinstance(original, TileSplitAllReduce):
        return []
    wrapper = TileSplitAllReduce(original)
    changed = []
    for candidate in list(sys.modules.values()):
        namespace = getattr(candidate, '__dict__', None)
        if isinstance(namespace, dict) and namespace.get(NAME) is original:
            namespace[NAME] = wrapper
            changed.append((namespace, NAME, original))
    if not any(namespace is module.__dict__ for namespace, _, _ in changed):
        module.__dict__[NAME] = wrapper
        changed.append((module.__dict__, NAME, original))
    if scope:
        changed += install_scope()
        changed += install_gather_options()
    return changed
