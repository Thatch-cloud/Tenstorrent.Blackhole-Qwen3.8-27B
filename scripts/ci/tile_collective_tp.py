"""The four-card block all-reduce, one 32-row tile at a time: the packed verify block's reduction order made the sequential engine's.

WHY (S3a at four cards, docs/tp4-exact-ring-parity.md). At the pair every cross-chip sum has two addends, which
commute, so the block's second 32-row tile can never differ from a one-tile call. At four cards the model's
tt_all_reduce (ccl.py: reduce_scatter_minimal_async, then all_gather_async) runs RING collectives, and the ring
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
which the process runs at QWEN_FAST_TP != 2 only, so the pair never sees it; model_batch.ModelBatch.run enters the
scope at four cards, and refuses the round unless every all-reduce the block issues was split.

The wrapper consumes its input, as tt_all_reduce does (ccl.py deallocates it after the gather).

Stdlib only; ttnn is imported on first use.
"""

from contextlib import contextmanager
import importlib
import sys

TILE = 32
CCL_MODULE = 'models.tt_transformers.tt.ccl'
NAME = 'tt_all_reduce'

# The one scope in force (the block's forward runs on one thread): the row count being split, and how many all-reduces
# this process has split so far.
_STATE = {'rows': None, 'splits': 0, 'tiles': 0}


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
        return self.split(tensor, shape, args, kwargs)

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
def block_scope(rows, expected=None, log=None):
    """Split every `rows`-row all-reduce issued inside. When it exits without an error, `expected` (when given) must
    equal the number split: a scope that engaged nothing means the wrapper was never bound where the block's
    layers look it up, and the round would run the unsplit, inexact reduction silently."""
    validate_rows(rows)
    if _STATE['rows'] is not None:
        raise ValueError('A block scope is already open (%d rows)' % _STATE['rows'])
    before = _STATE['splits']
    _STATE['rows'] = rows
    try:
        yield
    finally:
        _STATE['rows'] = None
    engaged = _STATE['splits'] - before
    if expected is not None and engaged != expected:
        raise AssertionError('The tile-split all-reduce engaged %d times in this %d-row forward, %d expected: the model '
                             'layers do not all reach the wrapper (tp_addresses.install binds it at QWEN_FAST_TP != 2)'
                             % (engaged, rows, expected))
    if log is not None:
        log('[PINDIAG] tile-split all-reduce: {} of {} rows in {} tiles each', engaged, rows, rows // TILE)


def install(module=None):
    """Bind the wrapper in place of the model's tt_all_reduce: on the ccl module and on every loaded module whose
    global of that name IS the original function (`from ccl import tt_all_reduce` binds a copy at import).
    -> [(namespace, name, original)] for each binding changed, so tp_addresses can put them back; [] when the ccl
    module is not importable here (no model tree: the CPU suite) or already carries the wrapper."""
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
    return changed
