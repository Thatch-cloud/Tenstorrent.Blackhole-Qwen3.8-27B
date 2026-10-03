"""The packed GDN block's odd-user slices from ONE row-major conversion, and a host log of each layer's launch order (tp4/gluefix).

WHY. The packed verify projects the four users' 16-row segments as one (1, 64, W) tile block and cuts one piece per user. Users 0
and 2 start on a tile row, so ttnn's Slice stays in tile layout. Users 1 and 3 start at row 16 and 48, half way through a tile:
ttnn's Slice then converts its WHOLE input to row-major, slices, and tilizes the piece back (opchain slice.cpp: `rm_only` when a
begin is not tile-aligned; `to_layout(input, ROW_MAJOR)` first, `to_layout(result, input_layout)` last). In the TP4 device profile
that is one UntilizeWithUnpadding of the whole 64 x 4128 projection per odd user, 18.4 us each, on each of the 48 GDN layers.

WHAT (QWEN_FAST_GDN_PAIR_SLICE=1). This proxy sits in front of the layer's `operations` for the duration of one packed decode
call. An exactly half-tile slice of a tile-layout tensor (start row 16 mod 32, 16 rows, every other axis whole) is served the way
ttnn serves it, op for op: `to_layout(ROW_MAJOR)` of the source, `slice` of the row-major tensor, `to_layout(TILE)` of the piece
(the same three ops on the same bytes, so the same bits), except that the row-major conversion of a given source is made ONCE and
shared by every odd user of the block. One whole-projection untilize per layer instead of two (about 18 us per GDN layer, 0.9 ms
per four-user verify). Every other slice, any slice with another shape, extra arguments or a non-tile source, goes to ttnn
unchanged and untouched.

Not used when QWEN_FAST_TP4_GDN_GLUE=1: V2's one DMA launch replaces these slices altogether (and its audit's served pieces
must be the plain ttnn slices). This is the fallback if V2 has to come off: it is exact by construction and needs no new kernel.

AUDIT (QWEN_FAST_TP4_VGLUE_AUDIT=1 beside the flag, gate arms only). Beside each served slice the proxy also makes ttnn's own slice
and holds DRAM copies of both, outside `owned`, as 'pair slice user k' entries on the layer's result; tp4_vglue.audit_round
compares them bit for bit on every chip after the replay (int16 patterns, so -0 and +0 differ) and fails a layer whose entries are
missing (a declined slice is a failure, not a pass on what is left).

QWEN_FAST_GDN_DISPATCH_DIAG=1 (diagnostic only, changes no op). The conv-gates launches of users 1-3 start about 16 us after the
op before them ends, on every chip, in the TP4 device profile. A host cause is ruled out by reading the trace replay (no program
cache lookup, argument update or sync per op), so this flag records, per packed GDN decode call, the ops enqueued and the host
time of each `gdn_decode_conv_gates` launch and the ops between consecutive ones, and logs the first DIAG_LINES calls as
'[PINDIAG] tp4 gdn dispatch diag ...'. It sees the eager warm forward and the capture (enqueue time, not device time: the device
side needs TT_METAL_DEVICE_PROFILER_DISPATCH=1 on a profiled arm). Nothing here waits, syncs or reorders anything.

Stdlib only, py 3.7. Both flags are strict (unset or 0 is off, 1 is on, anything else raises) and refused at the pair
(tp4_vglue reads them); with both off `wrap` returns None and the pinned call runs untouched.
"""

import time

import tp4_vglue

TILE = 32
USER_ROWS = 16
DIAG_LINES = 8

ENGAGED = '[PINDIAG] tp4 pair slice engaged'
FALLBACK = '[PINDIAG] tp4 pair slice fell back'
IDLE = '[PINDIAG] tp4 pair slice idle'
DIAG = '[PINDIAG] tp4 gdn dispatch diag'

_LOGGED = set()
_DIAG = dict(calls=0)


def log_once(line):
    if line not in _LOGGED:
        _LOGGED.add(line)
        tp4_vglue.log_line(line)


def half_tile_rows(starts, ends, shape):
    """The first row of a slice this module serves, or None: three axes, whole batch and width, exactly 16 rows starting half way
    through a tile (row 16 mod 32), inside a block whose row count is whole tiles."""
    if len(starts) != 3 or len(ends) != 3 or len(shape) != 3:
        return None
    if starts[0] != 0 or starts[2] != 0 or ends[0] != shape[0] or ends[2] != shape[2]:
        return None
    if shape[1] % TILE or starts[1] % TILE != USER_ROWS or ends[1] - starts[1] != USER_ROWS or ends[1] > shape[1]:
        return None
    return starts[1]


class PairSlice:
    """One packed decode call's shared row-major conversions and (audit) held copies."""

    def __init__(self, operations, audit):
        self.operations, self.audit = operations, audit
        self.converted = {}          # id(source) -> (source, its row-major copy); the source is held so its id stays unique
        self.held, self.entries = [], []
        self.served = 0

    def serve(self, tensor, starts, ends):
        """The slice, or None when it is not one this serves (the caller then makes the plain ttnn call)."""
        operations = self.operations
        try:
            shape = tuple(tensor.shape)
            start = half_tile_rows(tuple(starts), tuple(ends), shape)
            if start is None:
                return None
            if tensor.layout != operations.TILE_LAYOUT:
                log_once('%s site=gdn_slice reason=the block projection is not in tile layout' % FALLBACK)
                return None
        except (AttributeError, TypeError):
            return None
        entry = self.converted.get(id(tensor))
        if entry is None:
            entry = self.converted[id(tensor)] = (tensor, operations.to_layout(tensor, operations.ROW_MAJOR_LAYOUT))
        rows = operations.slice(entry[1], starts, ends)
        try:
            piece = operations.to_layout(rows, operations.TILE_LAYOUT)
        finally:
            operations.deallocate(rows)
        self.served += 1
        try:
            if self.audit:
                self.hold(tensor, starts, ends, piece, start // USER_ROWS)
        finally:
            if tuple(ends)[1] == shape[1]:
                # The last half-tile slice of this source: free the shared copy now, as ttnn's own slice frees its conversion, so the L1 layout
                # inside the trace does not carry it through the recurrence and seq-block launches.
                self.converted.pop(id(tensor), None)
                operations.deallocate(entry[1])
        return piece

    def hold(self, tensor, starts, ends, piece, user):
        operations = self.operations
        plain = operations.slice(tensor, starts, ends)
        try:
            copies = [operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG) for value in (piece, plain)]
        finally:
            operations.deallocate(plain)
        self.held.extend(copies)
        self.entries.append(dict(label='pair slice user %d' % user, mine=copies[0], served=copies[1]))

    def release_conversions(self):
        for _source, rows in self.converted.values():
            self.operations.deallocate(rows)
        self.converted.clear()

    def release_held(self):
        for value in self.held:
            self.operations.deallocate(value)
        self.held.clear()
        self.entries.clear()


class Recorder:
    """The ops enqueued during one packed decode call, by name, with the host time of each."""

    def __init__(self):
        self.events = []

    def note(self, name):
        self.events.append((name, time.perf_counter_ns()))

    def summary(self):
        events = self.events
        launches = [index for index, (name, _t) in enumerate(events) if name == 'gdn_decode_conv_gates']
        between = [launches[k + 1] - launches[k] - 1 for k in range(len(launches) - 1)]
        gaps = [(events[launches[k + 1]][1] - events[launches[k]][1]) // 1000 for k in range(len(launches) - 1)]
        total = (events[-1][1] - events[0][1]) // 1000 if events else 0
        return dict(ops=len(events), conv_gates=len(launches), ops_between=between, host_gap_us=gaps, host_span_us=total)


class Transformer:
    def __init__(self, inner, recorder):
        self._inner, self._recorder = inner, recorder

    def __getattr__(self, name):
        value = getattr(self._inner, name)
        if not callable(value) or isinstance(value, type):
            return value
        recorder = self._recorder

        def noted(*args, **keywords):
            recorder.note(name)
            return value(*args, **keywords)
        return noted


class Operations:
    """`operations` for one packed decode call: delegates every attribute, serves the half-tile slices (pair slice) and records
    the calls (diag). Either part can be absent."""

    def __init__(self, operations, pair, recorder):
        self._operations, self.pair, self.recorder = operations, pair, recorder

    def __getattr__(self, name):
        value = getattr(self._operations, name)
        recorder = self.recorder
        if recorder is None:
            return value
        if name == 'transformer':
            return Transformer(value, recorder)
        if not callable(value) or isinstance(value, type):
            return value

        def noted(*args, **keywords):
            recorder.note(name)
            return value(*args, **keywords)
        return noted

    def slice(self, tensor, starts, ends, *args, **keywords):
        if self.recorder is not None:
            self.recorder.note('slice')
        if self.pair is not None and not args and not keywords:
            piece = self.pair.serve(tensor, starts, ends)
            if piece is not None:
                return piece
        return self._operations.slice(tensor, starts, ends, *args, **keywords)


class Call:
    """What `wrap` hands the twin: the operations to use for the call and the closing steps."""

    def __init__(self, real, pair, recorder):
        self.real, self.pair, self.recorder = real, pair, recorder
        self.operations = Operations(real, pair, recorder)

    def abort(self):
        if self.pair is not None:
            self.pair.release_held()

    def close(self):
        if self.pair is not None:
            self.pair.release_conversions()

    def finish(self, result):
        """After a successful call: attach the audit entries, log the engaged line and the diag summary."""
        pair = self.pair
        if pair is not None:
            if pair.served:
                tp4_vglue.note('gdn_pair_slice', pair.served)
                log_once('%s site=gdn_slice' % ENGAGED)
            else:
                log_once('%s site=gdn_slice reason=no half-tile slice in this call (a one-user block)' % IDLE)
            if pair.entries:
                result['vglue_pair_audit'] = list(pair.entries)
                pair.entries.clear()
        if self.recorder is not None and _DIAG['calls'] < DIAG_LINES:
            _DIAG['calls'] += 1
            found = self.recorder.summary()
            tp4_vglue.log_line('%s call=%d ops=%d conv_gates=%d ops_between_conv_gates=%s host_gap_us=%s host_span_us=%d'
                               % (DIAG, _DIAG['calls'], found['ops'], found['conv_gates'], found['ops_between'],
                                  found['host_gap_us'], found['host_span_us']))


def wrap(operations, glue_on, environ=None):
    """The Call for one packed decode, or None when neither flag has anything to do (the pinned call then runs untouched).
    `glue_on` is QWEN_FAST_TP4_GDN_GLUE: with it the pair slice stays idle."""
    slicing = tp4_vglue.pair_slice_enabled(environ)
    diag = tp4_vglue.dispatch_diag_enabled(environ)
    if slicing and glue_on:
        log_once('%s reason=QWEN_FAST_TP4_GDN_GLUE=1 already replaces the slices' % IDLE)
        slicing = False
    if not slicing and not diag:
        return None
    pair = PairSlice(operations, tp4_vglue.audit_enabled(environ)) if slicing else None
    return Call(operations, pair, Recorder() if diag else None)
