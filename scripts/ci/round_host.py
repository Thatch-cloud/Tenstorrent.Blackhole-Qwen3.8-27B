"""tp4/round-host: the host work inside a verified round, taken off the critical path where it can be taken off EXACTLY.

Six flags, all default off, all 0 or 1 (anything else is a configuration error at the attach), all host only: no kernel, no trace, no
device command, no allocation. With every flag at 0 every path below is today's, argument for argument, and nothing is logged.

  QWEN_FAST_TP4_ROUND_HOST_LOG     the per-round ledger: ONE `[PACKED-ROUND-HOST]` line a step with the host time of every phase of the
                                   round (step entry, each block's verify split and commits, the early draft's launch, window, fence,
                                   collect, GDN flush, selection and tail). A control arm carries this flag alone, so the paired
                                   timing attributes the gain phase by phase against the same instrument.
  QWEN_FAST_TP4_ROUND_HOST_SELECT  the FP64 draft selection (draft_selector.select_active_candidates) with the operand checks that are a
                                   function of the CODEBOOKS made once, not twice a call: the gathered codebooks were scanned for
                                   non-finite values once per 8-position chunk (45% of the call on a host CPU), and converted to float64
                                   once per chunk. The arithmetic is the reference's, statement for statement.
  QWEN_FAST_TP4_ROUND_HOST_READ    the quad readback's host half: the 16 per-chunk candidate checks and merges done as one batch over the
                                   stacked chunks (merge_chunk_candidates), and the replicated selector-feature guard (the same tensor
                                   read back from every chip and compared) run on the first 64 reads of the process and every 64th after
                                   (chip 0's copy is what the selection uses either way).
  QWEN_FAST_TP4_ROUND_HOST_KEYED   the verify-time diff write keyed on (start, page table) per segment: when neither moved since the
                                   pre-stage, every staged value except the tokens is the snapshot's, so the 149-destination recompute and
                                   comparison is skipped and the tokens buffer alone is written. Any change takes today's diff.
  QWEN_FAST_TP4_ROUND_HOST_LEAN    the per-phase diagnostic lines nobody parses (the `[PHASE]` begin/end of propose, prepare_proposals,
                                   propose_quad, early_draft and packed_commit; `[PACKED-PUBLISH-SPLIT]`, `[PACKED-COMMIT-HOST]`,
                                   `[PACKED-COMMIT]`) are not written, about 50 of the 92 lines of an eight-live round; the ledger line
                                   replaces them. Every line a gate or a timing reader parses stays.
  QWEN_FAST_TP4_ROUND_HOST_AUDIT   (needs a lever) the reference runs beside each fast path on the live data and the two must agree
                                   byte for byte: the selection, the merge, and the keyed round's full staged values against the
                                   snapshot's. A disagreement is logged (`[ROUND-HOST-AUDIT] ... equal=0`), the REFERENCE bytes are used,
                                   and the smoke check fails the arm.

EXACTNESS. Every fast path is the reference with work removed that cannot change a result, and every fast path DECLINES to the reference
(whole call) the moment any check fails, so an error is raised by the reference's own code with its own message. SELECT and READ accept a
subset of what the reference accepts and return its bytes; KEYED writes the values the diff would have written (they are equal by
construction: the values are a pure function of (tokens, start, table) per segment, and only the tokens differ); LEAN and LOG write lines.
tests: test_tp4_round_host.py (random operands against the reference, bit for bit; flag-off identity; the real block over the fake device).
"""

import os
import sys
import time
import weakref

LOG_FLAG = 'QWEN_FAST_TP4_ROUND_HOST_LOG'
SELECT_FLAG = 'QWEN_FAST_TP4_ROUND_HOST_SELECT'
READ_FLAG = 'QWEN_FAST_TP4_ROUND_HOST_READ'
KEYED_FLAG = 'QWEN_FAST_TP4_ROUND_HOST_KEYED'
LEAN_FLAG = 'QWEN_FAST_TP4_ROUND_HOST_LEAN'
AUDIT_FLAG = 'QWEN_FAST_TP4_ROUND_HOST_AUDIT'
FLAGS = (LOG_FLAG, SELECT_FLAG, READ_FLAG, KEYED_FLAG, LEAN_FLAG, AUDIT_FLAG)
LEVER_FLAGS = (SELECT_FLAG, READ_FLAG, KEYED_FLAG)

ENGAGED_MARKER = '[PINDIAG] round host engaged'
REFUSED_MARKER = '[PINDIAG] round host refused'
LEDGER_MARKER = '[PACKED-ROUND-HOST]'
AUDIT_MARKER = '[ROUND-HOST-AUDIT]'
DECLINED_MARKER = '[PINDIAG] round host declined'

# The `[PHASE]` names LEAN does not write (serving_worker_hook.phase). `packed_verify` and `execute` stay: the gates and the timing reader parse them.
LEAN_PHASES = frozenset(('propose', 'prepare_proposals', 'propose_quad', 'early_draft', 'packed_commit'))
# The replicated selector-feature guard (READ): the first reads of the process and then every GUARD_EVERY-th.
GUARD_FIRST = 64
GUARD_EVERY = 64

# name -> bool, refreshed from the environment (refresh); read on the hot path as a dict lookup.
_STATE = {name: False for name in FLAGS}


def log_line(text):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def parse(environ=None):
    """{flag: bool} for every flag, strictly (a value but 0 or 1 raises ValueError), and the one cross rule: the audit needs a lever."""
    state = {name: _flag(name, environ) for name in FLAGS}
    if state[AUDIT_FLAG] and not any(state[name] for name in LEVER_FLAGS):
        raise ValueError('%s needs at least one of %s' % (AUDIT_FLAG, ', '.join(LEVER_FLAGS)))
    return state


def refresh(environ=None, strict=False):
    """Re-read the flags into the hot-path state. At import (strict=False) a malformed value leaves every flag off: the attach (engage) is
    where it raises, so a bad flag fails the attach and not the import of a module the hook needs."""
    try:
        state = parse(environ)
    except ValueError:
        if strict:
            raise
        state = {name: False for name in FLAGS}
    _STATE.update(state)
    ledger.on = _STATE[LOG_FLAG] or _STATE[LEAN_FLAG]
    return dict(_STATE)


def requested(environ=None):
    """Whether any flag is set to anything but 0 (the cheap test before the rest of this module is used)."""
    environ = os.environ if environ is None else environ
    return any(environ.get(name, '0') != '0' for name in FLAGS)


def log_enabled():
    return _STATE[LOG_FLAG] or _STATE[LEAN_FLAG]


def select_enabled():
    return _STATE[SELECT_FLAG]


def read_enabled():
    return _STATE[READ_FLAG]


def keyed_enabled():
    return _STATE[KEYED_FLAG]


def lean_enabled():
    return _STATE[LEAN_FLAG]


def audit_enabled():
    return _STATE[AUDIT_FLAG]


def engage(environ=None):
    """The attach (serving_packed_step.PackedStep): read the flags strictly, say once which are engaged, and reset the run's counters.
    Nothing set, nothing logged. Raises ValueError for a malformed flag or an audit without a lever."""
    state = refresh(environ, strict=True)
    reset_counters()
    if not any(state.values()):
        return state
    log_line('%s %s' % (ENGAGED_MARKER, ' '.join('%s=%d' % (name.rsplit('_', 1)[-1].lower(), int(state[name])) for name in FLAGS)))
    return state


# -- counters: which paths a round actually took (printed in the ledger line, read by the smoke check) -------------------------------------
COUNTS = dict(select_fast=0, select_declined=0, read_fast=0, read_declined=0, guard_full=0, guard_skipped=0, keyed=0, keyed_declined=0,
              audit_equal=0, audit_unequal=0)
_ROUND = dict(select=0, read=0, keyed=0, guard=0)


def reset_counters():
    for key in COUNTS:
        COUNTS[key] = 0
    for key in _ROUND:
        _ROUND[key] = 0
    _GUARD['reads'] = 0
    for key in _DECLINES:
        _DECLINES[key] = 0


def count(name, n=1):
    COUNTS[name] += n


def note_path(kind, n=1):
    """A fast path taken in the open round (ledger fields sel=, read=, keyed=)."""
    _ROUND[kind] += n


# -- the ledger --------------------------------------------------------------------------------------------------------------------------
class Ledger:
    """One `[PACKED-ROUND-HOST]` line per step, from stamps taken at the step's seams. Off, every method returns at once."""

    FIELDS = ('gap', 'entry', 'v0', 'v0_in', 'v0_tr', 'v0_sy', 'v0_rb', 'c0', 'v1', 'v1_in', 'v1_tr', 'v1_sy', 'v1_rb', 'c1',
              'ed', 'ed_pre', 'launch', 'window', 'fence', 'collect', 'flush', 'select', 'tail', 'step')

    def __init__(self):
        self.on = False
        self.rounds = 0
        self.open = False
        self.last_end = None
        self.clear()

    def clear(self):
        self.marks, self.values, self.verifies, self.commits = {}, {}, 0, 0
        self.live = self.position = None

    def begin(self, now=None):
        """The step's entry. A round still open (its drafts ran outside the step: no early draft) is written first, as it stands."""
        if not self.on:
            return
        if self.open:
            self.emit()
        self.clear()
        self.open = True
        now = time.perf_counter() if now is None else now
        self.marks['begin'] = now
        if self.last_end is not None:
            self.values['gap'] = (now - self.last_end) * 1000

    def mark(self, name, now=None):
        if self.on and self.open:
            self.marks[name] = time.perf_counter() if now is None else now

    def add(self, name, ms):
        if self.on and self.open:
            self.values[name] = self.values.get(name, 0.0) + ms

    def set_live(self, live, position=None):
        """The step's live users and their mean frontier (the timing reader pairs rounds by it)."""
        if self.on and self.open:
            self.live, self.position = live, position

    def note_verify(self, metrics, total_ms):
        """One block's verify (run_verified_block): the PACKED-PHASE split and the whole phase."""
        if not (self.on and self.open) or self.verifies > 1:
            return
        key = 'v%d' % self.verifies
        self.verifies += 1
        self.values[key] = total_ms
        if metrics:
            self.values[key + '_in'] = metrics.get('input_ms', 0.0)
            self.values[key + '_tr'] = metrics.get('blocking_trace_host_ms', 0.0)
            self.values[key + '_sy'] = metrics.get('replay_checks_sync_ms', 0.0)
            self.values[key + '_rb'] = metrics.get('output_readback_host_ms', 0.0)

    def note_commit(self, ms):
        if not (self.on and self.open) or self.commits > 1:
            return
        self.values['c%d' % self.commits] = ms
        self.commits += 1

    def span(self, name, start, end):
        """A value from two marks, when both were taken."""
        if start in self.marks and end in self.marks:
            self.values[name] = (self.marks[end] - self.marks[start]) * 1000

    def emit(self, now=None):
        """Write the open round's line (once) and close it."""
        if not (self.on and self.open):
            return None
        self.open = False
        now = time.perf_counter() if now is None else now
        marks = self.marks
        self.span('entry', 'begin', 'step')
        self.span('step', 'step', 'stepped')
        self.span('ed', 'draft0', 'draft1')
        self.span('ed_pre', 'draft0', 'quads0')
        self.span('launch', 'quads0', 'quads1')
        self.span('window', 'window0', 'window1')
        self.span('fence', 'fence0', 'fence1')
        self.span('collect', 'collect0', 'collect1')
        self.span('flush', 'collect1', 'flush1')
        self.span('select', 'flush1', 'select1')
        if 'select1' in marks and 'draft1' in marks:
            self.values['tail'] = (marks['draft1'] - marks['select1']) * 1000
        self.rounds += 1
        line = format_line(self.rounds, self.live, self.values, levers_taken(), COUNTS, self.position)
        _ROUND.update(select=0, read=0, keyed=0, guard=0)
        self.last_end = marks.get('draft1', now)
        self.marks = {}
        log_line(line)
        return line


def levers_taken():
    return dict(_ROUND)


def format_line(round_number, live, values, taken, counts, position=None):
    """`[PACKED-ROUND-HOST] round=N live=L pos=P f=ms ...` in FIELDS order (a missing field is -), then the paths taken this step."""
    fields = ' '.join('%s=%s' % (name, '%.2f' % values[name] if name in values else '-') for name in Ledger.FIELDS)
    return '%s round=%d live=%s pos=%s %s sel=%d read=%d keyed=%d guard=%d' % (
        LEDGER_MARKER, round_number, '-' if live is None else live, '-' if position is None else int(position), fields,
        taken['select'], taken['read'], taken['keyed'], taken['guard'])


ledger = Ledger()


# -- SELECT: the FP64 selection with the codebook checks made once -----------------------------------------------------------------------
_FINITE = {}
_SCAN_ROWS = 8192


def codebook_finite(tensor):
    """Whether every element of `tensor` is finite, from one full scan kept for the tensor object while its storage address, version, shape
    and dtype stand. The codebooks are the draft head's lent weights (one object for the life of the process); the scan runs on the first
    selection, in chunks of rows, and never again. A codebook with a non-finite value anywhere is reported False and every call then takes
    the reference's own per-call scan of the rows it gathers."""
    import torch

    key = id(tensor)
    stamp = (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tensor.dtype)
    entry = _FINITE.get(key)
    if entry is not None and entry[0]() is tensor and entry[1] == stamp:
        return entry[2]
    finite = True
    if tensor.ndim == 2:
        for start in range(0, tensor.shape[0], _SCAN_ROWS):
            if not bool(torch.isfinite(tensor[start:start + _SCAN_ROWS]).all()):
                finite = False
                break
    else:
        finite = bool(torch.isfinite(tensor).all())
    _FINITE[key] = (weakref.ref(tensor), stamp, finite)
    return finite


def forget_codebooks():
    _FINITE.clear()


class _Decline(Exception):
    """The fast path will not answer: the caller runs the reference, which raises (or answers) as it always has. `line` names the check
    that said so (this module's line number), for the declined line."""

    def __init__(self):
        super().__init__()
        self.line = sys._getframe(1).f_lineno


_DECLINES = dict(select=0, read=0)
DECLINE_LINES = 3


def note_decline(kind, failure):
    """A fast path that handed the call to the reference: counted, and the first DECLINE_LINES of each kind written (the smoke check fails
    an arm with one: the lever did not run on that call)."""
    count(kind + '_declined')
    _DECLINES[kind] += 1
    if _DECLINES[kind] <= DECLINE_LINES:
        declined(kind, 'check_line_%s' % getattr(failure, 'line', '?'))


def _check_chunk(hidden, candidates, unary, predecessors, successors, anchors):
    """draft_selector.validate_selector_operands for one chunk, with the finiteness of the two codebooks already established by the caller:
    the same conditions in the same order (a refusal raises _Decline and the reference then raises its own message)."""
    import torch

    operands = (hidden, candidates, unary, predecessors, successors, anchors)
    if any(value.device.type != 'cpu' for value in operands):
        raise _Decline()
    if hidden.ndim != 3 or predecessors.ndim != 2:
        raise _Decline()
    batch, positions, rank = hidden.shape
    vocabulary = predecessors.shape[0]
    if (batch < 1 or not 1 <= positions <= 8 or rank < 1 or vocabulary < 1
            or predecessors.shape[1] != rank or successors.shape != predecessors.shape
            or candidates.ndim != 3 or candidates.shape[:2] != (batch, positions)
            or not 1 <= candidates.shape[2] <= min(16, vocabulary)
            or unary.shape != candidates.shape or anchors.shape != (batch,)):
        raise _Decline()
    if candidates.dtype != torch.int64 or anchors.dtype != torch.int64:
        raise _Decline()
    if (any(not torch.is_floating_point(value) or not torch.isfinite(value).all() for value in (hidden, unary))
            or not torch.is_floating_point(predecessors) or not torch.is_floating_point(successors)):
        raise _Decline()
    if any(torch.any(value < 0) or torch.any(value >= vocabulary) for value in (candidates, anchors)):
        raise _Decline()
    sorted_candidates = candidates.sort(dim=-1).values
    if torch.any(sorted_candidates[..., 1:] == sorted_candidates[..., :-1]):
        raise _Decline()


def _fast_select(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    import torch

    if (projected_hidden.ndim != 3 or not 1 <= projected_hidden.shape[1] <= 31
            or candidates.ndim != 3 or candidates.shape[:2] != projected_hidden.shape[:2]
            or unary_logits.shape != candidates.shape or anchors.shape != (projected_hidden.shape[0],)
            or candidates.dtype != torch.int64 or anchors.dtype != torch.int64
            or candidates.device.type != 'cpu' or anchors.device.type != 'cpu'
            or predecessor_codes.ndim != 2 or successor_codes.shape != predecessor_codes.shape
            or predecessor_codes.device.type != 'cpu' or successor_codes.device.type != 'cpu'):
        raise _Decline()
    if any(torch.any(value < 0) or torch.any(value >= predecessor_codes.shape[0]) for value in (candidates, anchors)):
        raise _Decline()
    if not (codebook_finite(predecessor_codes) and codebook_finite(successor_codes)):
        raise _Decline()
    identifiers, inverse = torch.unique(torch.cat((anchors.flatten(), candidates.flatten())), sorted=True, return_inverse=True)
    local_anchors = inverse[:anchors.numel()].reshape(anchors.shape)
    local_candidates = inverse[anchors.numel():].reshape(candidates.shape)
    selected_rows, score_rows = [], []
    local_predecessors, local_successors = predecessor_codes[identifiers], successor_codes[identifiers]
    predecessor_double, successor_double = local_predecessors.double(), local_successors.double()
    for start in range(0, projected_hidden.shape[1], 8):
        hidden = projected_hidden[:, start:start + 8]
        chunk_candidates = local_candidates[:, start:start + 8]
        chunk_unary = unary_logits[:, start:start + 8]
        _check_chunk(hidden, chunk_candidates, chunk_unary, local_predecessors, local_successors, local_anchors)
        hidden, chunk_unary = hidden.double(), chunk_unary.double()
        predecessor = local_anchors
        path, scores_list = [], []
        for position in range(hidden.shape[1]):
            edges = (predecessor_double[predecessor, None, :] * hidden[:, position, None, :]
                     * successor_double[chunk_candidates[:, position]]).sum(dim=-1)
            scores = chunk_unary[:, position] + edges
            if not torch.isfinite(scores).all():
                raise _Decline()
            selected = scores.argmax(dim=-1, keepdim=True)
            predecessor = chunk_candidates[:, position].gather(1, selected).squeeze(1)
            path.append(predecessor)
            scores_list.append(scores)
        selected_chunk, score_chunk = torch.stack(path, dim=1), torch.stack(scores_list, dim=1)
        selected_rows.append(selected_chunk)
        score_rows.append(score_chunk)
        local_anchors = selected_chunk[:, -1]
    return identifiers[torch.cat(selected_rows, dim=1)], torch.cat(score_rows, dim=1)


def select_active_candidates(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    """draft_selector.select_active_candidates, SELECT's fast path: the same (tokens, scores), bit for bit, or the reference's own call."""
    from draft_selector import select_active_candidates as reference

    try:
        fast = _fast_select(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
    except _Decline as failure:
        note_decline('select', failure)
        return reference(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
    count('select_fast')
    note_path('select')
    if audit_enabled():
        expected = reference(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
        return audited('select', fast, expected)
    return fast


def audited(kind, fast, expected):
    """AUDIT: the fast result against the reference's, tensor for tensor and bit for bit. Returns the reference's on a difference."""
    import torch

    equal = all(a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b)) for a, b in zip(fast, expected))
    count('audit_equal' if equal else 'audit_unequal')
    log_line('%s kind=%s equal=%d' % (AUDIT_MARKER, kind, int(equal)))
    return fast if equal else expected


# -- READ: the quad readback's merge as one batch ---------------------------------------------------------------------------------------
def _fast_merge(chunks, block_rows):
    import torch
    import tp_shapes
    from draft_shared_head_tp import candidate_chunks, vocab_requested

    if type(block_rows) is not int or block_rows not in (8, 16, 32):
        raise _Decline()
    chips, ranges = tp_shapes.chip_count(), candidate_chunks()
    chunks = list(chunks)
    expected = {(chip, start, stop) for chip in range(chips) for start, stop in ranges}
    if len(chunks) != len(expected):
        raise _Decline()
    seen = set()
    for chunk in chunks:
        identity = chunk['chip'], chunk['start'], chunk['stop']
        if identity not in expected or identity in seen:
            raise _Decline()
        seen.add(identity)
    first_values, first_indices = chunks[0]['values'], chunks[0]['indices']
    for chunk in chunks:
        values, indices = chunk['values'], chunk['indices']
        if (values.device.type != 'cpu' or indices.device.type != 'cpu'
                or values.shape != (block_rows, 16) or indices.shape != values.shape
                or not torch.is_floating_point(values) or values.dtype != first_values.dtype
                or indices.dtype not in (torch.int32, torch.int64) or indices.dtype != first_indices.dtype):
            raise _Decline()
    stacked_values = torch.stack([chunk['values'] for chunk in chunks])
    stacked_indices = torch.stack([chunk['indices'] for chunk in chunks])
    if not torch.isfinite(stacked_values).all():
        raise _Decline()
    limits = torch.tensor([chunk['stop'] - chunk['start'] for chunk in chunks], dtype=torch.int64).reshape(-1, 1, 1)
    if torch.any(stacked_indices < 0) or torch.any(stacked_indices >= limits):
        raise _Decline()
    ordered = stacked_indices.sort(-1).values
    if torch.any(ordered[..., 1:] == ordered[..., :-1]):
        raise _Decline()
    shard = tp_shapes.vocab_shard()
    plan = None
    if vocab_requested():
        # QWEN_FAST_DRAFT_VOCAB (draft_vocab_tp, gate only): a chip's local index names a row of the shortlist, not a column of its vocabulary shard; the
        # same offsets locate the row in the list and the list gives the global token id. Off, none of this runs and `plan` stays None.
        import draft_vocab_tp

        plan = draft_vocab_tp.active_plan()
        shard = plan.per_chip
    offsets = torch.tensor([chunk['start'] + chunk['chip'] * shard for chunk in chunks], dtype=torch.int64).reshape(-1, 1, 1)
    count_chunks = len(chunks)
    values = stacked_values.permute(1, 0, 2).reshape(block_rows, count_chunks * 16)
    positions = stacked_indices.long() + offsets
    tokens = (positions if plan is None else plan.tensor()[positions]).permute(1, 0, 2).reshape(block_rows, count_chunks * 16)
    by_token = tokens.argsort(dim=-1, stable=True)
    values, tokens = values.gather(-1, by_token), tokens.gather(-1, by_token)
    selected = values.argsort(dim=-1, descending=True, stable=True)[:, :16]
    return tokens.gather(-1, selected)[None, 1:block_rows], values.gather(-1, selected)[None, 1:block_rows]


def merge_chunk_candidates(chunks, *, block_rows=8):
    """draft_shared_head_tp.merge_chunk_candidates, READ's fast path: the same (tokens, values), bit for bit, or the reference's own call."""
    from draft_shared_head_tp import merge_chunk_candidates as reference

    chunks = list(chunks)
    try:
        fast = _fast_merge(chunks, block_rows)
    except _Decline as failure:
        note_decline('read', failure)
        return reference(chunks, block_rows=block_rows)
    count('read_fast')
    note_path('read')
    if audit_enabled():
        return audited('merge', fast, reference(chunks, block_rows=block_rows))
    return fast


_GUARD = dict(reads=0)


def guard_full():
    """Whether this quad readback runs the replicated-feature guard on every chip (the first GUARD_FIRST reads of the process, then every
    GUARD_EVERY-th, and every read under the audit); False reads chip 0's copy alone. Always True with READ off."""
    if not _STATE[READ_FLAG]:
        return True
    _GUARD['reads'] += 1
    reads = _GUARD['reads']
    full = audit_enabled() or reads <= GUARD_FIRST or reads % GUARD_EVERY == 0
    count('guard_full' if full else 'guard_skipped')
    if not full:
        note_path('guard')
    return full


# -- the lines the arms and the gates read ----------------------------------------------------------------------------------------------
def declined(kind, reason):
    log_line('%s kind=%s reason=%s' % (DECLINED_MARKER, kind, str(reason).replace(' ', '_')[:96]))
refresh()
