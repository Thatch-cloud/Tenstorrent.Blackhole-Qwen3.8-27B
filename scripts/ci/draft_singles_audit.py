"""QWEN_FAST_DRAFT_SINGLES_AUDIT (default off): the batched draft against the users' own single-user drafts, in the same round.

WHY. A batched draft (a packed pair, the four-user quad) is worth having only if every user's proposal is byte for byte what its
own single-user capture proposes: the verifier keeps the TEXT exact whatever is drafted, so a batched draft that differs shows
up only as lower acceptance, and nothing but a comparison of the drafts themselves can prove it exact. The quad's own shadow
audit (QWEN_FAST_QUAD_DRAFT_AUDIT) compares the quad with the two pair traces; this compares every batched group, pair or quad,
with the SINGLES - the reference the whole exactness argument is about - and so covers the pair at four cards, whose fold has
never run on a card.

WHAT RUNS. In an audited round, after the coordinator has enqueued its batched traces and before its one fence, each member's own
single-user capture (rebuilt if a batched success released it: the audit needs the singles kept, so under the flag
_release_single_user releases nothing) is prepared with the SAME seed the batched pass got - the same K/V banks, positions and
anchors, one more replay on the in-order queue. After the fence, before anything else is enqueued (the batched readback, and
the deferred GDN commits that follow it, may write into freed buffers), every raw output of every trace is read to the host:
per chip, each candidate chunk's values and indices and the replicated selector features, the batched trace's rows for that
user and the single's own. After the round's batched selection the audit compares, per user, bit for bit:
    raw:<values|indices>:chunk<n>:chip<c>:u<user>   the per-chip top-k reads (the user's rows of the batched output)
    raw:projected:chip<c>:u<user>                    the per-chip selector features
    features / candidates / scores                   the parts the selector takes (batched readback against the single's)
    tokens                                           the draft tokens the round adopted against DFlashDevice.select_proposal
and logs one AUDIT_LINE per batched group. The first difference names the stage. A read or compare that fails is equal=0 with
its stage, never a failed round; the singles' pending replays are always discarded. A correctness arm only: it spends back what
batching saves.

READ AT EACH ROUND, never at import. Unset or empty: off, and nothing here is imported. 'all', or a positive N (the first N
rounds that ran a batched group). Anything else raises ValueError at the round.

Stdlib and torch only, torch inside functions.
"""

import os

FLAG = 'QWEN_FAST_DRAFT_SINGLES_AUDIT'
AUDIT_LINE = '[DRAFT-SINGLES-AUDIT] round=%s group=%s equal=%d stage=%s checks=%d'
BLOCK = 16


def audit_rounds(environ=None):
    """None (unset or empty), 'all', or a positive count N (the first N rounds with a batched group)."""
    value = (os.environ if environ is None else environ).get(FLAG)
    if value is None or value == '':
        return None
    if value == 'all':
        return value
    if not value.isdigit() or value != str(int(value)) or int(value) < 1:
        raise ValueError("%s must be 'all' or a positive count, got %r" % (FLAG, value))
    return int(value)


def enabled(environ=None):
    return audit_rounds(environ) is not None


def selected(audited_round, environ=None):
    """Whether the audited-candidate round numbered `audited_round` (1-based: rounds that ran a batched group) is audited."""
    rounds = audit_rounds(environ)
    return rounds == 'all' or (rounds is not None and audited_round <= rounds)


def log_line(message):
    """One INFO line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:  # noqa: BLE001 - a log line never fails a round
        pass


def same_bits(left, right):
    """Equal dtype, shape and bits (a float compared as its integer image)."""
    from dflash_packed_proposal import same_bits as bits

    return bits(left, right)


# ---------------------------------------------------------------------------------------------
# Raw reads.
# ---------------------------------------------------------------------------------------------

def raw_outputs(operations, outputs):
    """A proposal's raw outputs as host copies: every chunk's values and indices and the selector features, per chip."""
    def host(tensor):
        return [operations.to_torch(shard) for shard in operations.get_device_tensors(tensor)]

    return dict(chunks=[dict(start=chunk['start'], stop=chunk['stop'], values=host(chunk['values']),
                             indices=host(chunk['indices'])) for chunk in outputs.chunks],
                projected=host(outputs.projected))


def user_rows(raw, first, count=BLOCK):
    """One user's rows [first, first + count) of a raw read (a batched trace's block, or a single's own rows 0-15), the
    shapes the single's readback takes: values and indices (count, 16) and features (rows, 256) per chip."""
    rows = slice(first, first + count)
    return dict(
        chunks=[dict(start=chunk['start'], stop=chunk['stop'],
                     values=[value.reshape(-1, 16)[rows] for value in chunk['values']],
                     indices=[value.reshape(-1, 16)[rows] for value in chunk['indices']]) for chunk in raw['chunks']],
        projected=[value.reshape(-1, 256)[rows] for value in raw['projected']])


def read_parts(raw, block_rows=BLOCK):
    """The selector's parts of one user's rows, as DFlashDevice.select_proposal merges and slices them: (features
    (1, block_rows - 1, 256), candidates, scores) - the merge is the served merge_chunk_candidates (the four-card one at
    four cards), block_rows rows of one user."""
    import torch

    from draft_shared_head import merge_chunk_candidates

    host_chunks = []
    for chunk in raw['chunks']:
        for chip, (values, indices) in enumerate(zip(chunk['values'], chunk['indices'], strict=True)):
            host_chunks.append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                                    values=values.float().reshape(block_rows, 16),
                                    indices=indices.long().reshape(block_rows, 16)))
    candidates, unary = merge_chunk_candidates(host_chunks, block_rows=block_rows)
    projected = raw['projected']
    if any(not torch.equal(projected[0], other) for other in projected[1:]):
        raise AssertionError('Replicated learned selector features differ')
    hidden = projected[0].reshape(1, -1, 256)[:, 1:block_rows, :]
    return dict(hidden=hidden, candidates=candidates, unary=unary)


# ---------------------------------------------------------------------------------------------
# One round.
# ---------------------------------------------------------------------------------------------

class Round:
    """One audited round's state: the batched traces (their labels, trace and member devices) and, per member, its single capture
    and seed. Built by start(), read by read() after the fence, judged by finish() after the selection."""

    def __init__(self, round_number):
        self.round_number = round_number
        self.groups = []       # dict(labels=[slots], trace=, devices=[...], first_rows=[first row of each member], seeds=[...])
        self.singles = []      # dict(device=, capture=, seed=, slot=)
        self.reason = None     # why the round cannot be audited (start's or read's failure)
        self.raw = {}          # slot -> dict(batched=, single=) host copies

    def close(self):
        """Drop every single's pending replay (the device is fenced by now, or the round is failing): never raises."""
        while self.singles:
            entry = self.singles.pop()
            try:
                entry['capture'].discard_pending()
            except Exception:  # noqa: BLE001 - the audit never fails the round
                pass


def trace_members(labels, trace):
    """(devices, first rows) of one batched trace's members, in slot order: a pair's device_a and device_b at rows 0 and 16 of
    its 32-row block, the quad's four devices at rows 0, 16, 32 and 48."""
    devices = getattr(trace, 'devices', None)
    if devices is None:
        devices = (trace.device_a, trace.device_b)
    devices = list(devices)
    if len(devices) != len(labels):
        raise ValueError('The trace has %d devices for the slots %s' % (len(devices), list(labels)))
    return devices, [BLOCK * index for index in range(len(devices))]


def bucket_of(trace):
    """The pending bucket of a batched trace (its outputs and, after adopt(), its tokens): the pair keeps it third in its
    pending tuple, the quad second."""
    pending = trace._pending
    if pending is None:
        raise ValueError('The batched trace holds no pending round')
    return pending[2] if hasattr(trace, 'device_a') and not hasattr(trace, 'devices') else pending[1]


def start(batched, by_slot, round_number, ensure_single):
    """Prepare every member's own single-user capture with the seed the batched pass got. `batched` is the coordinator's list
    of (slots, trace); `by_slot` its {slot: entry}; `ensure_single(device)` rebuilds a released single capture. Returns the
    Round (its .reason set when it could not be built, its singles then discarded). Never raises."""
    state = Round(round_number)
    try:
        for labels, trace in batched:
            devices, first_rows = trace_members(labels, trace)
            group = dict(labels=list(labels), trace=trace, devices=devices, first_rows=first_rows, seeds=[])
            for slot, device in zip(labels, devices, strict=True):
                entry = by_slot[slot]
                ensure_single(device)
                capture = single_capture(device)
                if capture is None or not capture.prepare_device(entry['seed']):
                    raise ValueError('slot %s has no single-user capture to prepare' % slot)
                state.singles.append(dict(device=device, capture=capture, seed=entry['seed'], slot=slot))
                group['seeds'].append(entry['seed'])
            state.groups.append(group)
    except Exception as failure:  # noqa: BLE001 - the audit's verdict, never the round's
        state.reason = 'singles-unavailable:%s' % type(failure).__name__
        state.close()
    return state


def single_capture(device):
    """The device's own single-user capture: the original a _PackedCaptureView wraps, or the capture itself."""
    capture = device.proposal_capture
    return getattr(capture, '_original', None) if hasattr(capture, '_trace') else capture


def read(state):
    """After the round's fence and before anything else is enqueued (the batched readback, and the deferred GDN commits behind
    it, may write into freed buffers): every raw output of every batched trace and of every single as host copies, the batched
    readback's parts through the trace's own reader, and the single's tokens through DFlashDevice.select_proposal - all taken
    once, into state. A failure is the audit's verdict (state.reason)."""
    if state.reason is not None:
        return state
    try:
        singles = {entry['slot']: entry for entry in state.singles}
        for group in state.groups:
            operations = group['devices'][0].operations
            batched = raw_outputs(operations, bucket_of(group['trace']).outputs)
            group['parts'] = _batched_parts(group)
            for slot, device, first in zip(group['labels'], group['devices'], group['first_rows'], strict=True):
                entry = singles[slot]
                pending = entry['capture']._pending
                if pending is None:
                    raise ValueError('slot %s single capture holds no pending replay' % slot)
                outputs = pending[1].outputs
                single = user_rows(raw_outputs(device.operations, outputs), 0)
                state.raw[slot] = dict(batched=user_rows(batched, first), single=single, first=first,
                                       tokens=tuple(device.select_proposal(outputs, entry['seed'], BLOCK - 1)))
    except Exception as failure:  # noqa: BLE001 - the audit's verdict, never the round's
        state.reason = 'read-error:%s' % type(failure).__name__
    return state


def _batched_parts(group):
    """The batched readback of a group's outputs through the trace's own reader: per member (features, candidates, scores)."""
    trace, device = group['trace'], group['devices'][0]
    outputs = bucket_of(trace).outputs
    if hasattr(trace, 'devices'):
        import quad_draft

        return quad_draft.read_quad_outputs(device, outputs)
    from dflash_packed_proposal import read_device_outputs

    return read_device_outputs(device, outputs, len(group['devices']), BLOCK)


def _compare_group(state, group):
    """(equal, stage, checks) of one batched group against its members' singles."""
    checks = 0
    labels = group['labels']
    for user, slot in enumerate(labels):
        mine, theirs = state.raw[slot]['batched'], state.raw[slot]['single']
        for number, (left, right) in enumerate(zip(mine['chunks'], theirs['chunks'], strict=True)):
            for key in ('values', 'indices'):
                for chip, (a, b) in enumerate(zip(left[key], right[key], strict=True)):
                    checks += 1
                    if not same_bits(a.contiguous(), b.contiguous()):
                        return False, 'raw:%s:chunk%d:chip%d:u%d' % (key, number, chip, slot), checks
        for chip, (a, b) in enumerate(zip(mine['projected'], theirs['projected'], strict=True)):
            checks += 1
            if not same_bits(a.contiguous(), b.contiguous()):
                return False, 'raw:projected:chip%d:u%d' % (chip, slot), checks
    parts = group['parts']
    tokens = _adopted_tokens(group)
    for user, slot in enumerate(labels):
        single = read_parts(state.raw[slot]['single'])
        mine = parts[user]
        for stage, key in (('features', 'hidden'), ('candidates', 'candidates'), ('scores', 'unary')):
            checks += 1
            if not same_bits(mine[key], single[key]):
                return False, '%s:u%d' % (stage, slot), checks
        checks += 1
        if tuple(tokens[user]) != state.raw[slot]['tokens']:
            return False, 'tokens:u%d' % slot, checks
    return True, 'all', checks


def _adopted_tokens(group):
    """The draft tokens the round adopted, per member: the batched trace's bucket after adopt() (or its own selection)."""
    bucket = bucket_of(group['trace'])
    if bucket.tokens is None:
        raise ValueError('The batched round selected no tokens')
    return bucket.tokens


def finish(state):
    """After the round's selection: compare every group and log one AUDIT_LINE each; discard the singles' replays. Returns the
    [(labels, equal, stage, checks)] logged. Never raises."""
    results = []
    try:
        for group in state.groups:
            if state.reason is not None:
                result = (False, state.reason, 0)
            else:
                try:
                    result = _compare_group(state, group)
                except Exception as failure:  # noqa: BLE001 - the audit line is the verdict
                    result = (False, 'error:%s' % type(failure).__name__, 0)
            equal, stage, checks = result
            log_line(AUDIT_LINE % (state.round_number, list(group['labels']), int(bool(equal)), stage, checks))
            results.append((list(group['labels']), bool(equal), stage, checks))
        if not state.groups and state.reason is not None:
            log_line(AUDIT_LINE % (state.round_number, [], 0, state.reason, 0))
    finally:
        state.close()
    return results
