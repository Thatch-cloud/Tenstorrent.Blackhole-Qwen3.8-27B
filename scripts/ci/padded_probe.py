"""Variable-user packed rounds, M1: the G-pad probe (Option A's E1), QWEN_FAST_PADDED_PROBE=1.

Option A serves a round with two or three live users on the captured 64-row M3 trace, the
missing users' segments staged idle (packed_verifier.PackedVerifierEngine.idle_inputs: tokens 1,
the family start, page 0). That is exact only if the trace is segment-separable - a live row
depends on its own segment's inputs alone - and only if an idle segment writes nothing a live
user keeps. This probe measures both on hardware, inside a real 4 x 131k run, without serving
a single padded round.

On packed rounds ROUNDS (the verifier's own count, as [PACKED-PHASE] round=), after the round's
readback and before its commit (packed_verifier.verify):
  1. snapshot every segment: ids (every per-row output: the ids, and under QWEN_FAST_VERIFY_T1
     the per-chip maxima and the audited reference), the logits rows, the five tap rows, and
     the retained GDN `states` / `packed_conv_states` - each chip read back and hashed
     (sha256), never held;
  2. replay each live pattern of PATTERNS with the other segments from idle_inputs, and compare
     the live segments' rows bit for bit with step 1, per class and chip;
  3. hash each idle segment's carry (its pool slot's state: the verify trace reads it and must
     never write it) before and after that replay;
  4. restage the round's own inputs and replay once more - in a finally, so it runs whatever a
     pattern did - and compare the WHOLE snapshot with step 1: the determinism check, and the
     proof that what the commit reads is this round's own replay;
  5. commit normally (the caller).
A pattern needing three idle segments is refused by idle_inputs (page 0 holds two tile rows;
the T2 chained K/V write refuses a shared one) and logged exact=refused, never run. A round
whose live table holds page 0 inside its used range [0, (position + rows + 63) // 64) is not
probed, nor is a padded round (QWEN_FAST_PADDED_BLOCK, M2: some segments already idle). The
page-0 check itself runs on EVERY packed round while the flag is on, over the live segments.

What it can change: nothing a user sees. The predictions are read before step 2; the final
replay restages the round's inputs and step 4 proves it bit-identical, down to the states the
commit DMAs. If it is not - or a carry changed - the probe logs the line and raises, failing
the round's requests rather than serving a round the probe altered. A pattern whose live rows
differ is logged exact=0 and does not stop the round: step 4 still restores it.

Log lines (the gate parses the first five fields; lever_n_m3native_gate.PADDED_PROBE_LINE):
  [PINDIAG] padded probe round=R live=a,b exact=1|0|refused|error trace_ms=T idle_carry_intact=1|0|- idle=c,d differ=...
  (live= every segment is the final replay: the determinism check; its idle_carry_intact is
  every carry against the probe's start)
  [PINDIAG] padded probe page0 rounds=N hits=H         (each probe round: the running check)
  [PINDIAG] padded probe page0 hit round=R segment=S position=P page_index=I

Cost: a probe round reads back about 15 segment snapshots (at 4 x 131k the retained states are
~1.2 GB per segment across both chips), so it holds that round for tens of seconds. Diagnostic
arms only (R1); never a timed one.
"""

import hashlib
import os
import sys
import time

FLAG = 'QWEN_FAST_PADDED_PROBE'
ROUNDS = (3, 20)
PATTERNS = ((0,), (1,), (0, 1), (1, 3), (0, 1, 2))
MARKER = '[PINDIAG] padded probe'
PAGE0_LINE = '[PINDIAG] padded probe page0'
PAGE0_HIT = '[PINDIAG] padded probe page0 hit'
CLASSES = ('ids', 'logits', 'taps', 'states')

_STATE = dict(rounds=0, hits=0)


def enabled(environ=None):
    """QWEN_FAST_PADDED_PROBE=1."""
    return (os.environ if environ is None else environ).get(FLAG) == '1'


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


def used_pages(start, rows):
    """The page-table entries a user at `start` reads or writes this round."""
    return (int(start) + int(rows) + 63) // 64


def page_zero_hits(users, rows):
    """[(segment, start, first page index)] for every user whose table holds page 0 inside its
    used range. `users` in segment order, None for a segment with no user."""
    hits = []
    for segment, user in enumerate(users):
        if user is None:
            continue
        _, start, table = user
        row = table[0, :used_pages(start, rows)]
        zeros = (row == 0).nonzero()
        if len(zeros):
            hits.append((segment, int(start), int(zeros[0])))
    return hits


def patterns_for(users):
    """The live patterns a block of `users` segments can replay (every one is a proper subset)."""
    return tuple(pattern for pattern in PATTERNS if len(pattern) < users and all(s < users for s in pattern))


def _update(digest, value):
    import torch

    flat = value.detach().contiguous().reshape(-1)
    digest.update(repr((tuple(value.shape), str(value.dtype))).encode())
    digest.update(memoryview(flat.view(torch.uint8).numpy()))


def _chips(operations, tensor):
    return [operations.to_torch(part) for part in operations.get_device_tensors(tensor)]


def _rows(value, start, stop):
    return value.reshape(-1, value.shape[-1])[start:stop]


def snapshot(engine, segments):
    """{(class, segment, chip): sha256} of what the last replay left for `segments`."""
    from packed_verifier import segment_rows

    operations, shape = engine.operations, engine.shape
    spans = {segment: segment_rows(shape, segment) for segment in segments}
    digests = {}

    def add(kind, segment, chip, value):
        key = (kind, segment, chip)
        if key not in digests:
            digests[key] = hashlib.sha256()
        _update(digests[key], value)

    logits, *per_row = engine.output
    for tensor in per_row:
        if tensor is None:
            continue
        for chip, value in enumerate(_chips(operations, tensor)):
            flat = value.reshape(-1)[:shape.block_rows]
            for segment, (start, stop) in spans.items():
                add('ids', segment, chip, flat[start:stop])
    for chip, value in enumerate(_chips(operations, logits)):
        for segment, (start, stop) in spans.items():
            add('logits', segment, chip, _rows(value, start, stop))
    for tap in engine.feature_capture.outputs():
        for chip, value in enumerate(_chips(operations, tap)):
            for segment, (start, stop) in spans.items():
                add('taps', segment, chip, _rows(value, start, stop))
    for state, result, carries in engine.fixture.retained.records:
        pieces = result['segment_results']
        for segment in segments:
            piece = pieces[segment]
            for tensor in (piece['states'], *piece['packed_conv_states']):
                for chip, value in enumerate(_chips(operations, tensor)):
                    add('states', segment, chip, value)
    return {key: digest.hexdigest() for key, digest in digests.items()}


def carry_digest(engine, segment):
    """sha256 of one segment's carry (its pool slot's 48 layer states), both chips."""
    digest = hashlib.sha256()
    for layer in engine.carries[segment]:
        for tensor in layer:
            for value in _chips(engine.operations, tensor):
                _update(digest, value)
    return digest.hexdigest()


def differences(before, after, segments):
    """'class:chips' for every class whose digest moved for one of `segments`, '+'-joined chips."""
    moved = {}
    for key in sorted(set(before) | set(after)):
        kind, segment, chip = key
        if segment in segments and before.get(key) != after.get(key):
            moved.setdefault(kind, set()).add(chip)
    return ['%s:%s' % (kind, '+'.join(str(chip) for chip in sorted(moved[kind]))) for kind in CLASSES if kind in moved]


def _label(segments):
    return ','.join(str(segment) for segment in segments) or '-'


def line(round_number, live, exact, trace_ms, intact, idle, differ, reason=None):
    text = '%s round=%d live=%s exact=%s trace_ms=%s idle_carry_intact=%s idle=%s differ=%s' % (
        MARKER, round_number, _label(live), exact, '-' if trace_ms is None else '%.2f' % trace_ms,
        '-' if intact is None else int(intact), _label(idle), ','.join(differ) or '-')
    if reason:
        text += ' reason=%s' % str(reason).replace(' ', '_')[:160]
    return text


def stage(engine, users):
    from packed_verifier import stage_packed

    return stage_packed(engine.operations, engine.model, engine.fixture, engine.shape, users)


def replay(engine):
    """One blocking replay of the verify trace, then a fence. Returns the replay's host ms."""
    started = time.perf_counter()
    engine.operations.execute_trace(engine.mesh, engine.trace, cq_id=0, blocking=True)
    elapsed = (time.perf_counter() - started) * 1000
    engine.operations.synchronize_device(engine.mesh)
    return elapsed


def kv_refusal(engine, users):
    """The T2 guard's reason these users cannot share the chained K/V write, checked on the host
    BEFORE staging - stage_packed's own backstop would log KV_SHARED, which fails the arm."""
    if not getattr(engine, 'kv_chains', False):
        return None
    import verify_trace_t2

    rows = engine.rows_per_user
    conflict = verify_trace_t2.kv_conflict([(range(start, start + rows), table[0]) for _, start, table in users])
    return None if conflict is None else verify_trace_t2.kv_conflict_reason(conflict)


def run_pattern(engine, users, live, baseline, round_number):
    """Step 2 and 3 for one pattern. Returns its result dict (the line's fields)."""
    idle = tuple(segment for segment in range(engine.users) if segment not in live)
    try:
        fills = engine.idle_inputs(live)
    except ValueError as refusal:
        log_line(line(round_number, live, 'refused', None, None, idle, [], refusal))
        return dict(live=live, idle=idle, exact='refused', reason=str(refusal))
    padded = [users[segment] if segment in live else fills[segment] for segment in range(engine.users)]
    reason = kv_refusal(engine, padded)
    if reason is not None:
        log_line(line(round_number, live, 'refused', None, None, idle, [], reason))
        return dict(live=live, idle=idle, exact='refused', reason=reason)
    before = {segment: carry_digest(engine, segment) for segment in idle}
    try:
        stage(engine, padded)
    except ValueError as refusal:
        # stage_packed's host checks run before any copy: nothing was written.
        log_line(line(round_number, live, 'refused', None, None, idle, [], refusal))
        return dict(live=live, idle=idle, exact='refused', reason=str(refusal))
    trace_ms = replay(engine)
    after = snapshot(engine, live)
    intact = all(carry_digest(engine, segment) == before[segment] for segment in idle)
    differ = differences(baseline, after, live)
    log_line(line(round_number, live, int(not differ), trace_ms, intact, idle, differ))
    return dict(live=live, idle=idle, exact=int(not differ), trace_ms=trace_ms, idle_carry_intact=intact,
                differ=differ)


def after_readback(engine, users, round_number):
    """The per-round entry (packed_verifier.verify, flag on): the page-0 check every round, the
    probe on ROUNDS. `users`: the round's own (tokens, start, pages) in segment order. Returns
    the probe's results on a probe round, else None. Raises when the round's own state could not
    be shown restored (see the module docstring)."""
    rows = engine.rows_per_user
    hits = page_zero_hits(users, rows)
    _STATE['rounds'] += 1
    _STATE['hits'] += len(hits)
    for segment, start, index in hits:
        log_line('%s round=%d segment=%d position=%d page_index=%d' % (PAGE0_HIT, round_number, segment, start, index))
    if round_number not in ROUNDS:
        return None
    log_line('%s rounds=%d hits=%d' % (PAGE0_LINE, _STATE['rounds'], _STATE['hits']))
    everyone = tuple(range(engine.users))
    if hits:
        log_line(line(round_number, everyone, 'refused', None, None, (), [], 'page0 in a live table'))
        return []
    idle = tuple(segment for segment, user in enumerate(users) if user is None)
    if idle:
        # A padded round (QWEN_FAST_PADDED_BLOCK, M2): the patterns are compared with an all-live
        # round, which this is not, and its own inputs cannot be restaged without its idle ones.
        log_line(line(round_number, tuple(segment for segment in everyone if segment not in idle), 'refused', None,
                      None, idle, [], 'a padded round'))
        return []
    baseline = snapshot(engine, everyone)
    carries = {segment: carry_digest(engine, segment) for segment in everyone}
    results = []
    try:
        for live in patterns_for(engine.users):
            try:
                results.append(run_pattern(engine, users, live, baseline, round_number))
            except Exception as failure:
                # A pattern that failed on the device side is reported and the probe moves on;
                # the restage below still runs, and fails the round if the device cannot.
                reason = '%s: %s' % (type(failure).__name__, failure)
                log_line(line(round_number, live, 'error', None, None,
                              tuple(s for s in everyone if s not in live), [], reason))
                results.append(dict(live=live, exact='error', reason=reason))
    finally:
        # Step 4: this round's own inputs, restaged and replayed, whatever happened above.
        stage(engine, users)
        trace_ms = replay(engine)
    final = snapshot(engine, everyone)
    differ = differences(baseline, final, everyone)
    intact = all(carry_digest(engine, segment) == carries[segment] for segment in everyone)
    log_line(line(round_number, everyone, int(not differ), trace_ms, intact, (), differ))
    results.append(dict(live=everyone, idle=(), exact=int(not differ), trace_ms=trace_ms, idle_carry_intact=intact,
                        differ=differ))
    engine.validate_bindings()
    if differ or not intact or any(result.get('idle_carry_intact') is False for result in results):
        raise AssertionError('Padded probe round %d could not show the round restored: differ=%s carries_intact=%s'
                             % (round_number, differ, intact))
    return results
