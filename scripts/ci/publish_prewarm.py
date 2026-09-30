"""QWEN_FAST_PUBLISH_PREWARM (M3NATIVE_PUBLISH_PREWARM; default off): build the drafter
publication's eager programs once per process, at a request's admission, instead of at the
first sequential commit of each accepted prefix.

WHY. A sequential step (serving_sequential_step, beside a packed block at widths 1, 2 and 4)
commits through DFlashRequestRuntime.publish: the verified features, then
drafter.prepare_publication (DFlashDevice: project_features sliced, padded and trimmed to the
prefix, the history concat/slice/pad, DraftKVHistory.prepare's per-layer slice/concat/pad),
then the target's pre-captured commit trace. The target side is warmed and captured for every
prefix when the engine is built (verifier_engine.py, the per-prefix commit captures); the
drafter side is eager, shaped by the prefix, and nothing warms it, so the first rows=4 commit of
each prefix creates its programs. That cost 13-77 ms in fourteen runs and 0.35-4.1 s in the
K5-A B arms (v228, v230), where program creation was inflated; a second user at a prefix already
seen was fast (the k5dbg verdict, sections 1-2).

WHAT. warm(device, engine), called by serving_request_factory.from_prefill after the engine is
built and before the FastRequest binds it: for every captured bucket of this request's engine,
for each prefix 1..rows this process has not warmed at that width yet,
    publication = device.prepare_publication(features, prefix, position=device.position)
    device.discard_publication(publication)
with `features` that bucket's own feature taps (bucket['feature_capture'].outputs(), what
VerifierEngine.verified_features_for_publication hands the runtime for a ticket of that width).
That is DFlashRequestRuntime.publish's own call (dflash_request_runtime.py), argument for
argument: merge_release and fused_steady_state keep their defaults, so under
QWEN_FAST_ROUND_B1 it is the non-fused branch and never sets history_stale. The sequential step
installs no publish option around it (serving_packed_step.commit_entry does, for the packed
commits only), so this is the call the sequential commit makes. Only at history_rows == 2048,
where every later publication has these shapes; below that (the prefill ramp) the history grows
every step and a warm of today's shapes would not be the next step's, so it is skipped, logged.

WHY IT IS EXACT. A prepare writes only the drafter's SPARE feature history and the cache's SPARE
K/V banks, each a full 2048-row copy (dflash_device.prepare_publication / its B1 twin,
draft_kv_history.DraftKVHistory.prepare); a discard clears the two pending records and nothing
else (DFlashDevice.discard_publication, DraftKVHistory.discard). Nothing is committed, so no
pair swaps: the committed history, the active banks, both positions and history_rows, the target
engine and the session are untouched. What the spares hold afterwards is read by nobody before
it is written again in full: the next prepare rewrites both before commit_publication swaps them
in; the fused in-place commit (fused_commit, _INPLACE) never swaps; the pair trace's live-bank
normalisation and the fused audit write their spare before reading it; and a C7 history
(history_stale) is refused by every reader. The gate's real-text exact check stays the proof on
hardware; test_publish_prewarm runs the real publication code on a torch fake and holds the
committed history and banks bit for bit against a device that never warmed.

SIDE EFFECTS. The first publication's '[PINDIAG] round b1 engaged' names site=publication (the
gate checks only that it is there). Each warm runs project_features' eager CCL gather at
admission, the same pattern the drafter's own construction runs there already. About seven
(rows, prefix) pairs once per process, at the first request's admission (its TTFT only).

MARKER: one line per call, '[PINDIAG] publish prewarm pairs=<rows>:<first>-<last>,... count=<n>
ms=<wall> program_cache=<before>-><after>' (pairs=none count=0 when every width was warmed
earlier in this process), or '[PINDIAG] publish prewarm skipped history_rows=<rows>'. The gate
(lever_n_m3native_gate.PUBLISH_PREWARM_*) requires a line that warmed something when the flag
reaches the server, so a mounted-but-unexecuted change fails the arm.
"""

import time

FLAG = 'QWEN_FAST_PUBLISH_PREWARM'
MARKER = '[PINDIAG] publish prewarm'
HISTORY_ROWS = 2048

# The (rows, prefix) pairs this process has already published once. The programs are the
# mesh's (its program cache), not the request's, so a later request's engine at a width seen
# before needs nothing: its taps and spares have the same shapes and memory configs.
_WARMED = set()


def _log(template, *values):
    try:
        from loguru import logger
    except ImportError:
        print(template.format(*values), flush=True)
        return
    logger.info(template, *values)


def program_cache(device):
    """mesh.num_program_cache_entries(), as dflash_device's proposal audit reads it, or 'n/a'."""
    count = getattr(getattr(device, 'mesh', None), 'num_program_cache_entries', None)
    if not callable(count):
        return 'n/a'
    try:
        return int(count())
    except Exception:
        return 'n/a'


def describe_pairs(pairs):
    """'1:1,2:1-2,4:1-4' for [(1, 1), (2, 1), (2, 2), (4, 1), ...]: per width, its prefix run(s)."""
    runs = []
    for rows, prefix in pairs:
        if runs and runs[-1][0] == rows and runs[-1][2] == prefix - 1:
            runs[-1][2] = prefix
        else:
            runs.append([rows, prefix, prefix])
    return ','.join('%d:%d' % (rows, first) if first == last else '%d:%d-%d' % (rows, first, last)
                    for rows, first, last in runs) or 'none'


def _check_untouched(device, position, history_rows):
    kv_history = getattr(device, 'kv_history', None)
    if (device.pending is not None or device.position != position or device.history_rows != history_rows
            or (kv_history is not None and kv_history.pending is not None)):
        raise AssertionError('A publish prewarm must leave nothing pending and the frontier where it was: '
                             'pending=%r position=%r->%r history_rows=%r->%r' % (
                                 device.pending, position, device.position, history_rows, device.history_rows))


def warm(device, engine, *, log=None):
    """Prepare and discard the drafter publication of every (rows, prefix) of `engine`'s captured
    buckets this process has not published yet; returns the pairs warmed, in order."""
    log = _log if log is None else log
    history_rows = getattr(device, 'history_rows', None)
    if history_rows != HISTORY_ROWS:
        log('{} skipped history_rows={} (the publication shapes settle at {})', MARKER, history_rows, HISTORY_ROWS)
        return ()
    position = device.position
    _check_untouched(device, position, history_rows)
    warmed = []
    before = None
    started = time.perf_counter()
    for bucket in engine.buckets.values():
        # A packed segment's synthetic bucket (VerifierEngine.adopt_packed) exists only while a
        # packed ticket is being published; the sequential step never publishes from it.
        capture = bucket.get('feature_capture')
        if capture is None or bucket.get('packed') is not None:
            continue
        rows = bucket['rows']
        prefixes = [prefix for prefix in range(1, rows + 1) if (rows, prefix) not in _WARMED]
        if not prefixes:
            continue
        if before is None:
            before = program_cache(device)
        features = capture.outputs()
        for prefix in prefixes:
            publication = device.prepare_publication(features, prefix, position=position)
            device.discard_publication(publication)
            _check_untouched(device, position, history_rows)
            _WARMED.add((rows, prefix))
            warmed.append((rows, prefix))
    elapsed = (time.perf_counter() - started) * 1000
    after = program_cache(device) if warmed else 'n/a'
    log('{} pairs={} count={} ms={:.2f} program_cache={}->{}', MARKER, describe_pairs(warmed), len(warmed),
        elapsed if warmed else 0.0, 'n/a' if before is None else before, after)
    return tuple(warmed)
