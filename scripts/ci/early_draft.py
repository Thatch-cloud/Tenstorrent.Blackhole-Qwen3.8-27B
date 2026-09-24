"""Round-fence plan H2: the drafts inside the step, the GDN commits after the pairs. Two flags, both
default off:

  QWEN_FAST_EARLY_DRAFT       the worker hook drafts the next round inside execute_model, right after
                              the packed step has committed and applied this round's outputs, and
                              take_draft_token_ids hands vLLM the cached drafts once it has confirmed
                              nothing it drafted from has changed (else it discards them and drafts
                              again, today's way)
  QWEN_FAST_GDN_AFTER_PAIRS   (needs QWEN_FAST_EARLY_DRAFT and the block's QWEN_FAST_ROUND_FENCES) the
                              packed block's GDN commit traces are decided as today but enqueued only
                              after the next round's pairs: right after the drafts' fence F9 and the
                              pairs' readback, before the selection - or, when a single's readback is
                              still to come (a padded round) or the round has no batched selection, at
                              the end of the draft. Always inside the same execute_model (R1).

WHAT IT BUYS (round-fence plan section 3, H2, no events). Today the round's device chain after the
verify is T_proj + slides (5.5 ms), the four GDN commit traces (5.2 ms), then the pairs (31.2 ms) and
the host tail (collect, select, propose, vLLM, the hook: ~8 ms) with the device idle. The pairs do not
read what the GDN commits write (the target model's carries and native slot 0), so they need not wait
for them: under H2 the chain is T_proj + slides, pairs, F9, the pairs' readback (2.9 ms, device idle),
then the GDN commits under the rest of the host tail. The next replay pays the owed fence
(gdn_records.RetainedGDNBlock.fence_at_replay), by then mostly or entirely drained: ~-5 ms a round.

WHY THE DRAFT MOVES INTO THE STEP (R1). All device work of round r is enqueued inside
execute_model(r): the GDN commits of round r may not wait for take_draft_token_ids, a separate call
(and, with Lever N, a prefill chunk's step can run between two decode steps). So the drafts - the same
FastWorkerHook._drafts body, eligibility and all, one function - run at the end of the step that
decided them, and the deferred commits are flushed in that step's `finally` whatever happens.

THE DRAFTS ARE TODAY'S. The early draft is FastWorkerHook._drafts itself, called at the end of
execute_model instead of by vLLM a moment later. Between the two nothing the drafts read can move: the
runner's request states and the sessions were updated by the step before the draft (apply_committed_
output, session.commit), a detach only happens at the next execute_model's entry and an attach only in
a prefill step's sample_tokens. take_draft_token_ids still confirms it (draft_key: every bridge, its
failure and closure, its session's phase, pending ticket, position, emitted count, finished flag and
the runner's remaining budget) and on any difference discards the early tickets (the pair traces write
only pair-owned buffers, so a discard costs device time and nothing else) and drafts again. The pairs
read the drafter's banks and taps, never the target's GDN state, so moving the GDN commits after them
changes no draft: the [PACKED] request/position/prefix/emitted/predictions lines of a run must equal a
same-image control arm's, admission order fixed (acceptance_report.compare_packed).

A draft that raises is kept and re-raised by take_draft_token_ids, where today's would have raised; the
step's own outputs still reach vLLM. A cache vLLM never took is dropped at the next execute_model's
entry (reconcile), with a line.
"""

import os
import sys
import time

EARLY_DRAFT_FLAG = 'QWEN_FAST_EARLY_DRAFT'
GDN_AFTER_PAIRS_FLAG = 'QWEN_FAST_GDN_AFTER_PAIRS'
FLAGS = (EARLY_DRAFT_FLAG, GDN_AFTER_PAIRS_FLAG)

# Once, the first time a hook drafts early; one line per early draft taken, redrafted, failed or
# never taken; from the block: once at attach (engaged or refused, and why) and one line per round
# whose deferred commits were enqueued (site=window|end, or reconcile|verify - R1 broken: a problem).
ENGAGED_MARKER = '[PINDIAG] early draft engaged'
MARKER = '[PACKED-EARLY-DRAFT]'
GDN_ENGAGED_MARKER = '[PINDIAG] gdn after pairs engaged'
GDN_REFUSED_MARKER = '[PINDIAG] gdn after pairs refused'
GDN_MARKER = '[PACKED-GDN-AFTER-PAIRS]'
# The flush sites that keep R1 (inside the execute_model that decided the commits).
IN_STEP_SITES = ('window', 'end')

MISSING = object()


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_EARLY_DRAFT=1. Any value but 0 or 1 is a configuration error."""
    return _flag(EARLY_DRAFT_FLAG, environ)


def gdn_after_pairs_enabled(environ=None):
    """QWEN_FAST_GDN_AFTER_PAIRS=1 under QWEN_FAST_EARLY_DRAFT=1 (alone it defers nothing: nobody would
    flush inside the step, so the block never defers; the arm and the gate refuse it)."""
    return _flag(GDN_AFTER_PAIRS_FLAG, environ) and enabled(environ)


def requested(environ=None):
    """Whether either flag is set to anything but 0 - the cheap test before this module is imported."""
    environ = os.environ if environ is None else environ
    return any(environ.get(name, '0') != '0' for name in FLAGS)


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


def draft_key(hook):
    """Everything FastWorkerHook._drafts decides from, per bridge in the hook's order: the id, the
    bridge, its failure, the request's closure and cancellation, the session's phase, pending ticket
    (by value: request, epoch, position, tokens), position, emitted count and finished flag, and the
    runner's remaining budget (serving_worker_hook.real_remaining_budget) - with the packed step."""
    from serving_worker_hook import real_remaining_budget

    entries = []
    for request_id, bridge in hook.bridges.items():
        request = bridge.request
        session = request.session
        entries.append((str(request_id), id(bridge), bool(getattr(bridge, 'failed', False)),
                        bool(getattr(request, 'closed', False)), bool(getattr(request, 'cancelled', False)),
                        getattr(session, 'phase', None), getattr(session, 'pending', None),
                        getattr(session, 'position', None), len(getattr(session, 'emitted', None) or ()),
                        bool(getattr(session, 'finished', False)), real_remaining_budget(bridge)))
    return (id(hook.packed_step), tuple(entries))


KEY_FIELDS = ('request', 'bridge', 'failed', 'closed', 'cancelled', 'phase', 'pending', 'position', 'emitted',
              'finished', 'budget')


def key_difference(before, after):
    """Why two draft keys differ, short: 'packed-step', 'bridges' or 'request=<id>:<field>'."""
    if before[0] != after[0]:
        return 'packed-step'
    if [entry[:2] for entry in before[1]] != [entry[:2] for entry in after[1]]:
        return 'bridges'
    for old, new in zip(before[1], after[1]):
        for name, mine, theirs in zip(KEY_FIELDS, old, new):
            if mine != theirs:
                return 'request=%s:%s' % (old[0][:24], name)
    return 'unknown'


def discard_ticket(request):
    """A pending ticket this module drafted and vLLM will not be handed: serving_worker_hook.
    discard_stale_ticket's reset without its width test - the drafter's cached proposal dropped, the
    session back to idle. Host only (the pair traces wrote pair-owned buffers only)."""
    session = request.session
    if session.pending is None or session.phase != 'pending':
        return False
    discard = getattr(request.runtime, 'discard_proposal', None)
    if callable(discard):
        discard()
    session.pending, session.phase = None, 'idle'
    return True


class EarlyDraft:
    """One worker hook's early-draft state (QWEN_FAST_EARLY_DRAFT): the round's cached drafts and the
    key they were drafted from, a draft failure to re-raise at take_draft_token_ids, and the counts."""

    def __init__(self):
        self.cache = MISSING
        self.key = None
        self.failure = None
        self.drafting = False
        self.armed = False
        self.rounds = 0
        self.draft_ms = 0.0
        self.noted = False
        self.counts = dict(reuse=0, redo=0, failed=0, untaken=0)

    # -- the step ---------------------------------------------------------------------------------
    def execute(self, hook, step):
        """The packed step (`step`, execute_packed_decode) and then this round's drafts, inside one
        execute_model. Under QWEN_FAST_GDN_AFTER_PAIRS the block defers its GDN commit traces for the
        round and they are enqueued after the pairs' readback, or at the latest before this returns. The
        hook reconciles (below) at every execute_model entry, this one's included, before calling this."""
        packed_step = hook.packed_step
        self.armed = False
        if gdn_after_pairs_enabled():
            arm = getattr(packed_step, 'arm_deferred_commits', None)
            self.armed = bool(callable(arm) and arm())
        if not self.noted:
            self.noted = True
            log_line('%s gdn_after_pairs=%d' % (ENGAGED_MARKER, int(self.armed)))
        try:
            output = step()
        except BaseException:
            self.flush(hook, 'end', quiet=True)
            raise
        try:
            if not hook.closed:
                self.draft(hook)
        finally:
            self.flush(hook, 'end')
        return output

    def coordinator_options(self, hook):
        """What the drafts' coordinator.prepare gets besides today's options: the flush of the deferred
        GDN commits after the pairs' readback (after_reads), when the round deferred any."""
        if not self.armed:
            return {}
        return dict(after_reads=lambda: self.flush(hook, 'window'))

    def flush(self, hook, site, quiet=False):
        """Enqueue the round's deferred GDN commit traces (packed_step.flush_deferred_commits); a no-op
        once they are. `quiet` (a step already failing): a flush failure is swallowed, never masking
        the one propagating."""
        flush = getattr(hook.packed_step, 'flush_deferred_commits', None)
        if not callable(flush):
            return 0
        try:
            return flush(site)
        except BaseException:
            for bridge in list(getattr(hook, 'bridges', {}).values()):
                bridge.failed = True
            if quiet:
                return 0
            raise
        finally:
            if site != 'window':
                self.armed = False

    def draft(self, hook):
        """FastWorkerHook._drafts, now: its result cached with the key it was drafted from. A failure is
        kept for take_draft_token_ids to raise, where today's draft would have raised."""
        from serving_worker_hook import phase

        self.cache, self.key, self.failure = MISSING, None, None
        self.rounds += 1
        started = time.perf_counter()
        self.drafting = True
        try:
            # Under QWEN_FAST_PHASE_LOG the same begin/end lines as the hook's other phases, so the round
            # timeline sees the drafts inside the step.
            result = phase('early_draft', 'round=%d' % self.rounds, lambda: hook._drafts(hook.worker))
        except Exception as failure:
            self.failure = failure
            self.draft_ms = (time.perf_counter() - started) * 1000
            return
        finally:
            self.drafting = False
        self.draft_ms = (time.perf_counter() - started) * 1000
        self.cache = result
        self.key = draft_key(hook)

    # -- take_draft_token_ids -----------------------------------------------------------------------
    def take(self, hook):
        """The cached drafts when nothing they were drafted from has changed; MISSING - and the early
        tickets discarded - when something has (the caller drafts again, today's way) or when there is
        no cache (a step that was not the packed decode). A kept failure is raised here."""
        if self.failure is not None:
            failure, self.failure = self.failure, None
            self.counts['failed'] += 1
            self.note('failed', 0, '%s' % type(failure).__name__)
            raise failure
        if self.cache is MISSING:
            return MISSING
        cached, key = self.cache, self.key
        self.cache, self.key = MISSING, None
        live = len(getattr(cached, 'req_ids', None) or ()) if cached is not None else 0
        now = draft_key(hook)
        if now == key:
            self.counts['reuse'] += 1
            self.note('reuse', live, '-')
            return cached
        reason = key_difference(key, now)
        early = {id(entry[6]) for entry in key[1] if entry[6] is not None}
        for bridge in hook.bridges.values():
            if id(bridge.request.session.pending) in early:
                discard_ticket(bridge.request)
        self.counts['redo'] += 1
        self.note('redo', live, reason)
        return MISSING

    def reconcile(self, hook):
        """At every execute_model entry: a cache vLLM never took is dropped (its tickets stay pending,
        as a ticket drafted by take_draft_token_ids stays across a prefill step), a kept failure too,
        and - R1's backstop - any deferred GDN commit still held is enqueued now (site=reconcile: the
        gate fails the arm on it)."""
        if self.cache is not MISSING or self.failure is not None:
            self.counts['untaken'] += 1
            self.note('untaken', 0, 'failure' if self.failure is not None else '-')
            self.cache, self.key, self.failure = MISSING, None, None
        self.flush(hook, 'reconcile')

    def note(self, path, live, reason):
        log_line('%s round=%d path=%s live=%d draft_ms=%.2f reason=%s' % (
            MARKER, self.rounds, path, live, self.draft_ms, str(reason).replace(' ', '_')[:96]))

    def describe(self):
        return dict(rounds=self.rounds, armed=self.armed, counts=dict(self.counts))
