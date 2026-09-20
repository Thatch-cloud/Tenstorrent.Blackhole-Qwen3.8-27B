"""Unselected serving transaction driver for an already prepared greedy T16 request."""

from dataclasses import dataclass
import json
import time


@dataclass(frozen=True)
class CommittedOutput:
    request_id: str
    token_ids: tuple
    position: int
    finished: bool
    cancelled: bool = False


class FastRequest:
    def __init__(self, session, engine, runtime, *, release_drafter, collect_timings=False):
        if (session.phase != 'idle' or engine.phase != 'idle' or engine.session is not session
                or session.verifier_rows != 16 or not callable(release_drafter)
                or runtime.drafter_name not in ('dflash2', 'dspark') or type(collect_timings) is not bool):
            raise ValueError('Prepared request-owned greedy T16 engine and explicit cleanup required')
        runtime.bind(session, engine)
        self.session, self.engine, self.runtime = session, engine, runtime
        self.release_drafter = release_drafter
        self.closed = self.cancelled = False
        self.busy = False
        self.collect_timings = collect_timings
        self.timings = []
        self.prepared_timing = self.last_commit_time = None

    def prepare(self, request_id, packed_rows=None):
        """Prepare this request's ticket. `packed_rows` is the width of a round the packed
        block will serve (the worker hook asks the packed step before drafting); without it
        the engine proposes at its own captured width, exactly as before."""
        if (self.closed or self.busy or self.cancelled or request_id != self.session.request_id
                or self.session.phase != 'idle' or self.session.finished or self.engine.phase != 'idle'):
            raise ValueError('One unfinished idle owner required for draft preparation')
        self.busy = True
        try:
            started = time.perf_counter() if self.collect_timings else None
            rows = (self.engine.proposal_rows() if packed_rows is None
                    else self.engine.proposal_rows(packed_rows=packed_rows))
            if type(rows) is not int or rows not in (1, 2, 4, 8, 16):
                raise ValueError('Qualified T16 verifier bucket required')
            ticket = self.session.propose(request_id, max_rows=rows, selected=self.runtime.drafter_name)
            if self.collect_timings:
                self.prepared_timing = (started, time.perf_counter())
            return ticket
        finally:
            self.busy = False

    def step(self, request_id, *, cancelled):
        if (self.closed or self.busy or self.cancelled or request_id != self.session.request_id
                or self.session.phase not in ('idle', 'pending') or not callable(cancelled)):
            raise ValueError('One live owner and a cancellation callback required')
        if cancelled():
            if self.session.pending is not None:
                self.session.fail_verification(request_id, self.session.pending)
            self.cancelled = True
            return CommittedOutput(request_id, (), self.session.position, True, True)
        if self.session.finished:
            return CommittedOutput(request_id, (), self.session.position, True)
        ticket = self.session.pending or self.prepare(request_id)
        self.busy = True
        try:
            verify_started = time.perf_counter() if self.collect_timings else None
            predictions, verify_metrics = self.engine.verify(ticket)
            verified = time.perf_counter() if self.collect_timings else None
            if cancelled():
                self.session.abort(request_id, ticket, self.runtime.publish)
                self.cancelled = True
                return CommittedOutput(request_id, (), self.session.position, True, True)
            decision = self.session.commit(request_id, ticket, predictions, self.runtime.publish)
            if self.collect_timings:
                finished = time.perf_counter()
                prepared, drafted = self.prepared_timing
                cycle_start = prepared if self.last_commit_time is None else self.last_commit_time
                self.timings.append(dict(position=ticket.position, rows=len(ticket.tokens),
                    committed=len(decision.emitted), draft_ms=(drafted - prepared) * 1000,
                    verify_host_ms=(verified - verify_started) * 1000,
                    commit_host_ms=(finished - verified) * 1000,
                    cycle_ms=(finished - cycle_start) * 1000,
                    outside_phases_ms=((prepared - cycle_start) + (verify_started - drafted)) * 1000,
                    verifier=dict(verify_metrics)))
                self.last_commit_time = finished
                self.prepared_timing = None
            return CommittedOutput(request_id, tuple(decision.emitted), self.session.position, self.session.finished)
        except BaseException:
            if ticket is not None and self.session.phase == 'pending':
                if self.engine.phase == 'verified':
                    try:
                        self.session.abort(request_id, ticket, self.runtime.publish)
                    finally:
                        self.session.phase = 'failed'
                else:
                    self.session.fail_verification(request_id, ticket)
            raise
        finally:
            self.busy = False

    def close(self, request_id):
        if request_id != self.session.request_id or self.busy:
            raise ValueError('Only the idle owner can release request resources')
        if self.closed:
            return
        if self.session.phase == 'pending' and self.engine.phase == 'idle':
            self.session.fail_verification(request_id, self.session.pending)
        self.engine.close()
        self.release_drafter()
        self.session.close(request_id)
        self.closed = True
        if self.collect_timings:
            print(json.dumps(dict(stage='fast_serving_phases', blocks=self.timings,
                cancelled=self.cancelled, finished=self.session.finished,
                timing_scope='Host wall time, existing blocking boundaries; verifier detail is nested',
                added_device_fences=False, performance_qualified=False)), flush=True)
