"""Unselected serving transaction driver for an already prepared greedy T16 request."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommittedOutput:
    request_id: str
    token_ids: tuple
    position: int
    finished: bool
    cancelled: bool = False


class FastRequest:
    def __init__(self, session, engine, runtime, *, release_drafter):
        if (session.phase != 'idle' or engine.phase != 'idle' or engine.session is not session
                or session.verifier_rows != 16 or not callable(release_drafter)
                or runtime.drafter_name not in ('dflash2', 'dspark')):
            raise ValueError('Prepared request-owned greedy T16 engine and explicit cleanup required')
        runtime.bind(session, engine)
        self.session, self.engine, self.runtime = session, engine, runtime
        self.release_drafter = release_drafter
        self.closed = self.cancelled = False
        self.busy = False

    def prepare(self, request_id):
        if (self.closed or self.busy or self.cancelled or request_id != self.session.request_id
                or self.session.phase != 'idle' or self.session.finished or self.engine.phase != 'idle'):
            raise ValueError('One unfinished idle owner required for draft preparation')
        self.busy = True
        try:
            rows = self.engine.proposal_rows()
            if type(rows) is not int or rows not in (1, 2, 4, 8, 16):
                raise ValueError('Qualified T16 verifier bucket required')
            return self.session.propose(request_id, max_rows=rows, selected=self.runtime.drafter_name)
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
            predictions, _ = self.engine.verify(ticket)
            if cancelled():
                self.session.abort(request_id, ticket, self.runtime.publish)
                self.cancelled = True
                return CommittedOutput(request_id, (), self.session.position, True, True)
            decision = self.session.commit(request_id, ticket, predictions, self.runtime.publish)
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
