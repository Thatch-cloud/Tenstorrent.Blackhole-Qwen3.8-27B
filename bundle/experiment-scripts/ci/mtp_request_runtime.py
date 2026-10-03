"""Real MTP proposal/verification bridge; caller supplies prepared device execution."""


class MTPRequestRuntime:
    def __init__(self, step, initial_hidden, *, copy_hidden, verified_row=None, max_drafts=31, reuse_accepted_cache=False,
                 propose_chain=None):
        if (type(max_drafts) is not int or max_drafts not in (1, 3, 7, 15, 31)
                or not all(callable(callback) for callback in (step, copy_hidden))
                or (verified_row is not None and not callable(verified_row))
                or (propose_chain is not None and not callable(propose_chain))):
            raise ValueError('Prepared MTP step, hidden copy/row access and bounded draft count required')
        self.step, self.copy_hidden, self.verified_row = step, copy_hidden, verified_row
        self.anchor = initial_hidden
        self.max_drafts = max_drafts
        self.propose_chain = propose_chain
        if type(reuse_accepted_cache) is not bool:
            raise ValueError('Explicit draft-cache reuse selection required')
        self.reuse_accepted_cache = reuse_accepted_cache
        self.cache_accounting = dict(reused_rows=0, teacher_forced_rows=0)
        self.engine = self.session = None
        self.phase = 'unbound'
        self.proposed = ()
        self.drafted_inputs = ()

    def bind(self, session, engine):
        if self.phase != 'unbound' or not engine.retain_mtp_hidden or session.phase != 'idle':
            raise ValueError('Fresh runtime and prepared hidden-retaining verifier required')
        self.session, self.engine = session, engine
        if self.verified_row is None:
            self.verified_row = engine.mtp_row
        self.position = session.position
        self.phase = 'idle'

    def __call__(self, request_id, history, count):
        if (self.phase != 'idle' or self.session.request_id != request_id
                or self.session.phase != 'drafting' or self.position != self.session.position
                or not history or history[-1] != self.session.seed
                or type(count) is not int or count < 1):
            raise ValueError('MTP proposal must belong to the current committed request frontier')
        self.phase = 'drafting'
        try:
            hidden, token = self.anchor, self.session.seed
            proposed = []
            selected_count = min(count, self.max_drafts)
            if self.propose_chain is not None:
                proposed = list(self.propose_chain(token, hidden, self.position, selected_count))
                if len(proposed) != selected_count:
                    raise ValueError('MTP chain must account for every requested draft input')
            else:
                for offset in range(selected_count):
                    hidden, token = self.step(token, hidden, self.position + offset, select=True)
                    if type(token) is not int or not 0 <= token < self.session.vocab_size:
                        raise ValueError('MTP head returned an invalid global token ID')
                    proposed.append(token)
            if any(type(token) is not int or not 0 <= token < self.session.vocab_size for token in proposed):
                raise ValueError('MTP head returned an invalid global token ID')
            self.proposed = tuple(proposed)
            self.drafted_inputs = (self.session.seed, *self.proposed[:-1])
            self.phase = 'proposed'
            return self.proposed
        except BaseException:
            self.phase = 'failed'
            raise

    def publish(self, prefix):
        ticket = self.session.pending
        if (self.phase not in ('idle', 'proposed') or ticket is None
                or self.session.phase != 'committing' or self.engine.phase != 'verified'
                or ticket.position != self.position or type(prefix) is not int
                or not 0 <= prefix <= len(ticket.tokens)):
            raise ValueError('MTP publication requires the live verified target transaction')
        self.phase = 'committing'
        try:
            hidden_rows = self.engine.verified_mtp_hidden_for_publication(ticket)
            reused = min(prefix, len(self.drafted_inputs)) if self.reuse_accepted_cache else 0
            if tuple(ticket.tokens[:reused]) != self.drafted_inputs[:reused]:
                raise ValueError('Only the actual verified prefix of this MTP proposal can reuse draft KV')
            if prefix:
                previous = self.verified_row(hidden_rows, reused - 1) if reused else self.anchor
                for offset in range(reused, prefix):
                    self.step(ticket.tokens[offset], previous, ticket.position + offset, select=False)
                    previous = self.verified_row(hidden_rows, offset)
                self.copy_hidden(previous, self.anchor)
            self.engine.publish(prefix)
            self.cache_accounting['reused_rows'] += reused
            self.cache_accounting['teacher_forced_rows'] += prefix - reused
            self.position += prefix
            self.proposed = ()
            self.drafted_inputs = ()
            self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            self.engine.phase = 'failed'
            raise
