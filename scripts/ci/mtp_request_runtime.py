"""Real MTP proposal/verification bridge; caller supplies prepared device execution."""


class MTPRequestRuntime:
    def __init__(self, step, initial_hidden, *, copy_hidden, verified_row=None, max_drafts=31):
        if (type(max_drafts) is not int or max_drafts not in (1, 3, 7, 15, 31)
                or not all(callable(callback) for callback in (step, copy_hidden))
                or (verified_row is not None and not callable(verified_row))):
            raise ValueError('Prepared MTP step, hidden copy/row access and bounded draft count required')
        self.step, self.copy_hidden, self.verified_row = step, copy_hidden, verified_row
        self.anchor = initial_hidden
        self.max_drafts = max_drafts
        self.engine = self.session = None
        self.phase = 'unbound'
        self.proposed = ()

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
            for offset in range(min(count, self.max_drafts)):
                hidden, token = self.step(token, hidden, self.position + offset, select=True)
                if type(token) is not int or not 0 <= token < self.session.vocab_size:
                    raise ValueError('MTP head returned an invalid global token ID')
                proposed.append(token)
            self.proposed = tuple(proposed)
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
            if prefix:
                previous = self.anchor
                for offset in range(prefix):
                    self.step(ticket.tokens[offset], previous, ticket.position + offset, select=False)
                    previous = self.verified_row(hidden_rows, offset)
                self.copy_hidden(previous, self.anchor)
            self.engine.publish(prefix)
            self.position += prefix
            self.proposed = ()
            self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            self.engine.phase = 'failed'
            raise
