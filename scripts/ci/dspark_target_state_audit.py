"""Instrumented-only target-state boundary checks around learned draft execution."""


class TargetStateChanged(AssertionError):
    def __init__(self, phase, before, after):
        self.evidence = dict(kind='target_state_changed_by_drafter', phase=phase,
            before=before, after=after)
        super().__init__(f'Target state changed during {phase}')


class TargetStateAuditedDrafter:
    def __init__(self, drafter, snapshot):
        self.drafter, self.snapshot = drafter, snapshot
        self.initial_state = None
        self.initial_position = None

    def __getattr__(self, name):
        return getattr(self.drafter, name)

    def checked(self, phase, operation):
        before = self.snapshot()
        result = operation()
        after = self.snapshot()
        if before != after:
            raise TargetStateChanged(phase, before, after)
        return result

    def prepare_trace(self, anchor, *, audit=False):
        result = self.checked('proposal_capture', lambda: self.drafter.prepare_trace(anchor, audit=audit))
        self.initial_state = self.snapshot()
        self.initial_position = self.drafter.position
        return result

    def propose(self, anchor, count):
        if self.initial_state is not None and self.drafter.position == self.initial_position:
            current = self.snapshot()
            if current != self.initial_state:
                raise TargetStateChanged('before_proposal_at_initial_frontier', self.initial_state, current)
        return self.checked('proposal_replay', lambda: self.drafter.propose(anchor, count))
