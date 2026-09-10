"""Instrumented-only target-state boundary checks around learned draft execution."""


class TargetStateChanged(AssertionError):
    def __init__(self, phase, before, after):
        self.evidence = dict(kind='target_state_changed_by_drafter', phase=phase,
            before=before, after=after)
        super().__init__(f'Target state changed during {phase}')


class TargetStateAuditedDrafter:
    def __init__(self, drafter, snapshot):
        self.drafter, self.snapshot = drafter, snapshot

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
        return self.checked('proposal_capture', lambda: self.drafter.prepare_trace(anchor, audit=audit))

    def propose(self, anchor, count):
        return self.checked('proposal_replay', lambda: self.drafter.propose(anchor, count))
