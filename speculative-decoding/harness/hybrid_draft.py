"""Host-only lookup-first routing; proposals are not verified or emitted here."""

from dataclasses import dataclass

from lookup_draft import LookupDraft


@dataclass(frozen=True)
class Proposal:
    source: str
    tokens: tuple
    match_length: int = 0


class HybridDraft:
    """Adapters receive a bounded suffix, not full context or target feature tensors."""

    def __init__(self, request_id, prompt, *, vocab_size, neural=None, **lookup_options):
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.neural = dict(neural or {})
        if any(not isinstance(name, str) or not name or name in ("lookup", "target")
               or not callable(adapter) for name, adapter in self.neural.items()):
            raise ValueError("Expected named neural proposal callables")
        self.lookup = LookupDraft(request_id, self.validate(prompt), **lookup_options)

    def validate(self, tokens):
        tokens = tuple(tokens)
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in tokens):
            raise ValueError("Token outside target vocabulary")
        return tokens

    def propose(self, request_id, count, *, greedy=False, verifier_ready=False, selected=None):
        self.lookup.check_owner(request_id)
        if type(count) is not int or not 1 <= count <= self.lookup.max_proposals:
            raise ValueError('Proposal count exceeds the configured verifier capacity')
        if greedy is not True or verifier_ready is not True:
            return Proposal("target", ())
        tokens, match_length = self.lookup.propose_with_match(request_id, count)
        if tokens:
            return Proposal("lookup", tuple(tokens), match_length)
        adapter = self.neural.get(selected)
        if adapter is None:
            return Proposal("target", ())
        tokens = self.validate(adapter(request_id, tuple(self.lookup.history), count))
        if len(tokens) > count:
            raise ValueError("Neural proposal exceeds requested count")
        return Proposal(selected, tokens) if tokens else Proposal("target", ())

    def commit_verified(self, request_id, tokens):
        """Caller must finish target verification and all device-state rollback first."""
        self.lookup.check_owner(request_id)
        self.lookup.commit(request_id, self.validate(tokens))

    def close(self, request_id):
        self.lookup.close(request_id)
        self.neural.clear()
