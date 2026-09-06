"""Request-local exact token lookup proposals; never emit without target verification."""


class LookupDraft:
    def __init__(self, request_id, prompt, *, history_limit=32768, match_limit=128, max_proposals=15, prefer_full_suffix=False):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A request identity is required")
        if type(history_limit) is not int or history_limit < 2:
            raise ValueError("history_limit must be at least two tokens")
        if type(match_limit) is not int or match_limit < 1:
            raise ValueError("match_limit must be positive")
        if type(max_proposals) is not int or max_proposals not in (15, 31):
            raise ValueError('Explicit T16 or T32 proposal capacity required')
        if type(prefer_full_suffix) is not bool:
            raise ValueError('Known-suffix tie preference must be an explicit boolean')
        self.request_id = request_id
        self.history_limit = history_limit
        self.match_limit = match_limit
        self.max_proposals = max_proposals
        self.prefer_full_suffix = prefer_full_suffix
        self.closed = False
        self.history = []
        self.commit(request_id, prompt)

    def check_owner(self, request_id):
        if self.closed or request_id != self.request_id:
            raise ValueError("Closed or mismatched request identity")

    def commit(self, request_id, accepted_tokens):
        self.check_owner(request_id)
        tokens = list(accepted_tokens)
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Expected nonnegative integer token IDs")
        self.history = (self.history + tokens)[-self.history_limit:]

    def propose(self, request_id, count):
        return self.propose_with_match(request_id, count)[0]

    def propose_with_match(self, request_id, count):
        self.check_owner(request_id)
        if type(count) is not int or not 1 <= count <= self.max_proposals:
            raise ValueError('Proposal count exceeds the configured verifier capacity')
        history = self.history
        if len(history) < 2:
            return [], 0
        best_length, best_end = 0, None
        limit = min(self.match_limit, len(history) - 1)
        for end in range(len(history) - 2, -1, -1):
            if history[end] != history[-1]:
                continue
            length = 1
            while length < min(limit, end + 1) and history[end - length] == history[-1 - length]:
                length += 1
            longer_suffix = (self.prefer_full_suffix and length == best_length and best_end is not None
                             and min(count, len(history) - end - 1) > min(count, len(history) - best_end - 1))
            if length > best_length or longer_suffix:
                best_length, best_end = length, end
                if length == limit and (not self.prefer_full_suffix or len(history) - end - 1 >= count):
                    break
        return ([], 0) if best_end is None else (history[best_end + 1:best_end + 1 + count], best_length)

    def close(self, request_id):
        self.check_owner(request_id)
        self.history.clear()
        self.closed = True
