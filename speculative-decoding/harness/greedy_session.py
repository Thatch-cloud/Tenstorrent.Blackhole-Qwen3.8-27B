"""Request-scoped host coordinator; device execution and synchronization remain external."""

from dataclasses import dataclass

from greedy_verify import select_prefix
from hybrid_draft import HybridDraft


@dataclass(frozen=True)
class BlockTicket:
    request_id: str
    epoch: int
    position: int
    tokens: tuple
    source: str
    match_length: int = 0


class GreedySession:
    def __init__(self, request_id, prompt, seed, *, vocab_size, max_new_tokens, eos_ids=(), neural=None, verifier_rows=16,
                 prefer_full_suffix=False):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('Positive generation budget required')
        if type(verifier_rows) is not int or verifier_rows not in (16, 32):
            raise ValueError('Explicit T16 or T32 verifier capacity required')
        prompt, eos_ids = tuple(prompt), tuple(eos_ids)
        select_prefix((), (seed,), vocab_size=vocab_size, eos_ids=eos_ids)
        self.drafter = HybridDraft(request_id, prompt, vocab_size=vocab_size, neural=neural,
                                  max_proposals=verifier_rows - 1, prefer_full_suffix=prefer_full_suffix)
        self.drafter.commit_verified(request_id, (seed,))
        self.request_id, self.vocab_size = request_id, vocab_size
        self.max_new_tokens, self.eos_ids = max_new_tokens, eos_ids
        self.verifier_rows = verifier_rows
        self.position, self.seed = len(prompt), seed
        self.emitted = [seed]
        self.finished = seed in eos_ids or max_new_tokens == 1
        self.phase, self.epoch, self.pending = 'idle', 0, None
        self.committed_block_proposals = self.accepted_proposals = self.committed_blocks = 0
        self.committed_decode_tokens = self.aborted_blocks = 0

    def check_owner(self, request_id):
        if request_id != self.request_id or self.phase in ('failed', 'closed'):
            raise ValueError('Closed, failed or mismatched request')

    def begin_preparation(self, request_id):
        self.check_owner(request_id)
        if self.phase != 'idle' or self.pending is not None or self.finished:
            raise ValueError('An unfinished idle request is required before device preparation')
        self.phase = 'preparing'

    def finish_preparation(self, request_id):
        self.check_owner(request_id)
        if self.phase != 'preparing':
            raise ValueError('Request preparation is not active')
        self.phase = 'idle'

    def fail_preparation(self, request_id):
        self.check_owner(request_id)
        if self.phase != 'preparing':
            raise ValueError('Request preparation is not active')
        self.phase = 'failed'

    def propose(self, request_id, *, max_rows=16, selected=None):
        self.check_owner(request_id)
        if self.phase != 'idle' or self.finished:
            raise ValueError('An unfinished idle request is required')
        if type(max_rows) is not int or max_rows not in (1, 2, 4, 8, 16, 32) or max_rows > self.verifier_rows:
            raise ValueError('Supported verifier bucket required')
        remaining = self.max_new_tokens - len(self.emitted)
        limit = min(max_rows, remaining)
        self.phase = 'drafting'
        try:
            proposal = self.drafter.propose(request_id, limit - 1, greedy=True, verifier_ready=True,
                                            selected=selected) if limit > 1 else None
        except BaseException:
            self.phase = 'failed'
            raise
        tokens = () if proposal is None else proposal.tokens
        rows = max(width for width in (1, 2, 4, 8, 16, 32) if width <= min(limit, len(tokens) + 1))
        self.epoch += 1
        ticket = BlockTicket(request_id, self.epoch, self.position, (self.seed, *tokens[:rows - 1]),
                             proposal.source if rows > 1 else 'target', proposal.match_length if rows > 1 else 0)
        self.pending, self.phase = ticket, 'pending'
        return ticket

    def check_ticket(self, request_id, ticket):
        self.check_owner(request_id)
        if self.phase != 'pending' or ticket is not self.pending:
            raise ValueError('The current live block ticket is required')

    def commit(self, request_id, ticket, predictions, publish):
        self.check_ticket(request_id, ticket)
        if not callable(publish):
            raise ValueError('Synchronized state publication callback required')
        decision = select_prefix(ticket.tokens[1:], predictions, vocab_size=self.vocab_size, eos_ids=self.eos_ids,
                                 max_proposals=self.verifier_rows - 1)
        self.phase = 'committing'
        try:
            if publish(decision.state_rows) is not None:
                raise RuntimeError('Publication must return None after successful synchronization')
            self.drafter.commit_verified(request_id, decision.emitted)
        except BaseException:
            self.phase = 'failed'
            raise
        self.position += decision.state_rows
        self.seed = decision.next_input
        self.emitted.extend(decision.emitted)
        self.committed_block_proposals += len(ticket.tokens) - 1
        self.accepted_proposals += decision.accepted
        self.committed_blocks += 1
        self.committed_decode_tokens += len(decision.emitted)
        self.finished = decision.finished or len(self.emitted) == self.max_new_tokens
        self.pending, self.phase = None, 'idle'
        return decision

    def abort(self, request_id, ticket, restore):
        self.check_ticket(request_id, ticket)
        if not callable(restore):
            raise ValueError('Synchronized entry-state restore callback required')
        self.phase = 'committing'
        try:
            if restore(0) is not None:
                raise RuntimeError('Abort must return None after successful synchronization')
        except BaseException:
            self.phase = 'failed'
            raise
        self.aborted_blocks += 1
        self.pending, self.phase = None, 'idle'

    def fail_verification(self, request_id, ticket):
        self.check_ticket(request_id, ticket)
        self.phase = 'failed'

    def close(self, request_id):
        if request_id != self.request_id or self.phase not in ('idle', 'failed'):
            raise ValueError('Finish or abort the block before closing its request')
        self.drafter.close(request_id)
        self.emitted.clear()
        self.seed, self.pending, self.phase = None, None, 'closed'
