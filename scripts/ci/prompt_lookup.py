"""Prompt-lookup drafting for code: host-side proposals from a user's own token history (QWEN_FAST_LOOKUP_DRAFT).

Coding output copies its context: a function the user just pasted, an identifier defined a screen above, a tool
result quoted back. The lookup finds the most recent earlier occurrence of the last N tokens of the user's history
(prompt plus every committed token) and proposes the tokens that followed it. The target verifies every proposal, so
a wrong lookup costs nothing but the acceptance the DFlash2 proposal would have had: lossless greedy is unchanged.

THE POLICY STRING. 'n<N>m<M>', for example 'n3m12':
  N  the length of the key, 2..6 tokens: the last N tokens of the history must occur earlier;
  M  the confidence gate, N..64: the match is extended backwards from the key while the tokens before agree, up to
     64, and the lookup is used for a round only when that extended match length is at least M.
A round the gate rejects keeps the DFlash2 proposal as it is. Anything else (an unset flag, '', '0', 'off') is the
lookup off; a malformed policy is a ValueError, never a silent off.

WHAT A LOOKUP ROUND PROPOSES. The lookup's tokens fill the ticket's proposal rows from the first row on. A lookup
shorter than the ticket (the occurrence sits near the end of the history) is completed with DFlash2's own tokens at
the same rows, so the ticket keeps its width (the packed block, the capture and vLLM's scheduler all fix it): row i
carries the lookup's token while the lookup has one and DFlash2's token after. The later DFlash2 rows are
conditioned on DFlash2's chain, not on the lookup's, so they only matter when the lookup was right to its end.

No device state is touched here. The index is plain Python integers; it allocates nothing on a card and runs while
the verify replays or before the ticket is handed to vLLM, so there is no capture hazard to keep ahead of.
"""

import os

LOOKUP_FLAG = 'QWEN_FAST_LOOKUP_DRAFT'
TOKEN_BITS = 18                      # the vocabulary (248,320) fits in 18 bits: a key is an exact packed integer
MAX_BACK = 64                        # how far a match is extended backwards (and the largest gate)
MIN_N, MAX_N = 2, 6
OFF_VALUES = ('', '0', 'off')

ENGAGED = '[LOOKUP-DRAFT] engaged'
ROUND_MARKER = '[LOOKUP-ROUND]'


class LookupPolicy:
    """A parsed policy: key length `n` and gate `m`."""

    __slots__ = ('n', 'm')

    def __init__(self, n, m):
        if type(n) is not int or not MIN_N <= n <= MAX_N:
            raise ValueError('The lookup key length must be %d to %d tokens, got %r' % (MIN_N, MAX_N, n))
        if type(m) is not int or not n <= m <= MAX_BACK:
            raise ValueError('The lookup gate must be %d (the key length) to %d tokens, got %r' % (n, MAX_BACK, m))
        self.n, self.m = n, m

    def __repr__(self):
        return 'n%dm%d' % (self.n, self.m)

    def __eq__(self, other):
        return isinstance(other, LookupPolicy) and (self.n, self.m) == (other.n, other.m)

    def __hash__(self):
        return hash((self.n, self.m))


def parse_policy(text):
    """The policy named by `text`, or None when the lookup is off. Raises ValueError on anything else."""
    if text is None or text.strip().lower() in OFF_VALUES:
        return None
    text = text.strip().lower()
    if len(text) < 4 or text[0] != 'n' or 'm' not in text:
        raise ValueError('%s must be off or n<N>m<M> (for example n3m12), got %r' % (LOOKUP_FLAG, text))
    head, _, tail = text[1:].partition('m')
    if not head.isdigit() or not tail.isdigit():
        raise ValueError('%s must be off or n<N>m<M> (for example n3m12), got %r' % (LOOKUP_FLAG, text))
    return LookupPolicy(int(head), int(tail))


def policy_from_environment(environ=None):
    return parse_policy((os.environ if environ is None else environ).get(LOOKUP_FLAG))


class TokenLookup:
    """The incremental n-gram index over one user's history.

    `last` maps the packed key of every n-gram that ends before the final token to the end index of its most
    recent occurrence. The n-gram ending at the final token is the query, so it is inserted only when a later
    token arrives: a query never finds itself."""

    def __init__(self, prompt, n):
        if type(n) is not int or not MIN_N <= n <= MAX_N:
            raise ValueError('The lookup key length must be %d to %d tokens, got %r' % (MIN_N, MAX_N, n))
        self.n = n
        self.mask = (1 << (TOKEN_BITS * n)) - 1
        self.history = []
        self.last = {}
        self.rolling = 0
        self.key = None              # the packed key of the n-gram ending at the final token (None under n tokens)
        self.extend(prompt)

    def __len__(self):
        return len(self.history)

    def extend(self, tokens):
        history, last, n, mask = self.history, self.last, self.n, self.mask
        rolling, key = self.rolling, self.key
        for token in tokens:
            if type(token) is not int or not 0 <= token < (1 << TOKEN_BITS):
                raise ValueError('Expected a token id below %d, got %r' % (1 << TOKEN_BITS, token))
            if key is not None:
                last[key] = len(history) - 1
            history.append(token)
            rolling = ((rolling << TOKEN_BITS) | token) & mask
            key = rolling if len(history) >= n else None
        self.rolling, self.key = rolling, key

    def propose(self, count):
        """(tokens, match_length): up to `count` tokens that followed the most recent earlier occurrence of the
        last n tokens, and how many tokens (n to MAX_BACK) agree backwards from there. ((), 0) when the key has no
        earlier occurrence or the occurrence has nothing after it."""
        if type(count) is not int or count < 1:
            raise ValueError('A positive proposal count is required')
        if self.key is None:
            return (), 0
        end = self.last.get(self.key)
        if end is None:
            return (), 0
        history = self.history
        final = len(history) - 1
        length = self.n
        while length < MAX_BACK and end - length >= 0 and history[end - length] == history[final - length]:
            length += 1
        tokens = tuple(history[end + 1:end + 1 + count])
        return (tokens, length) if tokens else ((), 0)


def choose(policy, lookup_tokens, match, dflash_tokens):
    """(source, tokens) for one round: the ticket's proposal rows after the seed.

    `dflash_tokens` is what DFlash2 proposed (its length is the ticket's proposal width). The lookup is used when
    its match reaches the policy's gate; its tokens lead and DFlash2's fill the rest of the width."""
    width = len(dflash_tokens)
    if policy is None or not lookup_tokens or match < policy.m or width < 1:
        return 'dflash2', tuple(dflash_tokens)
    lead = tuple(lookup_tokens[:width])
    return 'lookup', lead + tuple(dflash_tokens[len(lead):])


class RequestLookup:
    """One request's lookup: its index, its policy, and the per-round log.

    `sync` brings the index up to the session's emitted tokens (the history is the prompt plus every committed
    token, so nothing hooks the commit paths); `apply` is called by FastRequest.prepare on the ticket the session
    proposed and returns the ticket the round verifies."""

    def __init__(self, policy, prompt, *, request_id, vocab_size, log=None):
        if not isinstance(policy, LookupPolicy):
            raise ValueError('A parsed lookup policy is required')
        self.policy, self.request_id, self.vocab_size = policy, request_id, vocab_size
        self.index = TokenLookup(prompt, policy.n)
        self.synced = 0              # how many of session.emitted are in the index
        self.open = None             # the round whose commit has not been logged yet
        self.rounds = self.lookup_rounds = 0
        self.log = print if log is None else log

    def sync(self, emitted):
        if len(emitted) < self.synced:
            raise ValueError('The session emitted fewer tokens than the lookup index holds')
        self.index.extend(emitted[self.synced:])
        self.synced = len(emitted)

    def apply(self, session, ticket):
        """The ticket the round will verify: `ticket` itself unless the policy picks the lookup."""
        emitted = session.emitted
        self.close_round(len(emitted))
        self.sync(emitted)
        width = len(ticket.tokens) - 1
        if width < 1:
            return ticket
        lookup_tokens, match = self.index.propose(width)
        if ticket.tokens[0] != self.index.history[-1] or any(token >= self.vocab_size for token in lookup_tokens):
            # The seed is the history's last token by construction; a ticket that disagrees (or a token outside the
            # target vocabulary) is never offered a lookup: the round keeps DFlash2's proposal.
            lookup_tokens, match = (), 0
        source, tokens = choose(self.policy, lookup_tokens, match, ticket.tokens[1:])
        proposed = min(len(lookup_tokens), width) if source == 'lookup' else width
        if source == 'lookup':
            ticket = replace_proposals(session, ticket, tokens, source, match)
        self.open = dict(position=ticket.position, source=source, match=match, proposed=proposed,
                         offered=len(lookup_tokens), emitted=len(emitted))
        return ticket

    def close_round(self, emitted_now):
        """Log the open round's committed count (the tokens the session emitted since it was proposed)."""
        round_ = self.open
        if round_ is None:
            return
        self.open = None
        committed = emitted_now - round_['emitted']
        if committed <= 0:
            return                   # discarded and re-proposed before it was verified: the next apply replaces it
        self.rounds += 1
        self.lookup_rounds += round_['source'] == 'lookup'
        self.log('%s request=%s position=%d source=%s match=%d offered=%d proposed=%d committed=%d' % (
            ROUND_MARKER, self.request_id, round_['position'], round_['source'], round_['match'], round_['offered'],
            round_['proposed'], committed), flush=True)

    def finish(self, emitted_now):
        self.close_round(emitted_now)


def replace_proposals(session, ticket, tokens, source, match):
    """The session's pending ticket with its proposal rows replaced: the same position and seed, a new epoch.

    The twin of GreedySession.narrow (which cuts a ticket in place), written here because the harness is pinned.
    Nothing was verified or published for the old ticket, so replacing it is host-only, as discarding a draft is;
    the drafter's own state advances from the target's features at publication, never from the proposal tokens."""
    from dataclasses import replace

    session.check_ticket(session.request_id, ticket)
    if len(tokens) != len(ticket.tokens) - 1:
        raise ValueError('A replacement must keep the ticket width')
    session.epoch += 1
    replaced = replace(ticket, epoch=session.epoch, tokens=(ticket.tokens[0],) + tuple(tokens), source=source,
                       match_length=int(match))
    session.pending, session.phase = replaced, 'pending'
    return replaced


def for_request(prompt, session, *, request_id, environ=None, log=None):
    """The RequestLookup for a new request, or None when QWEN_FAST_LOOKUP_DRAFT is off (the flag-off path
    builds nothing)."""
    policy = policy_from_environment(environ)
    if policy is None:
        return None
    lookup = RequestLookup(policy, prompt, request_id=request_id, vocab_size=session.vocab_size, log=log)
    lookup.sync(session.emitted)
    return lookup
