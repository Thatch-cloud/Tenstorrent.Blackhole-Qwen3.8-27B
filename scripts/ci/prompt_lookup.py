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

THE ADMISSION COST. The index is built synchronously when the request is created, after its prefill and on the step
thread, so it is NOT hidden behind the prefill. To bound the stall it indexes at most the last MAX_INDEXED_PROMPT
prompt tokens (32768; about 0.07 s), plus every committed token.
"""

import os

LOOKUP_FLAG = 'QWEN_FAST_LOOKUP_DRAFT'
TOKEN_BITS = 18                      # the vocabulary (248,320) fits in 18 bits: a key is an exact packed integer
MAX_BACK = 64                        # how far a match is extended backwards (and the largest gate)
MIN_N, MAX_N = 2, 6
# The index is built on the engine step thread right after the prefill (FastRequest is created from the prefilled
# session), so its cost stalls every decoding seat. Only the most recent MAX_INDEXED_PROMPT prompt tokens are
# indexed: about 0.07 s and 4 MB at the cap, however long the prompt (0.27 s for 123k tokens, 0.6 s for 254k
# uncapped). The code a reply copies is overwhelmingly in the recent context, and the committed tokens are
# always indexed whole.
MAX_INDEXED_PROMPT = 32768
OFF_VALUES = ('', '0', 'off')

ENGAGED = '[LOOKUP-DRAFT] engaged'
ROUND_MARKER = '[LOOKUP-ROUND]'
# QWEN_FAST_LOOKUP_SHADOW=1 (tp4/stream-1, default off, needs the lookup): one SHADOW line per round with both candidates the round could have verified (DFlash2's
# rows and the lookup's) and the tokens the round committed, so tree_gain_report.py can say offline what a two-candidate verify (docs/tp4-tree-verify.md) would
# have committed on the same text. Token ids of a user's own text: the line stays in the private container log, only the report's aggregates leave the rig.
SHADOW_FLAG = 'QWEN_FAST_LOOKUP_SHADOW'
SHADOW_MARKER = '[LOOKUP-SHADOW]'


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


def shadow_from_environment(environ=None):
    """Whether QWEN_FAST_LOOKUP_SHADOW=1. Unset, '' and '0' are off; anything but '1' beyond those is a ValueError, never a silent off."""
    value = (os.environ if environ is None else environ).get(SHADOW_FLAG, '')
    if value in ('', '0'):
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (SHADOW_FLAG, value))


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


def lookup_chain(lookup_tokens, dflash_tokens):
    """The ticket's proposal rows of a lookup round: the lookup's tokens lead and DFlash2's fill the rest of the width (cut to the width)."""
    width = len(dflash_tokens)
    lead = tuple(lookup_tokens[:width])
    return lead + tuple(dflash_tokens[len(lead):])


def choose(policy, lookup_tokens, match, dflash_tokens):
    """(source, tokens) for one round: the ticket's proposal rows after the seed.

    `dflash_tokens` is what DFlash2 proposed (its length is the ticket's proposal width). The lookup is used when
    its match reaches the policy's gate; its tokens lead and DFlash2's fill the rest of the width."""
    width = len(dflash_tokens)
    if policy is None or not lookup_tokens or match < policy.m or width < 1:
        return 'dflash2', tuple(dflash_tokens)
    return 'lookup', lookup_chain(lookup_tokens, dflash_tokens)


class RequestLookup:
    """One request's lookup: its index, its policy, and the per-round log.

    `sync` brings the index up to the session's emitted tokens (the history is the prompt plus every committed
    token, so nothing hooks the commit paths); `apply` is called by FastRequest.prepare on the ticket the session
    proposed and returns the ticket the round verifies."""

    def __init__(self, policy, prompt, *, request_id, vocab_size, log=None, shadow=False):
        if not isinstance(policy, LookupPolicy):
            raise ValueError('A parsed lookup policy is required')
        self.policy, self.request_id, self.vocab_size = policy, request_id, vocab_size
        self.shadow = shadow is True      # the SHADOW line per round (QWEN_FAST_LOOKUP_SHADOW); False is every caller as it always was
        self.emitted_ref = None           # the session's emitted list, kept only under the shadow (its tokens are what a round committed)
        self.index = TokenLookup(tuple(prompt)[-MAX_INDEXED_PROMPT:], policy.n)
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
        dflash_rows = tuple(ticket.tokens[1:]) if self.shadow else None      # before the replacement: DFlash2's own rows
        if source == 'lookup':
            ticket = replace_proposals(session, ticket, tokens, source, match)
        self.open = dict(position=ticket.position, source=source, match=match, proposed=proposed,
                         offered=len(lookup_tokens), emitted=len(emitted))
        if self.shadow:
            self.emitted_ref = emitted
            self.open.update(dflash=dflash_rows, lookup=tuple(lookup_tokens))
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
        if self.shadow:
            self.log(shadow_line(self.request_id, round_, tuple(self.emitted_ref[round_['emitted']:emitted_now])), flush=True)

    def finish(self, emitted_now):
        self.close_round(emitted_now)


def _csv(tokens):
    return ','.join(str(token) for token in tokens) or '-'


def shadow_line(request_id, round_, committed):
    """The SHADOW line of one round: both candidates' rows (DFlash2's `dflash`, the lookup's raw `lookup`, '-' when empty) and the committed tokens."""
    return '%s request=%s position=%d source=%s match=%d dflash=%s lookup=%s committed=%s' % (
        SHADOW_MARKER, request_id, round_['position'], round_['source'], round_['match'], _csv(round_['dflash']),
        _csv(round_['lookup']), _csv(committed))


def parse_shadow_line(line):
    """The dict a SHADOW line carries (request, position, source, match, dflash, lookup, committed as tuples of ints), or None when `line` is not one."""
    index = line.find(SHADOW_MARKER)
    if index < 0:
        return None
    fields = {}
    for word in line[index + len(SHADOW_MARKER):].split():
        name, separator, value = word.partition('=')
        if separator:
            fields[name] = value
    try:
        def tokens(name):
            return () if fields[name] == '-' else tuple(int(token) for token in fields[name].split(','))
        return dict(request=fields['request'], position=int(fields['position']), source=fields['source'], match=int(fields['match']),
                    dflash=tokens('dflash'), lookup=tokens('lookup'), committed=tokens('committed'))
    except (KeyError, ValueError):
        return None


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
    lookup = RequestLookup(policy, prompt, request_id=request_id, vocab_size=session.vocab_size, log=log,
                           shadow=shadow_from_environment(environ))
    lookup.sync(session.emitted)
    return lookup
