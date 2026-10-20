"""Two-candidate (tree) verify: the accept/commit logic on the host. A PROTOTYPE: nothing in serving imports it, nothing here touches a device.

docs/tp4-tree-verify.md is the design; this file is the part of it that can be settled on a CPU today: given several candidate chains for one round of one request
(every chain starts with the same seed row and carries its own proposals), which chain does the round commit, what does it emit, which segment's state is committed,
and does the winner's K/V and recurrent state have to be moved back to the request's own segment.

THE MODEL (design D1, "sibling segments"). Each candidate is an ordinary linear chain verified in a segment of its own, as a user is: the same position, the same
seed, the same context, its own working state and its own K/V cells. The target's predictions of a chain's rows depend only on that chain's own tokens, so every chain
is judged exactly as a one-candidate round would judge it (greedy_verify.select_prefix). What the tree adds is the CHOICE of one chain after the verify:

* every chain's emitted tokens are a prefix of the one greedy continuation of the seed (a chain's accepted rows equal the target's predictions, and its correction is
  the target's prediction at the first row that differs), so the emitted tokens of any two chains agree on their common length: whichever chain wins, the output is
  the target's own greedy output. The choice can only change HOW MANY tokens a round commits, never which;
* the winner is the chain that emits the most tokens, ties to the lowest index (index 0 is the primary, the chain the scheduler was handed and the request's own
  segment): with one candidate the round is today's, call for call; a branch segment (index > 0) wins only by STRICTLY more tokens, so `copy_back` is exactly the
  rounds where the winner's state and K/V live in a sibling segment;
* the session then commits as it always does, on the winner's ticket and the winner's predictions (GreedySession.commit): the position advances by the winner's
  state rows, the winner's last emitted token is the next seed, the drafter publishes the winner's features at the winner's prefix.

Python 3.7 syntax; imports the harness (speculative-decoding/harness) like the serving code does.
"""

from dataclasses import dataclass, replace

from greedy_verify import select_prefix

MAX_CANDIDATES = 4        # a 64-row block has four 16-row segments: a branch needs a segment of its own, so at most as many chains as the block has segments


@dataclass(frozen=True)
class Candidate:
    index: int            # 0: the primary (the chain the scheduler was handed, verified in the request's own segment); 1..: a branch, in a sibling segment
    source: str
    tokens: tuple         # (seed, *proposals)


@dataclass(frozen=True)
class TreeDecision:
    branch: int           # the winning candidate's index
    decision: object      # greedy_verify.Decision of the winner
    emitted_by_branch: tuple     # how many tokens each candidate would have emitted alone (the winner's is the maximum)
    copy_back: bool       # branch != 0: the winner's recurrent state and K/V rows are in a sibling segment and move to the request's own


@dataclass(frozen=True)
class TreeCommit:
    decision: object
    branch: int
    copy_back: bool
    emitted_by_branch: tuple


def build_candidates(primary, alternatives=(), *, max_candidates=MAX_CANDIDATES):
    """The round's candidates: `primary` (a BlockTicket, or any object with .tokens and .source) as index 0, then each (source, proposals) of `alternatives` as a chain
    on the primary's seed. An alternative equal to an earlier chain is dropped (it would only spend a segment); one that shares a first row is kept (the tree still
    wins on the rows after the first difference). More than `max_candidates` chains is a ValueError, not a silent cut. Every chain is at most as wide as the primary."""
    tokens = tuple(primary.tokens)
    if not tokens:
        raise ValueError('The primary ticket has no seed row')
    chains = [Candidate(0, primary.source, tokens)]
    for source, proposals in alternatives:
        chain = (tokens[0],) + tuple(proposals)
        if len(chain) > len(tokens):
            raise ValueError('A candidate may not be wider than the primary ticket (%d rows against %d)' % (len(chain), len(tokens)))
        if len(chain) < 2 or any(chain == other.tokens for other in chains):
            continue
        chains.append(Candidate(len(chains), source, chain))
    if len(chains) > max_candidates:
        raise ValueError('%d candidates do not fit %d segments' % (len(chains), max_candidates))
    return tuple(chains)


def row_cap(rows, remaining, max_rows):
    """GreedySession.commit's cut of a chain of `rows` tokens: the commit cap `max_rows` and, for a chain wider than the budget left, the budget."""
    if rows > remaining:
        max_rows = remaining if max_rows is None else min(max_rows, remaining)
    return max_rows


def decide(candidate, predictions, *, vocab_size, eos_ids, remaining, max_proposals, max_rows=None):
    """greedy_verify.Decision of one candidate alone, cut exactly as GreedySession.commit cuts a ticket of its width."""
    predictions = tuple(predictions)
    if len(predictions) < len(candidate.tokens):
        raise ValueError('Candidate %d has %d rows and %d predictions' % (candidate.index, len(candidate.tokens), len(predictions)))
    proposals, targets = candidate.tokens[1:], predictions[:len(candidate.tokens)]
    cap = row_cap(len(candidate.tokens), remaining, max_rows)
    if cap is not None:
        proposals, targets = candidate.tokens[1:cap], targets[:cap]
    return select_prefix(proposals, targets, vocab_size=vocab_size, eos_ids=eos_ids, max_proposals=max_proposals)


def select_branch(candidates, predictions, *, vocab_size, eos_ids=(), remaining=1 << 30, max_proposals=15, max_rows=None):
    """TreeDecision over `candidates` with `predictions[i]` the target's argmax rows of candidate i. `remaining` is the budget left (GreedySession.commit's cut)."""
    candidates, predictions = tuple(candidates), tuple(predictions)
    if not candidates or len(candidates) != len(predictions):
        raise ValueError('One prediction row set per candidate is required')
    if len(set(candidate.tokens[0] for candidate in candidates)) != 1:
        raise ValueError('Every candidate must start from the same seed')
    if [candidate.index for candidate in candidates] != list(range(len(candidates))):
        raise ValueError('Candidate indexes must be 0..n-1 in order')
    decisions = [decide(candidate, rows, vocab_size=vocab_size, eos_ids=eos_ids, remaining=remaining, max_proposals=max_proposals, max_rows=max_rows)
                 for candidate, rows in zip(candidates, predictions)]
    emitted = tuple(len(decision.emitted) for decision in decisions)
    best = max(emitted)
    branch = emitted.index(best)                  # the first maximum: ties go to the lowest index, the primary
    return TreeDecision(branch, decisions[branch], emitted, branch != 0)


def adopt(session, request_id, candidate):
    """Make `candidate` the session's pending ticket (a new epoch, the same position and seed), as prompt_lookup.replace_proposals does for a replaced chain.
    Nothing was verified or published for the old ticket in the session's view: the branch's verify ran in a sibling segment."""
    pending = session.pending
    session.check_ticket(request_id, pending)
    if candidate.tokens[0] != pending.tokens[0]:
        raise ValueError('A branch must start from the pending ticket\'s seed')
    session.epoch += 1
    adopted = replace(pending, epoch=session.epoch, tokens=tuple(candidate.tokens), source=candidate.source, match_length=0)
    session.pending, session.phase = adopted, 'pending'
    return adopted


def commit_tree(session, request_id, candidates, predictions, publish_for, *, max_rows=None):
    """Select the winner, commit it through the session, -> TreeCommit.

    `publish_for(branch)` returns the publication callback of the winner's segment (GreedySession.commit's `publish`: it receives the state rows and commits the
    segment's recurrent state at that prefix, and publishes the drafter's features); for a branch the caller also owes the copy-back (`copy_back` says so). The
    session's pending ticket must be the primary's chain."""
    candidates = tuple(candidates)
    if not callable(publish_for):
        raise ValueError('publish_for(branch) -> publication callback is required')
    session.check_ticket(request_id, session.pending)
    if candidates and tuple(session.pending.tokens) != candidates[0].tokens:
        raise ValueError('The session\'s pending ticket is not the primary candidate')
    remaining = session.max_new_tokens - len(session.emitted)
    selection = select_branch(candidates, predictions, vocab_size=session.vocab_size, eos_ids=session.eos_ids, remaining=remaining,
                              max_proposals=session.verifier_rows - 1, max_rows=max_rows)
    ticket = session.pending if selection.branch == 0 else adopt(session, request_id, candidates[selection.branch])
    decision = session.commit(request_id, ticket, tuple(predictions[selection.branch]), publish_for(selection.branch), max_rows=max_rows)
    if decision != selection.decision:
        raise RuntimeError('The session decided another prefix than the tree selected')
    return TreeCommit(decision, selection.branch, selection.copy_back, selection.emitted_by_branch)


def plan_branches(segments, live, wants, *, max_candidates=MAX_CANDIDATES):
    """{live segment: [free segments that carry its extra candidates]} for one block.

    `segments` is the block's segment count (4 for an M3 block, 8 for the octo block), `live` the segments that hold a request's primary chain, `wants` {live segment:
    extra candidates wanted} in priority order. A branch needs a FREE segment (one no live request holds), so a full block plans nothing; free segments go round-robin
    in `wants` order (every wanting request gets its first extra before any gets a second), at most `max_candidates - 1` extras each. Free segments no branch takes stay
    idle, as today."""
    live = tuple(live)
    if len(set(live)) != len(live) or any(not 0 <= segment < segments for segment in live):
        raise ValueError('Live segments must be distinct segments of the block')
    free = [segment for segment in range(segments) if segment not in live]
    plan = dict((segment, []) for segment in live if wants.get(segment, 0) > 0)
    order = [segment for segment in wants if segment in plan]
    progressed = True
    while free and progressed:
        progressed = False
        for segment in order:
            if free and len(plan[segment]) < min(wants[segment], max_candidates - 1):
                plan[segment].append(free.pop(0))
                progressed = True
    return dict((segment, branches) for segment, branches in plan.items() if branches)
