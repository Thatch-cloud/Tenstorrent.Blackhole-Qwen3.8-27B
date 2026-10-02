"""The teacher-forced walk of the A0 screen: one drafter over one logged turn (a0 design A.2). Python 3.7, stdlib only.

S2 is lossless greedy, so the text a turn produced does not depend on the drafter, and a drafter's state depends only on the
target's features, the anchor and its own weights. Two drafters can therefore walk the SAME logged text with the SAME features
and be compared round for round. This module is that walk, with the drafter behind a two-method interface so the harness, the
statistics and the tests never need a model:

    drafter.propose(sequence, start, count)   the `count` tokens the drafter proposes after the anchor at absolute position
                                              `start` of `sequence` (prompt + answer). It may read the features of rows below
                                              `start` and the token AT `start`, and nothing later: the walk hands it a view that ends at
                                              the anchor, so a look-ahead raises.

Round arithmetic (the served rule):
  anchor index j in the answer (j = 0 is the prefill seed); absolute start s = prompt_len + j; offset = j + 1 (the lab's
  completion-token offset); rows = proposals + 1 (16 for T16, 8 for T8); proposals are compared with answer[j+1 ...];
  accepted a = the matching prefix; left = answer_len - 1 - j; cap = min(extent_attention_replay.accept_limit(s, rows), left)
  committed = min(a + 1, cap); uncapped = min(a + 1, left) (the counterfactual without the extent cap).
  The round that reaches the end of the answer (the terminal round) and the prefill seed are not counted (the lab's rule).

Records carry counts only: no token, no text.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extent_attention_replay import accept_limit  # noqa: E402

MAX_PROPOSALS = 15


def round_geometry(prompt_len, index, answer_len, proposals):
    """(absolute start, offset, rows, left, cap) of the round whose anchor is answer[index]."""
    start = prompt_len + index
    rows = proposals + 1
    left = answer_len - 1 - index
    cap = min(accept_limit(start, rows), left)
    return start, index + 1, rows, left, cap


def accepted_prefix(proposed, answer, index):
    """How many of `proposed` match answer[index + 1 ...] in a row (the answer's end stops the match)."""
    count = 0
    for offset, token in enumerate(proposed):
        at = index + 1 + offset
        if at >= len(answer) or answer[at] != token:
            break
        count += 1
    return count


class VisibleSequence(object):
    """The sequence up to and including the anchor, as a view: an index past the anchor raises, so a drafter cannot read the
    answer it is meant to predict (no copy: a round at 120k tokens would otherwise copy 120k ids)."""
    __slots__ = ('_data', '_limit')

    def __init__(self, data, limit):
        self._data, self._limit = data, limit

    def __len__(self):
        return self._limit

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._limit)
            return [self._data[at] for at in range(start, stop, step)]
        if key < 0:
            key += self._limit
        if not 0 <= key < self._limit:
            raise IndexError('the walk shows a drafter rows up to the anchor only')
        return self._data[key]


def _check_turn(turn):
    prompt, answer = turn['prompt_ids'], turn['output_ids']
    if not prompt or not answer:
        raise ValueError('a turn needs a prompt and an answer')
    return prompt, answer


def _region(turn, offset):
    think = turn.get('think_tokens') or 0
    tool_at = turn.get('tool_at')
    if tool_at is not None and offset >= tool_at:
        return 'tool'
    return 'reasoning' if offset < think else 'content'


def _record(turn, index, prompt_len, answer_len, proposals, accepted, rows, cap, left, start, offset):
    committed = min(accepted + 1, cap) if cap > 0 else 0
    return dict(offset=offset, start=start, rows=rows, cap=cap, accepted=accepted,
                available=min(proposals, left), committed=committed, uncapped=min(accepted + 1, left),
                region=_region(turn, offset), terminal=(index + committed >= answer_len - 1))


def walk_free(turn, drafter, proposals, max_rounds=None):
    """The drafter running its own schedule over the logged answer. Returns the counted rounds (seed and terminal round left
    out), in order, as dicts of counts."""
    if not 1 <= proposals <= MAX_PROPOSALS:
        raise ValueError('proposals must be 1..%d' % MAX_PROPOSALS)
    prompt, answer = _check_turn(turn)
    sequence = list(prompt) + list(answer)
    out, index = [], 0
    while index < len(answer) - 1:
        start, offset, rows, left, cap = round_geometry(len(prompt), index, len(answer), proposals)
        proposed = list(drafter.propose(VisibleSequence(sequence, start + 1), start, proposals))
        if len(proposed) != proposals:
            raise ValueError('a drafter must return exactly the requested proposals')
        accepted = accepted_prefix(proposed, answer, index)
        entry = _record(turn, index, len(prompt), len(answer), proposals, accepted, rows, cap, left, start, offset)
        if entry['committed'] < 1:
            raise ValueError('a round must commit at least its anchor successor')
        if not entry['terminal']:
            out.append(entry)
        index += entry['committed']
        if max_rounds is not None and len(out) >= max_rounds:
            break
    return out


def walk_forced(turn, drafter, proposals, logged_emitted):
    """The drafter at the anchors the SERVED schedule visited: `logged_emitted` is the served emitted count of each logged
    round of the turn in order (the terminal round last). Each entry also carries `logged` and `exact` (the drafter's committed
    equals the served emitted), the calibration of V2. A schedule that does not reach the answer's end exactly is refused."""
    prompt, answer = _check_turn(turn)
    sequence = list(prompt) + list(answer)
    out, index = [], 0
    for position, emitted in enumerate(logged_emitted):
        if index >= len(answer) - 1:
            raise ValueError('the logged schedule runs past the answer')
        start, offset, rows, left, cap = round_geometry(len(prompt), index, len(answer), proposals)
        proposed = list(drafter.propose(VisibleSequence(sequence, start + 1), start, proposals))
        accepted = accepted_prefix(proposed, answer, index)
        entry = _record(turn, index, len(prompt), len(answer), proposals, accepted, rows, cap, left, start, offset)
        entry['logged'] = emitted
        entry['exact'] = entry['committed'] == emitted
        if not entry['terminal'] and position != len(logged_emitted) - 1:
            out.append(entry)
        index += emitted
    if index != len(answer) - 1:
        raise ValueError('the logged schedule does not cover the answer')
    return out


# -- tallies -----------------------------------------------------------------------------------------------------------

def tally(rounds, key='committed'):
    """(sum of `key`, number of rounds)."""
    return (sum(entry[key] for entry in rounds), len(rounds))


def tau(rounds, key='committed'):
    total, count = tally(rounds, key)
    return total / float(count) if count else None


def position_acceptance(rounds, proposals):
    """[(accepted at j, available at j)] for j = 1..proposals: position j is available when the answer has a token there and
    accepted when the whole prefix up to j matched."""
    out = []
    for position in range(1, proposals + 1):
        available = sum(1 for entry in rounds if entry['available'] >= position)
        accepted = sum(1 for entry in rounds if entry['available'] >= position and entry['accepted'] >= position)
        out.append((accepted, available))
    return out


def schedule_agreement(rounds):
    """Forced walk against the served counts: (exact rounds, rounds, drafter committed, served emitted)."""
    exact = sum(1 for entry in rounds if entry['exact'])
    return (exact, len(rounds), sum(entry['committed'] for entry in rounds), sum(entry['logged'] for entry in rounds))


# -- the compact round row (what the run driver writes and the report reads) ----------------------------------------------

ROW_COLUMNS = ('offset', 'start', 'rows', 'cap', 'accepted', 'available', 'committed', 'uncapped')


def to_row(entry):
    """A round as a list of ints (ROW_COLUMNS): a 35k-round arm stays a few MB."""
    return [entry[name] for name in ROW_COLUMNS]


def from_row(row, turn=None):
    """The round dict back from a row; the region comes from the turn's think / tool offsets."""
    if len(row) != len(ROW_COLUMNS):
        raise ValueError('a round row has %d columns' % len(ROW_COLUMNS))
    entry = dict(zip(ROW_COLUMNS, row))
    entry['region'] = _region(turn or {}, entry['offset'])
    return entry
