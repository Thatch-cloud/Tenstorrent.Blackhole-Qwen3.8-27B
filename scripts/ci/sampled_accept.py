"""Sampled verification, host side: keyed noise, the sampling filter, the draw and the coupled accept walk.

Pure host code (numpy only). Nothing here touches a device and nothing in serving imports it yet: it is the part of the sampled-verify design
(docs/tp4-sampled-verify.md) that can be settled on a CPU, and the contract the device step has to meet.

THE SCHEME. Every draft source of the fast path is deterministic (the drafter's top choice, prompt-lookup copies, the rows of a tree or octo branch), so
a draft never depends on the noise. At each verify row the target's logits give a filtered distribution (temperature, top_k, top_p over the row's top
K_MAX slice) and the row's token y is the inverse-CDF draw of that distribution at a uniform u keyed ONLY by (request seed, absolute position of the
token being sampled). The walk takes drafts while they equal y, emits the first y that differs (or the bonus y after the last draft) and stops.

  * Exact: the committed token at every position is y of the TRUE prefix at that position, so the output is a sample of the model's own distribution.
  * Reproducible: u is a pure function of (seed, position), so the committed text equals plain one-token-at-a-time sampling with the same seed, whatever
    the draft lengths, the round splits, the tree shape, the admission timing or the block a request shares. The audited "0 mismatches against the
    reference" gates stay meaningful: the reference calls the same functions (sample_logits) on the same logits.
  * A position's u is reused by a later round only when an earlier round saw a mismatch BEFORE that position, so the earlier round never observed it:
    nothing is conditioned on a u that is drawn again.

NAMING. In this code base the "seed" of a session is the first generated token (GreedySession.seed, the token the prefill sampled). Here `seed` is the
request's random seed (the PRNG key); the first token is at absolute position len(prompt), the row i of a ticket at `ticket.position` predicts the token
at position ticket.position + i + 1 (noise_positions).

THE ONE APPROXIMATION. The filter works on the top K_MAX candidates of the row (64: the model's own defaults are top_k 20, so they and every client
top_k up to 64 are exact). A request that sends top_k 0 (disabled) or above K_MAX is clamped to K_MAX; top_p is then taken over the renormalised slice,
which can only cut the nucleus EARLIER than the full vocabulary would (never adds a token outside the true nucleus). Exact whenever the true nucleus
fits in K_MAX, which at the model's default top_p 0.95 is the norm. See docs/tp4-sampled-verify.md section 5.

ORDER AND TIES. A row's candidates are ordered by (logit descending, token id ascending): the same first-occurrence rule as torch.argmax, the pinned
sampler and verify_trace_t1.combine_shards, so temperature 0 is exactly today's argmax. Arithmetic is float64 on the (bf16-valued) logits.
"""

from dataclasses import dataclass
import math

import numpy as np

# Candidates per row the filter sees. Justified in docs/tp4-sampled-verify.md section 5; the contract keeps a copy (serving_c2_contract.SAMPLED_K_MAX,
# held equal by a test) because the contract cannot import numpy at boot.
K_MAX = 64

# generation_config.json of the base snapshot (temperature, top_p, top_k): what a request that sends none of them gets. The contract keeps a copy
# (serving_c2_contract.SAMPLED_MODEL_DEFAULTS, held equal by a test).
MODEL_DEFAULTS = dict(temperature=1.0, top_p=0.95, top_k=20)

SEED_MASK = (1 << 64) - 1

_U = np.uint64
_SEED_SALT = _U(0x9E3779B97F4A7C15)
_POSITION_SALT = _U(0xD1B54A32D192ED03)
_MIX_1 = _U(0xBF58476D1CE4E5B9)
_MIX_2 = _U(0x94D049BB133111EB)


def _mix(x):
    """The SplitMix64 finaliser (Stafford variant 13) on a uint64 array: a bijection with full avalanche."""
    with np.errstate(over='ignore'):
        x = (x ^ (x >> _U(30))) * _MIX_1
        x = (x ^ (x >> _U(27))) * _MIX_2
        return x ^ (x >> _U(31))


def request_seed(value):
    """The 64-bit key of a request seed: any int (a negative one wraps), never a bool or a float."""
    if type(value) is not int:
        raise ValueError('A request seed must be an int, got %r' % (value,))
    return value & SEED_MASK


def noise_positions(ticket_position, rows):
    """The absolute positions of the tokens a ticket's rows sample: row i, which consumed the token at ticket_position + i, samples the token at
    ticket_position + i + 1. The first generated token (the prefill's) is at len(prompt), the position the first ticket starts from."""
    if type(ticket_position) is not int or ticket_position < 0 or type(rows) is not int or rows < 0:
        raise ValueError('A position and a row count of whole numbers are required')
    return [ticket_position + 1 + row for row in range(rows)]


def tree_positions(ticket_position, parents):
    """noise_positions for a tree: row j samples the token at ticket_position + depth(j) + 1, with parents[0] == -1 (the seed row) and parents[j] < j."""
    depths = []
    for row, parent in enumerate(parents):
        if row == 0 and parent == -1:
            depths.append(0)
        elif type(parent) is int and 0 <= parent < row:
            depths.append(depths[parent] + 1)
        else:
            raise ValueError('Row %d: a tree row needs a parent among the earlier rows (row 0 has parent -1), got %r' % (row, parent))
    return [ticket_position + 1 + depth for depth in depths]


def _as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, range)):
        return list(value)
    return [value]


def noise(seeds, positions):
    """Uniform float64 in [0, 1), one per (seed, position): a counter-based hash, a pure function of its two inputs. `seeds` are request seeds (Python
    ints of any size), `positions` whole numbers; a scalar or equal-length sequences (a scalar beside a sequence is repeated)."""
    seeds = np.array([request_seed(seed) for seed in _as_list(seeds)], dtype=np.uint64)
    raw = np.atleast_1d(np.asarray(_as_list(positions)))
    if raw.dtype.kind not in 'iu' or bool((raw < 0).any()):
        raise ValueError('Positions must be non-negative whole numbers')
    positions = raw.astype(np.uint64)
    if seeds.shape != positions.shape:
        if seeds.size == 1:
            seeds = np.broadcast_to(seeds, positions.shape)
        elif positions.size == 1:
            positions = np.broadcast_to(positions, seeds.shape)
        else:
            raise ValueError('Seeds and positions must have the same length')
    with np.errstate(over='ignore'):
        key = _mix(seeds + _SEED_SALT)
        counter = _mix(positions + _POSITION_SALT)
        bits = _mix(key ^ counter)
    return (bits >> _U(11)).astype(np.float64) * (1.0 / (1 << 53))


@dataclass(frozen=True)
class SamplingSpec:
    """A request's sampling terms as the verifier applies them. temperature 0 is greedy; top_k <= 0 means disabled (clamped to K_MAX)."""
    temperature: float = MODEL_DEFAULTS['temperature']
    top_p: float = MODEL_DEFAULTS['top_p']
    top_k: int = MODEL_DEFAULTS['top_k']

    def __post_init__(self):
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError('temperature must be a finite number >= 0, got %r' % (self.temperature,))
        if (isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float))
                or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError('top_p must be in (0, 1], got %r' % (self.top_p,))
        if type(self.top_k) is not int:
            raise ValueError('top_k must be an int, got %r' % (self.top_k,))

    @property
    def greedy(self):
        return self.temperature == 0

    @property
    def effective_top_k(self):
        """The candidates the filter keeps: top_k as sent when 1..K_MAX, K_MAX when it is disabled (<= 0) or larger: the one approximation."""
        return K_MAX if self.top_k <= 0 else min(self.top_k, K_MAX)

    @classmethod
    def from_request(cls, temperature=None, top_p=None, top_k=None):
        """Fields a request omitted (None) take the model's defaults."""
        return cls(MODEL_DEFAULTS['temperature'] if temperature is None else temperature,
                   MODEL_DEFAULTS['top_p'] if top_p is None else top_p,
                   MODEL_DEFAULTS['top_k'] if top_k is None else top_k)


GREEDY = SamplingSpec(0.0, 1.0, 0)


def order_candidates(values, ids):
    """Each row of (R, n) `values` and `ids` ordered by (value descending, id ascending). Non-finite values and a repeated id in a row are refused."""
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if values.ndim != 2 or values.shape != ids.shape or values.shape[1] < 1:
        raise ValueError('Candidates are (rows, n) values and ids of one shape, n >= 1')
    if not bool(np.isfinite(values).all()):
        raise ValueError('Non-finite logits among the candidates (a device fault, never a sampling input)')
    by_id = np.argsort(ids, axis=1, kind='stable')
    ids = np.take_along_axis(ids, by_id, axis=1)
    values = np.take_along_axis(values, by_id, axis=1)
    if ids.shape[1] > 1 and bool((ids[:, 1:] == ids[:, :-1]).any()):
        raise ValueError('A token id appears twice among one row\'s candidates')
    by_value = np.argsort(-values, axis=1, kind='stable')
    return np.take_along_axis(values, by_value, axis=1), np.take_along_axis(ids, by_value, axis=1)


def topk_slice(logits, k=K_MAX):
    """The top-k slice of each row of (R, V) full-vocabulary logits, (values, ids) ordered by (value descending, id ascending). Ties at the cut
    keep the lowest ids. The reference path's way to the slice the device returns."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or not 1 <= k <= logits.shape[1]:
        raise ValueError('(rows, V) logits and 1 <= k <= V required')
    if bool(np.isnan(logits).any()):
        raise ValueError('NaN logits (a device fault, never a sampling input)')
    values, ids = [], []
    for row in logits:
        floor = np.partition(row, row.shape[0] - k)[row.shape[0] - k]
        candidates = np.flatnonzero(row >= floor)
        order = np.lexsort((candidates, -row[candidates].astype(np.float64)))[:k]
        ids.append(candidates[order])
        values.append(row[candidates[order]])
    return np.stack(values), np.stack(ids).astype(np.int64)


def merge_shard_topk(chip_values, chip_ids, shard_width, k=K_MAX):
    """The global top-k slice from each chip's top slice of its own vocabulary shard: `chip_values[c]` and `chip_ids[c]` are (R, n_c) with ids local to
    the shard; the global id is local + c * shard_width. Exact: the global top-k is inside the union of the per-shard top-k."""
    if len(chip_values) != len(chip_ids) or not chip_values or type(shard_width) is not int or shard_width < 1:
        raise ValueError('One (values, ids) pair per chip and a shard width are required')
    values = np.concatenate([np.asarray(v, dtype=np.float64) for v in chip_values], axis=1)
    ids = np.concatenate([np.asarray(i, dtype=np.int64) + chip * shard_width for chip, i in enumerate(chip_ids)], axis=1)
    if bool(((np.concatenate([np.asarray(i) for i in chip_ids], axis=1)) < 0).any()):
        raise ValueError('Shard-local ids must be non-negative')
    values, ids = order_candidates(values, ids)
    return values[:, :k], ids[:, :k]


def _per_row(value, rows, dtype):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(rows, array, dtype=dtype)
    if array.shape != (rows,):
        raise ValueError('A per-row setting needs one value per row')
    return array


def draw_batch(values, ids, temperature, top_p, top_k, seeds, positions, active=None):
    """The sampled token of each row: (R,) int64, -1 for an inactive row.

    values, ids   (R, n) candidates of each row (any order; n up to a shard-merged slice). Only the first effective_top_k of the ordered row are used.
    temperature, top_p, top_k   one value or one per row (top_k as sent: <= 0 is disabled, clamped to K_MAX).
    seeds, positions   one request seed and one absolute position (the token's, see noise_positions) per row.
    active        optional boolean per row; an inactive row (an idle segment of a padded block) is neither checked nor drawn.
    Temperature 0 rows return the first candidate: today's argmax. Every row is computed on its own: a row's token does not depend on the other rows.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError('Candidates are (rows, n)')
    rows = values.shape[0]
    temperature = _per_row(temperature, rows, np.float64)
    top_p = _per_row(top_p, rows, np.float64)
    top_k = _per_row(top_k, rows, np.int64)
    seeds, positions = list(seeds), list(positions)
    if len(seeds) != rows or len(positions) != rows:
        raise ValueError('One seed and one position per row required')
    live = np.ones(rows, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if live.shape != (rows,):
        raise ValueError('One active flag per row required')
    result = np.full(rows, -1, dtype=np.int64)
    if not bool(live.any()):
        return result
    which = np.flatnonzero(live)
    temperature, top_p, top_k = temperature[which], top_p[which], top_k[which]
    if bool((temperature < 0).any()) or not bool(np.isfinite(temperature).all()) or bool(((top_p <= 0) | (top_p > 1)).any()):
        raise ValueError('Per-row temperature must be finite >= 0 and top_p in (0, 1]')
    ordered_values, ordered_ids = order_candidates(values[which], np.asarray(ids)[which])
    sampled = temperature > 0
    out = ordered_ids[:, 0].copy()                      # greedy rows: the first candidate, first occurrence on ties
    if bool(sampled.any()):
        rows_s = np.flatnonzero(sampled)
        v, i = ordered_values[rows_s], ordered_ids[rows_s]
        width = v.shape[1]
        keep_k = np.where(top_k[rows_s] <= 0, K_MAX, np.minimum(top_k[rows_s], K_MAX))
        keep_k = np.minimum(keep_k, width)
        column = np.arange(width)[None, :]
        inside = column < keep_k[:, None]
        with np.errstate(over='ignore', divide='ignore'):
            scaled = (v - v[:, :1]) / temperature[rows_s][:, None]      # <= 0 (or -inf at a tiny temperature), so exp cannot overflow
        weight = np.where(inside, np.exp(scaled), 0.0)
        probability = weight / weight.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probability, axis=1)
        before = np.concatenate((np.zeros((len(rows_s), 1)), cumulative[:, :-1]), axis=1)
        keep = inside & (before < top_p[rows_s][:, None])               # the smallest prefix reaching top_p; always the first
        kept = np.where(keep, probability, 0.0)
        kept = kept / kept.sum(axis=1, keepdims=True)
        cdf = np.cumsum(kept, axis=1)
        draw = noise([seeds[w] for w in which[rows_s]], [positions[w] for w in which[rows_s]])
        index = np.minimum((cdf <= draw[:, None]).sum(axis=1), keep.sum(axis=1) - 1)
        out[rows_s] = i[np.arange(len(rows_s)), index]
    result[which] = out
    return result


def draw(values, ids, spec, seed, position):
    """One row's sampled token: draw_batch on a single row, so a row alone and in a batch agree."""
    return int(draw_batch(np.asarray(values)[None, :], np.asarray(ids)[None, :], spec.temperature, spec.top_p, spec.top_k,
                          [seed], [position])[0])


def sample_logits(logits, spec, seed, position, k=K_MAX):
    """The reference path: the token plain one-token-at-a-time sampling takes from one row of full-vocabulary logits at `position` (the position of
    the token being sampled). A speculative run's committed token at that position is this, for the same logits."""
    values, ids = topk_slice(np.asarray(logits)[None, :], k)
    return draw(values[0], ids[0], spec, seed, position)


def walk_linear(drafts, ys):
    """The coupled accept walk for a linear draft: `drafts` are the k proposed tokens (rows 1..k), `ys[i]` (or ys(i), called lazily) the target's draw
    after row i, i = 0..k. Returns (accepted, emitted): the leading drafts equal to their y, then the first y that differs or, if all matched, the
    bonus y after the last draft. emitted == ys[:accepted + 1], and ys past the first mismatch are never read."""
    get = ys if callable(ys) else ys.__getitem__
    drafts, emitted = tuple(drafts), []
    for row, token in enumerate(drafts):
        y = get(row)
        emitted.append(y)
        if token != y:
            return row, tuple(emitted)
    emitted.append(get(len(drafts)))
    return len(drafts), tuple(emitted)


def walk_tree(tokens, parents, ys):
    """The coupled walk for a tree: tokens[0] is the seed row, parents[j] < j the parent row of row j (parents[0] == -1), `ys(row)` (or ys[row]) the
    draw from that row's logits. From the root: y = ys(row); descend to the child whose token equals y; with none, y is emitted and the walk stops.
    Returns (rows, emitted): the rows visited and the tokens emitted (len(emitted) == len(rows)); accepted drafts are len(rows) - 1."""
    tokens, parents = tuple(tokens), tuple(parents)
    if len(tokens) != len(parents) or not tokens or parents[0] != -1 or any(
            type(p) is not int or not 0 <= p < j for j, p in enumerate(parents) if j):
        raise ValueError('A tree is tokens and parents with parents[0] == -1 and parents[j] < j')
    get = ys if callable(ys) else ys.__getitem__
    children = {}
    for row in range(1, len(tokens)):
        children.setdefault(parents[row], []).append(row)
    row, rows, emitted = 0, [0], []
    while True:
        y = get(row)
        emitted.append(y)
        match = [child for child in children.get(row, ()) if tokens[child] == y]
        if not match:
            return tuple(rows), tuple(emitted)
        row = match[0]
        rows.append(row)
