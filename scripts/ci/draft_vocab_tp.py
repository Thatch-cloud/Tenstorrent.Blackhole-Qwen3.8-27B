"""QWEN_FAST_DRAFT_VOCAB (gate only, default off): the DFlash2 drafter proposes from a coding SHORTLIST of the vocabulary (docs/tp4-draft-vocab.md).

Today the drafter's head is the TARGET's LM head run on the draft rows: a (rows x 5,120) by (5,120 x 62,080) matmul per chip, two 32,768-wide top-16 launches
per chip, and 2 chunks x 4 chips x (values, indices) read back per head. With QWEN_FAST_DRAFT_VOCAB=<name or path> the head runs over a sliced copy of that
weight holding only the shortlist's rows (a sorted list of global token ids, 40,960 of 248,320 for `coding-40960`), re-sharded evenly over the four chips
(10,240 columns each, so no chip does more than another), and the top-16 runs over that: ONE chunk per chip. The local index a chip returns is mapped back to
the GLOBAL token id on the host before anything else sees it, so the candidate selector, its codebooks (indexed by global id), the FP64 selection, the
ties (lowest global id) and everything downstream are today's code on today's kind of operands.

EXACTNESS. The target verifies every proposal over the full 248,320 rows with its own weights; this module never touches the target's head, its sampler or its
verify. Committed text is the target's own greedy decode whatever the shortlist holds. The cost is tau only: a token outside the list can never be proposed,
so a round that needed one is rejected there (and a candidate list of 16 inside the shortlist replaces one that held an outside token). When the full top-16
of a row lies inside the shortlist, the shortlist run returns the same 16 tokens with the same scores in the same order (test_draft_vocab_tp).

The sliced weight is built ONCE, at the attach, before any trace is captured (serving_runtime, after the shared draft weights: a buffer allocated after a
trace was captured can sit in that trace's freed holes and be overwritten by its replay): each chip's shard of the target head is read back, the shortlist's
columns are picked out on the host and uploaded in the target head's own dtype, sharded on the vocabulary axis. In a block-float dtype the regrouped columns
share exponents differently from the target's, so a shortlist logit can differ from the target head's in the last mantissa bits: proposals only.

Flags (strict; read at the attach and at each capture, never at import):
  QWEN_FAST_DRAFT_VOCAB       unset, '' or '0': off, nothing below runs and nothing is imported. Otherwise a NAME (NAMED below: the list ships beside this module)
                              or an absolute PATH to a list written by draft_vocab_build.py (uint32 little endian, ascending). Anything else raises.
  QWEN_FAST_DRAFT_VOCAB_TOPK  the width the shortlist's logits are padded to (with -inf) for the top-16 launch: 32768 (default, the width every drafter top-16
                              has run at) or 16384 (half the work, a shape no card has run: its own arm). Needs the flag above.
Four cards only (QWEN_FAST_TP=4): at the pair the pinned draft_shared_head is never edited and the flag raises.

Markers (c2_smoke_check.draft_vocab_problems reads them): `[PINDIAG] draft vocab admitted ...` once at the attach, `[PINDIAG] draft vocab head built ...` once
(rows, per chip, dtype, bytes per chip), `[PINDIAG] draft vocab engaged ...` once at the first head launch. There is no fallback: a failure raises.

Stdlib at import (torch only inside the functions that need it), py 3.7 syntax: the contract imports it at boot.
"""

import hashlib
import os
import struct
import sys
import time

import tp_shapes

FLAG = 'QWEN_FAST_DRAFT_VOCAB'
TOPK_FLAG = 'QWEN_FAST_DRAFT_VOCAB_TOPK'
NAMES = (FLAG, TOPK_FLAG)
PREFIX = 'QWEN_FAST_DRAFT_VOCAB'
TOPK_WIDTHS = (16384, 32768)
TOPK_DEFAULT = 32768
TILE = tp_shapes.TILE
CHIPS = 4
TOP_CANDIDATES = 16
# name -> (file stem beside this module, rows, sha256 of the .ids file). The sidecar .json holds the statistics; test_draft_vocab_tp pins all three.
NAMED = {
    'coding-40960': ('draft_vocab_coding_40960', 40960, '94fbd2a832f655628cfae83b1016cb1148ed32d0610f4c6670c2c0eece1e40fa'),
}
# Files the shortlist needs at run time, by basename; both image copy lists and the CPU allowlist name every entry (test_draft_vocab_tp).
RUNTIME_FILES = ('draft_vocab_tp.py', 'draft_vocab_coding_40960.ids', 'draft_vocab_coding_40960.json')

ADMITTED = '[PINDIAG] draft vocab admitted'
BUILT = '[PINDIAG] draft vocab head built'
ENGAGED = '[PINDIAG] draft vocab engaged'
MARKERS = (ADMITTED, BUILT, ENGAGED)
UNQUALIFIED = 'GATE ONLY, UNQUALIFIED'

HERE = os.path.dirname(os.path.abspath(__file__))


def requested(environ=None):
    """Whether QWEN_FAST_DRAFT_VOCAB asks for a shortlist (anything but unset, '' and '0'). Cheap; nothing imported."""
    return (os.environ if environ is None else environ).get(FLAG, '') not in ('', '0')


def log_line(message):
    """One line into the server log (loguru where it exists, stderr otherwise). Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, file=sys.stderr, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


def emit(log, message):
    """`message` through the caller's brace-style logger (serving_runtime.pindiag: template, *values) or, with none, into the server log."""
    if log is None:
        log_line(message)
    else:
        log('{}', message)


class Plan(object):
    """The shortlist as the four-card head sees it: global ids ascending, split evenly (`per_chip` each) in that order over `chips`."""

    def __init__(self, ids, chips, topk_width, sha256, source):
        self.ids, self.chips, self.topk_width, self.sha256, self.source = tuple(ids), chips, topk_width, sha256, source
        self.rows = len(self.ids)
        self.per_chip = self.rows // chips
        self._tensor = None

    def tensor(self):
        """The ids as an int64 torch tensor (built once)."""
        if self._tensor is None:
            import torch

            self._tensor = torch.tensor(self.ids, dtype=torch.int64)
        return self._tensor

    def describe(self):
        return 'rows=%d per_chip=%d chips=%d topk_width=%d sha256=%s' % (self.rows, self.per_chip, self.chips, self.topk_width, self.sha256[:12])


def parse_ids(data, rows_limit=tp_shapes.VOCAB):
    """The uint32 little-endian list in `data`, validated: non-empty, strictly ascending, below the vocabulary."""
    if not data or len(data) % 4:
        raise ValueError('a draft vocabulary is a non-empty whole number of uint32 words')
    ids = struct.unpack('<%dI' % (len(data) // 4), data)
    if any(a >= b for a, b in zip(ids, ids[1:])) or ids[-1] >= rows_limit:
        raise ValueError('a draft vocabulary is strictly ascending global token ids below %d' % rows_limit)
    return ids


def resolve(spec):
    """(path, expected sha256 or None, expected rows or None) for a NAME or an absolute PATH."""
    if spec in NAMED:
        stem, rows, sha256 = NAMED[spec]
        return os.path.join(HERE, stem + '.ids'), sha256, rows
    if os.path.isabs(spec) and not spec.endswith('/'):
        return spec, None, None
    raise ValueError('%s=%r is neither a shortlist name (%s) nor an absolute path to an id list' % (FLAG, spec, ', '.join(sorted(NAMED))))


_PLANS = {}


def load(spec, chips=CHIPS, topk_width=TOPK_DEFAULT):
    """The Plan for a NAME or PATH at `chips` chips (cached). Raises ValueError for anything a head cannot serve: the wrong width, a list that is not a whole
    32-row tile per chip, a sha256 that is not the named list's, an id outside the vocabulary or unordered."""
    key = (spec, chips, topk_width)
    if key in _PLANS:
        return _PLANS[key]
    if chips != CHIPS:
        raise ValueError('%s is a four-card lever: this process serves %d chips' % (FLAG, chips))
    if topk_width not in TOPK_WIDTHS:
        raise ValueError('%s must be one of %s, got %r' % (TOPK_FLAG, ', '.join(str(width) for width in TOPK_WIDTHS), topk_width))
    path, sha256, rows = resolve(spec)
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
    except OSError as failure:
        raise ValueError('%s=%s: the id list cannot be read (%s)' % (FLAG, spec if spec in NAMED else 'a path', failure.__class__.__name__))
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError('%s=%s: the id list is not the pinned one (sha256 %s)' % (FLAG, spec, digest[:12]))
    ids = parse_ids(data)
    if rows is not None and len(ids) != rows:
        raise ValueError('%s=%s holds %d ids, not the named %d' % (FLAG, spec, len(ids), rows))
    if len(ids) % (chips * TILE):
        raise ValueError('%s: %d ids are not a whole %d-row tile on each of %d chips' % (FLAG, len(ids), TILE, chips))
    if len(ids) // chips > topk_width:
        raise ValueError('%s: %d columns per chip do not fit the %d-wide top-16 launch' % (FLAG, len(ids) // chips, topk_width))
    plan = _PLANS[key] = Plan(ids, chips, topk_width, digest, spec)
    return plan


def topk_width(environ=None):
    """QWEN_FAST_DRAFT_VOCAB_TOPK, strict; the default when unset."""
    environ = os.environ if environ is None else environ
    value = environ.get(TOPK_FLAG)
    if value is None:
        return TOPK_DEFAULT
    if not (value.isascii() and value.isdigit()) or int(value) not in TOPK_WIDTHS:
        raise ValueError('%s must be one of %s, got %r' % (TOPK_FLAG, ', '.join(str(width) for width in TOPK_WIDTHS), value))
    return int(value)


def active_plan(environ=None):
    """The Plan the environment asks for, None with the flag off. Raises with the flag on beside anything it cannot serve (the pair, a bad name, a bad width)."""
    environ = os.environ if environ is None else environ
    if not requested(environ):
        if environ.get(TOPK_FLAG) is not None:
            raise ValueError('%s needs %s' % (TOPK_FLAG, FLAG))
        return None
    return load(environ[FLAG], tp_shapes.chip_count(environ), topk_width(environ))


def profile_problems(profile):
    """Why a profile's draft-vocabulary names cannot be served, [] when sound or unset: an unknown QWEN_FAST_DRAFT_VOCAB name, a width flag without the list, a list
    the four-card head cannot serve. The contract calls this at boot (serving_c2_contract.draft_vocab_problems)."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    problems = ['%s is not a draft vocabulary setting (the names are %s)' % (name, ', '.join(NAMES))
                for name in sorted(env) if name.startswith(PREFIX) and name not in NAMES]
    if FLAG not in env and TOPK_FLAG not in env:
        return problems
    if env.get('QWEN_FAST_TP', '2') != '4':
        problems.append('%s is a four-card lever: the profile does not set QWEN_FAST_TP=4' % FLAG)
        return problems
    try:
        active_plan(env)
    except ValueError as failure:
        problems.append(str(failure))
    return problems


def admission(environ=None, log=None):
    """The attach's admission: the Plan (None with the flag off), logged once. Raises, never falls back."""
    environ = os.environ if environ is None else environ
    plan = active_plan(environ)
    if plan is None:
        return None
    emit(log, '%s: %s (%s; the target still verifies all %d rows: only tau can move)' % (ADMITTED, plan.describe(), UNQUALIFIED, tp_shapes.VOCAB))
    return plan


# ---- the head: what draft_shared_head_tp calls with the flag on --------------------------------------------------------------------------

class Head(object):
    """The sliced weight a model's drafter head runs on, and the Plan it was cut with."""

    def __init__(self, weight, plan, dtype_name, bytes_per_chip):
        self.weight, self.plan, self.dtype_name, self.bytes_per_chip = weight, plan, dtype_name, bytes_per_chip


_HEADS = {}
_ENGAGED = []


def dtype_label(operations, dtype):
    for attribute, label in (('bfloat16', 'bf16'), ('bfloat8_b', 'bf8'), ('bfloat4_b', 'bf4'), ('float32', 'fp32')):
        value = getattr(operations, attribute, None)
        if value is not None and dtype == value:
            return label
    return str(dtype)


ELEMENT_BYTES = {'bf16': 2.0, 'bf8': 1.0625, 'bf4': 0.5625, 'fp32': 4.0}


def cut_columns(shards, plan, width, hidden):
    """The shortlist's columns as a (hidden, rows) bfloat16 tensor, in shortlist order: shard `source` of `shards` (a callable index -> (hidden, width) host tensor, read
    one at a time so only one is in memory) holds global columns [source * width, (source + 1) * width). Raises if a column is not found in exactly one shard."""
    import torch

    ids = plan.tensor()
    out = torch.empty(hidden, plan.rows, dtype=torch.bfloat16)
    filled = torch.zeros(plan.rows, dtype=torch.bool)
    for source in range(plan.chips):
        positions = ((ids // width) == source).nonzero().flatten()
        if not positions.numel():
            continue
        host = shards(source)
        if host.numel() != hidden * width:
            raise ValueError('a head shard has %d elements, not %d x %d' % (host.numel(), hidden, width))
        host = host.reshape(hidden, width)
        out[:, positions] = host.index_select(1, ids[positions] - source * width).to(torch.bfloat16)
        filled[positions] = True
        del host
    if not bool(filled.all()):
        raise ValueError('%d shortlist columns lie outside the %d head shards' % (int((~filled).sum()), plan.chips))
    return out


def build_head(operations, model, environ=None, log=None, hidden=tp_shapes.HIDDEN, clock=time.time):
    """Cut the shortlist's rows out of the target's LM head and upload them, sharded on the vocabulary axis, in the head's own dtype. Once per model, at the attach,
    before any trace is captured. -> the Head (also registered for head_for)."""
    plan = active_plan(environ)
    if plan is None:
        raise ValueError('%s is off: there is no shortlist head to build' % FLAG)
    if id(model) in _HEADS:
        raise ValueError('the shortlist head of this model is already built')
    if (model.num_devices != plan.chips or model.vocab_size != tp_shapes.VOCAB or not model._lmhead_vocab_sharded):
        raise ValueError('The vocabulary-sharded four-card target head is required to cut a draft shortlist')
    started = clock()
    weight = model.lm_head_weight
    device_shards = tuple(operations.get_device_tensors(weight))
    if len(device_shards) != plan.chips:
        raise ValueError('%s chips required to read the target head' % plan.chips)
    width = tp_shapes.vocab_shard()
    columns = cut_columns(lambda source: operations.to_torch(device_shards[source]).float(), plan, width, hidden)
    mesh = model.mesh_device
    dtype = weight.dtype
    uploaded = operations.from_torch(columns.reshape(1, 1, hidden, plan.rows), device=mesh, dtype=dtype, layout=operations.TILE_LAYOUT,
                                     memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ShardTensorToMesh(mesh, dim=3))
    label = dtype_label(operations, dtype)
    size = int(hidden * plan.per_chip * ELEMENT_BYTES.get(label, 2.0))
    head = _HEADS[id(model)] = Head(uploaded, plan, label, size)
    emit(log, '%s: %s dtype=%s bytes_per_chip=%d full_head_bytes_per_chip=%d in %.1f s' % (
        BUILT, plan.describe(), label, size, int(hidden * width * ELEMENT_BYTES.get(label, 2.0)), clock() - started))
    return head


def head_for(model):
    """The Head built for `model`; raises when the attach built none (a shortlist head is never made in a request: its buffer would sit in a trace's holes)."""
    head = _HEADS.get(id(model))
    if head is None:
        raise ValueError('%s is on but no shortlist head was built for this model at the attach' % FLAG)
    return head


def release(operations, model=None):
    """Free the shortlist head(s): the one built for `model`, or all. Safe to call twice."""
    for key in [key for key in _HEADS if model is None or key == id(model)]:
        head = _HEADS.pop(key)
        try:
            operations.deallocate(head.weight)
        except Exception:
            pass
    if model is None:
        del _ENGAGED[:]


def candidate_chunks(environ=None):
    """The one chunk per chip the shortlist leaves: ((0, per_chip),). draft_shared_head_tp.candidate_chunks with the flag on."""
    plan = active_plan(environ)
    return ((0, plan.per_chip),)


def validate_normalized(operations, model, normalized, plan):
    rows = normalized.shape[2] if len(normalized.shape) == 4 else 0
    if (model.num_devices != plan.chips or model.vocab_size != tp_shapes.VOCAB or not model._lmhead_vocab_sharded
            or rows not in (8, 16, 32) or tuple(normalized.shape) != (1, 1, rows, tp_shapes.HIDDEN)
            or normalized.dtype != operations.bfloat16 or normalized.layout != operations.TILE_LAYOUT
            or normalized.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Replicated eight/16/32-row learned-normalized input and pinned TP%d vocabulary head required' % plan.chips)


def shared_head_candidates(operations, model, normalized, owned):
    """draft_shared_head_tp.shared_head_candidates over the shortlist's sliced weight: one linear, one (padded) top-16 per chip."""
    plan = active_plan()
    validate_normalized(operations, model, normalized, plan)
    head = head_for(model)
    if head.plan is not plan:
        raise ValueError('the shortlist head was cut for another list or width than the environment now names')
    logits = operations.linear(normalized, head.weight)
    owned.append(logits)
    if not _ENGAGED:
        _ENGAGED.append(True)
        log_line('%s %s dtype=%s chunks=1 (was 2) rows_per_launch=%d' % (ENGAGED, plan.describe(), head.dtype_name, normalized.shape[2]))
    return local_head_candidates(operations, logits, owned)


def local_head_candidates(operations, logits, owned):
    """draft_shared_head_tp.local_head_candidates for the shortlist's logits (1, 1, rows, per_chip): the whole shard is the one chunk."""
    plan = active_plan()
    rows = logits.shape[2] if len(logits.shape) == 4 else 0
    if rows not in (8, 16, 32) or tuple(logits.shape) != (1, 1, rows, plan.per_chip):
        raise ValueError('Expected local shortlist shards, not gathered logits')
    chunk = logits
    if plan.per_chip != plan.topk_width:
        chunk = operations.pad(logits, [(0, 0), (0, 0), (0, 0), (0, plan.topk_width - plan.per_chip)], float('-inf'))
        owned.append(chunk)
    values, indices = operations.topk(chunk, k=TOP_CANDIDATES, dim=-1, largest=True, sorted=True)
    owned.extend((values, indices))
    return [dict(start=0, stop=plan.per_chip, values=values, indices=indices)]


def map_tokens(plan, chip, start, indices):
    """The GLOBAL token ids of local top-16 `indices` (any integer tensor) a chip returned for the chunk starting at `start`."""
    return plan.tensor()[chip * plan.per_chip + start + indices.long()]


def merge_chunk_candidates(chunks, block_rows=8):
    """draft_shared_head_tp.merge_chunk_candidates over the shortlist's chunks: the same checks and the same ordering (ascending global id, then descending score,
    both stable, so a tie goes to the lowest global id), with each chip's local index mapped to its global token id through the list."""
    import torch

    plan = active_plan()
    if type(block_rows) is not int or block_rows not in (8, 16, 32):
        raise ValueError('Explicit eight/16/32-row candidate block required')
    expected = {(chip, start, stop) for chip in range(plan.chips) for start, stop in ((0, plan.per_chip),)}
    seen, scores, identifiers = set(), [], []
    for chunk in chunks:
        chip, start, stop = (chunk[key] for key in ('chip', 'start', 'stop'))
        identity = chip, start, stop
        if identity not in expected or identity in seen:
            raise ValueError('Exactly one candidate chunk per chip/range required')
        seen.add(identity)
        values, indices = chunk['values'], chunk['indices']
        if (values.device.type != 'cpu' or indices.device.type != 'cpu'
                or values.shape != (block_rows, TOP_CANDIDATES) or indices.shape != values.shape
                or not torch.is_floating_point(values) or not torch.isfinite(values).all()
                or indices.dtype not in (torch.int32, torch.int64)
                or torch.any(indices < 0) or torch.any(indices >= stop - start)):
            raise ValueError('Finite complete-block top16 values and in-range integer indices required')
        ordered = indices.sort(-1).values
        if torch.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError('Duplicate local candidate index')
        scores.append(values)
        identifiers.append(map_tokens(plan, chip, start, indices))
    if seen != expected:
        raise ValueError('Missing candidate chunks')
    values, tokens = torch.cat(scores, dim=-1), torch.cat(identifiers, dim=-1)
    by_token = tokens.argsort(dim=-1, stable=True)
    values, tokens = values.gather(-1, by_token), tokens.gather(-1, by_token)
    selected = values.argsort(dim=-1, descending=True, stable=True)[:, :TOP_CANDIDATES]
    return tokens.gather(-1, selected)[None, 1:block_rows], values.gather(-1, selected)[None, 1:block_rows]
