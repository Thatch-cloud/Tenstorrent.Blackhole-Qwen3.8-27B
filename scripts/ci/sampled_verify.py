"""Sampled verification on the TP4 fast path: the serving side of docs/tp4-sampled-verify.md (QWEN_FAST_SAMPLED=1; default off, gate-only profiles).

sampled_accept.py is the scheme (keyed noise, the filter, the draw, the coupled walk). This module is what puts it behind the verifiers:

  DEVICE (option b1). Each chip returns its vocabulary shard's top-K_MAX logits and shard-local ids for every verify row through STOCK ttnn.topk, the op the drafter's
  vocabulary head already runs on a card (draft_shared_head_tp.local_head_candidates: k=16, sorted, on 32,768-column chunks of the 62,080-wide shard, the short chunk padded
  with -inf, rows in tiles of 32). Same shapes here, k = K_MAX = 64: a shard is two chunks, a 64-row block is two 32-row pieces, so a block is four launches a chip and eight
  small tensors, read back by batched_reads_tp.read (one fence in the profile's `async` mode) and merged on the host (sampled_accept.merge_shard_topk: exact, the global top-64
  lies inside the union of the chunks' top-64). The launches sit AFTER the pinned shard argmax in the same verify trace; the argmax path is untouched and still serves every
  greedy request of a sampled profile.
  HOST. packed_predictions turns the merged slices into draws, one entry at a time with the entry's own spec, seed and absolute positions (the ticket's), idle segments inactive.
  The sequential TP4 engine (verifier_engine_tp) reads its few rows' logits shards instead and slices on the host (no extra device op: it is the exceptional path).
  FIRST TOKEN. The plugin samples token 0 from the prefill logits with its own (unkeyed) host sampler; FirstToken wraps `runner.host_sampler` for a sampled request and returns the
  keyed draw at position len(prompt) from the same top-64 filter, so the committed text is the plain sampler's from its first token on.
  GATE. accepts() is what serving_fast_policy.validate_request_sampling asks: a non-greedy request is admitted only with the switch on, after the attach proved every block can draw
  (attach), and when the request carries a valid keyed seed. Until then it is refused, ending that one request under C2-any.

Everything is behind QWEN_FAST_SAMPLED=1. With the switch unset or 0 nothing here is imported by a served path and every module that hooks it behaves as it did.
"""

import os
import time

import numpy as np

import sampled_accept as sa

FLAG = 'QWEN_FAST_SAMPLED'
AUDIT_FLAG = 'QWEN_FAST_SAMPLED_AUDIT'
SEED_KEY = 'qwen_sampled_seed'        # SamplingParams.extra_args key (serving_c2_contract.SAMPLED_SEED_KEY, held equal by a test)
K = sa.K_MAX                          # the candidates a chunk's topk returns
CHUNK = 32768                         # the drafter head's chunk width (draft_shared_head_tp.candidate_chunks)
PIECE_ROWS = 32                       # rows a topk launch takes (the drafter runs its head on 8/16/32-row tiles; a 64-row block is two pieces)
BLOCK_ROWS = (8, 16, 32, 64)
FIRST_LOGGED = 4                      # sampled rounds whose timing line is logged, then one in LOG_EVERY

ENGAGED_MARKER = '[PINDIAG] sampled verify engaged'
ROUND_MARKER = '[PINDIAG] sampled verify round'
AUDIT_MARKER = '[PINDIAG] sampled verify audit'
AUDIT_MISMATCH = '[PINDIAG] sampled verify audit mismatch'
REFUSED_MARKER = '[PINDIAG] sampled verify refused'
ATTACH_MARKER = '[PINDIAG] sampled verify attach'
FIRST_TOKEN_MARKER = '[PINDIAG] sampled first token'
SEQUENTIAL_MARKER = '[PINDIAG] sampled verify sequential'
LOG_EVERY = 256


class SampledRefused(ValueError):
    """The sampled path cannot be built or run as asked; the attach (or the request) fails closed, with the reason."""


def log_line(text):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def _strict(name, environ=None):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (name, value))
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_SAMPLED=1 (strict: anything but unset, 0 or 1 is a configuration error, never a silent off)."""
    return _strict(FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_SAMPLED_AUDIT=1 beside QWEN_FAST_SAMPLED=1 (gate profiles only): the audited rounds also read the logits back and compare the slice and the draw."""
    if not _strict(AUDIT_FLAG, environ):
        return False
    if not enabled(environ):
        raise ValueError('%s=1 needs %s=1' % (AUDIT_FLAG, FLAG))
    return True


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# the request
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

class Sampling(object):
    """One request's sampling terms as the verifier applies them: the filter (temperature, top_p, top_k, already clamped by the contract) and the 64-bit key."""

    __slots__ = ('spec', 'seed')

    def __init__(self, spec, seed):
        self.spec, self.seed = spec, seed

    def __eq__(self, other):
        return isinstance(other, Sampling) and (self.spec, self.seed) == (other.spec, other.seed)

    def __hash__(self):
        return hash((self.spec, self.seed))

    def __repr__(self):
        return 'Sampling(%r, seed=%d)' % (self.spec, self.seed)


def of_params(params):
    """The Sampling of a request's SamplingParams as serving_c2_contract.sampled_terms left them; None for a greedy request (temperature 0: the argmax path serves it
    unchanged). A sampled request that does not carry exactly what the verifier needs raises ValueError (it is refused, never served as something else): its seed in
    extra_args and not in the plugin's seed manager (the pinned samplers refuse an active request seed), a filter inside what the slice can serve, no penalties, no min_p."""
    temperature = params.temperature
    if type(temperature) not in (int, float) or not 0 <= temperature < float('inf'):
        raise ValueError('temperature must be a finite number >= 0, got %r' % (temperature,))
    if temperature == 0:
        return None
    if getattr(params, 'seed', None) is not None:
        raise ValueError('a sampled request\'s seed belongs in extra_args[%s]: the plugin\'s seed manager must not hold it' % SEED_KEY)
    if (getattr(params, 'min_p', 0.0) or 0.0) != 0 or (getattr(params, 'presence_penalty', 0.0) or 0.0) != 0 or (getattr(params, 'frequency_penalty', 0.0) or 0.0) != 0 \
            or (1.0 if getattr(params, 'repetition_penalty', None) is None else params.repetition_penalty) != 1.0:
        raise ValueError('penalties and min_p are forced off before a request reaches the verifier')
    seed = (getattr(params, 'extra_args', None) or {}).get(SEED_KEY)
    if type(seed) is not int or not 0 <= seed <= sa.SEED_MASK:
        raise ValueError('a sampled request carries a 64-bit seed in extra_args[%s], got %r' % (SEED_KEY, seed))
    top_p, top_k = getattr(params, 'top_p', None), getattr(params, 'top_k', None)
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ValueError('a sampled request\'s top_p is in (0, 1], got %r' % (top_p,))
    if type(top_k) is not int or not 1 <= top_k <= sa.K_MAX:
        raise ValueError('a sampled request\'s top_k is 1..%d (the contract clamps it), got %r' % (sa.K_MAX, top_k))
    return Sampling(sa.SamplingSpec(float(temperature), float(top_p), top_k), seed)


def sampling_of(entry):
    """The Sampling of a packed entry's request (None: greedy, or nothing attached)."""
    session = getattr(entry['request'], 'session', None)
    return getattr(session, 'sampling', None)


def attach_sampling(request, params):
    """serving_request_factory: the request's session carries its Sampling (None for a greedy one) for every verifier that serves it."""
    request.session.sampling = of_params(params)
    return request.session.sampling


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# readiness: the engine-side gate fails closed until the attach has proven the verifiers can draw
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

class _State(object):
    def __init__(self):
        self.ready = False
        self.blocks = 0


STATE = _State()


def reset():
    STATE.ready = False
    STATE.blocks = 0


def attach(blocks, log=None):
    """serving_runtime, after every block is built: with the switch on, each block must have engaged the sampled readback (a block that could not has already failed its own
    construction); only then does the gate admit a sampled request. -> the number of blocks. Off, nothing."""
    if not enabled():
        return 0
    log = log_line if log is None else log
    blocks = list(blocks)
    missing = [index for index, block in enumerate(blocks) if getattr(block, 'sampled', False) is not True]
    if missing:
        raise SampledRefused('%s=1 but packed block(s) %s did not engage the sampled readback' % (FLAG, missing))
    STATE.ready, STATE.blocks = True, len(blocks)
    log('%s blocks=%d k=%d audit=%d' % (ATTACH_MARKER, len(blocks), K, int(audit_enabled())))
    return len(blocks)


def accepts(params, environ=None):
    """The engine-side gate (serving_fast_policy.validate_request_sampling) for a request whose temperature is not 0: True only when the switch is on, the attach proved the
    verifiers can draw, and the request is a well-formed sampled one (of_params)."""
    if not enabled(environ) or not STATE.ready:
        return False
    try:
        return of_params(params) is not None
    except ValueError:
        return False


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# the device slice: stock ttnn.topk on the vocabulary shard, inside the verify trace
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def chunks(width):
    """The column ranges of a vocabulary shard of `width` columns (draft_shared_head_tp.candidate_chunks): 32,768 wide, the last the remainder."""
    return tuple((start, min(start + CHUNK, width)) for start in range(0, width, CHUNK))


def layout(rows, width):
    """The topk launches of a `rows`-row block over a `width`-column shard: [(row0, rows, chunk, start, stop)], pieces of 32 rows (a block of 8/16/32 rows is one piece)."""
    if rows not in BLOCK_ROWS:
        raise SampledRefused('the sampled readback serves blocks of %s rows, not %r' % ('/'.join(str(n) for n in BLOCK_ROWS), rows))
    pieces = ((0, rows),) if rows <= PIECE_ROWS else tuple((row, PIECE_ROWS) for row in range(0, rows, PIECE_ROWS))
    return tuple((row, count, index, start, stop) for row, count in pieces for index, (start, stop) in enumerate(chunks(width)))


def block_problem(block):
    """Why a packed block cannot take the sampled readback, or None. A block that cannot refuses its own construction (fail closed): a sampled profile never runs a verifier
    that would commit argmax for a sampled request."""
    import tp_shapes

    if tp_shapes.chip_count() != 4:
        return 'the sampled readback is the four-card path (QWEN_FAST_TP=4)'
    if not getattr(block, 'shard_argmax', False):
        reason = getattr(block, 'shard_problem', None)
        return 'the block needs the verify trace\'s shard argmax (QWEN_FAST_VERIFY_T1 and its shard_argmax cut) beside the slice%s' % (': %s' % reason if reason else '')
    try:
        layout(block.block_rows, tp_shapes.vocab_shard())
    except SampledRefused as error:
        return str(error)
    for name in ('slice', 'topk', 'deallocate'):
        if not callable(getattr(block.operations, name, None)):
            return 'this ttnn build has no %s' % name
    return None


def device_slice(operations, logits, rows):
    """In the verify trace (and its eager warm forward): the top-K of every chunk of the (1, 1, rows, shard) bf16 TILE logits, per row piece. -> the output tensors in launch
    order, (values, indices) per launch (layout()). The intermediates (the column slice, its -inf pad) are freed here; the outputs are held with the trace's other outputs."""
    import tp_shapes

    width = tp_shapes.vocab_shard()
    shape = tuple(logits.shape)
    if len(shape) != 4 or shape[:3] != (1, 1, rows) or shape[3] != width:
        raise SampledRefused('the sampled slice needs the pre-gather (1, 1, %d, %d) vocab shard; got %r' % (rows, width, shape))
    dtype, tile = getattr(logits, 'dtype', None), getattr(logits, 'layout', None)
    if dtype != operations.bfloat16 or tile != operations.TILE_LAYOUT:
        raise SampledRefused('the sampled slice needs bf16 TILE logits; got %r %r' % (dtype, tile))
    produced = []
    try:
        for row, count, index, start, stop in layout(rows, width):
            owned = []
            try:
                piece = operations.slice(logits, (0, 0, row, start), (1, 1, row + count, stop), (1, 1, 1, 1))
                owned.append(piece)
                if stop - start != CHUNK:
                    piece = operations.pad(piece, [(0, 0), (0, 0), (0, 0), (0, CHUNK - (stop - start))], float('-inf'))
                    owned.append(piece)
                try:
                    values, indices = operations.topk(piece, k=K, dim=-1, largest=True, sorted=True)
                except (TypeError, ValueError, RuntimeError) as error:
                    raise SampledRefused('ttnn.topk refused k=%d on a (1, 1, %d, %d) bf16 chunk: %s: %s'
                                         % (K, count, CHUNK, type(error).__name__, str(error).splitlines()[0][:200] if str(error) else '')) from None
                produced.extend((values, indices))
            finally:
                for value in reversed(owned):
                    operations.deallocate(value)
    except BaseException:
        for value in produced:
            operations.deallocate(value)
        raise
    return tuple(produced)


def read_tensors(operations, mesh, tensors):
    """[[chip 0 .. chip 3 host tensor] per device tensor]: one fence in batched reads' `async` mode, a composed read a tensor in `1`, a blocking read a shard otherwise."""
    import batched_reads_tp

    if batched_reads_tp.enabled():
        return batched_reads_tp.read(operations, mesh, tensors, site='sampled')
    return batched_reads_tp.served(operations, tensors)


def chip_slices(per_chip, rows, width):
    """The per-chip candidate arrays from the read tensors (launch order, values then indices): ([chip] (rows, chunks * K) float64 values, [chip] int64 shard-local ids).
    A local id past its chunk (the -inf pad) is a device fault and refused."""
    chips = len(per_chip[0])
    plan = layout(rows, width)
    if len(per_chip) != 2 * len(plan):
        raise SampledRefused('%d tensors read for %d launches' % (len(per_chip), len(plan)))
    columns = len(chunks(width)) * K
    values = [np.zeros((rows, columns), dtype=np.float64) for _ in range(chips)]
    ids = [np.zeros((rows, columns), dtype=np.int64) for _ in range(chips)]
    for number, (row, count, index, start, stop) in enumerate(plan):
        for chip in range(chips):
            part_values = per_chip[2 * number][chip].float().numpy().astype(np.float64).reshape(count, K)
            part_ids = per_chip[2 * number + 1][chip].long().numpy().reshape(count, K)
            if bool((part_ids < 0).any()) or bool((part_ids >= stop - start).any()):
                raise SampledRefused('a top-K index of chunk %d on chip %d lies outside its %d columns' % (index, chip, stop - start))
            values[chip][row:row + count, index * K:(index + 1) * K] = part_values
            ids[chip][row:row + count, index * K:(index + 1) * K] = part_ids + start
    return values, ids


def merged_slice(chip_values, chip_ids, which, width):
    """The global top-K slice (values, ids) of the rows `which` from the per-chip candidates (sampled_accept.merge_shard_topk)."""
    return sa.merge_shard_topk([value[which] for value in chip_values], [value[which] for value in chip_ids], width, k=K)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# the packed block's predictions
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def engaged_line(block):
    import tp_shapes

    plan = layout(block.block_rows, tp_shapes.vocab_shard())
    return '%s site=packed_verify rows=%d users=%d k=%d launches=%d tensors=%d chunk=%d audit=%d' % (
        ENGAGED_MARKER, block.block_rows, block.users, K, len(plan), 2 * len(plan), CHUNK, int(audit_enabled()))


def tensors_of(block):
    """The slice tensors at the end of the block's captured output (operation puts them last)."""
    import tp_shapes

    count = 2 * len(layout(block.block_rows, tp_shapes.vocab_shard()))
    return tuple(block.output[-count:])


def packed_predictions(block, entries, segments, host, predictions, audit_number):
    """The packed round's per-entry predictions with every sampled entry's rows replaced by its draws.

    `host` is the block's argmax ids (the shard argmax path), `predictions` the per-entry slices of it; a greedy entry keeps its slice, a sampled one gets, for each of its rows,
    the keyed draw at ticket.position + 1 + row from the merged top-K of the row's logits (its own spec and seed). Rows of idle segments and of greedy entries are never merged
    or read. -> (predictions, metrics); metrics carries the round's read, merge and draw milliseconds."""
    import tp_shapes
    from packed_shapes import segment_rows

    chosen = []
    for position, (entry, segment) in enumerate(zip(entries, segments)):
        sampling = sampling_of(entry)
        if sampling is not None:
            chosen.append((position, entry, segment, sampling))
    if not chosen:
        return predictions, dict(entries=0)
    started = time.perf_counter()
    width = tp_shapes.vocab_shard()
    rows_each = block.rows_per_user
    per_chip = read_tensors(block.operations, block.mesh, tensors_of(block))
    read_done = time.perf_counter()
    chip_values, chip_ids = chip_slices(per_chip, block.block_rows, width)
    which = np.concatenate([np.arange(*segment_rows(block.shape, segment)) for _, _, segment, _ in chosen])
    values, ids = merged_slice(chip_values, chip_ids, which, width)
    merged = time.perf_counter()
    seeds, positions, temperature, top_p, top_k = [], [], [], [], []
    for _, entry, segment, sampling in chosen:
        ticket = entry['ticket']
        if len(ticket.tokens) != rows_each:
            raise SampledRefused('a sampled ticket of %d rows in a block of %d rows a user' % (len(ticket.tokens), rows_each))
        seeds += [sampling.seed] * rows_each
        positions += sa.noise_positions(ticket.position, rows_each)
        temperature += [sampling.spec.temperature] * rows_each
        top_p += [sampling.spec.top_p] * rows_each
        top_k += [sampling.spec.top_k] * rows_each
    draws = sa.draw_batch(values, ids, np.array(temperature), np.array(top_p), np.array(top_k, dtype=np.int64), seeds, positions)
    drawn = time.perf_counter()
    argmax_rows = np.array([host[row] for row in which], dtype=np.int64)
    top1 = np.flatnonzero(ids[:, 0] != argmax_rows)
    if len(top1):
        message = '%s site=packed_verify top1 rows=%s slice=%s argmax=%s' % (AUDIT_MISMATCH, [int(which[row]) for row in top1[:8]], [int(ids[row, 0]) for row in top1[:8]],
                                                                          [int(argmax_rows[row]) for row in top1[:8]])
        log_line(message)
        if audit_enabled() or getattr(block, 'shard_audit', False):
            raise AssertionError(message)
    if audit_enabled() and audit_number is not None:
        audit_round(block, chosen, which, values, ids, draws, seeds, positions, temperature, top_p, top_k, audit_number)
    out = list(predictions)
    offset = 0
    for position, entry, segment, sampling in chosen:
        out[position] = [int(token) for token in draws[offset:offset + rows_each]]
        offset += rows_each
    block.sampled_rounds = getattr(block, 'sampled_rounds', 0) + 1
    metrics = dict(entries=len(chosen), rows=len(which), read_ms=(read_done - started) * 1000, merge_ms=(merged - read_done) * 1000, draw_ms=(drawn - merged) * 1000)
    if block.sampled_rounds <= FIRST_LOGGED or block.sampled_rounds % LOG_EVERY == 0:
        log_line('%s site=packed_verify n=%d entries=%d rows=%d read_ms=%.2f merge_ms=%.2f draw_ms=%.2f' % (
            ROUND_MARKER, block.sampled_rounds, len(chosen), len(which), metrics['read_ms'], metrics['merge_ms'], metrics['draw_ms']))
    return out, metrics


def slice_difference(device_values, device_ids, reference_values, reference_ids, row):
    """Why a device slice is not the reference's for one row (a full-vocabulary logits row), or None. The values must be equal; the ids must be equal where the value is
    strictly above the cut (the K-th value); inside the cut's tie group the device may hold any ids of the row with that value (ttnn.topk's tie order is its own)."""
    if not np.array_equal(device_values, reference_values):
        return 'values differ at slots %s' % np.flatnonzero(device_values != reference_values)[:6].tolist()
    cut = reference_values[-1]
    above = reference_values > cut
    if not np.array_equal(device_ids[above], reference_ids[above]):
        return 'ids differ above the cut at slots %s' % np.flatnonzero(above & (device_ids != reference_ids))[:6].tolist()
    tied = ~above
    allowed = set(np.flatnonzero(row.astype(np.float64) == cut).tolist())
    if not set(int(i) for i in device_ids[tied]) <= allowed or len(set(device_ids[tied].tolist())) != int(tied.sum()):
        return 'ids inside the cut\'s tie group are not that value\'s'
    return None


def full_rows(block, which):
    """The sampled rows' full-vocabulary logits off the four chips (float32, one 248,320-wide row each): the audited rounds' reference input."""
    import torch

    ops = block.operations
    parts = ops.get_device_tensors(block.output[0])
    rows = torch.as_tensor(np.asarray(which, dtype=np.int64))
    host = [ops.to_torch(part).reshape(block.block_rows, -1)[rows].float().numpy() for part in parts]
    return np.concatenate(host, axis=1)


def audit_round(block, chosen, which, values, ids, draws, seeds, positions, temperature, top_p, top_k, number):
    """QWEN_FAST_SAMPLED_AUDIT, on the audited rounds (audit_sampling's cadence): the sampled rows' logits are read back whole and the device slice and the draw are compared with
    the reference path's (topk_slice of the full row, draw_batch of that slice) - the functions full_request's gold calls. A difference is logged and raised."""
    full = full_rows(block, which)
    reference_values, reference_ids = sa.topk_slice(full, K)
    ties = []
    for row in range(len(which)):
        reason = slice_difference(values[row], ids[row], reference_values[row].astype(np.float64), reference_ids[row], full[row])
        if reason is not None:
            message = '%s site=packed_verify round=%d row=%d %s' % (AUDIT_MISMATCH, number, int(which[row]), reason)
            log_line(message)
            raise AssertionError(message)
        if not np.array_equal(ids[row], reference_ids[row]):
            ties.append(row)
    expected = sa.draw_batch(reference_values, reference_ids, np.array(temperature), np.array(top_p), np.array(top_k, dtype=np.int64), seeds, positions)
    different = [row for row in range(len(which)) if row not in ties and int(expected[row]) != int(draws[row])]
    if different:
        message = '%s site=packed_verify round=%d draws rows=%s served=%s reference=%s' % (
            AUDIT_MISMATCH, number, [int(which[row]) for row in different[:8]], [int(draws[row]) for row in different[:8]], [int(expected[row]) for row in different[:8]])
        log_line(message)
        raise AssertionError(message)
    log_line('%s site=packed_verify round=%d rows=%d slice=exact draws=exact tie_rows=%d' % (AUDIT_MARKER, number, len(which), len(ties)))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# the sequential TP4 engine: host slice from the read-back logits
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def sequential_predictions(engine, output, ticket, sampling):
    """A sampled request's predictions from the sequential engine's own verify (1, 2 or 4 rows): the output's logits (the vocab shard on each chip, or the replicated full
    row when the engine has no device sampler) are read back and sliced on the host - no extra device op in a trace that only the exceptional rounds replay. The draw at row i
    is the keyed draw at ticket.position + 1 + i, exactly the packed block's."""
    import tp_shapes

    ops = engine.operations
    rows = len(ticket.tokens)
    parts = ops.get_device_tensors(output[0])
    vocab = engine.model.args.vocab_size
    width = tp_shapes.vocab_shard()
    host = [ops.to_torch(part) for part in parts]
    last = int(host[0].shape[-1])
    if last == width and len(parts) == tp_shapes.chip_count():
        chip_values, chip_ids = [], []
        for part in host:
            values, ids = sa.topk_slice(part.reshape(-1, width)[:rows].float().numpy(), K)
            chip_values.append(values)
            chip_ids.append(ids)
        values, ids = sa.merge_shard_topk(chip_values, chip_ids, width, k=K)
    elif last == vocab:
        values, ids = sa.topk_slice(host[0].reshape(-1, vocab)[:rows].float().numpy(), K)
    else:
        raise SampledRefused('the sequential verify output holds %d columns a chip: neither the %d-column shard nor the %d-column row' % (last, width, vocab))
    spec = sampling.spec
    draws = sa.draw_batch(values, ids, spec.temperature, spec.top_p, spec.top_k, [sampling.seed] * rows, sa.noise_positions(ticket.position, rows))
    return [int(token) for token in draws]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# the first token
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

class FirstToken(object):
    """The plugin's `host_sampler` for a sampled request's prefill: token 0 is the keyed draw at position len(prompt), the same filter and noise as every later token, from
    the prefill logits (bf16 values in float32). Armed by the lifecycle at admission, consumed by the one host-sampler call of the request's final prefill chunk, checked after
    the plugin's sample_tokens (finish): a request that was armed and not drawn here fails the engine rather than start its text from an unkeyed token. Any other call (a
    greedy request, the warmup) passes through to the plugin's sampler untouched."""

    def __init__(self, inner):
        self.inner = inner
        self.armed = None          # (request_id, Sampling, position)
        self.drawn = None          # (request_id, token)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @classmethod
    def install(cls, runner):
        mode = getattr(runner, 'sample_on_device_mode', None)
        if mode == 'all':
            raise SampledRefused('%s needs the prefill sampled on the host (sample_on_device_mode %r samples it on the device)' % (FLAG, mode))
        if not callable(getattr(runner, 'host_sampler', None)):
            raise SampledRefused('%s needs the plugin runner\'s host_sampler to draw the first token' % FLAG)
        wrapper = cls(runner.host_sampler)
        runner.host_sampler = wrapper
        return wrapper

    def uninstall(self, runner):
        if getattr(runner, 'host_sampler', None) is self:
            runner.host_sampler = self.inner

    def arm(self, request_id, params, prompt_tokens):
        """The admission of a request: its Sampling (None for a greedy one) and the position of its first token."""
        sampling = of_params(params)
        self.drawn = None
        self.armed = None if sampling is None else (request_id, sampling, prompt_tokens)
        return sampling

    def disarm(self):
        self.armed = self.drawn = None

    def __call__(self, *args, **kwargs):
        armed = self.armed
        if armed is None:
            return self.inner(*args, **kwargs)
        import torch

        logits = kwargs.get('logits', args[0] if args else None)
        vocab = int(logits.shape[-1]) if logits is not None and logits.dim() == 2 else None
        if vocab is None or int(logits.shape[0]) != 1:
            raise SampledRefused('the first token of a sampled request is drawn from exactly one logits row, got %r' % (None if logits is None else tuple(logits.shape),))
        request_id, sampling, position = armed
        # drawn BEFORE the plugin's sampler runs: vLLM's sampler may rescale the logits in place
        row = np.array(logits[0].detach().to(torch.float32).cpu().numpy(), dtype=np.float32, copy=True)
        token = sa.sample_logits(row, sampling.spec, sampling.seed, position)
        output = self.inner(*args, **kwargs)
        sampled = output.sampled_token_ids
        output.sampled_token_ids = torch.full_like(sampled, token)
        self.drawn = (request_id, token)
        log_line('%s request=%s position=%d token=%d' % (FIRST_TOKEN_MARKER, str(request_id)[:48], position, token))
        return output

    def finish(self, request_id, result):
        """After the plugin's sample_tokens returned `result` for the request in its prefill phase: an armed request must have been drawn here and be the token the plugin
        returned. Raises ValueError otherwise (the lifecycle fails the engine)."""
        armed, drawn = self.armed, self.drawn
        self.armed = self.drawn = None
        if armed is None:
            return None
        if armed[0] != request_id or drawn is None or drawn[0] != request_id:
            raise SampledRefused('the sampled request %s reached its first token without the keyed draw (armed for %s, drawn for %s)'
                                 % (str(request_id)[:48], armed[0], None if drawn is None else drawn[0]))
        returned = result.sampled_token_ids[0][0] if result is not None and result.sampled_token_ids else None
        if returned != drawn[1]:
            raise SampledRefused('the sampled request %s first token %r is not the keyed draw %r' % (str(request_id)[:48], returned, drawn[1]))
        return drawn[1]
