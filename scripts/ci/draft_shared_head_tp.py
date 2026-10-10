"""draft_shared_head at any served width.

The pair's module is pinned (recorded evidence hashes its bytes; test_tp2_pins) and stays as it was;
tp_addresses rebinds the names below to these at four cards only. Each is the pair's function with its literal chip and
head counts read from tp_shapes; at two chips it would be call for call the pinned one."""

import tp_shapes


def candidate_chunks():
    width = tp_shapes.vocab_shard()
    return tuple((start, min(start + 32768, width)) for start in range(0, width, 32768))


def shared_head_candidates(operations, model, normalized, owned):
    rows = normalized.shape[2] if len(normalized.shape) == 4 else 0
    if (model.num_devices != tp_shapes.chip_count() or model.vocab_size != 248320 or not model._lmhead_vocab_sharded
            or rows not in (8, 16, 32) or tuple(normalized.shape) != (1, 1, rows, 5120)
            or normalized.dtype != operations.bfloat16 or normalized.layout != operations.TILE_LAYOUT
            or normalized.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Replicated eight/16/32-row learned-normalized input and pinned TP%d vocabulary head required'
                         % tp_shapes.chip_count())
    logits = operations.linear(normalized, model.lm_head_weight)
    owned.append(logits)
    return local_head_candidates(operations, logits, owned)


def local_head_candidates(operations, logits, owned, row=0, rows=None):
    """The 16 best candidates of each 32,768-column chunk of a (1, 1, R, vocabulary shard) logits tensor, for its rows [row, row + rows). Default: all R rows (8, 16 or 32). QWEN_FAST_DRAFT_HEAD64
    (draft_head64_tp) passes the two 32-row halves of the 64-row head's logits as `row` = 0 and 32: the chunk slice then cuts its rows as well as its columns, so the ops below the matmul are
    the served ones on the served 32-row blocks."""
    total = logits.shape[2] if len(logits.shape) == 4 else 0
    if rows is None:
        rows = total
        if rows not in (8, 16, 32) or tuple(logits.shape) != (1, 1, rows, tp_shapes.vocab_shard()):
            raise ValueError('Expected local vocabulary shards, not gathered logits')
    elif (rows not in (8, 16, 32) or row < 0 or row % rows or row + rows > total or total % 32
            or tuple(logits.shape) != (1, 1, total, tp_shapes.vocab_shard())):
        raise ValueError('Expected local vocabulary shards of whole tile rows, not gathered logits')
    outputs = []
    for start, stop in candidate_chunks():
        chunk = operations.slice(logits, (0, 0, row, start), (1, 1, row + rows, stop), (1, 1, 1, 1))
        owned.append(chunk)
        if stop - start != 32768:
            chunk = operations.pad(chunk, [(0, 0), (0, 0), (0, 0), (0, 32768 - (stop - start))], float('-inf'))
            owned.append(chunk)
        values, indices = operations.topk(chunk, k=16, dim=-1, largest=True, sorted=True)
        owned.extend((values, indices))
        outputs.append(dict(start=start, stop=stop, values=values, indices=indices))
    return outputs


def block_head_candidates(operations, model, normalized, owned, halves=2):
    """shared_head_candidates for a block of `halves` 32-row halves in ONE matmul (QWEN_FAST_DRAFT_HEAD64): the 64-row learned-normalized block against the vocabulary shard once, then each
    half's chunk candidates as the served path takes them. Returns one candidates list a half."""
    rows = normalized.shape[2] if len(normalized.shape) == 4 else 0
    if (model.num_devices != tp_shapes.chip_count() or model.vocab_size != 248320 or not model._lmhead_vocab_sharded
            or rows != 32 * halves or tuple(normalized.shape) != (1, 1, rows, 5120)
            or normalized.dtype != operations.bfloat16 or normalized.layout != operations.TILE_LAYOUT
            or normalized.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Replicated %d-row learned-normalized input and pinned TP%d vocabulary head required' % (32 * halves, tp_shapes.chip_count()))
    logits = operations.linear(normalized, model.lm_head_weight)
    owned.append(logits)
    return [local_head_candidates(operations, logits, owned, row=32 * half, rows=32) for half in range(halves)]


def merge_chunk_candidates(chunks, *, block_rows=8):
    import torch

    if type(block_rows) is not int or block_rows not in (8, 16, 32):
        raise ValueError('Explicit eight/16/32-row candidate block required')
    expected = {(chip, start, stop) for chip in range(tp_shapes.chip_count()) for start, stop in candidate_chunks()}
    seen, scores, identifiers = set(), [], []
    for chunk in chunks:
        chip, start, stop = (chunk[key] for key in ('chip', 'start', 'stop'))
        identity = chip, start, stop
        if identity not in expected or identity in seen:
            raise ValueError('Exactly one candidate chunk per chip/range required')
        seen.add(identity)
        values, indices = chunk['values'], chunk['indices']
        if (values.device.type != 'cpu' or indices.device.type != 'cpu'
                or values.shape != (block_rows, 16) or indices.shape != values.shape
                or not torch.is_floating_point(values) or not torch.isfinite(values).all()
                or indices.dtype not in (torch.int32, torch.int64)
                or torch.any(indices < 0) or torch.any(indices >= stop - start)):
            raise ValueError('Finite complete-block top16 values and in-range integer indices required')
        ordered = indices.sort(-1).values
        if torch.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError('Duplicate local candidate index')
        scores.append(values)
        identifiers.append(indices.long() + start + chip * tp_shapes.vocab_shard())
    if seen != expected:
        raise ValueError('Missing candidate chunks')
    values, tokens = torch.cat(scores, dim=-1), torch.cat(identifiers, dim=-1)
    by_token = tokens.argsort(dim=-1, stable=True)
    values, tokens = values.gather(-1, by_token), tokens.gather(-1, by_token)
    selected = values.argsort(dim=-1, descending=True, stable=True)[:, :16]
    return tokens.gather(-1, selected)[None, 1:block_rows], values.gather(-1, selected)[None, 1:block_rows]
