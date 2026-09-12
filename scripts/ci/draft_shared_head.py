"""Experimental borrowed target LM head and chunked candidates; no hardware or serving certification."""


def candidate_chunks():
    return tuple((start, min(start + 32768, 124160)) for start in range(0, 124160, 32768))


def shared_head_candidates(operations, model, normalized, owned):
    rows = normalized.shape[2] if len(normalized.shape) == 4 else 0
    if (model.num_devices != 2 or model.vocab_size != 248320 or not model._lmhead_vocab_sharded
            or rows not in (8, 32) or tuple(normalized.shape) != (1, 1, rows, 5120)
            or normalized.dtype != operations.bfloat16 or normalized.layout != operations.TILE_LAYOUT
            or normalized.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Replicated eight/32-row learned-normalized input and pinned TP2 vocabulary head required')
    logits = operations.linear(normalized, model.lm_head_weight)
    owned.append(logits)
    return local_head_candidates(operations, logits, owned)


def local_head_candidates(operations, logits, owned):
    rows = logits.shape[2] if len(logits.shape) == 4 else 0
    if rows not in (8, 32) or tuple(logits.shape) != (1, 1, rows, 124160):
        raise ValueError('Expected local vocabulary shards, not gathered logits')
    outputs = []
    for start, stop in candidate_chunks():
        chunk = operations.slice(logits, (0, 0, 0, start), (1, 1, rows, stop), (1, 1, 1, 1))
        owned.append(chunk)
        if stop - start != 32768:
            chunk = operations.pad(chunk, [(0, 0), (0, 0), (0, 0), (0, 32768 - (stop - start))], float('-inf'))
            owned.append(chunk)
        values, indices = operations.topk(chunk, k=16, dim=-1, largest=True, sorted=True)
        owned.extend((values, indices))
        outputs.append(dict(start=start, stop=stop, values=values, indices=indices))
    return outputs


def merge_chunk_candidates(chunks, *, block_rows=8):
    import torch

    if type(block_rows) is not int or block_rows not in (8, 32):
        raise ValueError('Explicit eight/32-row candidate block required')
    expected = {(chip, start, stop) for chip in range(2) for start, stop in candidate_chunks()}
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
        identifiers.append(indices.long() + start + chip * 124160)
    if seen != expected:
        raise ValueError('Missing candidate chunks')
    values, tokens = torch.cat(scores, dim=-1), torch.cat(identifiers, dim=-1)
    by_token = tokens.argsort(dim=-1, stable=True)
    values, tokens = values.gather(-1, by_token), tokens.gather(-1, by_token)
    selected = values.argsort(dim=-1, descending=True, stable=True)[:, :16]
    return tokens.gather(-1, selected)[None, 1:block_rows], values.gather(-1, selected)[None, 1:block_rows]
