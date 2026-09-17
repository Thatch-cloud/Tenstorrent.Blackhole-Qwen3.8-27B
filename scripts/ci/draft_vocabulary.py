"""Draft-only vocabulary precomputation; never restrict the target verifier vocabulary."""


def select_vocabulary(counts, size, *, required=()):
    counts, required = tuple(counts), tuple(required)
    vocabulary = len(counts)
    if (not vocabulary or any(type(count) is not int or count < 0 for count in counts)
            or type(size) is not int or not 0 < size <= vocabulary
            or any(type(token) is not int or not 0 <= token < vocabulary for token in required)):
        raise ValueError('Nonnegative integer frequencies and valid vocabulary bounds required')
    selected = set(required)
    if len(selected) > size:
        raise ValueError('Shortlist cannot discard required special tokens')
    for token in sorted(range(vocabulary), key=lambda token: (-counts[token], token)):
        if len(selected) == size:
            break
        selected.add(token)
    return tuple(sorted(selected))


def materialize_head(weight, token_ids):
    import torch

    token_ids = tuple(token_ids)
    if (weight.device.type != 'cpu' or weight.ndim != 2 or not torch.is_floating_point(weight)
            or not token_ids or any(type(token) is not int or not 0 <= token < weight.shape[0] for token in token_ids)
            or tuple(sorted(set(token_ids))) != token_ids):
        raise ValueError('CPU head and unique ascending global token IDs required')
    indices = torch.tensor(token_ids, dtype=torch.int64)
    return weight.index_select(0, indices).contiguous(), indices.to(torch.int32)


def global_argmax(logits, token_ids):
    import torch

    if (logits.device.type != 'cpu' or token_ids.device.type != 'cpu' or logits.ndim < 1
            or token_ids.ndim != 1 or token_ids.dtype not in (torch.int32, torch.int64)
            or not token_ids.numel() or logits.shape[-1] != token_ids.numel()
            or not torch.is_floating_point(logits) or not torch.isfinite(logits).all()
            or torch.any(token_ids < 0) or torch.any(token_ids[1:] <= token_ids[:-1])):
        raise ValueError('Finite draft logits and ascending global vocabulary mapping required')
    return token_ids[logits.argmax(dim=-1)]
