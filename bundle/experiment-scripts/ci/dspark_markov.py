"""CPU semantic reference for greedy vanilla Markov proposals; not a TT kernel or GPU rounding oracle."""


def greedy_proposals(base_logits, anchor_tokens, predecessor, successor):
    import torch

    tensors = (base_logits, anchor_tokens, predecessor, successor)
    if any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu' for value in tensors):
        raise ValueError('CPU tensors required for the semantic reference')
    if (base_logits.ndim != 3 or predecessor.ndim != 2 or successor.shape != predecessor.shape
            or anchor_tokens.shape != base_logits.shape[:1] or anchor_tokens.dtype != torch.int64
            or base_logits.dtype not in (torch.float32, torch.float64)
            or any(value.dtype != base_logits.dtype for value in (predecessor, successor))
            or predecessor.shape[0] != base_logits.shape[-1] or min(predecessor.shape) < 1
            or base_logits.shape[0] < 1):
        raise ValueError('Compatible batch/proposal/vocabulary and learned rank dimensions required')
    if (any(not torch.isfinite(value).all() for value in (base_logits, predecessor, successor))
            or torch.any(anchor_tokens < 0) or torch.any(anchor_tokens >= predecessor.shape[0])):
        raise ValueError('Finite operands and in-vocabulary anchors required')
    previous = anchor_tokens
    output = torch.empty(base_logits.shape[:2], dtype=torch.int64)
    for position in range(base_logits.shape[1]):
        bias = predecessor[previous] @ successor.T
        corrected = base_logits[:, position] + bias
        if not torch.isfinite(corrected).all():
            raise ValueError('Markov correction overflowed')
        previous = torch.argmax(corrected, dim=-1)
        output[:, position] = previous
    return output
