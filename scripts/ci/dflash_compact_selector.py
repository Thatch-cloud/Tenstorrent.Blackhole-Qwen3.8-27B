"""Experimental single-preparation FP64 selector; the existing selector remains default."""

from draft_selector import validate_selector_operands


def select_active_candidates(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    import torch

    operands = (projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
    if (any(value.device.type != 'cpu' for value in operands)
            or projected_hidden.ndim != 3 or projected_hidden.shape[0] < 1
            or not 1 <= projected_hidden.shape[1] <= 31
            or candidates.ndim != 3 or candidates.shape[:2] != projected_hidden.shape[:2]
            or not 1 <= candidates.shape[-1] <= 16 or unary_logits.shape != candidates.shape
            or anchors.shape != (projected_hidden.shape[0],)
            or candidates.dtype != torch.int64 or anchors.dtype != torch.int64
            or predecessor_codes.ndim != 2 or successor_codes.shape != predecessor_codes.shape):
        raise ValueError('Bounded CPU selector geometry and global INT64 IDs required')
    vocabulary = predecessor_codes.shape[0]
    if any(torch.any(value < 0) or torch.any(value >= vocabulary) for value in (candidates, anchors)):
        raise ValueError('Selector token IDs outside vocabulary')
    identifiers, inverse = torch.unique(torch.cat((anchors.flatten(), candidates.flatten())),
        sorted=True, return_inverse=True)
    local_anchors = inverse[:anchors.numel()].reshape(anchors.shape)
    local_candidates = inverse[anchors.numel():].reshape(candidates.shape)
    predecessors, successors = predecessor_codes[identifiers], successor_codes[identifiers]
    batch, positions, rank = projected_hidden.shape
    validate_selector_operands(projected_hidden.reshape(batch * positions, 1, rank),
        local_candidates.reshape(batch * positions, 1, -1), unary_logits.reshape(batch * positions, 1, -1),
        predecessors, successors, local_anchors.repeat_interleave(positions))
    hidden, unary = projected_hidden.double(), unary_logits.double()
    predecessors, successors = predecessors.double(), successors.double()
    previous = local_anchors
    path, score_rows = [], []
    for position in range(positions):
        edges = (predecessors[previous, None, :] * hidden[:, position, None, :]
            * successors[local_candidates[:, position]]).sum(dim=-1)
        scores = unary[:, position] + edges
        if not torch.isfinite(scores).all():
            raise ValueError('Selector score arithmetic overflowed')
        selected = scores.argmax(dim=-1, keepdim=True)
        previous = local_candidates[:, position].gather(1, selected).squeeze(1)
        path.append(previous)
        score_rows.append(scores)
    return identifiers[torch.stack(path, dim=1)], torch.stack(score_rows, dim=1)
