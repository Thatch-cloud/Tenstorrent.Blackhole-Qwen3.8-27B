"""Parallelizable DFlash2 transition-score oracle; FP64 math, not device or speed certification."""

from draft_selector import validate_selector_operands


def transition_scores_reference(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    import torch

    validate_selector_operands(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
    batch, positions, count = candidates.shape
    predecessor_ids = torch.cat((anchors[:, None, None].expand(batch, 1, count), candidates[:, :-1]), dim=1)
    predecessor = predecessor_codes[predecessor_ids].double()
    successor = successor_codes[candidates].double()
    weighted_predecessor = predecessor * projected_hidden.double()[:, :, None, :]
    edges = torch.matmul(weighted_predecessor, successor.transpose(-1, -2))
    scores = edges + unary_logits.double()[:, :, None, :]
    if not torch.isfinite(scores).all():
        raise ValueError('Transition score arithmetic overflowed')
    return scores


def select_transition_scores(scores, candidates):
    import torch

    if (scores.device.type != 'cpu' or candidates.device.type != 'cpu' or candidates.dtype != torch.int64
            or candidates.ndim != 3 or not torch.is_floating_point(scores)):
        raise ValueError('CPU floating transitions and INT64 candidate IDs required')
    batch, positions, count = candidates.shape
    if (batch < 1 or not 1 <= positions <= 8 or not 1 <= count <= 16
            or scores.shape != (batch, positions, count, count)
            or not torch.isfinite(scores).all() or torch.any(candidates < 0)):
        raise ValueError('Finite bounded candidate transition matrix required')
    sorted_candidates = candidates.sort(dim=-1).values
    if torch.any(sorted_candidates[..., 1:] == sorted_candidates[..., :-1]):
        raise ValueError('Unique candidates per position required')
    predecessor_index = torch.zeros(batch, dtype=torch.int64)
    batch_indices = torch.arange(batch)
    path, selected_scores = [], []
    for position in range(positions):
        row = scores[batch_indices, position, predecessor_index]
        predecessor_index = row.argmax(dim=-1)
        path.append(candidates[batch_indices, position, predecessor_index])
        selected_scores.append(row)
    return torch.stack(path, dim=1), torch.stack(selected_scores, dim=1)
