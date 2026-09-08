"""Ideal FP64 greedy selector oracle; not a BF16 device-equivalence or coding-quality gate."""


def validate_selector_operands(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    import torch

    operands = (projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
    if any(value.device.type != 'cpu' for value in operands):
        raise ValueError('CPU selector operands required')
    if projected_hidden.ndim != 3 or predecessor_codes.ndim != 2:
        raise ValueError('Batch/position/rank hidden and vocabulary/rank codebooks required')
    batch, positions, rank = projected_hidden.shape
    vocabulary = predecessor_codes.shape[0]
    if (batch < 1 or not 1 <= positions <= 8 or rank < 1 or vocabulary < 1
            or predecessor_codes.shape[1] != rank or successor_codes.shape != predecessor_codes.shape
            or candidates.ndim != 3 or candidates.shape[:2] != (batch, positions)
            or not 1 <= candidates.shape[2] <= min(16, vocabulary)
            or unary_logits.shape != candidates.shape or anchors.shape != (batch,)):
        raise ValueError('Matching bounded selector geometry required')
    if candidates.dtype != torch.int64 or anchors.dtype != torch.int64:
        raise ValueError('INT64 candidate and anchor token IDs required')
    if any(not torch.is_floating_point(value) or not torch.isfinite(value).all()
            for value in (projected_hidden, unary_logits, predecessor_codes, successor_codes)):
        raise ValueError('Finite floating selector operands required')
    if any(torch.any(value < 0) or torch.any(value >= vocabulary) for value in (candidates, anchors)):
        raise ValueError('Selector token IDs outside vocabulary')
    sorted_candidates = candidates.sort(dim=-1).values
    if torch.any(sorted_candidates[..., 1:] == sorted_candidates[..., :-1]):
        raise ValueError('Unique candidates per position required')


def greedy_selector_reference(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    import torch

    validate_selector_operands(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors)
    positions = projected_hidden.shape[1]
    hidden = projected_hidden.double()
    unary = unary_logits.double()
    predecessor_codes, successor_codes = predecessor_codes.double(), successor_codes.double()
    predecessor = anchors
    path, score_rows = [], []
    for position in range(positions):
        edges = (predecessor_codes[predecessor, None, :] * hidden[:, position, None, :]
            * successor_codes[candidates[:, position]]).sum(dim=-1)
        scores = unary[:, position] + edges
        if not torch.isfinite(scores).all():
            raise ValueError('Selector score arithmetic overflowed')
        selected = scores.argmax(dim=-1, keepdim=True)
        predecessor = candidates[:, position].gather(1, selected).squeeze(1)
        path.append(predecessor)
        score_rows.append(scores)
    return torch.stack(path, dim=1), torch.stack(score_rows, dim=1)


def select_active_candidates(projected_hidden, candidates, unary_logits, predecessor_codes, successor_codes, anchors):
    import torch

    if (candidates.dtype != torch.int64 or anchors.dtype != torch.int64
            or candidates.device.type != 'cpu' or anchors.device.type != 'cpu'
            or predecessor_codes.ndim != 2 or successor_codes.shape != predecessor_codes.shape
            or predecessor_codes.device.type != 'cpu' or successor_codes.device.type != 'cpu'):
        raise ValueError('CPU global candidate IDs and matching full codebooks required')
    if any(torch.any(value < 0) or torch.any(value >= predecessor_codes.shape[0]) for value in (candidates, anchors)):
        raise ValueError('Selector token IDs outside vocabulary')
    identifiers, inverse = torch.unique(torch.cat((anchors.flatten(), candidates.flatten())), sorted=True, return_inverse=True)
    local_anchors = inverse[:anchors.numel()].reshape(anchors.shape)
    local_candidates = inverse[anchors.numel():].reshape(candidates.shape)
    selected, scores = greedy_selector_reference(projected_hidden, local_candidates, unary_logits,
        predecessor_codes[identifiers], successor_codes[identifiers], local_anchors)
    return identifiers[selected], scores
