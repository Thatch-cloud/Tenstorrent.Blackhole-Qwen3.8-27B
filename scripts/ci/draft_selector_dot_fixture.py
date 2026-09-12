"""Host-prepared learned codebook dot fixture; candidates and hidden states are synthetic."""

from draft_selector_transitions import transition_scores_reference, select_transition_scores


def prepare_selector_dot(weights):
    import torch

    predecessor = weights['candidate_selector.predecessor_codebook']
    successor = weights['candidate_selector.successor_codebook']
    if (predecessor.shape != (248320, 256) or successor.shape != predecessor.shape
            or any(value.dtype != torch.bfloat16 or value.device.type != 'cpu' for value in (predecessor, successor))):
        raise ValueError('Pinned selector codebook geometry required')
    generator = torch.Generator().manual_seed(38256)
    candidates = torch.stack([torch.randperm(248320, generator=generator)[:16] for _ in range(7)])[None]
    anchors = torch.tensor([1596])
    hidden = torch.randn((1, 7, 256), generator=generator).bfloat16()
    unary = torch.randn((1, 7, 16), generator=generator).bfloat16()
    predecessor_ids = torch.cat((anchors[:, None, None].expand(1, 1, 16), candidates[:, :-1]), dim=1)
    left, right = [torch.zeros((1, 16, 32, 256), dtype=torch.float32) for _ in range(2)]
    left[:, :7, :16] = predecessor[predecessor_ids].float() * hidden.float()[:, :, None, :]
    right[:, :7, :16] = successor[candidates].float()
    transitions = transition_scores_reference(hidden, candidates, unary, predecessor, successor, anchors)
    path, _ = select_transition_scores(transitions, candidates)
    return dict(left=left, right=right, candidates=candidates, unary=unary.float(),
                expected_scores=transitions, expected_path=path)
