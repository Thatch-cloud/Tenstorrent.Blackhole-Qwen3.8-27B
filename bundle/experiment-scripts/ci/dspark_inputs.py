"""Published B1 DSpark serving boundary: sample the anchor query row, unlike the DFlash2 adapter."""


PROPOSALS = 7
VOCABULARY = 248320
MASK_TOKEN = 248070
MAX_POSITIONS = 262144


def query_inputs(anchor, prefix_tokens):
    import torch

    if (type(anchor) is not int or not 0 <= anchor < VOCABULARY or type(prefix_tokens) is not int
            or not 1 <= prefix_tokens <= MAX_POSITIONS - PROPOSALS - 1):
        raise ValueError('Valid anchor and room for the complete anchor-plus-seven verification block required')
    identifiers = torch.full((1, PROPOSALS), MASK_TOKEN, dtype=torch.int64)
    identifiers[0, 0] = anchor
    return dict(identifiers=identifiers,
        query_positions=torch.arange(prefix_tokens, prefix_tokens + PROPOSALS, dtype=torch.int64)[None],
        verifier_positions=torch.arange(prefix_tokens, prefix_tokens + PROPOSALS + 1, dtype=torch.int64)[None],
        first_sampled_query_row=0)


def verifier_inputs(anchor, proposals):
    import torch

    if (type(anchor) is not int or not 0 <= anchor < VOCABULARY or not isinstance(proposals, torch.Tensor)
            or proposals.device.type != 'cpu' or proposals.dtype != torch.int64
            or tuple(proposals.shape) != (1, PROPOSALS)
            or bool((proposals < 0).any()) or bool((proposals >= VOCABULARY).any())):
        raise ValueError('All seven sampled query rows and a valid anchor required; do not discard row zero')
    return torch.cat((torch.tensor([[anchor]], dtype=torch.int64), proposals), dim=1)
