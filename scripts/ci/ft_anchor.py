"""WP-F1: the host-side anchor pipeline of DFlash-family training (the SpecForge recipe), in torch on CPU: which anchors a sequence
trains on, the noise rows and position ids of its blocks, the labels and weights, and the dense visibility mask.

ANCHORS. A position i is a valid anchor when the clean token at i and the target at i + 1 are both supervised
(loss_mask[i] and loss_mask[i + 1]). Up to `num_anchors` (512) valid positions are drawn without replacement (the draw is
`rand` over positions with the invalid ones pushed to 2.0, then the smallest, so a given random tensor gives a given draw), sorted
ascending. A block is the anchor token followed by block_size - 1 mask tokens; block row k predicts the token at anchor + k.

MASK. Q = anchors x block_size rows, KV = S context rows then the same Q block rows. A block row sees the context rows strictly
before ITS ANCHOR (and, with a sliding window W, no further back than anchor + offset - (W - 1)), and the rows of its own block. The
block rule says which rows of its own block: 'specforge' is the training mask of the recipe (with a sliding window the block is
CAUSAL, row k sees block rows 0..k; without a window it is bidirectional); 'full' is bidirectional inside the block, which is what
the inference model does (z-lab, is_causal false). The two differ for every sliding-window draft: that difference is a finding to
confirm against the serving drafter, not a choice made here.

LABELS. target_ids[b, n, k] = input_ids[anchor_n + k]; predecessor_ids shifts them right by one with the anchor's own token in front
(the selector conditions on the previous TRUE token during training); weight_mask = keep x label in range x (k > 0) x loss_mask at the
label.
"""
import torch

BLOCK_RULES = ('specforge', 'full')


def sample_anchors(loss_mask, num_anchors, random_values=None, generator=None):
    """loss_mask [B, S] -> (anchors [B, W], keep [B, W]) with W = min(num_anchors, the most valid anchors of any sequence). Invalid
    slots hold anchor 0 and keep False. `random_values` [B, S - 1] (default: torch.rand from `generator`) is the draw."""
    batch, length = loss_mask.shape
    candidates = max(length - 1, 0)
    valid = (loss_mask[:, :candidates] > 0.5) & (loss_mask[:, 1:candidates + 1] > 0.5)
    counts = valid.sum(dim=1)
    width = min(num_anchors, int(counts.max().item()) if batch else 0)
    if width == 0:
        raise ValueError('training needs two consecutive supervised tokens')
    if random_values is None:
        random_values = torch.rand(valid.shape, generator=generator)
    draw = random_values.clone()
    draw.masked_fill_(~valid, 2.0)
    chosen = draw.argsort(dim=1)[:, :width]
    keep = torch.arange(width).unsqueeze(0) < counts.clamp(max=width).unsqueeze(1)
    sentinel = valid.shape[1]
    anchors = torch.where(keep, chosen, torch.full_like(chosen, sentinel)).sort(dim=1).values
    keep = anchors < sentinel
    return torch.where(keep, anchors, torch.zeros_like(anchors)), keep


def position_ids(anchors, block_size):
    """Absolute positions of the block rows: [B, N * block_size], anchor + 0 .. block_size - 1 per block."""
    offsets = torch.arange(block_size).view(1, 1, -1)
    return (anchors.unsqueeze(-1) + offsets).reshape(anchors.shape[0], -1)


def noise_ids(input_ids, anchors, keep, mask_token, block_size):
    """[B, N * block_size] token ids of the block rows: the anchor token first (the mask token for a dropped slot), mask tokens after."""
    batch, count = anchors.shape
    ids = torch.full((batch, count * block_size), mask_token, dtype=torch.long)
    clean = torch.gather(input_ids, 1, anchors.clamp(0, input_ids.shape[1] - 1))
    starts = torch.arange(count) * block_size
    rows = torch.arange(batch).unsqueeze(1).expand(batch, count)
    ids[rows, starts.unsqueeze(0).expand(batch, count)] = clean.masked_fill(~keep, mask_token)
    return ids


def block_labels(input_ids, loss_mask, anchors, keep, block_size):
    """(target_ids, predecessor_ids, weight_mask), each [B, N, block_size]."""
    batch, count = anchors.shape
    length = input_ids.shape[1]
    indices = anchors.unsqueeze(-1) + torch.arange(block_size).view(1, 1, -1)
    in_range = indices < length
    safe = indices.clamp(max=length - 1)
    expanded = input_ids.unsqueeze(1).expand(-1, count, -1)
    target = torch.gather(expanded, 2, safe)
    predecessor = torch.cat([target[:, :, :1], target[:, :, :-1]], dim=-1)
    weight = keep.unsqueeze(-1).expand(-1, -1, block_size).float() * in_range.float()
    weight = weight * (torch.arange(block_size).view(1, 1, -1) > 0).float()
    weight = weight * torch.gather(loss_mask.unsqueeze(1).expand(-1, count, -1), 2, safe).float()
    return target, predecessor, weight


def dense_mask(anchors, keep, length, block_size, sliding_window=None, block_rule='specforge'):
    """Boolean visibility [B, 1, Q, S + Q] (True = may attend), Q = N * block_size."""
    if block_rule not in BLOCK_RULES:
        raise ValueError('block_rule must be one of %s' % ', '.join(BLOCK_RULES))
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError('sliding_window must be > 0')
    batch, count = anchors.shape
    q_len, kv_len = count * block_size, length + count * block_size
    q = torch.arange(q_len).view(1, 1, -1, 1)
    kv = torch.arange(kv_len).view(1, 1, 1, -1)
    q_block, q_offset = q // block_size, q % block_size
    anchor = anchors.view(batch, 1, count, 1).repeat_interleave(block_size, dim=2)
    context = (kv < length) & (kv < anchor)
    if sliding_window is not None:
        context = context & (kv >= anchor + q_offset - (sliding_window - 1))
    own = (kv >= length) & (q_block == (kv - length) // block_size)
    if block_rule == 'specforge' and sliding_window is not None:
        own = own & ((kv - length) % block_size <= q_offset)
    valid = keep.view(batch, 1, count, 1).repeat_interleave(block_size, dim=2)
    return (context | own) & valid
