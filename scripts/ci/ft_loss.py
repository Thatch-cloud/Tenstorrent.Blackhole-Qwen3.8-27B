"""WP-F2: the DFlash2 training objective (SpecForge's Qwen3.8 recipe), in torch.

Per anchor block, row k > 0 predicts the TARGET's own token at anchor + k:
  * HARD CROSS-ENTROPY on the greedy tokens, weighted by a position decay exp(-(k - 1) / gamma) (gamma 7) and by the weight mask
    (anchor row, labels past the end and unsupervised labels carry no weight);
  * the SELECTOR cross-entropy over the strict top-k of the unary logits (k = 16): conditioned on the previous TRUE token, the
    candidate scores are the unary logits plus the low-rank predecessor / successor term, and the label is the candidate that equals
    the target. A target outside the top-k is a recall failure of the backbone, not a selector example: it carries no selector weight.
  loss = (sum(ce * w) + alpha * sum(selector_ce * w * covered)) / sum(w),   w = weight_mask * decay,   alpha 1.0.

`chunk_blocks` bounds the [blocks x block x vocab] logits tensor by summing the numerators and the denominator over chunks of blocks
(the sums, hence the loss and its gradient, equal the unchunked ones).
"""
import torch
from torch.nn import functional as F

GAMMA = 7.0
ALPHA = 1.0


def decay_weights(block_size, gamma=GAMMA):
    """[block_size] exp(-(k - 1).clamp(min=0) / gamma): rows 0 and 1 weigh 1."""
    positions = torch.arange(block_size).float()
    return torch.exp(-(positions - 1).clamp(min=0) / gamma)


def objective_terms(hidden, lm_head_weight, selector, target_ids, predecessor_ids, weight_mask, gamma=GAMMA):
    """The numerators and the denominator for one set of blocks. hidden [B, n, bs, H]; ids and mask [B, n, bs]."""
    batch, count, block, size = hidden.shape
    logits = (hidden.reshape(-1, size) @ lm_head_weight.T).float().reshape(batch, count, block, -1)
    neg_log_q = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction='none').reshape(target_ids.shape)
    weights = weight_mask * decay_weights(block, gamma).to(weight_mask.device).view(1, 1, -1)
    terms = dict(ce_num=(neg_log_q * weights).sum(), den=weights.sum(), selector_num=torch.zeros((), dtype=torch.float32),
                 covered_den=torch.zeros(()), correct=((logits.argmax(-1) == target_ids).float() * (weight_mask > 0.5).float()).sum(),
                 accuracy_den=(weight_mask > 0.5).float().sum())
    if selector is not None:
        unary, candidates = logits.topk(selector.top_k, dim=-1)
        matches = candidates.eq(target_ids.unsqueeze(-1))
        covered = matches.any(dim=-1)
        index = matches.long().argmax(dim=-1)
        scores = selector.score_candidates(candidate_ids=candidates, unary_logits=unary, hidden=hidden, predecessor_ids=predecessor_ids)
        selector_ce = F.cross_entropy(scores.float().reshape(-1, scores.shape[-1]), index.reshape(-1), reduction='none').reshape(target_ids.shape)
        selector_weights = weights * covered.float()
        terms['selector_num'] = (selector_ce * selector_weights).sum()
        terms['covered_den'] = selector_weights.sum()
    return terms


def dflash_loss(hidden, lm_head_weight, selector, target_ids, predecessor_ids, weight_mask, gamma=GAMMA, alpha=ALPHA, chunk_blocks=None):
    """-> (loss, terms). `selector` None trains the unary objective only."""
    count = hidden.shape[1]
    step = chunk_blocks or count
    total = None
    for start in range(0, count, step):
        part = objective_terms(hidden[:, start:start + step], lm_head_weight, selector, target_ids[:, start:start + step],
                               predecessor_ids[:, start:start + step], weight_mask[:, start:start + step], gamma)
        total = part if total is None else dict((name, total[name] + part[name]) for name in total)
    numerator = total['ce_num'] + alpha * total['selector_num']
    loss = numerator / total['den'].clamp_min(torch.finfo(total['den'].dtype).tiny)
    return loss, total
