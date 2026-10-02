"""WP-F2: the DFlash2 packed-anchor training forward, over dflash2_torch's model and ft_anchor's host-side pipeline.

One sequence trains on Q = anchors x block rows at once: the queries are the blocks' rows (the anchor token then mask tokens, at
their absolute positions), the keys are the S context rows (the target's tap rows, fc and hidden_norm inside the model) followed by
the same Q block rows, and a dense mask says what each row may read (ft_anchor.dense_mask: the context before ITS anchor inside the
sliding window, plus its own block). The grouped dynamic convolution is applied per block (`conv_block`), so it never crosses a block
boundary. Embedding and LM head are the target's and frozen; fc and hidden_norm are frozen too in the default recipe (`freeze`).

The block rule defaults to the recipe's ('specforge': causal inside a block for a sliding-window draft). The serving model is
bidirectional inside the block ('full'). `train_forward(..., block_rule='full')` trains the way the inference model reads; the two
are kept apart so a train / serve mismatch stays visible instead of being absorbed.
"""
from collections import namedtuple

import torch

import ft_anchor as fa
import ft_loss

Batch = namedtuple('Batch', 'anchors keep noise_ids positions target_ids predecessor_ids weight_mask')
FROZEN_DEFAULT = ('fc.', 'hidden_norm.')


def make_batch(input_ids, loss_mask, cfg, num_anchors, generator=None, block=None, random_values=None):
    block = block or cfg.block
    anchors, keep = fa.sample_anchors(loss_mask, num_anchors, random_values=random_values, generator=generator)
    target, predecessor, weight = fa.block_labels(input_ids, loss_mask, anchors, keep, block)
    return Batch(anchors, keep, fa.noise_ids(input_ids, anchors, keep, cfg.mask_token, block), fa.position_ids(anchors, block),
                 target, predecessor, weight)


def training_mask(batch, length, block, window, block_rule):
    """The dense mask with every query row guaranteed one key (a dropped block would otherwise be an all-False row: NaN in softmax)."""
    mask = fa.dense_mask(batch.anchors, batch.keep, length, block, window, block_rule)
    queries = mask.shape[2]
    empty = ~mask.any(dim=-1)                                    # [B, 1, Q]
    own = torch.arange(queries)
    mask = mask.clone()
    mask[:, 0, own, length + own] |= empty[:, 0]
    return mask


def draft_hidden(model, embed_weight, raw_rows, batch, block_rule='specforge', window='model'):
    """The draft's final rows for every block: [B, N, block, H]."""
    cfg = model.cfg
    block = batch.target_ids.shape[2]
    length = raw_rows.shape[1]
    window = cfg.window if window == 'model' else window
    noise = embed_weight[batch.noise_ids]
    ctx_positions = torch.arange(length).unsqueeze(0).expand(raw_rows.shape[0], -1)
    mask = training_mask(batch, length, block, window, block_rule)
    hidden = model(noise, batch.positions, raw_rows, ctx_positions, attention_mask=mask, conv_block=block)
    return hidden.reshape(raw_rows.shape[0], batch.anchors.shape[1], block, -1)


def train_forward(model, embed_weight, lm_head_weight, raw_rows, input_ids, loss_mask, num_anchors, generator=None, gamma=ft_loss.GAMMA,
                  alpha=ft_loss.ALPHA, block=None, block_rule='specforge', chunk_blocks=None, batch=None):
    """-> (loss, terms, batch)."""
    batch = batch or make_batch(input_ids, loss_mask, model.cfg, num_anchors, generator, block)
    hidden = draft_hidden(model, embed_weight, raw_rows, batch, block_rule)
    loss, terms = ft_loss.dflash_loss(hidden, lm_head_weight, model.candidate_selector, batch.target_ids, batch.predecessor_ids,
                                      batch.weight_mask, gamma, alpha, chunk_blocks)
    return loss, terms, batch


def freeze(model, prefixes=FROZEN_DEFAULT):
    """requires_grad False for every parameter whose name starts with a prefix. -> the frozen names."""
    frozen = []
    for name, parameter in model.named_parameters():
        if name.startswith(tuple(prefixes)):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen
