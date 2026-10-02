"""A plain-torch DFlash2 draft model (no transformers): the parameter names and arithmetic of the z-lab DFlash2DraftModel
(`incoai` checkpoint layout), written so that the A0 screen's walker and the fine-tune's training reference share ONE forward and
can be tested on CPU without a model download.

What it mirrors (z-lab model.py, read against SpecForge's DFlash2 draft):
  * a context-then-block key layout: every layer projects the target features (fc over the concatenated taps, then hidden_norm)
    to K and V, concatenates the block rows' own K and V, and the block rows (anchor + mask rows) are the queries;
  * Qwen3 attention pieces: q_norm / k_norm per head, rotary at ABSOLUTE positions, GQA, SiLU MLP, RMSNorm;
  * a grouped dynamic causal convolution around the attention and the MLP of each layer, over the rows of ONE block;
  * the candidate selector: strict top-k of the unary logits, then a greedy path with a low-rank predecessor / successor score.

Two block rules exist in the sources and they differ: the inference model (z-lab, `is_causal` false) lets the block rows see each
other both ways inside the sliding window (this file's `block_mask`); the training mask of SpecForge for a sliding layer is causal
inside a block (ft_dflash2_train builds it, with the rule as an argument so the difference stays visible).

The context's K and V depend only on the target features of the context rows (never on the draft's own depth), so they can be
computed once and reused: `context_kv` returns them, `forward(..., ctx_kv=...)` takes them. The walker's incremental ingest relies
on this and the tests pin that cached equals recomputed.
"""
import math
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

Config = namedtuple('Config', 'hidden intermediate layers heads kv_heads head_dim vocab eps rope_theta window block conv_taps '
                              'conv_group selector_rank selector_top_k mask_token tap_ids yarn')
Config.__new__.__defaults__ = (None,)       # yarn: None = plain rope; else (factor, beta_fast, beta_slow, original_max_position)


def tiny_config(hidden=64, layers=2, heads=4, kv_heads=2, head_dim=16, vocab=1024, window=32, block=8, taps=3, selector_rank=8,
                top_k=4, intermediate=96, conv_group=16, conv_taps=2, mask_token=1023):
    """A small config of the same shape as the real one (hidden 5120, 5 layers, 32 / 8 heads of 128, vocab 248320)."""
    return Config(hidden=hidden, intermediate=intermediate, layers=layers, heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                  vocab=vocab, eps=1e-6, rope_theta=10000.0, window=window, block=block, conv_taps=conv_taps,
                  conv_group=conv_group, selector_rank=selector_rank, selector_top_k=top_k, mask_token=mask_token,
                  tap_ids=tuple(range(taps)))


def real_dspark_config():
    """DSpark v2's backbone: the same body with full attention, YaRN rope and no convolutions and no selector (its Markov head
    lives outside, in a0_drafters)."""
    return Config(hidden=5120, intermediate=17408, layers=5, heads=32, kv_heads=8, head_dim=128, vocab=248320, eps=1e-6,
                  rope_theta=1e7, window=None, block=7, conv_taps=0, conv_group=16, selector_rank=0, selector_top_k=0,
                  mask_token=248070, tap_ids=(5, 19, 33, 47, 61), yarn=(32.0, 32.0, 1.0, 8192))


def real_config():
    return Config(hidden=5120, intermediate=17408, layers=5, heads=32, kv_heads=8, head_dim=128, vocab=248320, eps=1e-6,
                  rope_theta=1e7, window=2048, block=8, conv_taps=2, conv_group=16, selector_rank=256, selector_top_k=16,
                  mask_token=248070, tap_ids=(5, 19, 33, 47, 61))


class RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value):
        dtype = value.dtype
        value = value.to(torch.float32)
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(dtype)


def rotate_half(value):
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def yarn_inverse_frequency(head_dim, theta, factor, beta_fast, beta_slow, original):
    """YaRN's blended inverse frequencies and its attention scaling (the pinned DSpark checkpoint's tables, written as
    dspark_rope_tables.DSparkRotary writes them)."""
    def correction(rotations):
        return head_dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(correction(beta_fast)), 0)
    high = min(math.ceil(correction(beta_slow)), head_dim - 1)
    ramp = ((torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation = 1 - ramp
    wavelengths = theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    return (1 / (factor * wavelengths)) * (1 - extrapolation) + (1 / wavelengths) * extrapolation, 1 + 0.1 * math.log(factor)


def rotary(positions, head_dim, theta, dtype, yarn=None):
    """cos, sin of shape [B, L, head_dim] for absolute `positions` [B, L]: the default unscaled rope, or YaRN `yarn` =
    (factor, beta_fast, beta_slow, original_max_position) with its attention scaling folded into the tables."""
    scaling = 1.0
    if yarn is None:
        inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    else:
        inverse, scaling = yarn_inverse_frequency(head_dim, theta, *yarn)
    freqs = positions.to(torch.float32)[..., None] * inverse
    both = torch.cat((freqs, freqs), dim=-1)
    return (both.cos() * scaling).to(dtype), (both.sin() * scaling).to(dtype)


def apply_rope(value, cos, sin):
    """value [B, H, L, D]; cos / sin [B, L, D]."""
    return value * cos[:, None] + rotate_half(value) * sin[:, None]


def repeat_kv(value, groups):
    if groups == 1:
        return value
    batch, heads, length, dim = value.shape
    return value[:, :, None].expand(batch, heads, groups, length, dim).reshape(batch, heads * groups, length, dim)


def window_mask(query_positions, key_positions, window):
    """Boolean visibility [B, 1, Q, K] from absolute positions: |q - k| < window (both ways: the model is not causal)."""
    delta = query_positions[:, :, None] - key_positions[:, None, :]
    visible = torch.ones_like(delta, dtype=torch.bool)
    if window is not None:
        visible = visible & (delta < window) & (-delta < window)
    return visible[:, None]


def grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """hidden [B, L, H]; dynamic [B, L, taps, groups]; base [taps, H]. Causal along L (zero padding in front)."""
    batch, length, size = hidden.shape
    groups = size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        kernel = base[offset].reshape(1, 1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.reshape(batch, length, size)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden, taps, group):
        super().__init__()
        self.taps, self.group = taps, group
        base = torch.zeros(2, taps, hidden)
        base[:, 0] = 1.0                              # identity at initialisation (SpecForge)
        self.base_kernel = nn.Parameter(base)
        self.kernel_projection = nn.Linear(hidden, 2 * taps * (hidden // group), bias=False)

    def prepare(self, hidden, block=None):
        batch, length, size = hidden.shape
        groups = size // self.group
        dynamic = self.kernel_projection(hidden).reshape(batch, length, 2, self.taps, groups)
        pre, post = dynamic[:, :, 0], dynamic[:, :, 1]
        return self._run(hidden, pre, self.base_kernel[0], block), post

    def finish(self, hidden, post, block=None):
        return self._run(hidden, post, self.base_kernel[1], block)

    def _run(self, hidden, dynamic, base, block):
        if block is None:
            return grouped_dynamic_convolve(hidden, dynamic, base, self.group)
        batch, length, size = hidden.shape
        if length % block:
            raise ValueError('the rows must be whole blocks')
        blocks = length // block
        out = grouped_dynamic_convolve(hidden.reshape(batch * blocks, block, size),
                                       dynamic.reshape(batch * blocks, block, self.taps, -1), base, self.group)
        return out.reshape(batch, length, size)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden, cfg.heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.heads * cfg.head_dim, cfg.hidden, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps)

    def context_kv(self, target_hidden, positions):
        """K (normed, roped) and V of context rows: [B, KVH, C, D]."""
        cfg = self.cfg
        batch, length = target_hidden.shape[:2]
        k = self.k_norm(self.k_proj(target_hidden).reshape(batch, length, cfg.kv_heads, cfg.head_dim)).transpose(1, 2)
        v = self.v_proj(target_hidden).reshape(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        cos, sin = rotary(positions, cfg.head_dim, cfg.rope_theta, k.dtype, cfg.yarn)
        return apply_rope(k, cos, sin), v

    def forward(self, hidden, positions, ctx_k, ctx_v, mask):
        cfg = self.cfg
        batch, length = hidden.shape[:2]
        q = self.q_norm(self.q_proj(hidden).reshape(batch, length, cfg.heads, cfg.head_dim)).transpose(1, 2)
        k_new = self.k_norm(self.k_proj(hidden).reshape(batch, length, cfg.kv_heads, cfg.head_dim)).transpose(1, 2)
        v_new = self.v_proj(hidden).reshape(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        cos, sin = rotary(positions, cfg.head_dim, cfg.rope_theta, q.dtype, cfg.yarn)
        q, k_new = apply_rope(q, cos, sin), apply_rope(k_new, cos, sin)
        k = torch.cat([ctx_k, k_new], dim=2)
        v = torch.cat([ctx_v, v_new], dim=2)
        groups = cfg.heads // cfg.kv_heads
        out = F.scaled_dot_product_attention(q, repeat_kv(k, groups), repeat_kv(v, groups), attn_mask=mask,
                                             scale=cfg.head_dim ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden, cfg.eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden, cfg.eps)
        self.attention_conv = GroupedDynamicCausalConv(cfg.hidden, cfg.conv_taps, cfg.conv_group) if cfg.conv_taps else None
        self.mlp_conv = GroupedDynamicCausalConv(cfg.hidden, cfg.conv_taps, cfg.conv_group) if cfg.conv_taps else None

    def forward(self, hidden, positions, ctx_k, ctx_v, mask, conv_block):
        residual = hidden
        value = self.input_layernorm(hidden)
        post = None
        if self.attention_conv is not None:
            value, post = self.attention_conv.prepare(value, conv_block)
        value = self.self_attn(value, positions, ctx_k, ctx_v, mask)
        hidden = residual + (self.attention_conv.finish(value, post, conv_block) if post is not None else value)
        residual = hidden
        value = self.post_attention_layernorm(hidden)
        post = None
        if self.mlp_conv is not None:
            value, post = self.mlp_conv.prepare(value, conv_block)
        value = self.mlp(value)
        return residual + (self.mlp_conv.finish(value, post, conv_block) if post is not None else value)


class CandidateSelector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.selector_top_k
        self.predecessor_codebook = nn.Parameter(torch.zeros(cfg.vocab, cfg.selector_rank))
        self.successor_codebook = nn.Parameter(torch.zeros(cfg.vocab, cfg.selector_rank))
        self.hidden_projection = nn.Linear(cfg.hidden, cfg.selector_rank, bias=False)

    def score_candidates(self, candidate_ids, unary_logits, hidden, predecessor_ids):
        """Unary logits plus the low-rank predecessor-to-candidate transition (SpecForge's score_candidates)."""
        predecessor = self.predecessor_codebook[predecessor_ids]
        successor = self.successor_codebook[candidate_ids]
        context = predecessor * self.hidden_projection(hidden)
        return unary_logits + torch.einsum('...r,...kr->...k', context, successor)

    def greedy_path(self, candidate_ids, unary_logits, hidden, anchor_ids):
        """candidate_ids / unary_logits [B, P, K]; hidden [B, P, H]; anchor_ids [B]. -> proposals [B, P]."""
        predecessor, path = anchor_ids, []
        for position in range(hidden.shape[1]):
            scores = self.score_candidates(candidate_ids[:, position], unary_logits[:, position], hidden[:, position], predecessor)
            chosen = scores.argmax(dim=-1, keepdim=True)
            predecessor = candidate_ids[:, position].gather(1, chosen)[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1)


class Dflash2(nn.Module):
    """The draft backbone and selector. `embed` and the LM head belong to the TARGET and are passed in where needed."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.eps)
        self.fc = nn.Linear(len(cfg.tap_ids) * cfg.hidden, cfg.hidden, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.candidate_selector = CandidateSelector(cfg) if cfg.selector_rank else None

    # -- context ---------------------------------------------------------------------------------------------------------
    def context_features(self, raw_rows):
        """[B, C, taps * H] raw tap rows -> [B, C, H]: hidden_norm(fc(rows))."""
        return self.hidden_norm(self.fc(raw_rows))

    def context_kv(self, raw_rows, positions):
        """Per layer (K, V) of the context rows [B, C, taps * H] at absolute `positions` [B, C]."""
        target = self.context_features(raw_rows)
        return [layer.self_attn.context_kv(target, positions) for layer in self.layers]

    # -- the block -------------------------------------------------------------------------------------------------------------
    def block_mask(self, query_positions, ctx_positions):
        """Inference visibility of the block rows over [context; block]: windowed, block rows see each other both ways."""
        key_positions = torch.cat([ctx_positions, query_positions], dim=1)
        return window_mask(query_positions, key_positions, self.cfg.window)

    def forward(self, noise_embedding, positions, raw_rows=None, ctx_positions=None, ctx_kv=None, attention_mask=None,
                conv_block=None):
        """noise_embedding [B, Q, H]; positions [B, Q] absolute; the context as `raw_rows` [B, C, taps*H] with `ctx_positions`
        [B, C], or as a precomputed `ctx_kv`. `attention_mask` [B, 1, Q, C+Q] overrides the inference rule (training passes
        the packed-anchor mask); `conv_block` makes the convolutions stay inside blocks of that many rows. -> [B, Q, H]."""
        if ctx_kv is None:
            ctx_kv = self.context_kv(raw_rows, ctx_positions)
        if attention_mask is None:
            attention_mask = self.block_mask(positions, ctx_positions)
        hidden = noise_embedding
        for layer, (k, v) in zip(self.layers, ctx_kv):
            hidden = layer(hidden, positions, k, v, attention_mask, conv_block if conv_block is not None else hidden.shape[1])
        return self.norm(hidden)

    def propose(self, hidden, anchor_ids, lm_head_weight):
        """hidden [B, P, H] (the block's rows after the anchor row) -> proposals [B, P] by the selector's greedy path."""
        logits = hidden @ lm_head_weight.T
        unary, candidates = torch.topk(logits, self.candidate_selector.top_k, dim=-1, sorted=False)
        return self.candidate_selector.greedy_path(candidates, unary, hidden, anchor_ids)


def init_random(model, seed=0, scale=0.05):
    """Deterministic non-trivial weights for tests (the real ones come from a checkpoint)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if name.endswith('norm.weight') or name.endswith('layernorm.weight'):
                parameter.copy_(1.0 + 0.1 * torch.randn(parameter.shape, generator=generator))
            elif name.endswith('base_kernel'):
                base = torch.zeros_like(parameter)
                base[:, 0] = 1.0
                parameter.copy_(base + 0.1 * torch.randn(parameter.shape, generator=generator))
            else:
                parameter.copy_(scale * torch.randn(parameter.shape, generator=generator))
    return model


def parameter_names(layers):
    """The checkpoint tensor names of a draft with `layers` layers (the export contract; widths do not change names)."""
    return sorted(Dflash2(tiny_config(layers=layers)).state_dict())
