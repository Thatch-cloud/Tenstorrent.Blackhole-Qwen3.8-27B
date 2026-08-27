# SPDX-License-Identifier: Apache-2.0
"""K3 — the Qwen3.6/3.8 MTP module as a reusable class, with paged KV.

Packages what milestones 1-3 validated piecemeal (fusion PCC 0.9999, module 0.9974, sequence
with KV 0.9941) and adds the piece the model integration needs: its own **paged** KV pair,
allocated alongside the target's and driven by the *same* page table vLLM already passes to
`decode_forward`. That is the contract settled on tenstorrent/vllm#442 -- same block table, same
block ids, no new KV cache group, just one more layer over the same pool (~+6% KV).

The MTP decoder layer is shape-identical to a target full-attention layer, so `TPAttention` and
`Qwen36MLP` carry it unmodified; only the input fusion is new:

    x = fc(concat([pre_fc_norm_embedding(embed(tok)), pre_fc_norm_hidden(hidden)], -1))

Conventions pinned from artefacts rather than assumed, both with negative controls in the
accompanying tests: concat order is embedding-first (`qwen3_5_mtp.py`), and the norms are
zero-centered so weights are pre-offset by +1 (`tt/rms_norm.py`).
"""
import json
from pathlib import Path

import torch

import ttnn

PREFIX = "mtp.layers.0."
NAMES = [
    "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight", PREFIX + "input_layernorm.weight", PREFIX + "post_attention_layernorm.weight",
] + [PREFIX + f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")] \
  + [PREFIX + f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]


# Listed explicitly, NOT pattern-matched: `pre_fc_norm_embedding.weight` and
# `pre_fc_norm_hidden.weight` do not end in "norm.weight", so an endswith filter silently drops
# the two norms the fusion block needs -- which is exactly how this was first written.
NORM_NAMES = [
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    PREFIX + "input_layernorm.weight",
    PREFIX + "post_attention_layernorm.weight",
]


def load_mtp_weights(ckpt_dir):
    """Exact-name loader.

    NOT `load_layer_weights`: it matches with `k.endswith(f"layers.{i}.{leaf}")`, which is
    ambiguous against `mtp.layers.0.*` -- for the MLP it would silently return the *target's*
    layer 0. This is also why `weight_mapping.py` keeps the `mtp.` prefix rather than remapping.
    """
    from safetensors import safe_open

    ckpt_dir = Path(ckpt_dir)
    wm = json.load(open(ckpt_dir / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for n in NAMES:
        assert n in wm, f"{n} missing from checkpoint index"
        with safe_open(str(ckpt_dir / wm[n]), framework="pt") as sf:
            out[n] = sf.get_tensor(n).to(torch.bfloat16)
    return out


class Qwen36MTP:
    """MTP proposer head: fusion block + one decoder layer + final norm."""

    def __init__(self, mesh_device, args, sd, tt_ccl):
        from models.demos.blackhole.qwen36.tests.test_factory import replicate_to_device
        from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
        from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP

        self.mesh, self.args, self.tt_ccl = mesh_device, args, tt_ccl
        self.nd, self.eps, self.d = mesh_device.get_num_devices(), args.norm_eps, args.dim
        self._rep = replicate_to_device

        # zero-centered RMSNorm: pre-offset the weights by +1 (tt/rms_norm.py)
        self._nw = {
            n: replicate_to_device(mesh_device, (1.0 + sd[n].float()).to(torch.bfloat16).reshape(1, 1, 1, -1))
            for n in NORM_NAMES
        }
        # fc replicated rather than column-parallel: TT decode activations are replicated, so this
        # keeps the block replicated end to end and needs no gather. ~52 MiB/device vs ~26.
        self.w_fc = replicate_to_device(
            mesh_device, sd["mtp.fc.weight"].T.contiguous().reshape(1, 1, 2 * self.d, self.d))

        attn_sd = {k[len(PREFIX + "self_attn."):]: v for k, v in sd.items() if k.startswith(PREFIX + "self_attn.")}
        mlp_sd = {k[len(PREFIX + "mlp."):]: v for k, v in sd.items() if k.startswith(PREFIX + "mlp.")}
        self.attention = TPAttention(mesh_device, args, load_attention_weights_tp(mesh_device, attn_sd, args), tt_ccl)
        self.feed_forward = Qwen36MLP(mesh_device, mlp_sd, None, args=args, tt_ccl=tt_ccl)

    # -- KV ---------------------------------------------------------------------------------
    def allocate_kv(self, kv_cache_shape, dtype):
        """Allocate this layer's paged KV pair and bind it.

        Call after the target's 16, so vLLM sees 17 layers over the same block pool. The pair is
        allocated exactly as `_allocate_kv_caches_tp` does for the target's layers.
        """
        def _mk():
            return ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        k, v = _mk(), _mk()
        self.attention.set_paged_kv_cache(k, v)
        return [k, v]

    # -- forward ----------------------------------------------------------------------------
    def _gather(self, t):
        """TPAttention and Qwen36MLP reduce-scatter their output; the residual stream here is
        replicated, so gather rather than fracturing the stream. Two all-gathers on a layer that
        runs once per step."""
        if self.nd == 1:
            return t
        from models.tt_transformers.tt.ccl import tt_all_gather

        # dim=-1, not len(shape)-1: the hidden state arriving from the model can be rank 3 while
        # test inputs are rank 4, and a positive index computed from the wrong rank fails with
        # "Dimension input should be in between -3 and 2, but has 3". Negative is rank-agnostic.
        return tt_all_gather(t, self.mesh, self.tt_ccl, cluster_axis=None,
                             dim=-1, topology=self.args.ccl_topology())

    def forward(self, emb_tt, hidden_tt, cur_pos_tt, cos, sin, page_table=None):
        """One proposal step. Returns the pre-lm_head hidden state for the drafted position."""
        n = lambda k: ttnn.rms_norm(k[0], weight=self._nw[k[1]], epsilon=self.eps)  # noqa: E731
        e_n = n((emb_tt, "mtp.pre_fc_norm_embedding.weight"))
        h_n = n((hidden_tt, "mtp.pre_fc_norm_hidden.weight"))
        x = ttnn.matmul(ttnn.concat([e_n, h_n], dim=-1), self.w_fc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        a_in = n((x, PREFIX + "input_layernorm.weight"))
        x = ttnn.add(x, self._gather(self.attention.forward_decode(a_in, cur_pos_tt, cos, sin, page_table=page_table)))
        f_in = n((x, PREFIX + "post_attention_layernorm.weight"))
        x = ttnn.add(x, self._gather(self.feed_forward.forward(f_in)))
        return n((x, "mtp.norm.weight"))

    def propose(self, emb_tt, hidden_tt, cur_pos_tt, cos, sin, lm_head, page_table=None):
        """Propose one draft, returning **logits** rather than a token.

        `lm_head` is the target's own head: MTP shares it and the embedding table
        (`mtp_use_dedicated_embeddings = False`), so the head adds no vocab-sized weights.

        Logits rather than a sampled token, deliberately. Sampling policy belongs to the caller,
        and at TP=2 it must be on host anyway -- 248320/2 = 124,160 logits/device exceeds the
        65,536 on-device ceiling and `(1,2)` is not a certified topology (SBLK-4). It also
        matches the `return_verify_logits` shape discussed on #442.
        """
        h = self.forward(emb_tt, hidden_tt, cur_pos_tt, cos, sin, page_table=page_table)
        return lm_head(h)
