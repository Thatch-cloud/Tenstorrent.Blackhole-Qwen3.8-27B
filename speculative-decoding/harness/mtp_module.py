"""Existing Qwen36MTP implementation reused from the speculative-decoding branch."""

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

        self._nw = {
            n: replicate_to_device(mesh_device, (1.0 + sd[n].float()).to(torch.bfloat16).reshape(1, 1, 1, -1))
            for n in NORM_NAMES
        }
        self.w_fc = replicate_to_device(
            mesh_device, sd["mtp.fc.weight"].T.contiguous().reshape(1, 1, 2 * self.d, self.d))

        attn_sd = {k[len(PREFIX + "self_attn."):]: v for k, v in sd.items() if k.startswith(PREFIX + "self_attn.")}
        mlp_sd = {k[len(PREFIX + "mlp."):]: v for k, v in sd.items() if k.startswith(PREFIX + "mlp.")}
        self.attention = TPAttention(mesh_device, args, load_attention_weights_tp(mesh_device, attn_sd, args), tt_ccl)
        self.feed_forward = Qwen36MLP(mesh_device, mlp_sd, None, args=args, tt_ccl=tt_ccl)

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

    def _gather(self, t):
        """TPAttention and Qwen36MLP reduce-scatter their output; the residual stream here is
        replicated, so gather rather than fracturing the stream. Two all-gathers on a layer that
        runs once per step."""
        if self.nd == 1:
            return t
        from models.tt_transformers.tt.ccl import tt_all_gather

        return tt_all_gather(t, self.mesh, self.tt_ccl, cluster_axis=None,
                             dim=-1, topology=self.args.ccl_topology())

    def forward(self, emb_tt, hidden_tt, cur_pos_tt, cos, sin, page_table=None):
        """One proposal step. Returns the pre-lm_head hidden state for the drafted position."""
        n = lambda k: ttnn.rms_norm(k[0], weight=self._nw[k[1]], epsilon=self.eps)
        e_n = n((emb_tt, "mtp.pre_fc_norm_embedding.weight"))
        h_n = n((hidden_tt, "mtp.pre_fc_norm_hidden.weight"))
        x = ttnn.matmul(ttnn.concat([e_n, h_n], dim=-1), self.w_fc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        a_in = n((x, PREFIX + "input_layernorm.weight"))
        x = ttnn.add(x, self._gather(self.attention.forward_decode(a_in, cur_pos_tt, cos, sin, page_table=page_table)))
        f_in = n((x, PREFIX + "post_attention_layernorm.weight"))
        x = ttnn.add(x, self._gather(self.feed_forward.forward(f_in)))
        return n((x, "mtp.norm.weight"))
