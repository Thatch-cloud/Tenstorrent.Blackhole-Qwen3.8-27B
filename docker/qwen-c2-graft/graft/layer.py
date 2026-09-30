# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


# Lever N M2 prefill profile hook (QWEN_PREFILL_PROFILE_FLUSH=1); inert when unset.
_QWEN_PREFILL_PROFILE_FLAG = "QWEN_PREFILL_PROFILE_FLUSH"
_QWEN_PREFILL_PROFILE_EVERY = 16
_QWEN_PREFILL_PROFILE_SIGNPOST_CHUNKS = (0, 1, 31, 32, 62, 63)
_QWEN_PREFILL_PROFILE = {"chunk": -1, "prompt": 0, "calls": 0, "label": None, "flushes": 0, "signpost": None}


def _qwen_prefill_profile_on():
    import os

    return os.environ.get(_QWEN_PREFILL_PROFILE_FLAG) == "1"


def _qwen_prefill_signpost(label):
    state = _QWEN_PREFILL_PROFILE
    if state["signpost"] is None:
        try:
            from tracy import signpost
        except Exception as error:  # noqa: BLE001 - a missing tracy must not kill prefill
            from loguru import logger as _qwen_logger

            _qwen_logger.info("[PINDIAG] prefill profile flush: tracy signpost unavailable ({})", error)
            signpost = False
        state["signpost"] = signpost
    if state["signpost"]:
        state["signpost"](label)


def _qwen_prefill_profile_begin(layer, x, chunk_start_idx):
    """Layer 0 of a prefill forward starts a new chunk; the chosen chunks get a begin signpost.

    The chunk index is prompt-relative: a forward at chunk_start_idx 0 (or None, an unchunked
    prompt) opens a new prompt at chunk 0, so warm-up or probe prefills earlier in the process
    cannot shift it. The absolute count of layer-0 prefill calls is only logged."""
    if layer.layer_num != 0:
        return
    state = _QWEN_PREFILL_PROFILE
    state["calls"] += 1
    if chunk_start_idx is None or (isinstance(chunk_start_idx, int) and chunk_start_idx == 0):
        state["prompt"] += 1
        state["chunk"] = 0
    else:
        state["chunk"] += 1
    chunk = state["chunk"]
    state["label"] = None
    if chunk in _QWEN_PREFILL_PROFILE_SIGNPOST_CHUNKS:
        from loguru import logger as _qwen_logger

        state["label"] = "qwen_prefill_p%d_chunk_%d" % (state["prompt"], chunk)
        _qwen_prefill_signpost(state["label"] + "_begin")
        _qwen_logger.info("[PINDIAG] prefill profile flush: prompt {} chunk {} begin rows={} chunk_start_idx={} "
                          "(layer-0 prefill call {})",
                          state["prompt"], chunk, x.shape[-2], chunk_start_idx, state["calls"])


def _qwen_prefill_profile_end(layer):
    """Every 16th layer (and the last): drain the device profiler so a chunk cannot overflow it."""
    state = _QWEN_PREFILL_PROFILE
    n_layers = getattr(layer.args, "n_layers", None)
    last = n_layers is not None and layer.layer_num == n_layers - 1
    if not last and layer.layer_num % _QWEN_PREFILL_PROFILE_EVERY != _QWEN_PREFILL_PROFILE_EVERY - 1:
        return
    ttnn.synchronize_device(layer.device)
    if last and state["label"] is not None:
        _qwen_prefill_signpost(state["label"] + "_end")
        state["label"] = None
    ttnn.ReadDeviceProfiler(layer.device)
    state["flushes"] += 1
    if state["flushes"] == 1:
        from loguru import logger as _qwen_logger

        _qwen_logger.info("[PINDIAG] prefill profile flush: first flush at prompt {} chunk {} layer {} (every {} layers)",
                          state["prompt"], state["chunk"], layer.layer_num, _QWEN_PREFILL_PROFILE_EVERY)


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # Lever N M3native C1: under QWEN_FAST_SINGLE_GATEUP=1 the MLP builds no packed
        # gate/up copy and never fuses the gather, so the ff_norm must gather for itself.
        import os

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1"
        if os.environ.get("QWEN_FAST_SINGLE_GATEUP") == "1" and os.environ.get("QWEN_FAST_C1_AGMM") == "1":
            # Lever N M3native C1d: the MLP fuses the gather back (two all_gather_matmul_prefill
            # calls on w1/w3, mlp.py's _qwen_c1_agmm), so the ff_norm skips it as before C1.
            self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
            if self._fuse_ff_agmm:
                from loguru import logger as _qwen_logger

                _qwen_logger.info("[PINDIAG] C1d: ff_norm skips its all-gather for the MLP AGMM (layer {})", layer_num)
        if os.environ.get("QWEN_FAST_SINGLE_GATEUP") == "1" and os.environ.get("QWEN_FAST_C1_EXACT") == "1":
            # Lever N M3native C1e: the MLP runs the served fused SwiGLU AGMM again (on its per-layer packed
            # scratch, mlp.py's _qwen_c1e), so the ff_norm skips its all-gather exactly as served.
            self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
            if self._fuse_ff_agmm:
                from loguru import logger as _qwen_logger

                _qwen_logger.info("[PINDIAG] C1e: ff_norm skips its all-gather for the served fused op (layer {})", layer_num)
        if self.num_devices > 1 and not self._fuse_ff_agmm and not os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1":
            from loguru import logger as _qwen_logger

            _qwen_logger.info("[PINDIAG] single gate/up copy: ff_norm gathers its own input (layer {})", layer_num)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

        if self.num_devices > 1:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / f"layers.{layer_num}" / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        self.feed_forward = Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
    ):
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if mode == "prefill" and _qwen_prefill_profile_on():
            # Lever N M2 prefill profile (QWEN_PREFILL_PROFILE_FLUSH=1): layer 0 counts the chunk.
            _qwen_prefill_profile_begin(self, x, chunk_start_idx)
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.num_devices > 1:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        if mode == "prefill" and _qwen_prefill_profile_on():
            # Lever N M2 prefill profile: drain the device profiler every 16th layer.
            _qwen_prefill_profile_end(self)
        return output
