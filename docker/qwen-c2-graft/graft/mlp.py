# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP: down(silu(gate(x)) * up(x)).

9B (single device): dense matmuls, full weights.
27B TP (1,4 mesh): w1/w3 column-parallel, w2 row-parallel; tt_all_reduce
reduce-scatters on meshes with a dim-1 shape (e.g. P150x4), fracturing hidden.
"""
import os
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj [in, out], bfloat4_b
    w_gate_up: ttnn.Tensor = None  # TP prefill: tile-pair-interleaved packed [gate|up] for fused-swiglu AGMM


def _build_gate_up(gate_w, up_w, mesh, tp, cache_path):
    """Packed [gate|up] weight for all_gather_swiglu_prefill: prepare_for_fused_swiglu tile-pair
    interleave, then column-parallel shard on the 2N dim so each device holds its interleaved slice."""
    import torch

    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    gk = gate_w.to(torch.bfloat16).T.contiguous()  # [K=dim, N=hidden]
    uk = up_w.to(torch.bfloat16).T.contiguous()
    packed = torch.cat([gk, uk], dim=-1)  # [dim, 2*hidden], gate first
    il = prepare_for_fused_swiglu(packed, ndev=tp, gate_is_first=True)  # [dim, 2*hidden]
    return ttnn.as_tensor(
        il,
        dtype=ttnn.bfloat4_b,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
        # DRAM-sharded memcfgs from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # w1/w3 DRAM-WIDTH_SHARDED for decode (M=1 tile, ~+10% tok/s); w2 interleaved.
        # Cache uses `.dramshard` suffix — layout incompatible with interleaved cache
        # (as_tensor ignores requested memcfg on reload). Fallback if memcfgs absent.
        # 1D-decode (default) uses interleaved weights (its mcast decode matmul needs them).
        dram_sharded = (
            args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not getattr(args, "mlp_1d_decode", False)
        )

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        # Prefill-only packed [gate|up] AGMM weight (decode keeps w1/w3; extra DRAM ~w1+w3/layer).
        wgu = (
            _build_gate_up(
                state_dict["gate_proj.weight"],
                state_dict["up_proj.weight"],
                mesh_device,
                tp,
                cache("gate_up", ".swiglu"),
            )
            # Lever N M3native C1: no packed copy under QWEN_FAST_SINGLE_GATEUP=1.
            if tpc.mlp_gateup_agmm_enabled(tp) and os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1"
            else None
        )
        if wgu is None and not os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1":
            # Lever N M3native C1 marker: logged per layer, inside the build it skips.
            from loguru import logger as _qwen_logger

            load_mlp_weights._qwen_single_gateup = getattr(load_mlp_weights, "_qwen_single_gateup", 0) + 1
            _qwen_logger.info(
                "[PINDIAG] single gate/up copy: w_gate_up not built (layer {} of this process, cache {})",
                load_mlp_weights._qwen_single_gateup,
                tensor_cache_path,
            )
            if os.environ.get("QWEN_FAST_SINGLE_GATEUP") == "1" and os.environ.get("QWEN_FAST_C1_EXACT") == "1":
                # Lever N M3native C1e premise audit (QWEN_FAST_C1_EXACT_AUDIT=n, section J): for the first n
                # layers, the packed weight the served path builds is checked against the w1/w3 loaded here.
                _qwen_c1e_premise(mesh_device, state_dict, tp, cache, dram_sharded, load_mlp_weights._qwen_single_gateup)

        if dram_sharded:
            return MLPWeights(
                w1=tpc.shard_w(
                    state_dict["gate_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w1_weight_memcfg,
                    cache_path=cache("gate_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w3=tpc.shard_w(
                    state_dict["up_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w3_weight_memcfg,
                    cache_path=cache("up_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w2=tpc.shard_w(
                    state_dict["down_proj.weight"],
                    mesh_device,
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=cache("down_proj"),
                    dtype=ttnn.bfloat8_b,
                ),
                w_gate_up=wgu,
            )

        # Default: INTERLEAVED DRAM shards; ttnn.linear works for decode and prefill.
        return MLPWeights(
            w1=tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("gate_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w3=tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("up_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w2=tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("down_proj"),
                dtype=ttnn.bfloat8_b,
            ),
            w_gate_up=wgu,
        )

    def load(name, dtype):
        t = state_dict[f"{name}.weight"].T.contiguous()  # [in, out] for ttnn.linear
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # gate/up: bfloat4_b (bandwidth); down: bfloat8_b (accuracy).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", ttnn.bfloat8_b),
        w3=load("up_proj", ttnn.bfloat4_b),
    )


_QWEN_C1_SLICE_ROWS = 1024


def _qwen_c1_linear(args, tpc, w, max_cols, tuning):
    """Lever N M3native C1 (QWEN_FAST_SINGLE_GATEUP=1): the unfused prefill gate/up matmul
    at TP2, on row slices of _QWEN_C1_SLICE_ROWS with DRAM outputs, joined on rows. Same
    call shape as ttnn.linear; the passed program and memory configs are rebuilt per slice."""

    def linear(x, weight, compute_kernel_config=None, program_config=None, memory_config=None):
        seq = x.shape[-2]
        rank = len(x.shape)
        options = dict(max_cols=max_cols, tuning=tuning)
        if weight is w.w1:
            options["fused_activation"] = ttnn.UnaryOpType.SILU
        parts = []
        for start in range(0, seq, _QWEN_C1_SLICE_ROWS):
            rows = min(_QWEN_C1_SLICE_ROWS, seq - start)
            if rows == seq:
                part = x
            else:
                begins = [0] * rank
                ends = [x.shape[i] for i in range(rank)]
                begins[rank - 2], ends[rank - 2] = start, start + rows
                part = ttnn.slice(x, begins, ends)
            config = tpc.create_prefill_mlp_matmul_program_config(rows, args.dim, weight.shape[-1], **options)
            parts.append(ttnn.linear(part, weight, compute_kernel_config=compute_kernel_config,
                                     program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            if part is not x:
                ttnn.deallocate(part)
        if len(parts) == 1:
            return parts[0]
        joined = ttnn.concat(parts, dim=rank - 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for part in parts:
            ttnn.deallocate(part)
        return joined

    return linear


def _qwen_c1_swiglu(args, tpc, w, max_cols, tuning):
    """Lever N M3native C1c (QWEN_FAST_SINGLE_GATEUP=1): silu(x @ w1) * (x @ w3) on row slices
    of _QWEN_C1_SLICE_ROWS. Each slice of x is made once and feeds both matmuls, and only the
    products are joined. Per slice the matmuls and the mul are exactly _qwen_c1_linear's and
    the branch's own, so the result is bit-identical to C1 (see the patch module, section F)."""

    def swiglu(x, compute_kernel_config=None):
        seq = x.shape[-2]
        rank = len(x.shape)
        parts = []
        for start in range(0, seq, _QWEN_C1_SLICE_ROWS):
            rows = min(_QWEN_C1_SLICE_ROWS, seq - start)
            if rows == seq:
                part = x
            else:
                begins = [0] * rank
                ends = [x.shape[i] for i in range(rank)]
                begins[rank - 2], ends[rank - 2] = start, start + rows
                part = ttnn.slice(x, begins, ends)
            gate_config = tpc.create_prefill_mlp_matmul_program_config(
                rows, args.dim, w.w1.shape[-1], max_cols=max_cols, tuning=tuning,
                fused_activation=ttnn.UnaryOpType.SILU)
            up_config = tpc.create_prefill_mlp_matmul_program_config(
                rows, args.dim, w.w3.shape[-1], max_cols=max_cols, tuning=tuning)
            gate = ttnn.linear(part, w.w1, compute_kernel_config=compute_kernel_config,
                               program_config=gate_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up = ttnn.linear(part, w.w3, compute_kernel_config=compute_kernel_config,
                             program_config=up_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if part is not x:
                ttnn.deallocate(part)
            parts.append(ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
        if len(parts) == 1:
            return parts[0]
        joined = ttnn.concat(parts, dim=rank - 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for part in parts:
            ttnn.deallocate(part)
        return joined

    return swiglu


# Lever N M3native C1e (QWEN_FAST_C1_EXACT=1 with QWEN_FAST_SINGLE_GATEUP=1; lever_n_m3native_patch
# section J): the served fused gate/up + SwiGLU AGMM, fed each layer's packed weight rebuilt in ONE scratch.
_QWEN_C1E = {"packs": 0, "audited": 0, "premise": 0}
_QWEN_C1E_AUDIT_MAX = 64


def _qwen_c1e_audit_count():
    """QWEN_FAST_C1_EXACT_AUDIT=n (unset or empty: 0): how many layers' served weights (at load) and how
    many packs (in the first forward) are byte-checked. Refused outside 0..64: the audit reads back, so it
    must finish inside the first forward - an eager warmup - and never reach a trace capture."""
    text = (os.environ.get("QWEN_FAST_C1_EXACT_AUDIT") or "").strip() or "0"
    try:
        count = int(text)
    except ValueError:
        count = -1
    if not 0 <= count <= _QWEN_C1E_AUDIT_MAX:
        raise ValueError("QWEN_FAST_C1_EXACT_AUDIT must be an integer 0..64, got " + repr(text))
    return count


def _qwen_c1e_scratch(mesh_device, args, num_devices):
    """C1e at construction: refuse the other prefill gate/up paths, then return the mesh's ONE packed-weight
    scratch (the first MLP allocates it, before any KV cache) - or None where the served path never fused."""
    for _qwen_other in ("QWEN_FAST_C1_AGMM", "QWEN_FAST_C1_LEGACY"):
        if os.environ.get(_qwen_other) == "1":
            raise ValueError("QWEN_FAST_C1_EXACT=1 excludes " + _qwen_other + "=1: each replaces the prefill gate/up")
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    if not tpc.mlp_gateup_agmm_enabled(num_devices):
        return None
    audit = _qwen_c1e_audit_count()
    from models.demos.blackhole.qwen36.tt import mlp_c1e_pack as _c1e

    shape = (args.dim, args.hidden_dim // num_devices)
    first = not _c1e.has_scratch(mesh_device, shape)
    scratch = _c1e.allocate_scratch(ttnn, mesh_device, shape)
    if first:
        from loguru import logger as _qwen_logger

        _qwen_logger.info("[PINDIAG] C1e scratch allocated: per chip {} for w1/w3 {}, {} bytes; audit={}",
                          tuple(scratch.shape), shape, _c1e.traffic_bytes(shape)[1], audit)
    return scratch


def _qwen_c1e_pack(mlp, w, x):
    """C1e per prefill call: this layer's served packed gate/up, rebuilt in the shared scratch (one generic_op;
    nothing allocated, so trace-safe once compiled). The first QWEN_FAST_C1_EXACT_AUDIT packs are then
    byte-checked against w1/w3, raising on any difference."""
    from models.demos.blackhole.qwen36.tt import mlp_c1e_pack as _c1e

    scratch = _c1e.pack_gate_up(mlp.device, w.w1, w.w3, mlp._qwen_c1e)
    state = _QWEN_C1E
    state["packs"] += 1
    if state["packs"] == 1:
        # Once per process, inside the branch it names.
        from loguru import logger as _qwen_logger

        read, written = _c1e.traffic_bytes(w.w1.shape)
        _qwen_logger.info("[PINDIAG] prefill MLP C1e: the served fused SwiGLU AGMM on the per-layer packed scratch: rows={} k_local={} scratch={} copy read={} written={} bytes per layer per chip",
                          x.shape[-2], x.shape[-1], tuple(scratch.shape), read, written)
    if state["audited"] < _qwen_c1e_audit_count():
        state["audited"] += 1
        report = _c1e.audit_pairs(ttnn, mlp.device, scratch, w.w1, w.w3)
        from loguru import logger as _qwen_logger

        _qwen_logger.info("[PINDIAG] C1e audit {} exact={} pages={} mismatched_words={} rows={}",
                          state["audited"], report["exact"], report["pages"], report["mismatched_words"], x.shape[-2])
        if not report["exact"]:
            raise AssertionError("C1e: the packed scratch differs from this layer's w1/w3: " + repr(report))
    return scratch


def _qwen_c1e_premise(mesh_device, state_dict, tp, cache, dram_sharded, layer):
    """C1e premise, at load, for the first QWEN_FAST_C1_EXACT_AUDIT layers: the packed weight the served
    path builds (_build_gate_up, from its own cache) is the tile-pair interleave of the w1/w3 this layer loads
    (tp_common.shard_w, from theirs), byte for byte on every chip - so a scratch equal to that interleave is
    the served weight. All three are freed again before the layer's own weights load."""
    if layer > _qwen_c1e_audit_count():
        return
    if dram_sharded:
        raise ValueError("C1e needs DRAM-interleaved w1/w3 (mlp_1d_decode); this configuration shards them")
    from models.demos.blackhole.qwen36.tt import mlp_c1e_pack as _c1e
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    served = _build_gate_up(
        state_dict["gate_proj.weight"], state_dict["up_proj.weight"], mesh_device, tp, cache("gate_up", ".swiglu")
    )
    w1 = tpc.shard_w(state_dict["gate_proj.weight"], mesh_device, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                     cache_path=cache("gate_proj"), dtype=ttnn.bfloat4_b)
    w3 = tpc.shard_w(state_dict["up_proj.weight"], mesh_device, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                     cache_path=cache("up_proj"), dtype=ttnn.bfloat4_b)
    try:
        report = _c1e.audit_pairs(ttnn, mesh_device, served, w1, w3)
    finally:
        for tensor in (served, w1, w3):
            ttnn.deallocate(tensor)
    _QWEN_C1E["premise"] += 1
    from loguru import logger as _qwen_logger

    _qwen_logger.info("[PINDIAG] C1e premise audit {} exact={} layer={} pages={} mismatched_words={}",
                      _QWEN_C1E["premise"], report["exact"], layer, report["pages"], report["mismatched_words"])
    if not report["exact"]:
        raise AssertionError("C1e premise: the served packed gate/up is not the interleave of w1/w3: " + repr(report))


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        # 1D-decode (default): small-grid 1D matmuls beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny decode MLP matmuls (see test_mlp_matmul_sweep). Forces interleaved weights.
        self._mlp_1d_decode = args is not None and getattr(args, "mlp_1d_decode", False)
        # Match load_mlp_weights dram_sharded condition for layout consistency.
        self._dram_sharded = (
            self.num_devices > 1
            and args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not self._mlp_1d_decode
        )
        # Prefill fused-swiglu AGMM (ff_norm skips its AG; layer.py sets _fuse_ff_agmm to match).
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        if os.environ.get("QWEN_FAST_SINGLE_GATEUP") == "1" and os.environ.get("QWEN_FAST_C1_EXACT") == "1":
            # Lever N M3native C1e (QWEN_FAST_C1_EXACT=1): the served fused gate/up AGMM on a per-layer
            # rebuild of its packed weight in one shared scratch (layer.py skips the ff_norm gather again).
            self._qwen_c1e = _qwen_c1e_scratch(mesh_device, args, self.num_devices)
        if os.environ.get("QWEN_FAST_SINGLE_GATEUP") == "1" and os.environ.get("QWEN_FAST_C1_AGMM") == "1":
            # Lever N M3native C1d: two all_gather_matmul_prefill calls on w1/w3 take the ff_norm's
            # gather back (layer.py keeps _fuse_ff_agmm True under the same two flags).
            self._qwen_c1_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        # Lever N M3native C1: the fused branch goes with the packed copy.
        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1"
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path, args=args)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # fuse_swiglu AGMM: fp32 acc (subblock_w=4) to match GDN/attn in-proj.
        self.compute_kernel_config_agmm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x):
        if self.num_devices > 1:
            return self._forward_tp(x)
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        down_pc = None
        if (
            T > 1
            and getattr(self.args, "prefill_progcfg", None) is not None
            and os.environ.get("QWEN9B_MLP_DOWN_AUTO") != "1"
        ):
            down_pc = self.args.prefill_progcfg(T, hidden.shape[-1], w.w2.shape[-1])
        output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc, program_config=down_pc)
        ttnn.deallocate(hidden)
        return output

    def _forward_tp(self, x):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        _silu_fused = False
        # Prefill: x is K-sharded (ff_norm skipped AG); fused AG + [gate|up] + SwiGLU
        # Lever N M3native: the fused all-gather path is for the K-sharded prefill input;
        # a replicated (full-K) 64-row decode input takes the 1D decode branch below.
        _fused_gu = (self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None
                     and x.shape[-1] < w.w_gate_up.shape[-2])
        if _fused_gu:
            hidden = tpc.all_gather_swiglu_prefill(
                x, w.w_gate_up, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            _silu_fused = True
        elif getattr(self, "_qwen_c1e", None) is not None and x.shape[-2] > ttnn.TILE_SIZE and x.shape[-1] < w.w1.shape[-2]:
            # Lever N M3native C1e (QWEN_FAST_C1_EXACT=1): x is K-sharded (the ff_norm skipped its gather, as
            # served). This layer's w1/w3 are rebuilt in the served packing into the shared scratch, then the
            # served fused call runs on it unchanged: the served arithmetic on the served bytes.
            _qwen_c1e_weight = _qwen_c1e_pack(self, w, x)
            hidden = tpc.all_gather_swiglu_prefill(
                x, _qwen_c1e_weight, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            _silu_fused = True
            _fused_gu = True
        elif getattr(self, "_qwen_c1_agmm", False) and x.shape[-2] > ttnn.TILE_SIZE and x.shape[-1] < w.w1.shape[-2]:
            # Lever N M3native C1d (QWEN_FAST_C1_AGMM=1): x is K-sharded (the ff_norm skipped its
            # gather). Two fused all-gather + matmul calls on the separate w1 (SiLU fused) and w3,
            # then one mul; hidden is produced here, so the gate*up below is skipped.
            if not getattr(type(self), "_qwen_c1d_logged", False):
                type(self)._qwen_c1d_logged = True
                from loguru import logger as _qwen_logger

                _qwen_logger.info("[PINDIAG] prefill MLP C1d: two all_gather_matmul_prefill (w1+SiLU, w3) and a mul: rows={} k_local={}", x.shape[-2], x.shape[-1])
            _qwen_gate = tpc.all_gather_matmul_prefill(
                x, w.w1, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology(),
                fused_activation=ttnn.UnaryOpType.SILU,
            )
            _qwen_up = tpc.all_gather_matmul_prefill(
                x, w.w3, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            hidden = ttnn.mul(_qwen_gate, _qwen_up, memory_config=mc)
            ttnn.deallocate(_qwen_gate)
            ttnn.deallocate(_qwen_up)
            _silu_fused = True
            _fused_gu = True
        elif getattr(self, "_dram_sharded", False) and x.shape[-2] <= ttnn.TILE_SIZE:
            # DRAM-WIDTH_SHARDED w1/w3 decode (M=1 tile). Prefill uses fused AGMM above.
            x_sh = ttnn.to_memory_config(x, args.act_shard_hidden)
            w1_out = ttnn.linear(
                x_sh,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_sh,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_sh)
            # Keep gate/up in L1 for mul → w2 (avoid L1→DRAM→L1).
            w1_out = ttnn.to_memory_config(w1_out, ttnn.L1_MEMORY_CONFIG)
            w3_out = ttnn.to_memory_config(w3_out, ttnn.L1_MEMORY_CONFIG)
        elif self._mlp_1d_decode and x.shape[-2] <= 2 * ttnn.TILE_SIZE:
            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.
            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.
            # Lever N M3native: rows 1-32 keep the M = 1 configs; rows 33-64 select the
            # native 64-row (per_core_M 2) configs.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            _w1_1d_cfg = (args.mlp_w1_decode_1d_progcfg_64 if x.shape[-2] > ttnn.TILE_SIZE
                         else args.mlp_w1_decode_1d_progcfg)
            _w3_1d_cfg = (args.mlp_w3_decode_1d_progcfg_64 if x.shape[-2] > ttnn.TILE_SIZE
                         else args.mlp_w3_decode_1d_progcfg)
            w1_out = ttnn.linear(
                x_il,
                w.w1,
                compute_kernel_config=ckc,
                program_config=_w1_1d_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=ckc,
                program_config=_w3_1d_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            _silu_fused = True
        elif x.shape[-2] > ttnn.TILE_SIZE and os.environ.get("QWEN_FAST_SINGLE_GATEUP") == "1" and os.environ.get("QWEN_FAST_C1_LEGACY") != "1":
            # Lever N M3native C1c: the C1 prefill gate/up with one slice of x per 1024 rows feeding
            # both matmuls and only the product joined - bit-identical to C1 (patch module, F).
            # hidden is produced here, so the gate*up below is skipped.
            if not getattr(type(self), "_qwen_c1c_logged", False):
                type(self)._qwen_c1c_logged = True
                from loguru import logger as _qwen_logger

                _qwen_logger.info("[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows, product-only concat: rows={} slices of {} rows", x.shape[-2], _QWEN_C1_SLICE_ROWS)
            hidden = _qwen_c1_swiglu(
                args, tpc, w, getattr(args, "decode_grid_w", 8), getattr(args, "prefill_tuning", None)
            )(x, compute_kernel_config=ckc)
            _silu_fused = True
            _fused_gu = True
        elif x.shape[-2] > ttnn.TILE_SIZE:
            # Prefill (M>1 tile, compute-bound): FPU-tuned 2D config (grid width -> 1x4 subblock,
            # in0_block_w=4) beats ttnn-auto's 1x1 stall ~2.7x (test_mlp_matmul_sweep_prefill). SILU fused.
            seq = x.shape[-2]
            if not os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1" and not getattr(type(self), "_qwen_2d_logged", False):
                # Lever N M3native C1 marker: once per process, inside the branch it names.
                type(self)._qwen_2d_logged = True
                from loguru import logger as _qwen_logger

                _qwen_logger.info("[PINDIAG] prefill MLP via w1/w3 2D branch: rows={} fused={} x={} slices of {} rows, DRAM outputs",
                                  seq, self._fuse_gateup_agmm, getattr(x, "memory_config", lambda: None)(),
                                  _QWEN_C1_SLICE_ROWS)
            # max_cols = device worker-grid width (11 on BH): wide grid (gate/up -> 9x10) vs old 8-wide.
            _gw = getattr(args, "decode_grid_w", 8)
            # TP-selected prefill tuning; absent (single-device 9B) => frozen TP=4 behavior.
            _pt = getattr(args, "prefill_tuning", None)
            pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, max_cols=_gw, tuning=_pt
            )
            pc_up = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w3.shape[-1], max_cols=_gw, tuning=_pt
            )
            # L1 output (gate/up outputs; down output via mc_out below): +FPU, avoids the DRAM round-trip
            # (test_mlp_matmul_sweep_prefill *_outL1). The [seq,N] tensors fit L1 at the prefill chunk.
            # Lever N M3native C1: row-sliced, DRAM-output gate/up at TP2 under the flag (v100, v102).
            _qwen_linear = ttnn.linear if os.environ.get("QWEN_FAST_SINGLE_GATEUP") != "1" else _qwen_c1_linear(args, tpc, w, _gw, _pt)
            w1_out = _qwen_linear(
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            w3_out = _qwen_linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _silu_fused = True
        else:
            # Interleaved weights: auto matmul program for decode and prefill.
            w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
            _silu_fused = True

        # gated activation (down-proj INPUT): L1 in decode, DRAM in prefill. The L1 win is OUTPUT-only;
        # keeping both down input (hidden) and output (partial) in L1 at seq 2048 overflows L1.
        _prefill_tuned = x.shape[-2] > ttnn.TILE_SIZE and _silu_fused
        # gate * up (skipped when _fused_gu already produced `hidden` with SwiGLU in-kernel).
        if not _fused_gu:
            mc_out = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= ttnn.TILE_SIZE else mc
            # Standalone silu only on DRAM-sharded decode path (SILU not fused there).
            if _silu_fused:
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
            else:
                w1_act = ttnn.silu(w1_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
                hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out)
                ttnn.deallocate(w1_act)
            ttnn.deallocate(w3_out)
        # Prefill w2: 2D progcfg on (8,10); decode (M<=32) keeps ttnn-auto.
        w2_pc = None
        if self._mlp_1d_decode and hidden.shape[-2] <= 2 * ttnn.TILE_SIZE:
            # 1D mcast decode down-proj on a small explicit grid (~16 cores).
            # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the
            # native 64-row (per_core_M 2) config.
            w2_pc = (args.mlp_w2_decode_1d_progcfg_64 if hidden.shape[-2] > ttnn.TILE_SIZE
                     else args.mlp_w2_decode_1d_progcfg)
        elif hidden.shape[-2] > ttnn.TILE_SIZE:
            # Prefill down-proj: subblock-tuned 2D config with the wide grid (max_cols=device width),
            # off the generic 8-wide prefill_progcfg. Output L1 via mc_w2_out below.
            w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                hidden.shape[-2],
                hidden.shape[-1],
                w.w2.shape[-1],
                max_cols=getattr(args, "decode_grid_w", 8),
                tuning=getattr(args, "prefill_tuning", None),
            )
        # down-proj OUTPUT in L1 for the tuned prefill path (DRAM input `hidden` + L1 output = the
        # validated sweep outL1 config; tt_all_reduce already consumes an L1 partial).
        mc_w2_out = ttnn.L1_MEMORY_CONFIG if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned) else mc
        partial = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out, program_config=w2_pc)
        ttnn.deallocate(hidden)

        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
