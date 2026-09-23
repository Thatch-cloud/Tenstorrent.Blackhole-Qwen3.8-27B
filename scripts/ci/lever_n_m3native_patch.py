"""Lever N M3native: graft the native 64-row (four-user) decode path onto the pinned
model sources.

Read from the image's own dump (probe run 35503727180: model_config.py, tp_common.py,
attention/tp.py, gdn/tp.py, mlp.py), the model's decode-time 1D matmul configs
(tp_common.create_matmul_1d_decode_progcfg) are all built at M = 1 (model_config.py
_init_tp_config), and the seven gates that select them (attention/tp.py's _qkv,
_wo_proj and forward_decode's prep gate; gdn/tp.py's _row_proj, _project_qkvzab and
_project_qkvzab_raw; mlp.py's w1/w3 and w2 sites) all cap at one 32-row tile
(`x.shape[-2] <= tpc.TILE_SIZE` or the ttnn spelling of the same constant). Beyond
that they fall through to a slow PREFILL 2D program config - the two-call wrapper in
two_tile_decode.py exists to avoid that fall-through by calling the fast one-tile arm
twice and joining the halves.

create_matmul_1d_decode_progcfg's per_core_M is ceil(m / TILE_SIZE) (tp_common.py:161)
with no other shape-dependent branch, so the SAME builder call at M = 64 produces the
per_core_M = 2 config the two-tile wrapper's own runtime rebuild
(two_tile_decode.two_tile_matmul_1d_progcfg) already proved on the device for the
fused QKV projection (run 35507675630, image v51). This module adds that M = 64
sibling for every one of the seven configs, alongside - never replacing - the
existing M = 1 ones, then widens each of the seven gates to also select it for rows
33..64, so the model's own decode path is what runs a 64-row block natively once the
graft is mounted (lever_n_m3native_run_arm.sh), while a stock image (no _64 attrs)
takes the exact byte-identical path it does today.

NOT touched here (both C++-bounded, not a program-config question): attn_decode_prep
(hangs at batch 64 on the device, run 35502452429) and nlp_concat_heads_decode
(TT_FATAL input_shape[1] <= 32 on the host). Those stay two-call regardless of the
graft - two_tile_decode.py's own overlay switch (model_batch.py's native_m3
detection) keeps exactly those two wrapped and drops everything else.

Every edit here is scoped to a named function's AST line range and asserts its
literal anchor text matched exactly once within that scope, mirroring
lever_n_model_patch.py: a source change fails the patch loudly instead of silently
touching the wrong lookalike line (attention/tp.py's `_project_qkvzab`-shaped gate at
gdn/tp.py:431 and the `_project_qkvzab_raw`-shaped one at :1035 read alike outside
their own functions).

Applying nothing on import: stage() is explicit, mirroring lever_n_model_patch.
"""

import argparse
import ast
from pathlib import Path

import gdn_prefill_conv_exact as _pcx


def function_span(source, name):
    """Line span [start, end) of a method, by AST sibling order (see lever_n_model_patch).

    end_lineno needs Python 3.8; this has to run under 3.7 locally as well as 3.10 in
    the image, so the end is taken as the next sibling definition's start (or the end
    of the class body) rather than from the node.
    """
    tree = ast.parse(source)
    total = len(source.splitlines())
    for body in tree.body:
        if not isinstance(body, ast.ClassDef):
            continue
        members = [node for node in body.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for index, node in enumerate(members):
            if node.name != name:
                continue
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list) - 1
            else:
                start = node.lineno - 1
            if index + 1 < len(members):
                following = members[index + 1]
                end = (min(d.lineno for d in following.decorator_list)
                       if following.decorator_list else following.lineno) - 1
            else:
                end = total
            return start, end
    raise ValueError('no method named %s' % name)


def replace_once(lines, span, old, new, what):
    """Replace old with new inside a line span, requiring exactly one occurrence."""
    start, end = span
    region = ''.join(lines[start:end])
    if region.count(old) != 1:
        raise ValueError('%s: expected one occurrence of %r in %s, found %d'
                         % (what, old[:60], span, region.count(old)))
    lines[start:end] = (region.replace(old, new, 1)).splitlines(keepends=True)
    return lines


# ---------------------------------------------------------------------------------
# A. model_config.py: seven M = 64 siblings, alongside the M = 1 originals.
# ---------------------------------------------------------------------------------

TP_CONFIG_FUNCTION = '_init_tp_config'

KV_SHARD_ANCHOR = (
    '        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(\n'
    '            shape=(tpc.TILE_SIZE, self.head_dim),\n'
    '            core_grid=ttnn.CoreGrid(x=_cols, y=_rows),\n'
    '            strategy=ttnn.ShardStrategy.HEIGHT,\n'
    '            orientation=ttnn.ShardOrientation.ROW_MAJOR,\n'
    '            use_height_and_width_as_shard_shape=True,\n'
    '        )\n'
)

# Every argument here matches its M = 1 original above byte for byte; only the M
# changes (a new M64 local, never touching the M = 1 the DRAM-sharded configs above
# still read). Order mirrors the originals: mlp w1/w3/w2, attn_qkv (for symmetry -
# already proven on the device by the two-tile wrapper's own runtime rebuild),
# gdn_qkvz, attn_wo, gdn_out.
NATIVE_64_BLOCK = (
    '\n'
    '        # Lever N M3native: the same seven 1D decode matmul configs above, rebuilt at\n'
    '        # M = 64 (per_core_M = 2) for the native 64-row (four T16 user) decode graft.\n'
    '        # create_matmul_1d_decode_progcfg\'s per_core_M is ceil(m / TILE_SIZE)\n'
    '        # (tp_common.py:161), so this needs no builder change - only a second M. Every\n'
    '        # M = 1 config above is left byte-identical.\n'
    '        M64 = 64\n'
    '        self.mlp_w1_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64,\n'
    '            self.dim,\n'
    '            self.hidden_dim // tp,\n'
    '            num_cores=44,\n'
    '            fused_activation=ttnn.UnaryOpType.SILU,\n'
    '            grid_w=self.decode_grid_w,\n'
    '        )\n'
    '        self.mlp_w3_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64, self.dim, self.hidden_dim // tp, num_cores=44, grid_w=self.decode_grid_w\n'
    '        )\n'
    '        self.mlp_w2_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64, self.hidden_dim // tp, self.dim, num_cores=33, grid_w=self.decode_grid_w\n'
    '        )\n'
    '        self.attn_qkv_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64, self.dim, self.attn_qkv_fused_dim_tp, num_cores=64\n'
    '        )\n'
    '        self.gdn_qkvz_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64, self.dim, self.gdn_qkvzab_dim_tp, num_cores=44, grid_w=self.decode_grid_w\n'
    '        )\n'
    '        self.attn_wo_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64, self.attn_out_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w\n'
    '        )\n'
    '        self.gdn_out_decode_1d_progcfg_64 = tpc.create_matmul_1d_decode_progcfg(\n'
    '            M64, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w\n'
    '        )\n'
)


def patch_model_config(source):
    """Append the seven M = 64 progcfgs right after kv_update_shard_cfg, inside
    _init_tp_config. Every M = 1 assignment above stays untouched."""
    if 'mlp_w1_decode_1d_progcfg_64' in source:
        raise ValueError('model_config.py already carries the M3native _64 progcfgs')
    lines = source.splitlines(keepends=True)
    span = function_span(source, TP_CONFIG_FUNCTION)
    lines = replace_once(lines, span, KV_SHARD_ANCHOR, KV_SHARD_ANCHOR + NATIVE_64_BLOCK,
                         'model_config kv_update_shard_cfg anchor')
    result = ''.join(lines)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------------
# B. attention/tp.py: _qkv, _wo_proj, forward_decode's prep gate.
# ---------------------------------------------------------------------------------

QKV_FUNCTION = '_qkv'
WO_PROJ_FUNCTION = '_wo_proj'
FORWARD_DECODE_FUNCTION = 'forward_decode'


def patch_attention_tp(source):
    lines = source.splitlines(keepends=True)

    span = function_span(source, QKV_FUNCTION)
    lines = replace_once(
        lines, span,
        '        elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:\n'
        '            # Decode: small-grid 1D matmul (interleaved weight). Output DRAM so _make_heads_decode\'s\n'
        '            # to_memory_config(.,L1) stays a real copy before it deallocates the source.\n'
        '            qkv = tpc.matmul_1d_decode(\n'
        '                x,\n'
        '                tw["wqkv_fused"],\n'
        '                self.args.attn_qkv_decode_1d_progcfg,\n'
        '                self.compute_cfg,\n'
        '                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,\n'
        '            )\n',
        '        elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= 2 * tpc.TILE_SIZE:\n'
        '            # Decode: small-grid 1D matmul (interleaved weight). Output DRAM so _make_heads_decode\'s\n'
        '            # to_memory_config(.,L1) stays a real copy before it deallocates the source.\n'
        '            # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the\n'
        '            # native 64-row (per_core_M 2) config the model_config graft adds alongside it.\n'
        '            qkv = tpc.matmul_1d_decode(\n'
        '                x,\n'
        '                tw["wqkv_fused"],\n'
        '                self.args.attn_qkv_decode_1d_progcfg_64 if x.shape[-2] > tpc.TILE_SIZE\n'
        '                else self.args.attn_qkv_decode_1d_progcfg,\n'
        '                self.compute_cfg,\n'
        '                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,\n'
        '            )\n',
        'attention _qkv 1D decode gate')

    span = function_span(''.join(lines), WO_PROJ_FUNCTION)
    lines = replace_once(
        lines, span,
        '        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:\n'
        '            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.\n'
        '            return tpc.matmul_1d_decode(\n'
        '                x,\n'
        '                weight,\n'
        '                self.args.attn_wo_decode_1d_progcfg,\n'
        '                self.compute_cfg,\n'
        '                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,\n'
        '            )\n',
        '        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= 2 * tpc.TILE_SIZE:\n'
        '            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.\n'
        '            # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the\n'
        '            # native 64-row (per_core_M 2) config.\n'
        '            return tpc.matmul_1d_decode(\n'
        '                x,\n'
        '                weight,\n'
        '                self.args.attn_wo_decode_1d_progcfg_64 if x.shape[-2] > tpc.TILE_SIZE\n'
        '                else self.args.attn_wo_decode_1d_progcfg,\n'
        '                self.compute_cfg,\n'
        '                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,\n'
        '            )\n',
        'attention _wo_proj 1D decode gate')

    span = function_span(''.join(lines), FORWARD_DECODE_FUNCTION)
    lines = replace_once(
        lines, span,
        '        _prep = (\n'
        '            os.environ.get("QWEN_ATTN_PREP", "0") == "1"\n'
        '            and use_paged\n'
        '            and self._fused_qkv\n'
        '            and x.shape[-2] <= ttnn.TILE_SIZE\n'
        '        )\n',
        '        _prep = (\n'
        '            os.environ.get("QWEN_ATTN_PREP", "0") == "1"\n'
        '            and use_paged\n'
        '            and self._fused_qkv\n'
        '            # Lever N M3native: widened alongside _qkv/_wo_proj so a 64-row prep-path\n'
        '            # forward stays gated the same way its own projections are.\n'
        '            and x.shape[-2] <= 2 * ttnn.TILE_SIZE\n'
        '        )\n',
        'attention forward_decode prep gate')

    # Lever N M3native (run 35558196643): the fused all-gather + matmul branch is the
    # K-sharded PREFILL input path (the norm skipped its all-gather), and it was
    # selected on rows alone. A 64-row decode input is replicated at full K, and
    # all_gather_minimal_matmul_async gathers it to 2K and refuses (K=10240 vs
    # K_w=5120), so the fused branch now also requires x narrower than the weight K;
    # the 64-row decode input falls through to the 1D decode branch below it.
    span = function_span(''.join(lines), QKV_FUNCTION)
    lines = replace_once(
        lines, span,
        '        if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE:\n'
        '            qkv = tpc.all_gather_matmul_prefill(\n',
        '        # Lever N M3native: the fused all-gather path is for the K-sharded prefill input;\n'
        '        # a replicated (full-K) 64-row decode input takes the 1D decode branch below.\n'
        '        if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE and x.shape[-1] < tw["wqkv_fused"].shape[-2]:\n'
        '            qkv = tpc.all_gather_matmul_prefill(\n',
        'attention _qkv fused-prefill layout guard')
    result = ''.join(lines)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------------
# C. gdn/tp.py: _row_proj, _project_qkvzab, _project_qkvzab_raw.
# ---------------------------------------------------------------------------------

ROW_PROJ_FUNCTION = '_row_proj'
PROJECT_QKVZAB_FUNCTION = '_project_qkvzab'
PROJECT_QKVZAB_RAW_FUNCTION = '_project_qkvzab_raw'


def patch_gdn_tp(source):
    lines = source.splitlines(keepends=True)

    span = function_span(source, ROW_PROJ_FUNCTION)
    lines = replace_once(
        lines, span,
        '        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:\n'
        '            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.\n'
        '            return tpc.matmul_1d_decode(\n'
        '                x, weight, self.args.gdn_out_decode_1d_progcfg, self.cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG\n'
        '            )\n',
        '        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= 2 * tpc.TILE_SIZE:\n'
        '            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.\n'
        '            # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the\n'
        '            # native 64-row (per_core_M 2) config.\n'
        '            _row_proj_cfg = (self.args.gdn_out_decode_1d_progcfg_64 if x.shape[-2] > tpc.TILE_SIZE\n'
        '                             else self.args.gdn_out_decode_1d_progcfg)\n'
        '            return tpc.matmul_1d_decode(\n'
        '                x, weight, _row_proj_cfg, self.cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG\n'
        '            )\n',
        'gdn _row_proj 1D decode gate')

    span = function_span(''.join(lines), PROJECT_QKVZAB_FUNCTION)
    lines = replace_once(
        lines, span,
        '            elif getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:\n'
        '                # Decode: small-grid 1D matmul on the interleaved fused weight (beats the DRAM-sharded grid).\n'
        '                qkvzab = tpc.matmul_1d_decode(\n'
        '                    x,\n'
        '                    self.tw["qkvz"],\n'
        '                    self.args.gdn_qkvz_decode_1d_progcfg,\n'
        '                    self.cfg,\n'
        '                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,\n'
        '                )\n',
        '            elif getattr(self.args, "proj_1d_decode", False) and S <= 2 * tpc.TILE_SIZE:\n'
        '                # Decode: small-grid 1D matmul on the interleaved fused weight (beats the DRAM-sharded grid).\n'
        '                # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the\n'
        '                # native 64-row (per_core_M 2) config.\n'
        '                qkvzab = tpc.matmul_1d_decode(\n'
        '                    x,\n'
        '                    self.tw["qkvz"],\n'
        '                    self.args.gdn_qkvz_decode_1d_progcfg_64 if S > tpc.TILE_SIZE\n'
        '                    else self.args.gdn_qkvz_decode_1d_progcfg,\n'
        '                    self.cfg,\n'
        '                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,\n'
        '                )\n',
        'gdn _project_qkvzab 1D decode gate')

    span = function_span(''.join(lines), PROJECT_QKVZAB_RAW_FUNCTION)
    lines = replace_once(
        lines, span,
        '        if getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:\n'
        '            return tpc.matmul_1d_decode(\n'
        '                x,\n'
        '                self.tw["qkvz"],\n'
        '                self.args.gdn_qkvz_decode_1d_progcfg,\n'
        '                self.cfg,\n'
        '                out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,\n'
        '            )\n',
        '        if getattr(self.args, "proj_1d_decode", False) and S <= 2 * tpc.TILE_SIZE:\n'
        '            # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the\n'
        '            # native 64-row (per_core_M 2) config - the single call\n'
        '            # gdn_device_loop_state.project_qkvzab_by_tile makes over a whole packed\n'
        '            # 64-row block once native_m3 raises its tile cap.\n'
        '            return tpc.matmul_1d_decode(\n'
        '                x,\n'
        '                self.tw["qkvz"],\n'
        '                self.args.gdn_qkvz_decode_1d_progcfg_64 if S > tpc.TILE_SIZE\n'
        '                else self.args.gdn_qkvz_decode_1d_progcfg,\n'
        '                self.cfg,\n'
        '                out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,\n'
        '            )\n',
        'gdn _project_qkvzab_raw 1D decode gate')

    # Lever N M3native (run 35558196643): the fused all-gather + matmul branch is the
    # K-sharded PREFILL input path (the norm skipped its all-gather), and it was
    # selected on rows alone. A 64-row decode input is replicated at full K, and
    # all_gather_minimal_matmul_async gathers it to 2K and refuses (K=10240 vs
    # K_w=5120), so the fused branch now also requires x narrower than the weight K;
    # the 64-row decode input falls through to the 1D decode branch below it.
    span = function_span(''.join(lines), PROJECT_QKVZAB_FUNCTION)
    lines = replace_once(
        lines, span,
        '            if self._fuse_agmm and S > tpc.TILE_SIZE:\n'
        '                qkvzab = tpc.all_gather_matmul_prefill(\n',
        '            # Lever N M3native: the fused all-gather path is for the K-sharded prefill input;\n'
        '            # a replicated (full-K) 64-row decode input takes the 1D decode branch below.\n'
        '            if self._fuse_agmm and S > tpc.TILE_SIZE and x.shape[-1] < self.tw["qkvz"].shape[-2]:\n'
        '                qkvzab = tpc.all_gather_matmul_prefill(\n',
        'gdn _project_qkvzab fused-prefill layout guard')
    result = ''.join(lines)
    # remap_slots reads self.B entries from a remap vLLM sizes to the LIVE batch, so a
    # short remap indexed out of bounds the moment a batch condense happened with two
    # users decoding - run 35717866188. Latent until then: nothing had got two users
    # decoding together before, so no condense had ever run.
    import lever_n_model_patch as _m1
    result = _m1.patch_gdn_slot_remap(result)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------------
# D. mlp.py: the w1/w3 site and the w2 site inside _forward_tp.
# ---------------------------------------------------------------------------------

FORWARD_TP_FUNCTION = '_forward_tp'


def patch_mlp(source):
    lines = source.splitlines(keepends=True)
    span = function_span(source, FORWARD_TP_FUNCTION)

    lines = replace_once(
        lines, span,
        '        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:\n'
        '            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.\n'
        '            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.\n'
        '            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)\n'
        '            w1_out = ttnn.linear(\n'
        '                x_il,\n'
        '                w.w1,\n'
        '                compute_kernel_config=ckc,\n'
        '                program_config=args.mlp_w1_decode_1d_progcfg,\n'
        '                memory_config=ttnn.L1_MEMORY_CONFIG,\n'
        '            )\n'
        '            w3_out = ttnn.linear(\n'
        '                x_il,\n'
        '                w.w3,\n'
        '                compute_kernel_config=ckc,\n'
        '                program_config=args.mlp_w3_decode_1d_progcfg,\n'
        '                memory_config=ttnn.L1_MEMORY_CONFIG,\n'
        '            )\n'
        '            ttnn.deallocate(x_il)\n'
        '            _silu_fused = True\n',
        '        elif self._mlp_1d_decode and x.shape[-2] <= 2 * ttnn.TILE_SIZE:\n'
        '            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.\n'
        '            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.\n'
        '            # Lever N M3native: rows 1-32 keep the M = 1 configs; rows 33-64 select the\n'
        '            # native 64-row (per_core_M 2) configs.\n'
        '            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)\n'
        '            _w1_1d_cfg = (args.mlp_w1_decode_1d_progcfg_64 if x.shape[-2] > ttnn.TILE_SIZE\n'
        '                         else args.mlp_w1_decode_1d_progcfg)\n'
        '            _w3_1d_cfg = (args.mlp_w3_decode_1d_progcfg_64 if x.shape[-2] > ttnn.TILE_SIZE\n'
        '                         else args.mlp_w3_decode_1d_progcfg)\n'
        '            w1_out = ttnn.linear(\n'
        '                x_il,\n'
        '                w.w1,\n'
        '                compute_kernel_config=ckc,\n'
        '                program_config=_w1_1d_cfg,\n'
        '                memory_config=ttnn.L1_MEMORY_CONFIG,\n'
        '            )\n'
        '            w3_out = ttnn.linear(\n'
        '                x_il,\n'
        '                w.w3,\n'
        '                compute_kernel_config=ckc,\n'
        '                program_config=_w3_1d_cfg,\n'
        '                memory_config=ttnn.L1_MEMORY_CONFIG,\n'
        '            )\n'
        '            ttnn.deallocate(x_il)\n'
        '            _silu_fused = True\n',
        'mlp w1/w3 1D decode gate')

    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(
        lines, span,
        '        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:\n'
        '            # 1D mcast decode down-proj on a small explicit grid (~16 cores).\n'
        '            w2_pc = args.mlp_w2_decode_1d_progcfg\n',
        '        if self._mlp_1d_decode and hidden.shape[-2] <= 2 * ttnn.TILE_SIZE:\n'
        '            # 1D mcast decode down-proj on a small explicit grid (~16 cores).\n'
        '            # Lever N M3native: rows 1-32 keep the M = 1 config; rows 33-64 select the\n'
        '            # native 64-row (per_core_M 2) config.\n'
        '            w2_pc = (args.mlp_w2_decode_1d_progcfg_64 if hidden.shape[-2] > ttnn.TILE_SIZE\n'
        '                     else args.mlp_w2_decode_1d_progcfg)\n',
        'mlp w2 1D decode gate')

    # Lever N M3native (run 35558196643): the fused all-gather + matmul branch is the
    # K-sharded PREFILL input path (the norm skipped its all-gather), and it was
    # selected on rows alone. A 64-row decode input is replicated at full K, and
    # all_gather_minimal_matmul_async gathers it to 2K and refuses (K=10240 vs
    # K_w=5120), so the fused branch now also requires x narrower than the weight K;
    # the 64-row decode input falls through to the 1D decode branch below it.
    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(
        lines, span,
        '        _fused_gu = self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None\n',
        '        # Lever N M3native: the fused all-gather path is for the K-sharded prefill input;\n'
        '        # a replicated (full-K) 64-row decode input takes the 1D decode branch below.\n'
        '        _fused_gu = (self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None\n'
        '                     and x.shape[-1] < w.w_gate_up.shape[-2])\n',
        'mlp _forward_tp fused-prefill layout guard')
    result = ''.join(lines)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------------
# E. C1 of the 4 x 131k DRAM plan: one gate/up copy under QWEN_FAST_SINGLE_GATEUP=1.
# ---------------------------------------------------------------------------------
#
# The model builds a SECOND, packed copy of every layer's gate/up weight (w_gate_up,
# bfloat4_b, ~w1 + w3 per layer) only for the fused all-gather + SwiGLU PREFILL matmul.
# Decode never reads it on the four-user path (it takes the 1D w1/w3 arm), so dropping it
# is worth ~3.21 GB per chip. tp_common.mlp_gateup_agmm_enabled decides it, read at
# construction by layer.py (the ff_norm skips its all-gather), mlp.py load_mlp_weights
# (w_gate_up is built) and Qwen36MLP.__init__ (the fused branch is selected).
#
# tp_common.py itself is NOT grafted. The serving path's MLP-down qualification
# (mlp_down_grid_gate, entered by dflash_combined_request) pins its sha256 from the
# simulator run, and run 35802496949 (v95) refused a grafted copy at attach: 'Simulator-
# qualified source changed: .../tp_common.py'. Editing even an unrelated function voids
# that evidence, so the flag is applied at the three CALL SITES instead - two in mlp.py
# (already grafted) and one in layer.py (pinned by no gate). All three must flip together:
# a ff_norm that skips its gather in front of an MLP that no longer fuses one would feed a
# K-sharded input to the w1/w3 matmul.
#
# With the flag unset every call site reads exactly what it did ('x and True' is x for a
# bool x). With it set, prefill runs the unfused w1/w3 2D branch on a gathered input:
# DIFFERENT ARITHMETIC (separate SiLU and multiply instead of the in-kernel SwiGLU), so it
# ships only if the byte-exact gates pass. Image A's QWEN_FAST_SINGLE_GATEUP support skips
# the two scripts/ci consumers of w_gate_up (the block stream and FusedT16Arm).
#
# Markers sit inside the executed paths (memory: graft mounted is not graft executed): one
# line per layer whose w_gate_up was not built, one per layer whose ff_norm gathers for
# itself, and one the first time any MLP runs its prefill through the 2D branch.

SINGLE_GATEUP_FLAG = 'QWEN_FAST_SINGLE_GATEUP'
MARKER_SINGLE_GATEUP = '[PINDIAG] single gate/up copy: w_gate_up not built'
MARKER_FF_NORM_GATHER = '[PINDIAG] single gate/up copy: ff_norm gathers its own input'
MARKER_PREFILL_2D = '[PINDIAG] prefill MLP via w1/w3 2D branch'
LOAD_WEIGHTS_FUNCTION = 'load_mlp_weights'
INIT_FUNCTION = '__init__'
FLAG_OFF = 'os.environ.get("' + SINGLE_GATEUP_FLAG + '") != "1"'


def module_function_span(source, name):
    """Line span [start, end) of a MODULE-level function, by AST sibling order.

    function_span only walks class bodies; mlp.py's load_mlp_weights is a module
    function. The end is the next top-level statement's first line (its decorators
    included), or the end of the file - the same 3.7-safe rule function_span uses.
    """
    tree = ast.parse(source)
    total = len(source.splitlines())
    for index, node in enumerate(tree.body):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != name:
            continue
        start = (min(d.lineno for d in node.decorator_list) if node.decorator_list else node.lineno) - 1
        if index + 1 < len(tree.body):
            following = tree.body[index + 1]
            decorators = getattr(following, 'decorator_list', None) or []
            end = (min(d.lineno for d in decorators) if decorators else following.lineno) - 1
        else:
            end = total
        return start, end
    raise ValueError('no module function named %s' % name)


def refuse_if_grafted(source, what):
    """Each C1 edit keeps its anchor inside the replacement, so a second pass would match
    again and silently double up. The flag name is absent from every original."""
    if SINGLE_GATEUP_FLAG in source:
        raise ValueError('%s: already grafted (%s present)' % (what, SINGLE_GATEUP_FLAG))


C1_SLICE_ROWS = 1024

# Module-level helper added to mlp.py under C1. At TP2 the unfused 2D prefill program for
# gate/up (N=8704 per device) does not fit L1 at 2048 rows: v100 (run 35807762937) found its
# L1 outputs in the way, and with DRAM outputs v102 (run 35808212287) still found the
# program's own circular buffers reaching 1,559,424 bytes against persistent L1 buffers from
# 1,539,072. Row slices of 1024 halve per_core_M and the buffers with it. Each slice is the
# model's own matmul (the same builder, the same activation) on fewer rows, output in DRAM,
# and the slices are joined on the row axis. Up to 1024 rows the builder is called with the
# very arguments the branch uses, so only the output placement differs.
C1_LINEAR_HELPER = (
    '\n'
    '\n'
    '_QWEN_C1_SLICE_ROWS = ' + str(C1_SLICE_ROWS) + '\n'
    '\n'
    '\n'
    'def _qwen_c1_linear(args, tpc, w, max_cols, tuning):\n'
    '    """Lever N M3native C1 (QWEN_FAST_SINGLE_GATEUP=1): the unfused prefill gate/up matmul\n'
    '    at TP2, on row slices of _QWEN_C1_SLICE_ROWS with DRAM outputs, joined on rows. Same\n'
    '    call shape as ttnn.linear; the passed program and memory configs are rebuilt per slice."""\n'
    '\n'
    '    def linear(x, weight, compute_kernel_config=None, program_config=None, memory_config=None):\n'
    '        seq = x.shape[-2]\n'
    '        rank = len(x.shape)\n'
    '        options = dict(max_cols=max_cols, tuning=tuning)\n'
    '        if weight is w.w1:\n'
    '            options["fused_activation"] = ttnn.UnaryOpType.SILU\n'
    '        parts = []\n'
    '        for start in range(0, seq, _QWEN_C1_SLICE_ROWS):\n'
    '            rows = min(_QWEN_C1_SLICE_ROWS, seq - start)\n'
    '            if rows == seq:\n'
    '                part = x\n'
    '            else:\n'
    '                begins = [0] * rank\n'
    '                ends = [x.shape[i] for i in range(rank)]\n'
    '                begins[rank - 2], ends[rank - 2] = start, start + rows\n'
    '                part = ttnn.slice(x, begins, ends)\n'
    '            config = tpc.create_prefill_mlp_matmul_program_config(rows, args.dim, weight.shape[-1], **options)\n'
    '            parts.append(ttnn.linear(part, weight, compute_kernel_config=compute_kernel_config,\n'
    '                                     program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG))\n'
    '            if part is not x:\n'
    '                ttnn.deallocate(part)\n'
    '        if len(parts) == 1:\n'
    '            return parts[0]\n'
    '        joined = ttnn.concat(parts, dim=rank - 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)\n'
    '        for part in parts:\n'
    '            ttnn.deallocate(part)\n'
    '        return joined\n'
    '\n'
    '    return linear\n'
)


def patch_mlp_single_gateup(source):
    """C1 in mlp.py: both switch call sites, the sliced prefill gate/up, and two markers.
    Inert unless the flag is 1."""
    refuse_if_grafted(source, 'mlp single gate/up')
    lines = source.splitlines(keepends=True)
    span = module_function_span(source, LOAD_WEIGHTS_FUNCTION)
    lines = replace_once(
        lines, span,
        '            if tpc.mlp_gateup_agmm_enabled(tp)\n'
        '            else None\n'
        '        )\n',
        '            # Lever N M3native C1: no packed copy under QWEN_FAST_SINGLE_GATEUP=1.\n'
        '            if tpc.mlp_gateup_agmm_enabled(tp) and ' + FLAG_OFF + '\n'
        '            else None\n'
        '        )\n'
        '        if wgu is None and not ' + FLAG_OFF + ':\n'
        '            # Lever N M3native C1 marker: logged per layer, inside the build it skips.\n'
        '            from loguru import logger as _qwen_logger\n'
        '\n'
        '            load_mlp_weights._qwen_single_gateup = getattr(load_mlp_weights, "_qwen_single_gateup", 0) + 1\n'
        '            _qwen_logger.info(\n'
        '                "' + MARKER_SINGLE_GATEUP + ' (layer {} of this process, cache {})",\n'
        '                load_mlp_weights._qwen_single_gateup,\n'
        '                tensor_cache_path,\n'
        '            )\n',
        'mlp load_mlp_weights single gate/up')

    span = function_span(''.join(lines), INIT_FUNCTION)
    lines = replace_once(
        lines, span,
        '        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)\n',
        '        # Lever N M3native C1: the fused branch goes with the packed copy.\n'
        '        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and ' + FLAG_OFF + '\n',
        'mlp __init__ fused gate/up switch')

    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(
        lines, span,
        '            seq = x.shape[-2]\n',
        '            seq = x.shape[-2]\n'
        '            if not ' + FLAG_OFF + ' and not getattr(type(self), "_qwen_2d_logged", False):\n'
        '                # Lever N M3native C1 marker: once per process, inside the branch it names.\n'
        '                type(self)._qwen_2d_logged = True\n'
        '                from loguru import logger as _qwen_logger\n'
        '\n'
        '                _qwen_logger.info("' + MARKER_PREFILL_2D + ': rows={} fused={} x={} slices of {} rows, DRAM outputs",\n'
        '                                  seq, self._fuse_gateup_agmm, getattr(x, "memory_config", lambda: None)(),\n'
        '                                  _QWEN_C1_SLICE_ROWS)\n',
        'mlp prefill 2D branch marker')
    selection = (
        '            # Lever N M3native C1: row-sliced, DRAM-output gate/up at TP2 under the flag (v100, v102).\n'
        '            _qwen_linear = ttnn.linear if ' + FLAG_OFF + ' else _qwen_c1_linear(args, tpc, w, _gw, _pt)\n')
    for weight, config in (('w1', 'pc_gate'), ('w3', 'pc_up')):
        span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
        lines = replace_once(
            lines, span,
            '            %s_out = ttnn.linear(\n'
            '                x, w.%s, compute_kernel_config=ckc, program_config=%s, memory_config=ttnn.L1_MEMORY_CONFIG\n'
            % (weight, weight, config),
            (selection if weight == 'w1' else '') +
            '            %s_out = _qwen_linear(\n'
            '                x, w.%s, compute_kernel_config=ckc, program_config=%s, memory_config=ttnn.L1_MEMORY_CONFIG\n'
            % (weight, weight, config),
            'mlp prefill 2D %s call' % weight)
    result = ''.join(lines)
    anchor = '\n\nclass Qwen36MLP:\n'
    if result.count(anchor) != 1:
        raise ValueError('mlp C1 helper: expected one Qwen36MLP class anchor, found %d' % result.count(anchor))
    result = result.replace(anchor, C1_LINEAR_HELPER + anchor)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------------
# F. C1c and C1d: getting back the C1 prefill penalty (prefill ranking, levers 3 and 3a).
# ---------------------------------------------------------------------------------
#
# C1 costs +71-76 ms per 2048-row chunk against the fused v98 path (v99 vs v104 at 32k):
# the unfused w1/w3 run as two row-sliced matmuls, each joined by its own concat, and a
# separate mul follows. Two recoveries, both inside mlp.py:
#
# C1c (EXACT; the default whenever QWEN_FAST_SINGLE_GATEUP=1). Per 1024-row slice of x,
# slice ONCE, run the w1 matmul and the w3 matmul on that one slice, multiply the two, and
# join only the products. It is bit-identical to C1 by construction, op for op:
#   - every matmul is the same program: the same builder
#     (create_prefill_mlp_matmul_program_config) with the same (rows, dim, N) and the same
#     max_cols / tuning / fused_activation (SiLU on w1 only), the same compute kernel config
#     and the same DRAM output, on the same rows of x (ttnn.slice is a copy, and C1 made the
#     same slice twice where C1c makes it once);
#   - the mul is elementwise with no broadcast (both operands [.., rows, N], bf16, TILE,
#     DRAM interleaved in both cases), so each output element is the product of the same
#     two input elements under the same compute config. Multiplying per slice and joining
#     is the same values as joining and multiplying; only the tile count per call changes;
#   - ttnn.concat moves the same rows into the same order.
# What changes is only the number of ops (per chunk: two concats of [2048, 8704] become one)
# and the peak DRAM held between them. QWEN_FAST_C1_LEGACY=1 keeps today's C1 path for a
# byte comparison of the two on hardware.
#
# C1d (QWEN_FAST_C1_AGMM=1, only together with QWEN_FAST_SINGLE_GATEUP=1). The ff_norm skips
# its all-gather again (layer.py's _fuse_ff_agmm stays True), and the MLP fuses that gather
# into two tpc.all_gather_matmul_prefill calls on the separate w1 (SiLU fused) and w3, then
# one mul. tp_common is untouched: all_gather_matmul_prefill is already its public helper.
# NOT bitwise vs C1 (a different matmul program and accumulation order); the gate decides.
# The branch sits where v98's fused branch sits (first, K-sharded input only), so the same
# rows reach it that reached v98's fused path.
#
# Markers sit inside each executed branch, once per process.

C1_AGMM_FLAG = 'QWEN_FAST_C1_AGMM'
C1_LEGACY_FLAG = 'QWEN_FAST_C1_LEGACY'
C1D_ON = ('os.environ.get("' + SINGLE_GATEUP_FLAG + '") == "1" and os.environ.get("'
          + C1_AGMM_FLAG + '") == "1"')
MARKER_C1C = '[PINDIAG] prefill MLP C1c: one slice of x per 1024 rows, product-only concat'
MARKER_C1D = '[PINDIAG] prefill MLP C1d: two all_gather_matmul_prefill (w1+SiLU, w3) and a mul'
MARKER_C1D_FF_NORM = '[PINDIAG] C1d: ff_norm skips its all-gather for the MLP AGMM'

# Module-level helper, placed after _qwen_c1_linear (inside the same helper region).
C1C_SWIGLU_HELPER = (
    '\n'
    '\n'
    'def _qwen_c1_swiglu(args, tpc, w, max_cols, tuning):\n'
    '    """Lever N M3native C1c (QWEN_FAST_SINGLE_GATEUP=1): silu(x @ w1) * (x @ w3) on row slices\n'
    '    of _QWEN_C1_SLICE_ROWS. Each slice of x is made once and feeds both matmuls, and only the\n'
    '    products are joined. Per slice the matmuls and the mul are exactly _qwen_c1_linear\'s and\n'
    '    the branch\'s own, so the result is bit-identical to C1 (see the patch module, section F)."""\n'
    '\n'
    '    def swiglu(x, compute_kernel_config=None):\n'
    '        seq = x.shape[-2]\n'
    '        rank = len(x.shape)\n'
    '        parts = []\n'
    '        for start in range(0, seq, _QWEN_C1_SLICE_ROWS):\n'
    '            rows = min(_QWEN_C1_SLICE_ROWS, seq - start)\n'
    '            if rows == seq:\n'
    '                part = x\n'
    '            else:\n'
    '                begins = [0] * rank\n'
    '                ends = [x.shape[i] for i in range(rank)]\n'
    '                begins[rank - 2], ends[rank - 2] = start, start + rows\n'
    '                part = ttnn.slice(x, begins, ends)\n'
    '            gate_config = tpc.create_prefill_mlp_matmul_program_config(\n'
    '                rows, args.dim, w.w1.shape[-1], max_cols=max_cols, tuning=tuning,\n'
    '                fused_activation=ttnn.UnaryOpType.SILU)\n'
    '            up_config = tpc.create_prefill_mlp_matmul_program_config(\n'
    '                rows, args.dim, w.w3.shape[-1], max_cols=max_cols, tuning=tuning)\n'
    '            gate = ttnn.linear(part, w.w1, compute_kernel_config=compute_kernel_config,\n'
    '                               program_config=gate_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)\n'
    '            up = ttnn.linear(part, w.w3, compute_kernel_config=compute_kernel_config,\n'
    '                             program_config=up_config, memory_config=ttnn.DRAM_MEMORY_CONFIG)\n'
    '            if part is not x:\n'
    '                ttnn.deallocate(part)\n'
    '            parts.append(ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG))\n'
    '            ttnn.deallocate(gate)\n'
    '            ttnn.deallocate(up)\n'
    '        if len(parts) == 1:\n'
    '            return parts[0]\n'
    '        joined = ttnn.concat(parts, dim=rank - 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)\n'
    '        for part in parts:\n'
    '            ttnn.deallocate(part)\n'
    '        return joined\n'
    '\n'
    '    return swiglu\n'
)

C1D_INIT_BLOCK = (
    '        if ' + C1D_ON + ':\n'
    '            # Lever N M3native C1d: two all_gather_matmul_prefill calls on w1/w3 take the ff_norm\'s\n'
    '            # gather back (layer.py keeps _fuse_ff_agmm True under the same two flags).\n'
    '            self._qwen_c1_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)\n'
)

C1D_BRANCH = (
    '        elif getattr(self, "_qwen_c1_agmm", False) and x.shape[-2] > ttnn.TILE_SIZE and x.shape[-1] < w.w1.shape[-2]:\n'
    '            # Lever N M3native C1d (QWEN_FAST_C1_AGMM=1): x is K-sharded (the ff_norm skipped its\n'
    '            # gather). Two fused all-gather + matmul calls on the separate w1 (SiLU fused) and w3,\n'
    '            # then one mul; hidden is produced here, so the gate*up below is skipped.\n'
    '            if not getattr(type(self), "_qwen_c1d_logged", False):\n'
    '                type(self)._qwen_c1d_logged = True\n'
    '                from loguru import logger as _qwen_logger\n'
    '\n'
    '                _qwen_logger.info("' + MARKER_C1D + ': rows={} k_local={}", x.shape[-2], x.shape[-1])\n'
    '            _qwen_gate = tpc.all_gather_matmul_prefill(\n'
    '                x, w.w1, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology(),\n'
    '                fused_activation=ttnn.UnaryOpType.SILU,\n'
    '            )\n'
    '            _qwen_up = tpc.all_gather_matmul_prefill(\n'
    '                x, w.w3, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()\n'
    '            )\n'
    '            hidden = ttnn.mul(_qwen_gate, _qwen_up, memory_config=mc)\n'
    '            ttnn.deallocate(_qwen_gate)\n'
    '            ttnn.deallocate(_qwen_up)\n'
    '            _silu_fused = True\n'
    '            _fused_gu = True\n'
)

C1C_BRANCH = (
    '        elif x.shape[-2] > ttnn.TILE_SIZE and os.environ.get("' + SINGLE_GATEUP_FLAG + '") == "1" and os.environ.get("'
    + C1_LEGACY_FLAG + '") != "1":\n'
    '            # Lever N M3native C1c: the C1 prefill gate/up with one slice of x per 1024 rows feeding\n'
    '            # both matmuls and only the product joined - bit-identical to C1 (patch module, F).\n'
    '            # hidden is produced here, so the gate*up below is skipped.\n'
    '            if not getattr(type(self), "_qwen_c1c_logged", False):\n'
    '                type(self)._qwen_c1c_logged = True\n'
    '                from loguru import logger as _qwen_logger\n'
    '\n'
    '                _qwen_logger.info("' + MARKER_C1C + ': rows={} slices of {} rows", x.shape[-2], _QWEN_C1_SLICE_ROWS)\n'
    '            hidden = _qwen_c1_swiglu(\n'
    '                args, tpc, w, getattr(args, "decode_grid_w", 8), getattr(args, "prefill_tuning", None)\n'
    '            )(x, compute_kernel_config=ckc)\n'
    '            _silu_fused = True\n'
    '            _fused_gu = True\n'
)


def patch_mlp_c1_fused(source):
    """C1c and C1d in mlp.py, on top of C1 (patch_mlp_single_gateup must already be applied)."""
    if C1_AGMM_FLAG in source:
        raise ValueError('mlp C1c/C1d: already grafted (%s present)' % C1_AGMM_FLAG)
    if SINGLE_GATEUP_FLAG not in source:
        raise ValueError('mlp C1c/C1d: needs the C1 graft first (%s absent)' % SINGLE_GATEUP_FLAG)
    lines = source.splitlines(keepends=True)
    # Ahead of C1's own switch (not after it), so the block is followed by code, not a blank line.
    init_lines = ('        # Lever N M3native C1: the fused branch goes with the packed copy.\n'
                  '        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and '
                  + FLAG_OFF + '\n')
    span = function_span(source, INIT_FUNCTION)
    lines = replace_once(lines, span, init_lines, C1D_INIT_BLOCK + init_lines, 'mlp __init__ C1d switch')
    dram_sharded_branch = '        elif getattr(self, "_dram_sharded", False) and x.shape[-2] <= ttnn.TILE_SIZE:\n'
    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(lines, span, dram_sharded_branch, C1D_BRANCH + dram_sharded_branch,
                         'mlp C1d branch (after the fused gate/up branch)')
    prefill_2d_branch = '        elif x.shape[-2] > ttnn.TILE_SIZE:\n'
    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(lines, span, prefill_2d_branch, C1C_BRANCH + prefill_2d_branch,
                         'mlp C1c branch (before the prefill 2D branch)')
    result = ''.join(lines)
    anchor = '\n\nclass Qwen36MLP:\n'
    if result.count(anchor) != 1 or result.count('\ndef _qwen_c1_linear(') != 1:
        raise ValueError('mlp C1c helper: expected one Qwen36MLP class anchor after _qwen_c1_linear')
    if result.index('\ndef _qwen_c1_linear(') > result.index(anchor):
        raise ValueError('mlp C1c helper: _qwen_c1_linear must precede Qwen36MLP')
    result = result.replace(anchor, C1C_SWIGLU_HELPER + anchor)
    ast.parse(result)
    return result


def patch_mlp_full(source):
    """The table maps ONE function per file: the 64-row decode gates, then C1, then C1c/C1d."""
    return patch_mlp_c1_fused(patch_mlp_single_gateup(patch_mlp(source)))


def patch_layer(source):
    """C1 in layer.py: the ff_norm keeps its own all-gather when the MLP no longer fuses it,
    and skips it again under C1d (QWEN_FAST_C1_AGMM=1), where the MLP fuses it once more."""
    refuse_if_grafted(source, 'layer ff_norm gather')
    lines = source.splitlines(keepends=True)
    span = function_span(source, INIT_FUNCTION)
    lines = replace_once(
        lines, span,
        '        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)\n',
        '        # Lever N M3native C1: under QWEN_FAST_SINGLE_GATEUP=1 the MLP builds no packed\n'
        '        # gate/up copy and never fuses the gather, so the ff_norm must gather for itself.\n'
        '        import os\n'
        '\n'
        '        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices) and ' + FLAG_OFF + '\n'
        '        if ' + C1D_ON + ':\n'
        '            # Lever N M3native C1d: the MLP fuses the gather back (two all_gather_matmul_prefill\n'
        '            # calls on w1/w3, mlp.py\'s _qwen_c1_agmm), so the ff_norm skips it as before C1.\n'
        '            self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)\n'
        '            if self._fuse_ff_agmm:\n'
        '                from loguru import logger as _qwen_logger\n'
        '\n'
        '                _qwen_logger.info("' + MARKER_C1D_FF_NORM + ' (layer {})", layer_num)\n'
        '        if self.num_devices > 1 and not self._fuse_ff_agmm and not ' + FLAG_OFF + ':\n'
        '            from loguru import logger as _qwen_logger\n'
        '\n'
        '            _qwen_logger.info("' + MARKER_FF_NORM_GATHER + ' (layer {})", layer_num)\n',
        'layer ff_norm fused all-gather switch')
    result = ''.join(lines)
    ast.parse(result)
    return result


# ---------------------------------------------------------------------------------
# G. M2 of the prefill ranking: the prefill device-profile flush hook in layer.py.
# ---------------------------------------------------------------------------------
#
# The device profiler's per-core buffers overflow inside ONE 2048-row prefill chunk (gate12b
# lost rows after about scan 22 of a forward; run 35422536834 caught only 37-45% of its
# forwards), because prefill reads the profiler only at the end or on the packed-verify dump
# round. Under QWEN_PREFILL_PROFILE_FLUSH=1 every 16th decoder layer of a prefill forward
# synchronises the mesh and calls ttnn.ReadDeviceProfiler, so no more than 16 layers of ops
# are ever buffered. A module counter, bumped on each layer-0 prefill call and reset to 0 when
# that call's chunk_start_idx is 0 (or None), is the PROMPT-RELATIVE chunk index, so warm-up
# or probe prefills earlier in the process cannot shift it (the absolute call count is only
# logged). Chunks 0, 1, 31, 32, 62 and 63 of every prompt are bracketed by tracy signposts
# qwen_prefill_p<prompt>_chunk_<n>_begin / _end. The counter assumes one prompt's chunks run
# back to back (users=1, the M2 arm). Unset, the only difference is one env lookup per layer
# call on the prefill path: no device op, no synchronisation, no state change.
#
# The arm's tracy flags (--disable-device-data-dump-to-files, no ops report) produce no
# tracy_ops_data.csv, so the signposts reach only the host .tracy capture; the report's
# primary per-chunk split is device-only (lever_n_prefill_profile_report.split_by_sdpa).

PREFILL_PROFILE_FLAG = 'QWEN_PREFILL_PROFILE_FLUSH'
PREFILL_PROFILE_EVERY = 16
PREFILL_PROFILE_SIGNPOST_CHUNKS = (0, 1, 31, 32, 62, 63)
MARKER_PREFILL_FLUSH = '[PINDIAG] prefill profile flush'
FORWARD_FUNCTION = 'forward'

PREFILL_PROFILE_HELPERS = (
    '\n'
    '\n'
    '# Lever N M2 prefill profile hook (QWEN_PREFILL_PROFILE_FLUSH=1); inert when unset.\n'
    '_QWEN_PREFILL_PROFILE_FLAG = "' + PREFILL_PROFILE_FLAG + '"\n'
    '_QWEN_PREFILL_PROFILE_EVERY = ' + str(PREFILL_PROFILE_EVERY) + '\n'
    '_QWEN_PREFILL_PROFILE_SIGNPOST_CHUNKS = ' + repr(PREFILL_PROFILE_SIGNPOST_CHUNKS) + '\n'
    '_QWEN_PREFILL_PROFILE = {"chunk": -1, "prompt": 0, "calls": 0, "label": None, "flushes": 0, "signpost": None}\n'
    '\n'
    '\n'
    'def _qwen_prefill_profile_on():\n'
    '    import os\n'
    '\n'
    '    return os.environ.get(_QWEN_PREFILL_PROFILE_FLAG) == "1"\n'
    '\n'
    '\n'
    'def _qwen_prefill_signpost(label):\n'
    '    state = _QWEN_PREFILL_PROFILE\n'
    '    if state["signpost"] is None:\n'
    '        try:\n'
    '            from tracy import signpost\n'
    '        except Exception as error:  # noqa: BLE001 - a missing tracy must not kill prefill\n'
    '            from loguru import logger as _qwen_logger\n'
    '\n'
    '            _qwen_logger.info("' + MARKER_PREFILL_FLUSH + ': tracy signpost unavailable ({})", error)\n'
    '            signpost = False\n'
    '        state["signpost"] = signpost\n'
    '    if state["signpost"]:\n'
    '        state["signpost"](label)\n'
    '\n'
    '\n'
    'def _qwen_prefill_profile_begin(layer, x, chunk_start_idx):\n'
    '    """Layer 0 of a prefill forward starts a new chunk; the chosen chunks get a begin signpost.\n'
    '\n'
    '    The chunk index is prompt-relative: a forward at chunk_start_idx 0 (or None, an unchunked\n'
    '    prompt) opens a new prompt at chunk 0, so warm-up or probe prefills earlier in the process\n'
    '    cannot shift it. The absolute count of layer-0 prefill calls is only logged."""\n'
    '    if layer.layer_num != 0:\n'
    '        return\n'
    '    state = _QWEN_PREFILL_PROFILE\n'
    '    state["calls"] += 1\n'
    '    if chunk_start_idx is None or (isinstance(chunk_start_idx, int) and chunk_start_idx == 0):\n'
    '        state["prompt"] += 1\n'
    '        state["chunk"] = 0\n'
    '    else:\n'
    '        state["chunk"] += 1\n'
    '    chunk = state["chunk"]\n'
    '    state["label"] = None\n'
    '    if chunk in _QWEN_PREFILL_PROFILE_SIGNPOST_CHUNKS:\n'
    '        from loguru import logger as _qwen_logger\n'
    '\n'
    '        state["label"] = "qwen_prefill_p%d_chunk_%d" % (state["prompt"], chunk)\n'
    '        _qwen_prefill_signpost(state["label"] + "_begin")\n'
    '        _qwen_logger.info("' + MARKER_PREFILL_FLUSH + ': prompt {} chunk {} begin rows={} chunk_start_idx={} "\n'
    '                          "(layer-0 prefill call {})",\n'
    '                          state["prompt"], chunk, x.shape[-2], chunk_start_idx, state["calls"])\n'
    '\n'
    '\n'
    'def _qwen_prefill_profile_end(layer):\n'
    '    """Every 16th layer (and the last): drain the device profiler so a chunk cannot overflow it."""\n'
    '    state = _QWEN_PREFILL_PROFILE\n'
    '    n_layers = getattr(layer.args, "n_layers", None)\n'
    '    last = n_layers is not None and layer.layer_num == n_layers - 1\n'
    '    if not last and layer.layer_num % _QWEN_PREFILL_PROFILE_EVERY != _QWEN_PREFILL_PROFILE_EVERY - 1:\n'
    '        return\n'
    '    ttnn.synchronize_device(layer.device)\n'
    '    if last and state["label"] is not None:\n'
    '        _qwen_prefill_signpost(state["label"] + "_end")\n'
    '        state["label"] = None\n'
    '    ttnn.ReadDeviceProfiler(layer.device)\n'
    '    state["flushes"] += 1\n'
    '    if state["flushes"] == 1:\n'
    '        from loguru import logger as _qwen_logger\n'
    '\n'
    '        _qwen_logger.info("' + MARKER_PREFILL_FLUSH + ': first flush at prompt {} chunk {} layer {} (every {} layers)",\n'
    '                          state["prompt"], state["chunk"], layer.layer_num, _QWEN_PREFILL_PROFILE_EVERY)\n'
)

PREFILL_PROFILE_BEGIN = (
    '        if mode == "prefill" and _qwen_prefill_profile_on():\n'
    '            # Lever N M2 prefill profile (QWEN_PREFILL_PROFILE_FLUSH=1): layer 0 counts the chunk.\n'
    '            _qwen_prefill_profile_begin(self, x, chunk_start_idx)\n'
)

PREFILL_PROFILE_END = (
    '        if mode == "prefill" and _qwen_prefill_profile_on():\n'
    '            # Lever N M2 prefill profile: drain the device profiler every 16th layer.\n'
    '            _qwen_prefill_profile_end(self)\n'
)


def patch_layer_profile_flush(source):
    """M2 in layer.py: the prefill profile flush hook, inert unless QWEN_PREFILL_PROFILE_FLUSH=1."""
    if PREFILL_PROFILE_FLAG in source:
        raise ValueError('layer prefill profile flush: already grafted (%s present)' % PREFILL_PROFILE_FLAG)
    lines = source.splitlines(keepends=True)
    norm_mode = '        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE\n'
    span = function_span(source, FORWARD_FUNCTION)
    lines = replace_once(lines, span, norm_mode, norm_mode + PREFILL_PROFILE_BEGIN,
                         'layer forward prefill profile begin')
    span = function_span(''.join(lines), FORWARD_FUNCTION)
    lines = replace_once(lines, span, '        return output\n', PREFILL_PROFILE_END + '        return output\n',
                         'layer forward prefill profile flush')
    result = ''.join(lines)
    anchor = '\n\nclass Qwen36DecoderLayer:\n'
    if result.count(anchor) != 1:
        raise ValueError('layer profile helpers: expected one Qwen36DecoderLayer class anchor, found %d'
                         % result.count(anchor))
    result = result.replace(anchor, PREFILL_PROFILE_HELPERS + anchor)
    ast.parse(result)
    return result


def patch_layer_full(source):
    """The table maps ONE function per file: C1 (and C1d) in __init__, then the M2 flush hook."""
    return patch_layer_profile_flush(patch_layer(source))


# ---------------------------------------------------------------------------------
# H. Lever #2: the GDN prefill conv as one generic_op (gdn_prefill_conv_exact).
# ---------------------------------------------------------------------------------
#
# forward_prefill's valid_len FIR (_causal_conv1d_fir: concat, a host one-hot + matmul for the
# carry, three untilize/slice/tilize windows, multiply, three addcmul, SiLU) and the three q/k/v
# slices after it are 22 device ops, 2.28 ms per layer-chunk at T=2048 (gate12b). Under
# QWEN_FAST_GDN_PREFILL_CONV=1 one launch of gdn_prefill_conv_exact (mounted beside gdn/tp.py,
# PREFILL_CONV_FILES) replaces them with the same LLK calls on the same bytes.
#
# Only the valid_len FIR is replaced. It writes a host one-hot (ttnn.from_torch), which TT_FATALs
# under a trace capture, so the call site this branch takes over can never be inside one; the
# valid_len-None conv1d branch ahead of it (the trace-safe one) is untouched and keeps priority.
#
# Flag unset: _qwen_prefill_conv_on returns False on its first line - no import, no device op -
# and forward_prefill runs the FIR, the three slices and the conv deallocate exactly as before.
# The marker sits inside the executed branch, once per prefill chunk (the first GDN layer of
# each forward), carrying the previous chunk's call count. From chunk 2 on, a completion line is
# logged when a chunk's calls reach chunk 1's count, so the gate can count the LAST chunk too
# (no later marker reports it). QWEN_FAST_GDN_PREFILL_CONV_AUDIT=<n> additionally runs the FIR
# beside the first n calls and raises on any byte difference.

PREFILL_CONV_FLAG = 'QWEN_FAST_GDN_PREFILL_CONV'
PREFILL_CONV_AUDIT_FLAG = 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT'
MARKER_PREFILL_CONV = '[PINDIAG] GDN prefill conv engaged'
MARKER_PREFILL_CONV_FALLBACK = '[PINDIAG] GDN prefill conv fell back to the FIR'
MARKER_PREFILL_CONV_AUDIT = '[PINDIAG] GDN prefill conv audit'
MARKER_PREFILL_CONV_COMPLETE = '[PINDIAG] GDN prefill conv chunk complete'
FORWARD_PREFILL_FUNCTION = 'forward_prefill'
PREFILL_CONV_MODULE = 'models.demos.blackhole.qwen36.tt.gdn.gdn_prefill_conv_exact'

# The unpatched files the graft mounts next to the patched gdn/tp.py: {graft-relative path:
# scripts/ci source}. The ONE table: the workflow stages graft/<path> from it and the arm
# (lever_n_m3native_run_arm.sh) derives its single-file mounts from it, so neither list can drift.
PREFILL_CONV_FILES = {'gdn/' + name: name for name in _pcx.RUNTIME_FILES}

PREFILL_CONV_HELPERS = (
    '\n'
    '\n'
    '# Lever #2 (' + PREFILL_CONV_FLAG + '=1): gdn_prefill_conv_exact replaces the valid_len FIR + q/k/v slices.\n'
    '_QWEN_PREFILL_CONV = {"calls": 0, "first": None, "chunk": 0, "chunk_calls": 0, "expected": None, "audited": 0}\n'
    '\n'
    '\n'
    'def _qwen_prefill_conv_on(layer, qkv, cstate, T, valid_len):\n'
    '    if os.environ.get("' + PREFILL_CONV_FLAG + '") != "1":\n'
    '        return False  # flag off: no import, no device op, the FIR path unchanged\n'
    '    if valid_len is None:\n'
    '        return False  # only the valid_len FIR (a host one-hot write: never under trace capture)\n'
    '    from ' + PREFILL_CONV_MODULE.rsplit('.', 1)[0] + ' import ' + PREFILL_CONV_MODULE.rsplit('.', 1)[1] + ' as _pcx\n'
    '\n'
    '    reason = _pcx.unsupported(ttnn, layer.mesh, qkv, cstate, layer.tw["conv_taps"], valid_len, layer.key_dim_tp,\n'
    '                              flat=layer._gdn_flat_qkv, kernel_size=layer.K)\n'
    '    if reason is not None:\n'
    '        from loguru import logger as _qwen_logger\n'
    '\n'
    '        _qwen_logger.warning("' + MARKER_PREFILL_CONV_FALLBACK + ': {} (T={} valid_len={})", reason, T, valid_len)\n'
    '    return reason is None\n'
    '\n'
    '\n'
    'def _qwen_prefill_conv(layer, qkv, cstate, valid_len, T):\n'
    '    """(q, k, v, new_state) from one gdn_prefill_conv_exact launch, byte-identical to the FIR + slices."""\n'
    '    from ' + PREFILL_CONV_MODULE.rsplit('.', 1)[0] + ' import ' + PREFILL_CONV_MODULE.rsplit('.', 1)[1] + ' as _pcx\n'
    '\n'
    '    out = _pcx.gdn_prefill_conv_exact(layer.mesh, qkv, cstate, layer.tw["conv_taps"], valid_len=valid_len,\n'
    '                                      key_dim_tp=layer.key_dim_tp)\n'
    '    state = _QWEN_PREFILL_CONV\n'
    '    state["calls"] += 1\n'
    '    if state["first"] is None:\n'
    '        state["first"] = id(layer)\n'
    '    if id(layer) == state["first"]:\n'
    '        # The first GDN layer of a forward starts a chunk: one marker per prefill chunk, inside the\n'
    '        # branch it names, with the calls the previous chunk made (every GDN layer, if it engaged).\n'
    '        from loguru import logger as _qwen_logger\n'
    '\n'
    '        state["chunk"] += 1\n'
    '        _qwen_logger.info("' + MARKER_PREFILL_CONV + ': chunk {} previous_chunk_calls={} calls={} T={} valid_len={} carry={}",\n'
    '                          state["chunk"], state["chunk_calls"], state["calls"], T, valid_len, cstate is not None)\n'
    '        if state["chunk"] == 2:\n'
    '            state["expected"] = state["chunk_calls"]  # chunk 1\'s count: what a complete chunk makes\n'
    '        state["chunk_calls"] = 0\n'
    '    state["chunk_calls"] += 1\n'
    '    if state["expected"] is not None and state["chunk_calls"] == state["expected"]:\n'
    '        # The last chunk has no later marker to report it: say when a chunk completes.\n'
    '        from loguru import logger as _qwen_logger\n'
    '\n'
    '        _qwen_logger.info("' + MARKER_PREFILL_CONV_COMPLETE + ': chunk {} calls={}", state["chunk"], state["chunk_calls"])\n'
    '    if state["audited"] < int(os.environ.get("' + PREFILL_CONV_AUDIT_FLAG + '", "0") or 0):\n'
    '        # qkv and the carry still hold this chunk\'s inputs: the FIR runs beside the op, bytes compared.\n'
    '        state["audited"] += 1\n'
    '        report = _pcx.audit_against_fir(ttnn, _causal_conv1d_fir, layer.mesh, qkv, cstate, layer.tw["conv_taps"],\n'
    '                                        valid_len, layer.key_dim_tp, out, kernel_size=layer.K)\n'
    '        from loguru import logger as _qwen_logger\n'
    '\n'
    '        _qwen_logger.info("' + MARKER_PREFILL_CONV_AUDIT + ' {} exact={} T={} valid_len={} carry={} mismatches={}",\n'
    '                          state["audited"], report["exact"], T, valid_len, cstate is not None, report["mismatches"])\n'
    '        if not report["exact"]:\n'
    '            raise AssertionError("GDN prefill conv differs from the FIR: %r" % (report,))\n'
    '    return out\n'
)

PREFILL_CONV_A_OLD = (
    '        if self._gdn_conv1d and valid_len is None:\n'
    '            # Native depthwise ttnn.conv1d (masked buckets keep the MAC FIR: valid_len new_state differs)\n'
    '            conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)\n'
    '        else:\n'
)
PREFILL_CONV_A_NEW = (
    '        _qwen_pc = None\n'
    '        if self._gdn_conv1d and valid_len is None:\n'
    '            # Native depthwise ttnn.conv1d (masked buckets keep the MAC FIR: valid_len new_state differs)\n'
    '            conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)\n'
    '        elif _qwen_prefill_conv_on(self, qkv, _cstate, T, valid_len):\n'
    '            # Lever #2 (' + PREFILL_CONV_FLAG + '=1): one tiled launch replaces the FIR + q/k/v slices.\n'
    '            _qwen_pc = _qwen_prefill_conv(self, qkv, _cstate, valid_len, T)\n'
    '            conv, conv_new_state = None, _qwen_pc[3]\n'
    '        else:\n'
)
PREFILL_CONV_B_OLD = (
    '        kd = self.key_dim_tp\n'
    '        if self._gdn_flat_qkv:\n'
)
PREFILL_CONV_B_NEW = (
    '        kd = self.key_dim_tp\n'
    '        if _qwen_pc is not None:\n'
    '            # Lever #2: q/k/v come straight from the op (the flat layout; no conv tensor to slice).\n'
    '            q, k, v = _qwen_pc[0], _qwen_pc[1], _qwen_pc[2]\n'
    '            _qkv_head_dims = (Nk, Dk, Nv, Dv)\n'
    '        elif self._gdn_flat_qkv:\n'
)
PREFILL_CONV_C_OLD = (
    '            _qkv_head_dims = None\n'
    '        ttnn.deallocate(conv)\n'
)
PREFILL_CONV_C_NEW = (
    '            _qkv_head_dims = None\n'
    '        if _qwen_pc is None:\n'
    '            ttnn.deallocate(conv)\n'
)


def patch_gdn_tp_prefill_conv(source):
    """Lever #2 in gdn/tp.py forward_prefill: three anchored edits and the module helpers."""
    if PREFILL_CONV_FLAG in source:
        raise ValueError('gdn prefill conv: already grafted (%s present)' % PREFILL_CONV_FLAG)
    lines = source.splitlines(keepends=True)
    for old, new, what in ((PREFILL_CONV_A_OLD, PREFILL_CONV_A_NEW, 'gdn forward_prefill conv branch'),
                           (PREFILL_CONV_B_OLD, PREFILL_CONV_B_NEW, 'gdn forward_prefill q/k/v split'),
                           (PREFILL_CONV_C_OLD, PREFILL_CONV_C_NEW, 'gdn forward_prefill conv deallocate')):
        span = function_span(''.join(lines), FORWARD_PREFILL_FUNCTION)
        lines = replace_once(lines, span, old, new, what)
    result = ''.join(lines)
    anchor = '\n\nclass TPGatedDeltaNet:\n'
    if result.count(anchor) != 1:
        raise ValueError('gdn prefill conv helpers: expected one TPGatedDeltaNet class anchor, found %d'
                         % result.count(anchor))
    result = result.replace(anchor, PREFILL_CONV_HELPERS + anchor)
    ast.parse(result)
    return result


def patch_gdn_tp_full(source):
    """The table maps ONE function per file: the 64-row decode gates (and the slot remap), then lever #2."""
    return patch_gdn_tp_prefill_conv(patch_gdn_tp(source))


PATCHES = {
    'model_config.py': patch_model_config,
    'attention/tp.py': patch_attention_tp,
    'gdn/tp.py': patch_gdn_tp_full,
    'mlp.py': patch_mlp_full,
    'layer.py': patch_layer_full,
}

# The M1 files, added for Lever N. They live in TWO image trees, and their patches
# belong to lever_n_model_patch rather than this module - this table only says which
# file comes from where, so the workflow can derive its docker cp and sha256sum lists
# instead of carrying four hardcoded copies of the same knowledge. That duplication is
# exactly what let the 65536 hardware lane ship a runner nothing staged.
MODEL_ROOT = '/opt/tt-metal/models/demos/blackhole/qwen36/tt'
PLUGIN_ROOT = '/opt/qwen-fast-plugin/src/vllm_tt_plugin'

# relative path -> (image directory, patch callable)
SOURCES = {name: (MODEL_ROOT, patch) for name, patch in PATCHES.items()}


def with_lever_n():
    """SOURCES plus the three M1 files. Separate so an arm can graft the decode-side
    four alone, which is every arm that does not set M3NATIVE_PREFILL_CHUNK_TOKENS."""
    import lever_n_model_patch as m1
    extra = {
        'model.py': (MODEL_ROOT, m1.patch_model),
        'qwen36_vllm.py': (MODEL_ROOT, m1.patch_vllm_entry),
        'platform.py': (PLUGIN_ROOT, m1.patch_platform),
        # M2 item 2, one prefill in flight. A PLUGIN file, so it is delivered by
        # bind-mount like platform.py and needs no image rebuild. The overlay route
        # (serving_one_in_flight.install setting scheduler_cls) cannot work:
        # platform.check_and_update_config overwrites that attribute afterwards, which
        # run 35690327326 proved with a marker that never fired.
        # BOTH M2 edits: one-in-flight in _schedule_prefill_only, and the
        # alternation of section 3.3 in schedule()'s default branch. The table
        # maps ONE function per file, so patch_scheduler_full composes them -
        # mapping only patch_scheduler is how the alternation goes missing.
        'scheduler.py': (PLUGIN_ROOT, m1.patch_scheduler_full),
        # The same alternation for a LANE-MODE deployment. Run 35707860782 proved
        # this class is inert here: check_and_update_config builds TTLaneCoordinator
        # only when uses_tt_lane_coordinator() is true, and that run logged
        # data_parallel_size=1 and loaded vllm_tt_plugin.scheduler.TTScheduler, so
        # the graft was mounted, correct, and never executed. It stays because the
        # two seats are mutually exclusive - in lane mode _forced_mode is set every
        # step, so TTScheduler's default branch is unreachable - and whichever class
        # the platform picks, exactly one alternation policy is live.
        'lane_scheduler.py': (PLUGIN_ROOT, m1.patch_lane_scheduler),
    }
    overlap = set(extra) & set(SOURCES)
    if overlap:
        raise ValueError('M1 and m3native graft the same file: %r' % sorted(overlap))
    return dict(SOURCES, **extra)


def stage(root, output=None):
    """Read the four originals from `root`, apply the four patches, write the four
    patched files under `output` (default: in place, same as `root`).

    Returns {relative path: written Path}. Every output is ast.parse'd (both inside
    each patch_* function and again here) and every replace_once inside the patches
    already asserts its own one-occurrence match, so a mismatch anywhere - a moved
    anchor, a source that no longer matches the probe dump - fails loudly here rather
    than producing a graft that silently does nothing.
    """
    root = Path(root)
    output = Path(output) if output else root
    written = {}
    for relative, patch in PATCHES.items():
        source = (root / relative).read_text(encoding='utf-8')
        patched = patch(source)
        ast.parse(patched)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patched, encoding='utf-8', newline='\n')
        written[relative] = target
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='directory holding model_config.py, attention/tp.py, '
                                     'gdn/tp.py and mlp.py (the qwen36/tt originals)')
    parser.add_argument('--output', help='directory to write the patched files into '
                                         '(default: alongside the originals)')
    options = parser.parse_args()
    for relative, target in stage(options.root, options.output).items():
        print('wrote %s -> %s' % (relative, target))
