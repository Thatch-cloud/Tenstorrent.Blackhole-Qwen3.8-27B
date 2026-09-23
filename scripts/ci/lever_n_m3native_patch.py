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


def patch_mlp_single_gateup(source):
    """C1 in mlp.py: both switch call sites, plus two markers. Inert unless the flag is 1."""
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
        '                _qwen_logger.info("' + MARKER_PREFILL_2D + ': rows={} fused={} x={} outputs=DRAM", seq,\n'
        '                                  self._fuse_gateup_agmm, getattr(x, "memory_config", lambda: None)())\n'
        '            # Lever N M3native C1: at TP2 the gate/up outputs (N=8704 per device) do not fit L1 (v100).\n'
        '            _qwen_c1_out = ttnn.L1_MEMORY_CONFIG if ' + FLAG_OFF + ' else ttnn.DRAM_MEMORY_CONFIG\n',
        'mlp prefill 2D branch marker')
    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(
        lines, span,
        '                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG\n',
        '                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=_qwen_c1_out\n',
        'mlp prefill 2D w1 output placement')
    span = function_span(''.join(lines), FORWARD_TP_FUNCTION)
    lines = replace_once(
        lines, span,
        '                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG\n',
        '                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=_qwen_c1_out\n',
        'mlp prefill 2D w3 output placement')
    result = ''.join(lines)
    ast.parse(result)
    return result


def patch_mlp_full(source):
    """The table maps ONE function per file: the 64-row decode gates, then C1."""
    return patch_mlp_single_gateup(patch_mlp(source))


def patch_layer(source):
    """C1 in layer.py: the ff_norm keeps its own all-gather when the MLP no longer fuses it."""
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
        '        if self.num_devices > 1 and not ' + FLAG_OFF + ':\n'
        '            from loguru import logger as _qwen_logger\n'
        '\n'
        '            _qwen_logger.info("' + MARKER_FF_NORM_GATHER + ' (layer {})", layer_num)\n',
        'layer ff_norm fused all-gather switch')
    result = ''.join(lines)
    ast.parse(result)
    return result


PATCHES = {
    'model_config.py': patch_model_config,
    'attention/tp.py': patch_attention_tp,
    'gdn/tp.py': patch_gdn_tp,
    'mlp.py': patch_mlp_full,
    'layer.py': patch_layer,
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
