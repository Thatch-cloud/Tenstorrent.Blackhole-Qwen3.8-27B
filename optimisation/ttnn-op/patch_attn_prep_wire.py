"""Wire ttnn.transformer.attn_decode_prep into the attention layer's decode path (QWEN_ATTN_PREP=1).

Input: the image's models/demos/blackhole/qwen36/tt/attention/tp.py. Output: the wrapped file the
runners bind-mount over it. Idempotent on the marker.

With the flag on and the paged KV path active, forward_decode takes the fused [q|k|v|gate]
projection output straight into attn_decode_prep, which returns q (normed, weighted, roped),
the gate block, and k/v already padded and in the KV update's height-sharded config -- so the
two slices, the head-split op and its three reshards, the two rms_norms and weight multiplies,
the two partial-RoPE chains and the two pad+reshards are gone. Prints one engagement line.
"""
import io
import os
import sys

MARK = "QWEN_ATTN_PREP"


def patch(s):
    if MARK in s:
        return s
    # 1. a raw-projection helper next to _qkv: the decode matmul output, unsliced
    anchor = "    def _qkv(self, x):\n"
    assert s.count(anchor) == 1, "_qkv anchor"
    helper = '''    def _qkv_raw_decode(self, x):
        """Decode-only: the fused [q|k|v|gate] projection output, unsliced (for attn_decode_prep)."""
        tw = self.tw
        if getattr(self.args, "proj_1d_decode", False):
            return tpc.matmul_1d_decode(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)

'''
    s = s.replace(anchor, helper + anchor, 1)

    # 2. forward_decode: fused prologue when the flag is on and the paged path is active
    old = "        qg, kp, vp = self._qkv(x)\n\n        if self._use_nlp_decode_heads:\n"
    assert s.count(old) == 1, "forward_decode qkv anchor"
    new = '''        _prep = (
            os.environ.get("QWEN_ATTN_PREP", "0") == "1"
            and use_paged
            and self._fused_qkv
            and x.shape[-2] <= ttnn.TILE_SIZE
        )
        if _prep:
            qkv_raw = self._qkv_raw_decode(x)
            _kv_cfg = self._kv_shard_cfg(B)
            q, gate, k_sh, v_sh = ttnn.transformer.attn_decode_prep(
                qkv_raw,
                cos_tt,
                sin_tt,
                tw["q_norm"],
                tw["k_norm"],
                NH,
                NKV,
                HD,
                self.rope_dim,
                _kv_cfg,
                batch=B,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_raw)
            if not getattr(self, "_prep_announced", False):
                self._prep_announced = True
                print(
                    f"QWEN_ATTN_PREP engaged: attn_decode_prep -> q {list(q.shape)}, gate {list(gate.shape)}, "
                    f"k/v {list(k_sh.shape)} height-sharded (B={B}, NH={NH}, NKV={NKV}, HD={HD}, RD={self.rope_dim})",
                    flush=True,
                )
            return self._decode_from_prep(q, gate, k_sh, v_sh, cur_pos_tt, page_table, B)

        qg, kp, vp = self._qkv(x)

        if self._use_nlp_decode_heads:
'''
    s = s.replace(old, new, 1)

    # 3. the paged tail (KV update, SDPA, gate, concat, out-proj, all-reduce) as a method fed by the prep op.
    #    Built from the existing paged branch: from 'keys, values = self.paged_k, self.paged_v' to the return.
    start = s.index("        if use_paged:\n            # External paged KV: update at cur_pos, then paged SDPA-decode\n")
    tail_start = s.index("            keys, values = self.paged_k, self.paged_v\n", start)
    sdpa_i = s.index("            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(", tail_start)
    else_i = s.index("        else:\n", sdpa_i)
    # the post-SDPA common tail: from 'gated = ' to the end of forward_decode's return
    gated_i = s.index("        gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)\n", else_i)
    # end of forward_decode = the next 'def ' at class indentation after gated_i
    nxt = s.find("\n    def ", gated_i)
    assert nxt > 0
    ret_block = s[gated_i:nxt + 1]
    sdpa_block = s[sdpa_i:else_i]
    # the SDPA program config forward_decode builds just before the paged branch (already 8-space indented)
    cfg_i = s.index("        _sdpa_grid = self.mesh.compute_with_storage_grid_size()\n")
    sdpa_cfg_block = s[cfg_i:start]
    method = (
        "    def _decode_from_prep(self, q, gate, k_sh, v_sh, cur_pos_tt, page_table, B):\n"
        '        """Paged decode tail fed by attn_decode_prep: KV update, SDPA, gate, concat heads, o_proj, all-reduce."""\n'
        "        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD\n"
        "        _L1 = ttnn.L1_MEMORY_CONFIG\n"
        "        keys, values = self.paged_k, self.paged_v\n"
        "        ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)\n"
        "        ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)\n"
        "        ttnn.deallocate(k_sh)\n"
        "        ttnn.deallocate(v_sh)\n"
        + sdpa_cfg_block
        + "".join(("    " + l[8:] if l.startswith("            ") else l) + "\n" for l in sdpa_block.rstrip("\n").split("\n"))
        + ret_block
        + "\n"
    )
    # dedent sdpa_block from 12 to 8 spaces was done above; ret_block is already at 8.
    s = s[:nxt + 1] + method + s[nxt + 1:]
    if "import os\n" not in s:
        s = s.replace("import torch\n", "import os\nimport torch\n", 1)
    return s


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    s = io.open(src, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    s2 = patch(s)
    io.open(dst, "w", encoding="utf-8", newline="\n").write(s2)
    print("wrote", dst, "engaged" if MARK in s2 else "UNCHANGED")
