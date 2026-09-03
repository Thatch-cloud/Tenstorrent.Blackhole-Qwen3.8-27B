#!/bin/bash
# K's last slice wiring: QWEN_GDN_NORM_GATE=1 now means the norm + gate are folded INTO the
# recurrence op (packed mode with z and norm_w): the op returns the gated [1,B,Nv*Dv] tiles,
# so the to_layout, both reshapes, rms_norm, silu and multiply go, and in direct mode the z
# slice goes too (z is a column window of the projection output).
set -u
cp ~/kwork/ttnn_delta_rule_ops.py ~/wrap-K/ttnn_delta_rule_ops.py
python3 - <<'PYEOF'
import os
p = os.path.expanduser("~/wrap-K/tp.py"); s = open(p).read()
if "_ng_fold" in s:
    print("tp.py already wired for the fold"); raise SystemExit
# 1. decide the fold early (its conditions are all known before the projection)
old = """        _direct = (
            self._conv_gates_enabled()
            and self._fuse_ab
            and os.environ.get("QWEN_GDN_PROJ_DIRECT", "0") == "1"
        )
        if _direct:
            qkvzab = self._project_qkvzab_raw(x, B, _L1)
            _qz, _az = self.qkv_dim_tp, self.qkvz_dim_tp
            z = ttnn.slice(qkvzab, (0, 0, _qz), (1, B, _az), memory_config=_L1)
"""
assert s.count(old) == 1, "direct anchor"
new = """        _direct = (
            self._conv_gates_enabled()
            and self._fuse_ab
            and os.environ.get("QWEN_GDN_PROJ_DIRECT", "0") == "1"
        )
        # K's last slice (QWEN_GDN_NORM_GATE=1): the recurrence op finishes the output norm and
        # gate itself and returns the gated [1,B,Nv*Dv] tiles; z goes in as a column window.
        _ng_fold = (
            self._conv_gates_enabled()
            and os.environ.get("QWEN_GDN_PACKED_QKV", "0") == "1"
            and os.environ.get("QWEN_GDN_NORM_GATE", "0") == "1"
            and hasattr(getattr(ttnn, "transformer", None), "decode_gated_delta_rule_packed")
        )
        _z_off = 0
        if _direct:
            qkvzab = self._project_qkvzab_raw(x, B, _L1)
            _qz, _az = self.qkv_dim_tp, self.qkvz_dim_tp
            if _ng_fold:
                z, _z_off = qkvzab, _qz  # no slice: the op reads z's window straight from qkvzab
            else:
                z = ttnn.slice(qkvzab, (0, 0, _qz), (1, B, _az), memory_config=_L1)
"""
s = s.replace(old, new, 1)
# 2. the packed recurrence call: pass z / norm_w when folding; drop the standalone-op decision
old = """        _ng = (
            _packed
            and os.environ.get("QWEN_GDN_NORM_GATE", "0") == "1"
            and hasattr(getattr(ttnn, "transformer", None), "gdn_decode_norm_gate")
        )
"""
assert s.count(old) == 1, "ng anchor"
new = """        _ng = _packed and _ng_fold
"""
s = s.replace(old, new, 1)
old = """                inplace_state=_want_inplace,
                return_row_major=_ng,
            )
            ttnn.deallocate(conv)
"""
assert s.count(old) == 1, "packed call anchor"
new = """                inplace_state=_want_inplace,
                z=z if _ng else None,
                norm_w=tw["norm_w"] if _ng else None,
                z_col_offset=_z_off,
            )
            ttnn.deallocate(conv)
"""
s = s.replace(old, new, 1)
# 3. the output block: when folded, o IS the gated [1,B,Nv*Dv] tensor
old = """        if _ng:
            gated = ttnn.transformer.gdn_decode_norm_gate(
                o, z, tw["norm_w"], Nv, batch=B, epsilon=1e-6, memory_config=_L1
            )
            ttnn.deallocate(o)
            ttnn.deallocate(z)
            if not getattr(self, "_norm_gate_logged", False):
                self._norm_gate_logged = True
                print(
                    f"QWEN_GDN_NORM_GATE engaged: ttnn.transformer.gdn_decode_norm_gate B={B} H={Nv} V={Dv}"
                    f" -> gated {list(gated.shape)}",
                    flush=True,
                )
"""
assert s.count(old) == 1, "norm block anchor"
new = """        if _ng:
            gated = o  # the recurrence op returned rms_norm(o) * norm_w * silu(z), [1,B,Nv*Dv] TILE
            if z is not qkv_src_for_z_check(z, _direct):
                ttnn.deallocate(z)
            if not getattr(self, "_norm_gate_logged", False):
                self._norm_gate_logged = True
                print(
                    f"QWEN_GDN_NORM_GATE engaged: norm+gate folded into decode_gated_delta_rule_packed"
                    f" (z_col_offset={_z_off}) -> gated {list(gated.shape)}",
                    flush=True,
                )
"""
s = s.replace(old, new, 1)
# the z buffer: in direct+fold mode z IS qkvzab, already freed with qkv after the conv kernel
s = s.replace("            if z is not qkv_src_for_z_check(z, _direct):\n                ttnn.deallocate(z)\n",
              "            if not (_direct and _ng_fold):\n                ttnn.deallocate(z)\n", 1)
open(p, "w").write(s)
print("wrap-K/tp.py wired for the fold")
PYEOF
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/tp.py')).read());print('wrap-K/tp.py parses')"
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/ttnn_delta_rule_ops.py')).read());print('wrap-K wrapper parses')"
grep -n "_ng_fold\|z_col_offset=_z_off\|gated = o" ~/wrap-K/tp.py | head -8
