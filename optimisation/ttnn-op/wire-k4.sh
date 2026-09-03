#!/bin/bash
# Milestone-4 wiring: QWEN_GDN_PROJ_DIRECT=1 -- the conv+gates kernel reads the fused
# projection output directly (x = qkvzab with channels=C; a/b = column windows at az, az+Nv),
# so the qkv / ab / a / b slices disappear (z still needs one). Needs conv+gates engaged and
# the fused a/b projection (self._fuse_ab).
set -u
python3 - <<'PYEOF'
import os
p = os.path.expanduser("~/wrap-K/tp.py"); s = open(p).read()
if "QWEN_GDN_PROJ_DIRECT" in s:
    print("tp.py already wired for direct projection"); raise SystemExit
# 1. projection: raw output when direct
old = """        qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)
"""
assert s.count(old) == 1, "projection anchor"
new = """        # Milestone 4 (QWEN_GDN_PROJ_DIRECT=1): keep the fused [qkv|z|a|b] projection output whole
        # and let the conv+gates kernel read x, a and b out of it by column -- the qkv, ab, a
        # and b slices disappear (z keeps one slice for the output gate).
        _direct = (
            self._conv_gates_enabled()
            and self._fuse_ab
            and os.environ.get("QWEN_GDN_PROJ_DIRECT", "0") == "1"
        )
        if _direct:
            qkvzab = self._project_qkvzab_raw(x, B, _L1)
            _qz, _az = self.qkv_dim_tp, self.qkvz_dim_tp
            z = ttnn.slice(qkvzab, (0, 0, _qz), (1, B, _az), memory_config=_L1)
            qkv, a, b = qkvzab, qkvzab, qkvzab
            _cg_kw = {"channels": _qz, "a_col": _az, "b_col": _az + Nv}
            if not getattr(self, "_proj_direct_logged", False):
                self._proj_direct_logged = True
                print(
                    f"QWEN_GDN_PROJ_DIRECT engaged: conv+gates reads qkvzab {list(qkvzab.shape)} directly"
                    f" (channels={_qz} a_col={_az} b_col={_az + Nv})",
                    flush=True,
                )
        else:
            qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)
            _cg_kw = {}
"""
s = s.replace(old, new, 1)
# 2. the kernel call takes the column kwargs; in direct mode a/b ARE qkv (freed once)
old = """                tw["neg_exp_A"],
                batch=B,
                memory_config=_L1,
            )
            ttnn.deallocate(qkv)
"""
assert s.count(old) == 1, "kernel call anchor"
new = """                tw["neg_exp_A"],
                batch=B,
                memory_config=_L1,
                **_cg_kw,
            )
            ttnn.deallocate(qkv)
            if _direct:
                a = b = None  # same buffer as qkv: already freed
"""
s = s.replace(old, new, 1)
# 3. the gate blocks must not free a/b twice
old = """        if _packed:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            beta, g = _cg_beta, _cg_g  # [1,B,Nv]: the packed reader's own layout
        elif _cg_beta is not None:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
"""
assert s.count(old) == 1, "gate free anchor"
new = """        if _packed:
            if a is not None:
                ttnn.deallocate(a)
                ttnn.deallocate(b)
            beta, g = _cg_beta, _cg_g  # [1,B,Nv]: the packed reader's own layout
        elif _cg_beta is not None:
            if a is not None:
                ttnn.deallocate(a)
                ttnn.deallocate(b)
"""
s = s.replace(old, new, 1)
# 4. the raw projection helper (decode branches of _project_qkvzab, fused a/b weight)
helper = '''
    def _project_qkvzab_raw(self, x, S, out_mc):
        """Decode-only: the fused [qkv|z|a|b] projection output, unsliced (milestone 4)."""
        _proj_mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG
        if getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:
            return tpc.matmul_1d_decode(
                x,
                self.tw["qkvz"],
                self.args.gdn_qkvz_decode_1d_progcfg,
                self.cfg,
                out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvzab_progcfg, out_memory_config=_proj_mc)

    def _conv_gates_enabled(self):'''
assert s.count("\n    def _conv_gates_enabled(self):") == 1
s = s.replace("\n    def _conv_gates_enabled(self):", helper, 1)
open(p, "w").write(s)
print("wrap-K/tp.py wired for direct projection")
PYEOF
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/tp.py')).read());print('wrap-K/tp.py parses')"
for o in ~/arunk.sh ~/arunk8.sh; do
  grep -q "QWEN_GDN_PROJ_DIRECT" $o || sed -i 's|-e QWEN_GDN_NORM_GATE="\${NORMGATE:-0}"|-e QWEN_GDN_NORM_GATE="${NORMGATE:-0}" -e QWEN_GDN_PROJ_DIRECT="${PROJDIRECT:-0}"|' $o
  grep -q "DIRECT engaged" $o || sed -i 's|^echo "  ENGAGED  : |echo "  DIRECT   : $(grep -ohE '"'"'QWEN_GDN_PROJ_DIRECT engaged[^\\"]{0,110}'"'"' "$L" \| head -1)"\necho "  ENGAGED  : |' $o
  bash -n $o && echo "$o syntax OK: $(grep -c 'PROJDIRECT\|PROJ_DIRECT' $o) hooks"
done
