#!/bin/bash
# Milestone-2 wiring: the packed recurrence path behind QWEN_GDN_PACKED_QKV=1 (only when the
# conv+gates kernel is engaged). Edits ~/wrap-K/tp.py in place, installs the packed-capable
# wrapper as ~/wrap-K/ttnn_delta_rule_ops.py, and points arunk*.sh at it with PACKED=0|1.
set -u
cp ~/kwork/ttnn_delta_rule_ops.py ~/wrap-K/ttnn_delta_rule_ops.py
python3 - <<'PYEOF'
import os
p = os.path.expanduser("~/wrap-K/tp.py"); s = open(p).read()
if "QWEN_GDN_PACKED_QKV" in s:
    print("tp.py already wired for packed"); raise SystemExit
# import the packed entry next to the existing one
old = "from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (\n    recurrent_gated_delta_rule_decode_ttnn,\n"
assert s.count(old) == 1, "import anchor"
s = s.replace(old, old + "    recurrent_gated_delta_rule_decode_packed_ttnn,\n", 1)
# 1. skip the q/k/v split + GQA expansion when packed
old = """        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))"""
assert s.count(old) == 1, "kd anchor"
start = s.index(old)
end = s.index("        v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)\n", start) + len("        v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)\n")
block = s[start:end]
indented = "".join(("    " + l if l.strip() else l) for l in block.splitlines(True))
new = """        # Milestone 2 (QWEN_GDN_PACKED_QKV=1, needs the conv+gates kernel engaged): the recurrence
        # op's reader takes conv [1,B,C] and beta/g [1,B,Nv] directly, GQA included -- the 3
        # slices, 8 reshapes and 2 repeat_interleaves below disappear.
        _packed = (
            _cg_beta is not None
            and os.environ.get("QWEN_GDN_PACKED_QKV", "0") == "1"
            and hasattr(getattr(ttnn, "transformer", None), "decode_gated_delta_rule_packed")
        )
        if not _packed:
""" + indented
s = s[:start] + new + s[end:]
# 2. gates: packed keeps [1,B,Nv]
old = """        if _cg_beta is not None:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            beta = ttnn.reshape(_cg_beta, (B, 1, Nv))
            g = ttnn.reshape(_cg_g, (B, 1, Nv))
"""
assert s.count(old) == 1, "gates anchor"
new = """        if _packed:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            beta, g = _cg_beta, _cg_g  # [1,B,Nv]: the packed reader's own layout
        elif _cg_beta is not None:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            beta = ttnn.reshape(_cg_beta, (B, 1, Nv))
            g = ttnn.reshape(_cg_g, (B, 1, Nv))
"""
s = s.replace(old, new, 1)
# 3. the recurrence call
old = """        o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q,
            k,
            v,
            beta,
            g,
            scale=self.scale,
            initial_state=init_state,
            device=self.mesh,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
            inplace_state=_want_inplace,
        )
"""
assert s.count(old) == 1, "recurrence anchor"
new = """        if _packed:
            o, new_rec = recurrent_gated_delta_rule_decode_packed_ttnn(
                conv,
                beta,
                g,
                Nk,
                Nv,
                Dk,
                Dv,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
                inplace_state=_want_inplace,
            )
            ttnn.deallocate(conv)
        else:
            o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
                inplace_state=_want_inplace,
            )
"""
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("wrap-K/tp.py wired for packed")
PYEOF
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/tp.py')).read());print('wrap-K/tp.py parses')"
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/ttnn_delta_rule_ops.py')).read());print('wrap-K wrapper parses')"
grep -n "if not _packed:\|_packed = (\|decode_packed_ttnn(\|beta, g = _cg_beta" ~/wrap-K/tp.py | head
for o in ~/arunk.sh ~/arunk8.sh; do
  sed -i -e 's|WRAPSRC=\$HOME/wrap-3c/ttnn_delta_rule_ops.py|WRAPSRC=$HOME/wrap-K/ttnn_delta_rule_ops.py|' \
         -e 's|-e QWEN_GDN_CONV_GATES="\${CONVGATES:-0}"|-e QWEN_GDN_CONV_GATES="${CONVGATES:-0}" -e QWEN_GDN_PACKED_QKV="${PACKED:-0}"|' $o
  grep -q "PACKED engaged" $o || sed -i 's|^echo "  ENGAGED  : |echo "  PACKED   : $(grep -ohE '"'"'QWEN_GDN_PACKED_QKV engaged[^\\"]{0,100}'"'"' "$L" \| head -1)"\necho "  ENGAGED  : |' $o
  bash -n $o && echo "$o syntax OK: $(grep -c 'wrap-K/ttnn_delta\|PACKED' $o) hooks"
done
