#!/bin/bash
# Milestone-3 wiring: the norm+gate kernel behind QWEN_GDN_NORM_GATE=1 (needs the packed path).
#   - ~/wrap-K/ttnn_delta_rule_ops.py: the packed entry gains return_row_major (skips to_layout)
#   - ~/wrap-K/tp.py: gdn_decode_norm_gate replaces relayout/reshape/rms_norm/reshape/silu/mul
#   - arunk*.sh: NORMGATE=0|1 passthrough + engagement line
set -u
cp ~/kwork/ttnn_delta_rule_ops.py ~/wrap-K/ttnn_delta_rule_ops.py
python3 - <<'PYEOF'
import os
p = os.path.expanduser("~/wrap-K/tp.py"); s = open(p).read()
if "QWEN_GDN_NORM_GATE" in s:
    print("tp.py already wired for norm+gate"); raise SystemExit
# 1. decide before the recurrence call (the packed wrapper must return o ROW_MAJOR)
old = """        if _packed:
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
"""
assert s.count(old) == 1, "packed call anchor"
new = """        # Milestone 3 (QWEN_GDN_NORM_GATE=1, needs the packed path): the norm+gate kernel takes o
        # as the recurrence writes it (ROW_MAJOR sticks) and z, and writes the gated, normed
        # [1,B,Nv*Dv] tiles the out-projection wants -- the to_layout, two reshapes, rms_norm,
        # silu and multiply below disappear.
        _ng = (
            _packed
            and os.environ.get("QWEN_GDN_NORM_GATE", "0") == "1"
            and hasattr(getattr(ttnn, "transformer", None), "gdn_decode_norm_gate")
        )
        if _packed:
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
                return_row_major=_ng,
            )
"""
s = s.replace(old, new, 1)
# 2. the output norm + gate block
old = """        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f, z, _L1)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
"""
assert s.count(old) == 1, "norm block anchor"
new = """        if _ng:
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
        else:
            out_r = ttnn.reshape(o, (B, Nv, Dv))
            out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
            ttnn.deallocate(out_r)
            out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
            ttnn.deallocate(out_n)
            gated = _silu_mul(out_f, z, _L1)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z)
"""
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("wrap-K/tp.py wired for norm+gate")
PYEOF
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/tp.py')).read());print('wrap-K/tp.py parses')"
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/ttnn_delta_rule_ops.py')).read());print('wrap-K wrapper parses')"
grep -n "return_row_major" ~/wrap-K/ttnn_delta_rule_ops.py | head -3
python3 - <<'PYM'
import os, re
for f in ("~/arunk.sh", "~/arunk8.sh"):
    p = os.path.expanduser(f); s = open(p).read()
    if "gdn_norm_gate" in s: continue
    out = []; done = False
    for l in s.split("
"):
        out.append(l)
        if not done and re.search(r'G="\$G -v \$O/gdn_conv_gates:\$OPS/gdn_conv_gates:ro"', l):
            out.append('  G="$G -v $O/gdn_norm_gate:$OPS/gdn_norm_gate:ro"'); done = True
    assert done, f
    open(p, "w").write("
".join(out)); print(f, "norm_gate mount added")
PYM
for o in ~/arunk.sh ~/arunk8.sh; do
  grep -q "QWEN_GDN_NORM_GATE" $o || sed -i 's|-e QWEN_GDN_PACKED_QKV="\${PACKED:-0}"|-e QWEN_GDN_PACKED_QKV="${PACKED:-0}" -e QWEN_GDN_NORM_GATE="${NORMGATE:-0}"|' $o
  grep -q "NORMGATE engaged" $o || sed -i 's|^echo "  ENGAGED  : |echo "  NORMGATE : $(grep -ohE '"'"'QWEN_GDN_NORM_GATE engaged[^\\"]{0,100}'"'"' "$L" \| head -1)"\necho "  ENGAGED  : |' $o
  bash -n $o && echo "$o syntax OK: $(grep -c 'NORMGATE' $o) hooks"
done
