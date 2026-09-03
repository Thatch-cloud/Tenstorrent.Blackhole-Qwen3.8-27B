#!/bin/bash
# Milestone-1 wiring for K's conv+gates kernel:
#   ~/wrap-K/tp.py   = wrap-3c/tp.py (fused decode + in-place) + the conv+gates call behind
#                      QWEN_GDN_CONV_GATES=1 (default off: the arm with the flag off is the control)
#   ~/arunk.sh / ~/arunk8.sh = arun.sh / arun8.sh mounting opgraft-K (which carries every
#                      graft op) and wrap-K/tp.py, with CONVGATES=0|1 passed through.
set -u
mkdir -p ~/wrap-K
python3 - <<'PYEOF'
import os
src = os.path.expanduser("~/wrap-3c/tp.py"); dst = os.path.expanduser("~/wrap-K/tp.py")
s = open(src).read()
if "QWEN_GDN_CONV_GATES" in s:
    print("tp.py already wired"); raise SystemExit
# 1. conv block -> fused op or the composed path
old = """        # Conv1d shift-register + weighted sum + SiLU
        st = self.conv_states
        if B < Bmax:"""
assert s.count(old) == 1, "conv anchor"
conv_start = s.index(old)
conv_end = s.index("        kd = self.key_dim_tp", conv_start)
composed_conv = s[conv_start:conv_end]
fused = """        # K's first kernel (QWEN_GDN_CONV_GATES=1): conv shift-register + FIR + SiLU + both
        # gates in ONE op, replacing the K copies, multiply + (K-1) mac + silu, the B<Bmax pad,
        # sigmoid, add+softplus and multiply -- 12 ops/layer. Conv states advance in place
        # (same buffers: the decode trace's address discipline holds).
        _cg_beta = _cg_g = None
        if self._conv_gates_enabled():
            conv, _cg_beta, _cg_g = ttnn.transformer.gdn_decode_conv_gates(
                qkv,
                self.conv_states,
                [tw["conv_taps"][j] for j in range(self.K)],
                a,
                b,
                tw["dt_bias"],
                tw["neg_exp_A"],
                batch=B,
                memory_config=_L1,
            )
            ttnn.deallocate(qkv)
            if not getattr(self, "_conv_gates_logged", False):
                self._conv_gates_logged = True
                print(
                    f"QWEN_GDN_CONV_GATES engaged: ttnn.transformer.gdn_decode_conv_gates B={B} Bmax={Bmax}"
                    f" C={self.qkv_dim_tp} Nv={Nv} K={self.K} -> conv {list(conv.shape)} beta {list(_cg_beta.shape)}",
                    flush=True,
                )
        else:
            conv = self._conv_composed(qkv, B, Bmax, _L1)

"""
s = s[:conv_start] + fused + s[conv_end:]
# 2. gates block -> take the fused outputs or the composed path
old_g = """        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))
"""
assert s.count(old_g) == 1, "gates anchor"
new_g = """        if _cg_beta is not None:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            beta = ttnn.reshape(_cg_beta, (B, 1, Nv))
            g = ttnn.reshape(_cg_g, (B, 1, Nv))
        else:
            beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
            ttnn.deallocate(b)
            g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
            ttnn.deallocate(a)
            g = ttnn.reshape(g, (B, 1, Nv))
"""
s = s.replace(old_g, new_g, 1)
# 3. helper methods before forward_decode (the composed conv block verbatim, as a method)
helper = """
    def _conv_gates_enabled(self):
        return os.environ.get("QWEN_GDN_CONV_GATES", "0") == "1" and self.K == 4 and hasattr(
            getattr(ttnn, "transformer", None), "gdn_decode_conv_gates"
        )

    def _conv_composed(self, qkv, B, Bmax, _L1):
        # The original conv shift-register + FIR + SiLU (the control arm's path).
        tw = self.tw
""" + composed_conv + """        return conv

    def forward_decode(self, x):"""
assert s.count("\n    def forward_decode(self, x):") == 1
s = s.replace("\n    def forward_decode(self, x):", helper, 1)
open(dst, "w").write(s)
print("wrap-K/tp.py written")
PYEOF
python3 -c "import ast,os;ast.parse(open(os.path.expanduser('~/wrap-K/tp.py')).read());print('wrap-K/tp.py parses')"
grep -n "_conv_composed\|return conv$\|QWEN_GDN_CONV_GATES engaged\|_cg_beta is not None" ~/wrap-K/tp.py | head

for f in arun.sh arun8.sh; do
  o=~/${f/arun/arunk}
  sed -e 's|O=\$HOME/opgraft-53587|O=$HOME/opgraft-K|' \
      -e 's|  G="\$G -v \$O/decode_gated_delta_rule:\$OPS/decode_gated_delta_rule:ro"|  G="$G -v $O/decode_gated_delta_rule:$OPS/decode_gated_delta_rule:ro"\n  G="$G -v $O/gdn_conv_gates:$OPS/gdn_conv_gates:ro"|' \
      -e 's|  G="\$G -v \$HOME/wrap-3c/tp.py:\$Q/tt/gdn/tp.py:ro"|  G="$G -v $HOME/wrap-K/tp.py:$Q/tt/gdn/tp.py:ro"|' \
      -e 's|-e QWEN_GDN_FUSED_DECODE="\$FLAG"|-e QWEN_GDN_FUSED_DECODE="$FLAG" -e QWEN_GDN_CONV_GATES="${CONVGATES:-0}"|' \
      -e 's|L="\$HOME/a-\$TAG.log"|L="$HOME/ak-$TAG.log"|' -e 's|L="\$HOME/a8-\$TAG.log"|L="$HOME/ak8-$TAG.log"|' \
      -e 's|^echo "  ENGAGED  : |echo "  CONVGATES: $(grep -ohE '"'"'QWEN_GDN_CONV_GATES engaged[^\\"]{0,100}'"'"' "$L" \| head -1)"\necho "  ENGAGED  : |' \
      ~/$f > $o
  chmod +x $o; bash -n $o && echo "$o syntax OK"; echo "  hooks: $(grep -c 'opgraft-K\|gdn_conv_gates\|wrap-K\|CONVGATES' $o)"
done
