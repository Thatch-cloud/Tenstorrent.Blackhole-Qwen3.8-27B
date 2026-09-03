set -u
pkill -f "runshipk" 2>/dev/null; true
python3 - <<'PYEOF'
import os
p = os.path.expanduser("~/runship.sh"); s = open(p).read(); o = os.path.expanduser("~/runshipk.sh")
old = '''if [ "$SHIP" = "1" ]; then
  TTCFG='''
assert s.count(old) == 1, "ship anchor"
new = '''KM=""
if [ "$SHIP" = "2" ]; then
  # K: both rebuilt libraries, the three grafted op dirs (JIT kernel sources), the K-wired tp.py
  # and the wrapper carrying the packed + in-place entries, mounted over the vLLM image (same
  # tt-metal rev as the serving image the graft was built against).
  O=$HOME/opgraft-K; OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer
  DRO=/opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
  KM="-v $O/_ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so:ro -v $O/_ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so:ro"
  KM="$KM -v $O/gdn_decay:$OPS/gdn_decay:ro -v $O/decode_gated_delta_rule:$OPS/decode_gated_delta_rule:ro -v $O/gdn_conv_gates:$OPS/gdn_conv_gates:ro"
  KM="$KM -v $HOME/wrap-K/tp.py:$Q/tt/gdn/tp.py:ro -v $HOME/wrap-K/ttnn_delta_rule_ops.py:$DRO:ro"
fi
if [ "$SHIP" != "0" ]; then
  TTCFG='''
s = s.replace(old, new, 1)
old = '''  EXTRA=(-e QWEN36_SHARD_GREEDY=1 -e QWEN35_GDN_DECODE_BF16=1 -e QWEN35_GDN_STATE_BF16=1 -e QWEN_SDPA_BF8=1)'''
assert s.count(old) == 1, "extra anchor"
s = s.replace(old, old + '''
  [ "$SHIP" = "2" ] && EXTRA+=(-e QWEN_GDN_CONV_GATES=1 -e QWEN_GDN_PACKED_QKV=1 -e QWEN_GDN_FUSED_INPLACE=1)''', 1)
old = '''  -v "$HOME/wrap-3e/model.py:$Q/tt/model.py:ro" \\
'''
assert s.count(old) == 1, "mount anchor"
s = s.replace(old, '''  -v "$HOME/wrap-3e/model.py:$Q/tt/model.py:ro" $KM \\
''', 1)
old = '''echo "  FLAGS: '''
assert s.count(old) == 1, "flags anchor"
s = s.replace(old, '''echo "  KFLAGS: $(grep -ohE 'QWEN_GDN_CONV_GATES engaged|QWEN_GDN_PACKED_QKV engaged|in-place happened=[A-Za-z]+' "$L" | sort -u | tr '\\n' '|')"
''' + old, 1)
s = s.replace('L="$HOME/rship-$TAG.log"', 'L="$HOME/rshipk-$TAG.log"', 1)
open(o, "w").write(s)
print("runshipk.sh written")
PYEOF
chmod +x ~/runshipk.sh; bash -n ~/runshipk.sh && echo "syntax OK"
grep -n 'KM=\|"$SHIP" = "2"\|KFLAGS\|rshipk' ~/runshipk.sh | cut -c1-110 | head -8
nohup bash -c '~/runshipk.sh ks-a 1; ~/runshipk.sh kk-a 2; ~/runshipk.sh kk-b 2; ~/runshipk.sh ks-b 1; echo "=== KSHIP COMPLETE ==="' > ~/kship.out 2>&1 &
echo "queued endpoint K A/B (ship vs ship+K, interleaved; pid=$!)"
