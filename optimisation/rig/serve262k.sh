#!/bin/bash
# Section 2.3: serve the full 262,144-token window through vLLM. Diff vs the README serve
# command is exactly the plan's: QWEN36_MAX_TOKENS_ALL_USERS pins the KV pool to one full-
# context user instead of max_model_len x max_num_seqs (2.1M tokens, 134 GB, allocator dies),
# max_model_len sizes RoPE + the page table the prefill trace is captured against, and
# VLLM_RPC_TIMEOUT must exceed the 220 s prefill. Lever A (fused decode) is on: it is shipped.
#   serve262k.sh <tag> <sdpa_bf8 0|1>
set -u
TAG=$1; BF8=${2:-0}
IMG=zot.thatch.local:5000/tt-vllm:qwen38-fused-decode
DESC=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
TTCFG='{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'
D1=$(readlink -f /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D)
F3=$(readlink -f /dev/tenstorrent/by-id/blackhole-3707293C249A5E67)
L="$HOME/s262k-$TAG.log"
echo "### $TAG sdpa_bf8=$BF8 load=[$(cut -d' ' -f1-3 /proc/loadavg)] start=$(date -Is)"
docker rm -f epserve >/dev/null 2>&1 || true
docker run -d --name epserve -p 8001:8000 -w /opt/vllm-tt-plugin \
  --device "$D1" --device "$F3" \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v "$HOME/hf-cache:/root/.cache/huggingface" -v "$HOME/ttcache:/ttcache" \
  -e QWEN_GDN_FUSED_DECODE=1 -e QWEN_SDPA_BF8="$BF8" \
  -e QWEN36_MAX_TOKENS_ALL_USERS=262144 \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 -e TT_MESH_GRAPH_DESC_PATH=$DESC \
  -e TT_METAL_HOME=/opt/tt-metal -e TT_METAL_CACHE=/ttcache \
  -e VLLM_RPC_TIMEOUT=600000 -e VLLM_PLUGINS=tt,tt_model_registry -e QWEN36_BATCHED_DECODE_MODE=host \
  "$IMG" python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B --served-model-name qwen3.8-27b \
    --max_model_len 262144 --max-num-seqs 8 --max-num-batched-tokens 262144 --no-enable-prefix-caching \
    --block-size 64 --reasoning-parser qwen3 --port 8000 --host 0.0.0.0 \
    --additional-config "$TTCFG" >/dev/null
ready=0
for i in $(seq 1 120); do   # readiness may exceed 510 s: full-width page table prefill trace
  curl -sf http://localhost:8001/v1/models >/dev/null 2>&1 && { ready=$((i*15)); break; }
  docker ps --filter name=epserve --format '{{.Names}}' | grep -q epserve || break
  sleep 15
done
docker logs epserve > "$L" 2>&1
echo "  ready after ${ready}s   (acceptance 1) $(grep -oE 'max_tokens_all_users=[0-9]+[^\"]{0,40}' "$L" | head -1)"
if [ "$ready" = "0" ]; then echo "  NOT READY"; grep -oE "Error[^\"]{0,120}|TT_FATAL[^)]{0,90}|Out of Memory" "$L" | sort -u | head -5; docker rm -f epserve >/dev/null; echo "=== DONE $TAG ==="; exit 0; fi
echo "  (acceptance 2) short request, decode should be ~18 tok/s:"
python3 - <<'PY'
import json,time,urllib.request
b=json.dumps({"model":"qwen3.8-27b","prompt":"Explain how a CPU pipeline works. "*60,"max_tokens":64,"temperature":0,"ignore_eos":True,"stream":True}).encode()
r=urllib.request.Request("http://localhost:8001/v1/completions",data=b,headers={"Content-Type":"application/json"})
ts=[];t0=time.perf_counter()
with urllib.request.urlopen(r,timeout=900) as resp:
    for raw in resp:
        l=raw.decode("utf-8","ignore").strip()
        if l.startswith("data:") and l[5:].strip()!="[DONE]":
            try:
                if json.loads(l[5:])["choices"][0].get("text"): ts.append(time.perf_counter())
            except Exception: pass
import statistics as st
itl=st.median(b-a for a,b in zip(ts,ts[1:]))
print(f"    ttft={ts[0]-t0:.2f}s  tokens={len(ts)}  median_ITL={itl*1000:.1f}ms  decode={1/itl:.2f} tok/s")
PY
echo "  (acceptance 3) the ~262k prompt, streamed: expect TTFT ~220 s, decode ~12.8 tok/s"
python3 - "$PROMPT262K" <<'PY'
import json,sys,time,urllib.request,statistics as st
p=sys.argv[1]
try: text=open(p,encoding="utf-8",errors="ignore").read()
except Exception as e: print(f"    prompt file unavailable ({e}); skipping"); sys.exit()
b=json.dumps({"model":"qwen3.8-27b","prompt":text,"max_tokens":64,"temperature":0,"ignore_eos":True,"stream":True}).encode()
r=urllib.request.Request("http://localhost:8001/v1/completions",data=b,headers={"Content-Type":"application/json"})
ts=[];out=[];t0=time.perf_counter()
try:
  with urllib.request.urlopen(r,timeout=1800) as resp:
    for raw in resp:
        l=raw.decode("utf-8","ignore").strip()
        if l.startswith("data:") and l[5:].strip()!="[DONE]":
            try:
                tx=json.loads(l[5:])["choices"][0].get("text")
                if tx: ts.append(time.perf_counter()); out.append(tx)
            except Exception: pass
except Exception as e: print(f"    request failed: {str(e)[:160]}"); sys.exit()
itl=st.median(b-a for a,b in zip(ts,ts[1:])) if len(ts)>2 else float("nan")
print(f"    prompt_bytes={len(text)}  ttft={ts[0]-t0:.1f}s  tokens={len(ts)}  median_ITL={itl*1000:.1f}ms  decode={1/itl:.2f} tok/s")
import hashlib, os; full="".join(out); open(os.path.expanduser("~/lc262k."+os.environ.get("ARM","x")+".txt"),"w").write(full)
print(f"    TEXT md5={hashlib.md5(full.encode()).hexdigest()[:12]}  {full[:110]!r}")
PY
docker logs epserve > "$L" 2>&1
echo "  preemptions: $(curl -s http://localhost:8001/metrics 2>/dev/null | grep -E '^vllm:num_preemptions_total' | head -1)"
docker rm -f epserve >/dev/null 2>&1
echo "=== DONE $TAG ==="
