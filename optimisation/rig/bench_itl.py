#!/usr/bin/env python3
"""Endpoint decode throughput via inter-token latency.

The previous version timed whole requests, so TTFT and vLLM's per-user blocking
prefill were inside the number; its two control arms disagreed by 41%. This one
streams, drops the first token of each stream (that one carries TTFT), and reports
aggregate only over the window in which ALL streams are concurrently decoding.
"""
import json, statistics, sys, threading, time, urllib.request

URL = "http://localhost:8001/v1/completions"
TEXTS = {}
N = 8
PROMPT = "Write a detailed explanation of how a modern CPU pipeline works, step by step."

def stream(i, out, maxtok):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": PROMPT, "max_tokens": maxtok,
                       "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    ts = []; txt = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d["choices"][0].get("text", ""):
                ts.append(time.perf_counter()); txt.append(d["choices"][0]["text"])
    out[i] = ts; TEXTS[i] = "".join(txt)

def run(maxtok, label):
    out = [None] * N
    th = [threading.Thread(target=stream, args=(i, out, maxtok)) for i in range(N)]
    [t.start() for t in th]; [t.join() for t in th]
    if any(not t or len(t) < 5 for t in out):
        print(f"BENCH_{label} FAILED counts={[len(t or []) for t in out]}", flush=True); return
    # per-stream inter-token gaps; none of these includes TTFT
    med = [statistics.median(b - a for a, b in zip(t, t[1:])) for t in out]
    itl = statistics.mean(med)
    # aggregate over the all-concurrent window only
    t0 = max(t[0] for t in out); t1 = min(t[-1] for t in out)
    toks = sum(sum(1 for x in t if t0 < x <= t1) for t in out)
    agg = toks / (t1 - t0) if t1 > t0 else float("nan")
    print(f"BENCH_{label} n={N} maxtok={maxtok} counts={[len(t) for t in out]} "
          f"median_ITL={itl*1000:.2f}ms per-user={1/itl:.2f} tok/s "
          f"window={t1-t0:.2f}s toks_in_window={toks} aggregate={agg:.1f} tok/s", flush=True)
    import hashlib
    h = hashlib.md5("".join(TEXTS.get(i, "") for i in range(N)).encode()).hexdigest()[:12]
    print(f"BENCH_TEXT_{label} md5={h} lens={[len(TEXTS.get(i, "")) for i in range(N)]} first={TEXTS.get(0, "")[:70]!r}", flush=True)

run(48, "WARM")   # captures the width-8 decode trace before anything is timed
run(200, "MAIN")
