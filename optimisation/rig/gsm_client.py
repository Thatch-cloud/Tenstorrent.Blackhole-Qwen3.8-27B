import json, re, urllib.request, concurrent.futures, time
N = 60
rows = [json.loads(l) for l in open('/home/thatch/gsm8k_test.jsonl')][:N]
def truth(a): return a.split('####')[-1].strip().replace(',','')
def ask(r):
    body = json.dumps({"model":"qwen3.8-27b",
        "messages":[{"role":"user","content":r["question"]+"\n\nGive the final numeric answer on the last line as: #### <number>"}],
        "max_tokens":2048,"temperature":0}).encode()
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        m = d["choices"][0]["message"]
        return (m.get("content") or ""), (m.get("reasoning") or "")
    except Exception as e:
        return f"ERR {e}", ""
t0=time.time(); ok=0; bad=[]; noans=0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for r,(content,reason) in zip(rows, ex.map(ask, rows)):
        nums = re.findall(r'-?\d+\.?\d*', (content or "").replace(',',''))
        got = nums[-1].rstrip('.') if nums else None
        exp = truth(r["answer"])
        if got is None: noans += 1
        try: hit = got is not None and abs(float(got)-float(exp)) < 1e-6
        except Exception: hit = False
        if hit: ok += 1
        elif len(bad) < 5: bad.append((r["question"][:66], exp, got))
print(f"  {N} items, 8 concurrent, max_tokens=2048, {time.time()-t0:.0f}s")
print(f"  CORRECT: {ok}/{N} = {100*ok/N:.1f}%   (no parseable answer: {noans})")
if N-noans: print(f"  excluding unanswered: {ok}/{N-noans} = {100*ok/(N-noans):.1f}%")
for q,e,g in bad: print(f"    MISS exp={e} got={g} | {q}...")
