"""Greedy endpoint isolation gate using exact streamed token IDs, not SSE counts."""

import argparse
import hashlib
import json
import math
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def stream(base_url, model, prompt, count, ready=None, timeout=1800):
    payload = dict(model=model, prompt=prompt, max_tokens=count, temperature=0,
                   ignore_eos=True, stream=True, return_token_ids=True,
                   stream_options={"include_usage": True}, seed=123)
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/completions",
                                     json.dumps(payload).encode(), {"Content-Type": "application/json"})
    started = time.perf_counter()
    events, tokens, fragments = [], [], []
    done = False
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            message = json.loads(data)
            if "error" in message:
                raise RuntimeError(message["error"])
            if message.get("usage"):
                usage = message["usage"]
            for choice in message.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise RuntimeError("Expected exactly one completion")
                fragments.append(choice.get("text", ""))
                emitted = choice.get("token_ids") or []
                if any(type(token) is not int or token < 0 for token in emitted):
                    raise ValueError("Invalid streamed token IDs")
                if emitted:
                    events.append((time.perf_counter(), len(emitted)))
                    tokens.extend(emitted)
                    if ready is not None:
                        ready.set()
    if (not done or len(tokens) != count or len(events) < 2 or usage is None
            or usage.get("completion_tokens") != count or usage.get("prompt_tokens") != len(prompt)):
        raise RuntimeError(f"Incomplete stream or token accounting: done={done}, tokens={len(tokens)}, expected={count}")
    gaps = [later[0] - earlier[0] for earlier, later in zip(events, events[1:])]
    duration = events[-1][0] - events[0][0]
    text = "".join(fragments)
    return dict(tokens=tokens, text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                started=started, first=events[0][0], last=events[-1][0], events=events,
                ttft_s=events[0][0] - started,
                decode_tok_s=(len(tokens) - events[0][1]) / duration,
                event_gap_p99_s=quantile(gaps, 0.99), event_gap_max_s=max(gaps),
                coalesced_events=sum(size > 1 for _, size in events))


def wait_ready(futures, events, timeout):
    deadline = time.monotonic() + timeout
    while not all(event.is_set() for event in events):
        for future in futures:
            if future.done():
                future.result()
                raise RuntimeError("A decoder ended before all streams became active")
        if time.monotonic() >= deadline:
            raise TimeoutError("Decoders did not become active")
        time.sleep(0.02)


def measure(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    unit = tokenizer.encode("def stable_sort(records):\n    return sorted(records, key=lambda row: row[0])\n", add_special_tokens=False)
    prompts = {}
    for index in range(args.decoders):
        prompts[f"B{index}"] = tokenizer.encode(
            f"Write Python code and tests for a bounded LRU cache. Variant {index}. Explain invariants.\n",
            add_special_tokens=False)
    prompts["A"] = (unit * math.ceil(args.prompt_length / len(unit)))[:args.prompt_length]
    prompts["C"] = tokenizer.encode("Write a Python binary search function and describe its edge cases.\n", add_special_tokens=False)
    counts = {name: args.decode_tokens if name.startswith("B") else 32 for name in prompts}
    stream(args.base_url, args.model, prompts["C"], 8)
    baseline = {name: stream(args.base_url, args.model, prompt, counts[name]) for name, prompt in prompts.items()}
    concurrent = {}
    with ThreadPoolExecutor(max_workers=args.decoders + 2) as executor:
        ready = [threading.Event() for _ in range(args.decoders)]
        decoders = [executor.submit(stream, args.base_url, args.model, prompts[f"B{index}"],
                                    counts[f"B{index}"], ready[index]) for index in range(args.decoders)]
        wait_ready(decoders, ready, args.ready_timeout)
        if any(future.done() for future in decoders):
            raise RuntimeError("Decoder finished before long-prefill injection")
        injected = time.perf_counter()
        long_request = executor.submit(stream, args.base_url, args.model, prompts["A"], counts["A"])
        time.sleep(args.short_delay)
        short_request = executor.submit(stream, args.base_url, args.model, prompts["C"], counts["C"])
        for index, future in enumerate(decoders):
            concurrent[f"B{index}"] = future.result()
        concurrent["A"], concurrent["C"] = long_request.result(), short_request.result()
    checks = {f"greedy_equal_{name}": baseline[name]["tokens"] == concurrent[name]["tokens"]
              and baseline[name]["text"] == concurrent[name]["text"] for name in prompts}
    checks["decoders_overlap_long_prefill"] = all(
        result["last"] > concurrent["A"]["first"] for name, result in concurrent.items() if name.startswith("B"))
    checks["short_arrives_before_long_first_token"] = concurrent["C"]["started"] < concurrent["A"]["first"]
    return dict(passed=all(checks.values()), checks=checks, prompt_length=args.prompt_length,
                decoders=args.decoders, injected=injected, baseline=baseline, concurrent=concurrent,
                timing_scope="client-observed; SSE event gaps are not per-token latency when coalesced",
                scope="Endpoint greedy-output isolation; not direct KV/GDN snapshot or trace-address validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--prompt-length", type=int, default=2049)
    parser.add_argument("--decoders", type=int, choices=(1, 7), default=1)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--short-delay", type=float, default=0.1)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_length <= 0 or args.decode_tokens < 2 or args.short_delay < 0:
        parser.error("Positive prompt length, at least two decode tokens and nonnegative delay required")
    report = dict(passed=False)
    try:
        report = measure(args)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ("baseline", "concurrent")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
