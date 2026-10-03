"""Fixed-workload endpoint baseline; client timing is not engine commit timing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import urllib.request


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def request_stream(prompt, count, label, output, *, logprobs=False, seed=123, request_id=None,
                   temperature=0, top_k=None, top_p=None, presence_penalty=None,
                   frequency_penalty=None, repetition_penalty=None):
    sampling_options = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                            repetition_penalty=repetition_penalty)
    sampling_options = {key: value for key, value in sampling_options.items() if value is not None}
    payload = dict(model="qwen3.8-27b", prompt=prompt, max_tokens=count,
                   ignore_eos=True, stream=True, stream_options={"include_usage": True},
                   return_token_ids=True)
    payload.update(sampling_options)
    if seed is not None:
        payload["seed"] = seed
    if request_id is not None:
        payload["request_id"] = request_id
    if logprobs:
        payload["logprobs"] = 1
    request = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
                                     json.dumps(payload).encode(), {"Content-Type": "application/json"})
    started = time.perf_counter()
    events, tokens, fragments = [], [], []
    usage, finish, done = None, None, False
    logprob_count = 0
    with urllib.request.urlopen(request, timeout=1800) as response:
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
                    raise RuntimeError("Expected one completion per request")
                emitted = choice.get("token_ids") or []
                if any(type(token) is not int or token < 0 for token in emitted):
                    raise ValueError("Invalid streamed token IDs")
                logprob_count += len((choice.get("logprobs") or {}).get("tokens") or [])
                if emitted:
                    events.append([time.perf_counter(), len(emitted)])
                    tokens.extend(emitted)
                fragments.append(choice.get("text", ""))
                finish = choice.get("finish_reason") or finish
    ended = time.perf_counter()
    gaps = [later[0] - earlier[0] for earlier, later in zip(events, events[1:])]
    duration = events[-1][0] - events[0][0] if len(events) > 1 else 0
    passed = (done and len(tokens) == count and usage is not None and usage.get("completion_tokens") == count
              and usage.get("prompt_tokens") == len(prompt) and (not logprobs or logprob_count == count))
    report = dict(label=label, passed=passed, prompt_tokens=len(prompt), requested_tokens=count,
                  seed=seed, request_id=request_id, sampling_options=sampling_options,
                  usage=usage, finish_reason=finish, started=started, ended=ended,
                  total_s=ended - started, ttft_s=events[0][0] - started if events else None,
                  client_decode_estimate_tok_s=(len(tokens) - events[0][1]) / duration if duration else None,
                  engine_committed_tok_s=None, engine_timing_unavailable=True,
                  event_gap_p50_s=quantile(gaps, .5), event_gap_p95_s=quantile(gaps, .95),
                  event_gap_p99_s=quantile(gaps, .99), event_gap_max_s=max(gaps) if gaps else None,
                  coalesced_events=sum(size > 1 for _, size in events), events=events,
                  output_sha256=hashlib.sha256("".join(fragments).encode()).hexdigest(),
                  token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                  token_ids=tokens, token_count_source="streamed token IDs", logprobs_requested=logprobs,
                  reasoning_answer_counts="unavailable for raw completion stream")
    (output / f"{label}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ("events", "token_ids")}), flush=True)
    if not passed:
        raise AssertionError(f"Incomplete generation/token accounting: {label}")
    return report


def make_prompt(tokenizer, target, variant):
    unit = "def lookup(records, key):\n    return next((row for row in records if row[0] == key), None)\n"
    def encode(repeats):
        encoded = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a careful coding assistant."},
            {"role": "user", "content": unit * repeats +
             f"\nVariant {variant}. Refactor this code into a tested indexed lookup API. Explain edge cases and write tests."}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise TypeError("Expected a flat list of template token IDs")
        return tokens
    lower, upper = 0, 1
    while upper < target and len(encode(upper)) <= target:
        upper = min(target, upper * 2)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if len(encode(middle)) <= target:
            lower = middle
        else:
            upper = middle - 1
    prompt = encode(lower)
    if len(prompt) > target:
        raise ValueError("Template exceeds prompt budget")
    return prompt


def scrape(stop, output):
    with (output / "metrics.jsonl").open("w") as stream:
        while not stop.is_set():
            sample = dict(wall_time=time.time(), monotonic=time.perf_counter())
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=2) as response:
                    sample["metrics"] = response.read().decode()
            except Exception as error:
                sample["error"] = str(error)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            stop.wait(.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--logprobs", action="store_true")
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1 or args.context <= args.tokens + 128:
        parser.error("Invalid generation, repetition or context budget")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    collector = threading.Thread(target=scrape, args=(stop, args.output), daemon=True)
    collector.start()
    report = dict(passed=False, scope="E0 container-local client estimates; not the 200 committed-token/s gate",
                  engine_timing="unavailable", context_limit=args.context, results=[])
    try:
        time.sleep(2)
        lengths = sorted(set(min(length, args.context - args.tokens) for length in (128, 4096, 32768, 65536)))
        workloads = [(length, 1) for length in lengths] + [(4096, batch) for batch in (2, 8)]
        for length, batch in workloads:
            prompts = [make_prompt(tokenizer, length, index) for index in range(batch)]
            with ThreadPoolExecutor(max_workers=batch) as pool:
                list(pool.map(lambda item: request_stream(item[1], 8,
                     f"warm-length-{length}-batch-{batch}-user-{item[0]}", args.output, logprobs=args.logprobs), enumerate(prompts)))
        for repeat in range(args.repeats):
            for length, batch in (workloads if repeat % 2 == 0 else list(reversed(workloads))):
                prompts = [make_prompt(tokenizer, length, index) for index in range(batch)]
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=batch) as pool:
                    results = list(pool.map(lambda item: request_stream(item[1], args.tokens,
                        f"run-{repeat}-length-{length}-batch-{batch}-user-{item[0]}", args.output, logprobs=args.logprobs), enumerate(prompts)))
                report["results"].append(dict(repeat=repeat, target_prompt=length, batch=batch,
                    end_to_end_aggregate_tok_s=sum(result["usage"]["completion_tokens"] for result in results) /
                    (time.perf_counter() - started), requests=[result["label"] for result in results]))
        time.sleep(2)
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        collector.join(timeout=5)
        (args.output / "baseline-summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
