"""Full-model boundary and cancellation observations before mixed-traffic adoption."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("baseline-client.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=True)
    unit = tokenizer.encode("def stable_sort(records):\n    return sorted(records)\n", add_special_tokens=False)
    report = dict(passed=False, scope="Raw token-prompt exact output-ID/cancellation probe; not direct state snapshots",
                  results=[])
    stop = threading.Event()
    collector = threading.Thread(target=baseline.scrape, args=(stop, args.output), daemon=True)
    collector.start()
    try:
        if os.environ.get("QWEN_BOUNDARY_DIAGNOSTICS") == "1":
            prompt = (unit * (2049 // len(unit) + 1))[:2049]
            for repeat in range(3):
                label = f"isolated-2049-{repeat}"
                baseline.request_stream(prompt, 32, label, args.output)
                report["results"].append(label)
        for length in (63, 64, 65, 2047, 2048, 2049, 4096, 8193):
            prompt = (unit * (length // len(unit) + 1))[:length]
            result = baseline.request_stream(prompt, 32, f"boundary-{length}", args.output)
            report["results"].append(result["label"])
        prompt = (unit * (4096 // len(unit) + 1))[:4096]
        request = urllib.request.Request("http://127.0.0.1:8000/v1/completions", json.dumps(dict(
            model="qwen3.8-27b", prompt=prompt, max_tokens=1024, temperature=0, ignore_eos=True,
            stream=True, return_token_ids=True)).encode(), {"Content-Type": "application/json"})
        cancelled = False
        with urllib.request.urlopen(request, timeout=1800) as response:
            for line in response:
                if line.startswith(b"data:") and b'"token_ids"' in line:
                    message = json.loads(line[5:])
                    if any(choice.get("token_ids") for choice in message.get("choices", [])):
                        cancelled = True
                        break
        if not cancelled:
            raise AssertionError("Cancellation stream produced no token")
        time.sleep(2)
        reused = baseline.request_stream(prompt, 32, "after-cancel", args.output)
        original = json.loads((args.output / "boundary-4096.json").read_text())
        if any(reused[key] != original[key] for key in ("output_sha256", "token_ids_sha256")):
            raise AssertionError("Greedy continuation changed after cancellation")
        report["results"].append("after-cancel")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        collector.join(timeout=5)
        (args.output / "interleave-summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
