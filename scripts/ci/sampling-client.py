"""Three ABBA blocks per prompt length, exact IDs, and actual sampler engagement."""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import threading

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("baseline-client.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def engagement(output, label, device):
    entries = [json.loads(line) for line in (output / "sampling-engagement.jsonl").read_text().splitlines()]
    matched = [entry for entry in entries if label in entry["request_id"] and entry["is_decode"]]
    if len(matched) != 1 or matched[0]["selected"] != device:
        raise AssertionError(f"Unexpected sampling engagement: {label}, {matched}")
    if device and (not matched[0]["force_argmax"] or matched[0]["trace_mode"] != "all"):
        raise AssertionError("Device arm did not use force-argmax with traced model decode")
    return matched[0]


def measure(prompt, count, label, output, device, **kwargs):
    report = baseline.request_stream(prompt, count, label, output, seed=None, request_id=label, **kwargs)
    engagement(output, label, device)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    stop = threading.Event()
    collector = threading.Thread(target=baseline.scrape, args=(stop, args.output), daemon=True)
    collector.start()
    summary = dict(passed=False, results=[], scope="B1 client-estimated ABBA sampling comparison; not engine-commit timing")
    try:
        for length in (128, 4096):
            prompt = baseline.make_prompt(tokenizer, length, 0)
            warm = []
            for arm in ("host", "device"):
                label = f"qwen-sampling-{arm}-warm-{length}"
                warm.append(measure(prompt, 8, label, args.output, arm == "device"))
            if warm[0]["token_ids"] != warm[1]["token_ids"]:
                raise AssertionError("Warm-up host/device tokens differ")
            fallback = measure(prompt, 8, f"qwen-sampling-device-logprobs-{length}", args.output, False, logprobs=True)
            if fallback["token_ids"] != warm[0]["token_ids"]:
                raise AssertionError("Host fallback changed greedy tokens")
            reference = None
            for block in range(3):
                records = []
                for order, arm in enumerate(("host", "device", "device", "host")):
                    label = f"qwen-sampling-{arm}-length-{length}-block-{block}-order-{order}"
                    result = measure(prompt, 512, label, args.output, arm == "device")
                    fingerprint = (result["token_ids_sha256"], result["output_sha256"])
                    if reference is None:
                        reference = fingerprint
                    elif fingerprint != reference:
                        raise AssertionError(f"Host/device output divergence: {label}")
                    records.append(result)
                host = statistics.mean(records[index]["client_decode_estimate_tok_s"] for index in (0, 3))
                device = statistics.mean(records[index]["client_decode_estimate_tok_s"] for index in (1, 2))
                summary["results"].append(dict(target_prompt=length, actual_prompt=len(prompt), block=block,
                    host_client_tok_s=host, device_client_tok_s=device, device_change_fraction=device / host - 1,
                    requests=[record["label"] for record in records]))
        summary["passed"] = True
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        collector.join(timeout=5)
        (args.output / "sampling-summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
