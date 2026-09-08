# PP / CTX / TG: two-card coding benchmark

## What the current number means

| Path | PP tok/s | CTX tokens | TG tok/s | Streams | Verify rows | Decode tokens/request | Total request time |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Captured DFlash2 + commit-only GDN | 517.89 | 170 | 72.21 | 1 | Up to 8 | 150, 150 | 7.52 s |
| Same + fused convolution | **510.65** | **170** | **78.06** | **1** | Up to 8 | 150, 150 | **6.42 s** |

Matched ABBA, separate correctness audits, both responses reach EOS. Candidate
TG samples are 77.96 / 78.17; mean prefill is 332.91 ms. These are offline
requests on an already-loaded model, not a streaming server measurement.
[Hardware run 34246322267](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34246322267).
Artifact SHA256: `b9947cffc7e596f51215da3ec0e585bafbaddd9e76366168e251510b6415b197`.

| Metric | Definition |
| --- | --- |
| PP | Prompt processing: actual input tokens / timed prefill seconds. Includes target feature capture and first-token selection; excludes draft setup. |
| CTX | Actual prompt tokens including chat template, before generation. Not the configured maximum cache capacity. |
| TG | Committed decode tokens / decode seconds, after the prefill seed. Includes drafting, verification/readback and publication; excludes prefill/setup. |
| B | Concurrent user streams, not repetitions or speculative positions. All lead measurements are B1. |
| T | Maximum target verification positions per speculative block, not batch size. |
| Total | Prefill + fresh drafter/verifier setup + decode; excludes model loading. Not TTFT. |

Rates use summed tokens divided by summed time, not an average of rates. Audit
timings never contribute. Keep contexts, arms, repetitions and output lengths
visible; do not present component rates or aggregate B8 throughput as B1 TG.

## Next matrix

| CTX target | B | Lead PP / TG | Next action |
| ---: | ---: | --- | --- |
| 170 | 1 | 510.65 / 78.06 measured | Retain regression anchor |
| 4,096 | 1 | Not measured | Qualify long-prefill initialization first |
| 8,192 | 1 | Not measured | Run after 4K correctness passes |
| 16,384 | 1 | Not measured | Same runtime and timing boundaries |
| 32,768 | 1 | Not measured | Same runtime and timing boundaries |
| 64,504 | 1 | Not measured | Reserve space for the 513-token output budget and verification |
| 4,096 | 2 / 4 / 8 | Not measured | Separate serving/concurrency work; report per-stream and aggregate TG |

Report the actual tokenized CTX, not a rounded label. Use fixed, recorded coding
inputs; no answer-derived prompts or repeated-code lookup speed claims. Keep
checkpoint, precision, four-link fabric and sampler fixed across context rows.
Run a full native-reference correctness audit before uninstrumented repetitions.
Record EOS and actual output lengths rather than assuming every prompt produces
the short pilot's 150 decode tokens. This matrix is not held-out coding quality.

## Long-context gate before hardware

| Finding | Required change / evidence |
| --- | --- |
| DFlash constructor, request wrapper and proposal bucket selector reject prefill >2,048 | Separate absolute target position from bounded draft history length. |
| Initial feature projection starts at row zero | Initialize from the final 2,048 valid prefill feature rows, excluding padding; keep ownership bounded. |
| Proposal input builder already supports absolute positions with a rolling window | Test initial 4K position and continuation, not only a short prompt that later crosses 2K. |
| Target KV is independent of the draft window | Preserve the full target context; compare tokens, GDN, valid KV and inactive slots against native decoding. |

Do not remove the guards and call the result qualified. Simulator coverage is
for window/position/ownership correctness, never a hardware speed claim. Then
run the 4K complete-request hardware gate before extending the matrix.

## Rebuild a table from artifacts

From the experiment worktree with its Python dependencies installed:

```sh
python scripts/ci/dflash_benchmark_report.py \
  --artifact 34246322267 hardware-evidence.local/34246322267/artifacts/qwen-hardware-inventory-34246322267/full-dflash-request.json
```

Repeat `--artifact RUN_ID REPORT` to include other runs; add `--format json` for
machine-readable rows with artifact hashes. Old evidence is read, never rewritten.
The reporter rejects incomplete/failed requests and mismatched recorded rates;
it keeps each run/context/arm separate. New DFlash summaries embed a `benchmark`
object containing PP, CTX, TG, individual rates, output counts and timing scope.
