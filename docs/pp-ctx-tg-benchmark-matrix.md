# PP / CTX / TG: two-card coding benchmark

## What the current number means

| Path | PP tok/s | CTX tokens | TG tok/s | Streams | Verify rows | Decode tokens/request | Total request time |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Captured DFlash2 + commit-only GDN | 517.89 | 170 | 72.21 | 1 | Up to 8 | 150, 150 | 7.52 s |
| Same + fused convolution | **510.65** | **170** | **78.06** | **1** | Up to 8 | 150, 150 | **6.42 s** |
| Lead at 4K, separate context pilot | **3,355.04** | **4,096** | **58.18** | **1** | Up to 8 | 121, 121 | **7.05 s** |
| Lead at 8K, separate context pilot | **3,149.33** | **8,192** | **46.20** | **1** | Up to 8 | 121, 121 | **10.24 s** |
| Lead at 8K, same-code repeat | **3,298.59** | **8,192** | **53.48** | **1** | Up to 8 | 121, 121 | **8.70 s** |

CTX170 uses matched ABBA, separate correctness audits, both responses reach EOS. Candidate
TG samples are 77.96 / 78.17; mean prefill is 332.91 ms. These are offline
requests on an already-loaded model, not a streaming server measurement.
[Hardware run 34246322267](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34246322267).
Artifact SHA256: `b9947cffc7e596f51215da3ec0e585bafbaddd9e76366168e251510b6415b197`.

The 4K row passes one correctness audit and two timed requests, TG57.28 / 59.10.
It is not a matched comparison with the shorter prompt or its different output.
[4K run34285614832](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34285614832).
Artifact SHA256: `2816e7c8e2b93eef9e8848c06f3d451c296897e555154bf86c49127f8241ab87`.

The 8K row passes the same correctness gate, but TG samples41.94 /51.43 include
a424 ms publication stall in the first request. Do not treat this as isolated
context-scaling evidence. [8K result and artifact](dflash-8k-context-2026-09-09.md).
The unchanged-code repeat34289671592 measures53.40 /53.57 TG, combined53.48.
Both runs remain separate: the repeat is not an optimization gain and does not
explain the original stall.

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
| 4,096 | 1 | 3,355.04 / 58.18 measured | Retain long-context control |
| 8,192 | 1 | Repeat3,298.59 /53.48; first3,149.33 /46.20 | Retain both runs; original stall remains unexplained |
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
| Former constructor/request/bucket guards limited prefill to 2,048 | Absolute target position and bounded draft history are now separate; corrected 4K hardware qualification passed. |
| Initial feature projection previously received the full prompt | It now receives only the final 2,048 valid feature rows, with an explicit absolute start; padding is excluded. |
| Proposal input builder already supports absolute positions with a rolling window | Test initial 4K position and continuation, not only a short prompt that later crosses 2K. |
| Target KV is independent of the draft window | Preserve the full target context; compare tokens, GDN, valid KV and inactive slots against native decoding. |

Do not remove the guards and call the result qualified. Simulator coverage is
for window/position/ownership correctness, never a hardware speed claim. Then
run the 4K complete-request hardware gate before extending the matrix.
[Implementation and gate evidence](dflash-long-context-2026-09-09.md).

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
