# Same-recipe context ladder

The ladder measures the complete single-stream generation pipeline, not isolated kernels.

| Setting | Value |
|---|---|
| Hardware | Two P150A cards, four fabric links |
| Prompt contexts | 4,096; 8,192; 16,384; 32,768; 65,536; 131,072; 262,144 |
| Generation headroom | 256 tokens |
| Runtime | Shared Q/K, norm prefetch, incremental publication, tail-first draft assembly, scalar reciprocal, T16 verifier and compact target scratch |
| Per context | One fresh full-model feature/state audit, then two complete timed requests |
| Changed runtime setting | `QWEN_DSPARK_REQUEST_CONTEXT` only |
| Report | PP (prefill tokens/s), CTX (actual prompt tokens), TG (committed generation tokens/s) |

All rows stage identical source files. The pinned corpus prevents harness edits from changing the workload. Longer contexts repeat that corpus where necessary, preserving the final coding task. This is a synthetic latency workload, not held-out coding-quality certification, and its numbers are not directly interchangeable with older live-source prompts.

The retained 32K component reports verify unchanged kernel sources; they do **not** establish correctness at another geometry. Every row must pass its own full-request audits, exact emitted tokens, state checks and clean shutdown. Positional limits and capacity failures remain failures, never implicit truncation or RoPE extension.

The matrix runs one hardware job at a time and continues after a failed row. Matrix execution order is not guaranteed. Explicit benchmark timers are removed; GitHub's platform job limit and setup/cleanup safeguards still apply. Serving defaults are unchanged.

Workflow: `.github/workflows/qwen-combined-ladder.yml`, triggered by `experiment/combined-ladder-v1`.

## Results so far

[Hardware run 35165321998](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35165321998), runtime revision `b2a2dd8`. Snapshot: 17 September 2026; pending rows are not measurements.

| CTX | Streams | PP tok/s | Committed TG tok/s | Status |
| ---: | ---: | ---: | ---: | --- |
| 4,096 | 1 | — | — | Pre-load host I/O gate failed: 33.94% full stall; model not run |
| 8,192 | 1 | — | — | Pre-load host I/O gate failed: 6.07% full stall; model not run |
| 16,384 | 1 | 3,170.94 | **87.11** | Passed |
| 32,768 | 1 | — | — | Running |
| 65,536 | 1 | — | — | Pending |
| 131,072 | 1 | — | — | Pending |
| 262,144 | 1 | — | — | Pending |

### 16K evidence and next bottleneck

- One fresh feature audit plus two timed requests; 242 committed tokens in total. Draft acceptance: 218/390 (55.90%).
- Exact output, target state and inactive state; stable source fingerprints, closed checkpoint/device and process exit 0. PP/TG independently recomputed from timed requests.
- Pre-load full I/O stall: 0.36%; request cgroup counters show no CPU throttling or OOM. This is not proof of an exclusive host.
- Mean block: **30.82 ms draft + 69.58 ms verification/readback + 5.41 ms selection/commit**, within a **106.80 ms** cycle.
- At 9.31 committed tokens/block, 200 TG requires approximately **46.54 ms/block**, excluding remaining request overhead. Even zero-cost drafting would only yield about **122.5 block-level tok/s** at unchanged acceptance. Verification must improve too.

Report: `dspark-captured-publication-request-hardware.json`, SHA-256 `cc76d9341a03ece238f51537ec65a77763a7618688c0f9e71aabaad4c23a4b31`. The result is a short coding fixture, not sustained serving or held-out quality acceptance.
