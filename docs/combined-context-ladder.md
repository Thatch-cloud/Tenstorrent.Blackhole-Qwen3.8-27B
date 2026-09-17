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
| 32,768 | 1 | 2,968.87 | **89.96** | Passed |
| 65,536 | 1 | — | — | Prompt length rejected before model execution; retry required |
| 131,072 | 1 | — | — | Verifier cache-writer page-table guard failed |
| 262,144 | 1 | — | — | Prompt plus output exceeds target positional limit |

### 16K evidence and next bottleneck

- One fresh feature audit plus two timed requests; 242 committed tokens in total. Draft acceptance: 218/390 (55.90%).
- Exact output, target state and inactive state; stable source fingerprints, closed checkpoint/device and process exit 0. PP/TG independently recomputed from timed requests.
- Pre-load full I/O stall: 0.36%; request cgroup counters show no CPU throttling or OOM. This is not proof of an exclusive host.
- Mean block: **30.82 ms draft + 69.58 ms verification/readback + 5.41 ms selection/commit**, within a **106.80 ms** cycle.
- At 9.31 committed tokens/block, 200 TG requires approximately **46.54 ms/block**, excluding remaining request overhead. Even zero-cost drafting would only yield about **122.5 block-level tok/s** at unchanged acceptance. Verification must improve too.

Report: `dspark-captured-publication-request-hardware.json`, SHA-256 `cc76d9341a03ece238f51537ec65a77763a7618688c0f9e71aabaad4c23a4b31`. The result is a short coding fixture, not sustained serving or held-out quality acceptance.

### 32K evidence

One fresh audit and two timed requests pass exact output/state checks, unchanged source fingerprints, checkpoint/device close and process exit 0. Independent recomputation gives **PP 2,968.87 / CTX 32,768 / TG 89.96**. The timed requests commit 234 tokens and accept 214/330 proposals (64.85%). Mean block: 38.24 ms draft, 73.05 ms verification/readback, 5.76 ms selection/commit and 118.16 ms overall. At unchanged acceptance, 200 TG requires approximately 53.18 ms/block.

Report SHA-256: `fac4e6663f001808fc183136e21edad99a8c740436cf032f7a5f3f6c63544e17`.

### Avoidable build overhead

16K and 32K rebuilt the host runtime in 265 and 264 seconds respectively. Their retained build inputs differ **only** in runtime capacity, not compiled-source hashes. Commit `4625732` separates that metadata from the build key while retaining all other inputs and capacity checks. The fix is for subsequent runs; this ladder remains on its original revision. Shape-specific device kernel compilation is separate and may still be needed.

### 64K preparation failure

The host-pressure check passed (0.097% full stall), but Python preflight rejected the generated prompt because its token count was not exactly 65,536. The builder previously allowed a 32-token shortfall while the runtime correctly required an exact context. No native build, target model execution, OOM or throughput result occurred; container exit was 1.

The retry builder preserves already-exact prompts and searches adjacent excerpt boundaries when tokenization is non-monotonic. It never pads/truncates token IDs or weakens the exact-length runtime guard. Six local prompt tests pass; real-tokenizer and hardware retry remain necessary.

Retry tag `experiment/combined-ladder-retry-v2` selects only 4K, 8K and 64K, carrying the prompt-boundary and build-cache fixes. It shares the hardware concurrency lock, so it waits for the original larger-context jobs rather than interrupting them. The validated 16K/32K rows are not rerun; model math and serving defaults remain unchanged.

### 131K and 262K failures

131K completes prefill but fails during verifier warmup in `ordered_cache.validate_shapes`: the legacy validator permits at most 1,024 pages, while the allocated page table has 2,052. The selected 64K geometry also exceeds that guard at 1,028 pages. Container exit 1, no OOM, no timed request result. The underlying writer already sizes its page-table buffer and compile arguments from the tensor dimensions.

`frozen_ladder_ordered_cache.py` prepares an explicitly scoped geometry validator, retaining bounds against both selected context and actual cache storage. Three host tests pass, including restoring the original guard on exceptions. It is **not yet wired into hardware**; larger-page writer correctness/replay must be checked before deployment. Default serving validation remains unchanged.

262K fails before model loading: a 262,144-token prompt plus 256 output slots requires 262,400 positions, exceeding the target's declared 262,144 limit. A full-window test would instead use 261,888 prompt tokens plus 256 output slots and must be labelled separately; no implicit truncation or positional extension has been applied.
