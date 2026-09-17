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
| 4,096 | 1 | 3,249.81 | **118.22** | Retry passed; original attempt blocked by host pressure |
| 8,192 | 1 | 3,293.61 | **106.50** | Retry passed; original attempt blocked by host pressure |
| 16,384 | 1 | 3,170.94 | **87.11** | Passed |
| 32,768 | 1 | 2,968.87 | **89.96** | Passed |
| 65,536 | 1 | 2,564.54 | **50.45** | Passed after exact-prompt and page-table fixes |
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

### 4K retry

[Run 35167726511](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35167726511), revision `f907f9b`: exact output/state, source immutability, clean close and exit 0. Independent recomputation confirms **PP 3,249.81 / CTX 4,096 / TG 118.22**. Two timed requests commit 242 tokens, accepting 224/300 draft proposals (74.67%). Mean cycle 102.28 ms with 12.1 committed tokens; verifier/readback accounts for 67.25 ms. This is not a matched speedup claim over older prompts.

Report SHA-256: `833b09c6a975a53374fd558c24b8947f9e760c91972619def2bd38c362fbc209`.

### 8K retry and long-context qualification

The same retry run passes 8K: **PP 3,293.61 / CTX 8,192 / TG 106.50**. One fresh audit and two timed requests pass exact output/state, matching prompts and outputs, source immutability, clean close and exit 0. PP/TG independently recomputed. The timed requests commit 242 tokens, accepting 222/330 proposals (67.27%). Report SHA-256: `b3c0140d8ff282f034f1333f1f830dffe49f679fc5b262d3c4d690c146d2ec48`.

Native build-cache reuse is confirmed: build setup takes **3 seconds**, versus 263 seconds on the preceding 4K cache miss. The entire 8K job completes in about 5m39s. Mean block: 28.76 ms drafting, 67.74 ms verification/readback, 5.73 ms selection/commit, 103.23 ms overall. With 11 committed tokens/block, 200 TG requires at most 55 ms/block before other request overhead; verification remains the main bottleneck.

The remaining 64K retry was cancelled during job setup because it still carried the known 1,024-page guard; completed 4K/8K results are retained. The weight-free wider page-table check [35168996959](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35168996959) failed immediately: the shell launcher's allowed-case list omitted `ladder-cache`. No simulator or model ran. The launcher fix includes a local executable regression test for valid and invalid case selection; retry tag `experiment/ladder-cache-sim-v2`. This check loads no model weights and does not measure TG.

The v2 retry exposed missing simulator support-file staging, fixed with a staging regression test. [v3 run 35169977291](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35169977291) ran the probe: all ten 64K checks and both 131K eager checks were exact, but the six-minute probe limit expired during 131K replay. No complete qualification or clean shutdown is claimed.

The v4 retry separates contexts into serial jobs and saves each full native reference once, instead of reading it back again for unchanged replay. Both changed-input reference snapshots are produced before trace capture, removing post-capture sharded-buffer allocations. Full-cache comparison on both chips, eager checks and changed-input replay remain required. This is a test-harness change, not a kernel or throughput improvement.

[v4 run 35170767742](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35170767742) passes both contexts: ten exact checks each, process exit 0, unchanged source fingerprints and clean close. The 64K job takes 2m40s; the 131K simulator reports about 270 seconds. Source-bound report digests are pinned in `scripts/ci/frozen-ladder-cache-reports.json`.

Hardware retry tag `experiment/combined-ladder-wide-cache-v1` selects only 64K/131K. It scopes the qualified page-table validator to the existing combined request sequence, checks generated kernel hashes, and restores the default validator afterward. Each context still requires a fresh full-model audit and two timed requests. This simulator pass establishes cache-write/replay correctness only, not combined runtime correctness or TG.

In hardware run `35171542810`, 64K still fails tokenizer preflight: no prefix within the searched boundary produces exactly 65,536 tokens (best: 65,535). No model loading or timed request occurs. The follow-up builder preserves all already-exact prompts and, only when prefix search fails, tries bounded shifts of the pinned source excerpt's starting boundary. It records that offset and excerpt hash; the coding task and chat template are unchanged, with no token padding or truncation. Eight local tests pass, including a tokenizer that skips the requested length for every unshifted prefix. Real-tokenizer qualification remains required. Retry tag `experiment/combined-ladder-wide-cache-v2` selects only 64K; it does not interrupt the separate 131K row.

### 131K audit-memory failure

The 131K row in `35171542810` advances past the cache guard (512 scoped validation calls), but fails during the first request's eager proposal audit. `PreparedDSparkProposal.propose` holds the captured trace's buffers while a second eager execution retains all five layers' temporaries. A 134,742,016-byte V-padding allocation fails: each DRAM bank needs 16,842,752 bytes, but only 11,020,864 bytes are free. Device close succeeds; container exit 1, no host OOM kill. There is no qualified TG result.

`frozen_ladder_reference_memory.py` bounds only the eager audit's per-layer temporary lifetime. It executes the same layer backend, synchronizes before releasing scratch, and preserves the output and borrowed tensors. Captured execution, timed requests, full proposal-output comparisons and state audits are unchanged. Host tests cover ownership, exception restoration and audit-only routing; hardware acceptance remains pending under the 131K-only retry tag `experiment/combined-ladder-wide-cache-v3`.

### 64K combined result

[Run 35172072077](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35172072077), revision `aa1cad7`, passes: **PP 2,564.54 / CTX 65,536 / committed TG 50.45**, one stream. A fresh feature audit and two timed requests pass exact output/state/inactive-state checks, source immutability, checkpoint/device close and exit 0. PP/TG are independently recomputed. The selected source excerpt starts at character 4, producing exactly 65,536 tokens with the task/template unchanged.

| Timed observation | Result |
| --- | ---: |
| Committed tokens, two requests | 268 |
| Draft acceptance | 234/540 (43.33%) |
| Mean committed tokens/block | 7.44 |
| Draft | 55.50 ms/block |
| Verification/readback | 81.12 ms/block |
| Selection/commit | 8.71 ms/block |
| Complete cycle | 147.49 ms/block |

At this acceptance, 200 TG needs at most **37.22 ms/block** before remaining request overhead. Removing drafting entirely would still yield only about 80.93 block-level tok/s; this is a bound, not a speedup prediction. Longer-context kernel cost and lower accepted-token yield both matter. The result is not a matched speedup over older, differently configured 64K fixtures, nor held-out coding-quality or serving acceptance.

Report SHA-256: `9f69fda886d0bb1ce7fe493920471683e0906a597acbaf8faa79cf9d4a9d6854`. Cache validation executes 1,536 times and restores the default guard. This run does not include the later audit-memory lifetime change. The 131K retry is run `35172695250`.
