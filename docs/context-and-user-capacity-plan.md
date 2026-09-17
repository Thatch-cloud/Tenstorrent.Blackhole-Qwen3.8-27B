# Long context and concurrent-user capacity

Status: planned, not runtime admission or measured capacity.

## Context coverage

Extend the single-stream ladder through 131072 and 262144 tokens, after resolving
the existing 32K numerical failure. Retain 128/4K/8K/32K/64K comparisons.
Do not dispatch the largest cases by simply bypassing the current shape guards.

| Window label | Total token budget | Input budget with 1024 output tokens |
|---|---:|---:|
| 128K (131K) | 131072 | 130048 |
| 256K (262K) | 262144 | 261120 |

Count chat-template and special tokens inside the input budget. If testing exactly
131072 or 262144 **input** tokens, reserve 1024 more and qualify those larger total
capacities separately. Verify the pinned model's positional configuration and
runtime limits; do not silently change RoPE or truncate history to fit a label.

Order: host geometry/memory checks → bounded synthetic arithmetic/replay → real
target and drafter correctness → combined PP/CTX/TG → long-context coding quality.
Check target KV, recurrent state, drafter history, rollback buffers, traces and
workspace together, per card. Include distant-history sensitivity and retained
state after accepted/rejected drafts. Fit in memory is not correctness or speed.

The existing simulator ladder still ends at 65536 input tokens. New large-window
fixtures, exact native selectors and runtime admission remain implementation work.
Use independently bounded CI probes for these sizes, not full-model simulation.

### Runtime integration audit

Current source inspection identifies these separate prerequisites; a passing
attention probe alone does not remove them.

| Path | Existing boundary | Required follow-up |
|---|---|---|
| `dspark_context_selection.py` | Only 4096/8192 request selection | Evidence-backed per-size request admission |
| `dspark_history.py:FullHistoryKV` | Initial frontier limited to 8192 | Qualify full-history initialization and publication beyond 8K |
| `dspark_stable_history.py` / `dspark_8k_scope.py` | 8192 default, exact 8448 scoped allocation | Preserve active/spare ownership, cleanup and rollback at new capacities |
| `dspark_history.py:project_chunks` | Projects history in 32-row blocks, with table uploads per block | Profile request setup independently; qualify a larger-block projection before replacing it |
| `dspark_ladder_build.py` | CPU-simulator-only build route | Matching physical build provenance and numerical evidence before hardware admission |

The 32-row projection loop is a source-observed scaling concern, not a measured
TG bottleneck: it initializes drafter history and must be timed in setup/TTFT,
not silently excluded from end-to-end results or blamed for steady decode.

## Concurrent users

Measure simultaneous active generating requests, not connected idle sessions.
Static batch size and concurrent client count are separate fields: requests might
queue or execute serially even when many clients are connected.

| Stage | Workload | Sweep |
|---|---|---|
| Short coding | 4K and 8K inputs, fixed output budget | 1, 2, 4, 8, then 16 if safe |
| Long coding | Each qualified 32K through 256K window | Start at 1; increase only within measured memory headroom |
| Mixed arrivals | Short edits alongside long repository prompts | Repeat at sustainable request rates |
| Prefix reuse | Identical workload with controlled shared prefixes | Cache off, then cold/warm cache separately |

Preflight scheduler/batching support and prove per-request KV/recurrent-state
isolation, cancellation cleanup, slot reuse and speculative rollback. Keep weights
shared where supported; do not equate one full model process per user with batching.
Repeat controls and test both container-local and gateway streaming paths.

At each row record PP, actual CTX, output count, per-user committed TG, aggregate
TG, completed requests/s, TTFT p50/p95/p99, streaming gaps, queue delay, failures,
peak memory per card and prefix-cache state. Include slow users, not just averages.
Stop escalation at allocation failures, correctness errors, unstable queues or the
agreed latency limit; preserve failure evidence and recover only this experiment.

Use closed-loop clients to find saturation, then paced arrivals to verify queue
stability. Agree TTFT and streaming-gap budgets before declaring a supported user
count. Until then publish a capacity/latency curve, not a single capacity promise.
The ≥200 TG single-user goal remains independent of aggregate throughput.

## Results table

| CTX input / total limit | Active users / batch | PP tok/s | Per-user TG min / median | Aggregate TG | TTFT p99 | Gap p99 | Peak memory/card | Decision |
|---|---|---|---|---|---|---|---|---|
| Not measured | Not measured | — | — | — | — | — | — | Planned |

Attach exact runtime/model revisions, run IDs, duration, actual output counts and
latency-budget definitions using the [experiment template](tuning-experiment-template.md).
