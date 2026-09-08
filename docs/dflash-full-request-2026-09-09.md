# Complete DFlash2 request integration

**Target: 200 committed tok/s for one coding stream. Not achieved.**
Latest measured hardware result remains **58.33 TG**, device-chained MTP,
[run 34216164140](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34216164140).
This change connects a parallel learned drafter to complete generation instead
of treating another isolated operator pass as a speed improvement.

## What is connected

| Part | Opt-in implementation |
| --- | --- |
| Drafter | All five BF16 layers of `incoai/Qwen3.8-27B-DFlash2` |
| Checkpoint | `dedf8df68adfb1afeaf7b7480c0a0243108177b4`; every fixture hash checked before device open |
| Target features | Post-layer taps 5, 19, 33, 47, 61; preallocated verifier destinations |
| Feature publication | Project only processed, committed input rows; rejected rows never enter history |
| History | Rolling 2048 projected rows; two buffers allocated before verifier capture |
| Proposals | Seed plus seven mask tokens; five parallel-position layers, not seven serial model calls |
| Attention | Existing composed precise control; unqualified native SDPA remains disabled |
| Embedding/head | Borrowed target weights; full-vocabulary chunked top16 candidates |
| Selection | Learned FP64 greedy selector on CPU; gather only the active candidate codebook entries |
| Target verification | Existing norm-batched GDN verifier, native-row full-vocabulary force argmax, four physical links |

This first integration is eager. It still recomputes per-layer history K/V,
dispatches operations from the host and reads candidate IDs back. It is not a
claimed 200-TG implementation or the final optimized serving path.

## One meaningful hardware suite

Dispatch `qwen-experiments.yml` with `suite=full-dflash-request` and explicit
exclusive card allocation. Serving defaults are unchanged.

| Request | Workload | Required evidence | Timing use |
| --- | --- | --- | --- |
| 1 | Complete `merge_intervals_v1`, through EOS | Every published tap/row on both chips equals native serial features | Audit only; TG suppressed |
| 2–3 | Same complete coding request, fresh prefill/setup each time | Exact native tokens, active GDN, valid target KV and inactive slots | Committed TG; prefill and setup separate |

The report is `full-dflash-request.json`. It records each block's proposal,
acceptance, draft time, verifier time and publication time. Summary TG divides
total committed tokens by total decode time for requests 2–3 only. Prefill seeds,
rejected drafts, setup and the instrumented audit do not inflate TG. Complete
prefill/setup/decode latency is also reported, with no setup amortization claim.

## Local evidence and limits

- Prepared feature-copy trace: 30 changed-input tap checks, 40 first-publication
  checks and 20 second-publication checks pass at T8 on both simulated chips.
  Report SHA256: `1be89583f72f0a2c70fa81b401bcbe8a2c751cbb0ed9967673656cde5afcfa57`.
- History simulation exercises CTX170 and the 2048-row sliding boundary, accepted
  prefixes 1/2/7/8, aborts and replay across the preallocated buffers: all 16
  checks pass. Report SHA256: `9ae52881272f70ca5da036af4dcaad209c2d5b9fe38d1816119e90dad31f2d01`.
- All 835 host tests pass (775 CI helpers + 60 speculative harness tests).
  They cover transaction failures, stale tickets, ownership, complete
  fixture loading, exact selector equivalence and exclusion of audited timing.
- These are integration/correctness checks, **not** full-model simulator speed,
  hardware fabric validation or a held-out coding-quality score.

## After the first complete result

Use the actual block breakdown to choose the next change: cache new history K/V
once per layer, capture the fixed proposal computation, then address verifier
cost and proposal width. Preserve the same full-request correctness boundary.
The failed native-attention numerical experiment remains a separate candidate,
not a prerequisite for measuring the integrated composed drafter.
