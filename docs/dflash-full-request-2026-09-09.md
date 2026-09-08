# Complete DFlash2 request integration

**Target: 200 committed tok/s for one coding stream. Not achieved.**
Latest measured hardware result remains **58.33 TG**, device-chained MTP,
[run 34216164140](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34216164140).
This change connects a parallel learned drafter to complete generation instead
of treating another isolated operator pass as a speed improvement.

**First hardware attempt: [34225857819](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34225857819), cancelled after a stall.**
At CTX170, its first T8 block accepts all seven drafts and commits eight tokens;
all published target-feature checks pass. First-block draft time is5429.31ms,
including compilation, and publication/audit is835.20ms. The next block stalls
without compiler activity. This is **not** a complete request or a new TG result.
Live log SHA256: `9272ee4f4f2ed6b305fe509ef48dbca4f7d0eca715a269cad6ce025ae854da17`.
The follow-up retains the full request, adds stage diagnostics during its audit,
and dumps Python stacks/exits after180 seconds without a committed block instead
of consuming the80-minute outer timeout. A timed-out device process requires
CI recovery before another hardware test.

Follow-up [34227754156](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34227754156)
repeats that first exact block with cached kernels (draft238.09ms,
target verification/readback64.33ms, publication/audit17.48ms). It then reaches
the final device synchronization in the second proposal and times out after180s;
the watchdog captures the Python stack and the container exits normally through
CI cleanup. Enqueued stage logs alone do not identify the blocked kernel.
Next audit adds synchronization between draft stages; those fenced timings are
excluded from TG, and the two measurement requests keep the unfenced path.

[34229264462](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34229264462)
narrows the stall to completion of layer-zero attention at CTX178. Input upload,
embedding, its four-link gather, and history/mask/RoPE preparation all complete.
A targeted simulator replay of key-input concatenation at170/178/170 passes all
six chip checks; that isolated assembly does not reproduce the hardware stall.
Simulator report SHA256: `e8426ef0742314e63436365789283fc3a5757b7a512ae8803e798caaf0a72cd5`.
The next full-request audit fences individual attention operations at the failing
block. It does not change attention math or the unfenced measurement requests.

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
- All 837 host tests pass (777 CI helpers + 60 speculative harness tests).
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
