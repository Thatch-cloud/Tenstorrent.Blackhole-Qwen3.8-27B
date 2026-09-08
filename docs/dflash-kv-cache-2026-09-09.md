# Historical K/V cache for DFlash2

**Status: integrated simulator gate passed; opt-in 4K hardware ABBA running.**
No cached-drafter hardware speed or correctness result exists yet. Serving
defaults and the measured lead are unchanged.

[Hardware run34291073085](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34291073085)
tests code `20915fe` on the dedicated two-card runner. Its gate is two separate
audits followed by four uninstrumented requests, in uncached/cached/cached/
uncached order. A running job is not a result.

## Why this experiment

At CTX4096, the current lead measures **58.18 committed TG**: drafting costs
54.57 ms/block and verification/readback 61.87 ms/block. The drafter projects the
same committed history through all five layers' K/V weights on every proposal.
The target KV cache is already used; this experiment concerns the separate
drafter's historical projections.

| Component | Candidate behavior |
| --- | --- |
| `draft_kv_projection.py` | Original HiFi4 FP32-accumulating K/V projection, BF16 rounding, native head split, K normalization and absolute RoPE, on only the required rows |
| `draft_kv_history.py` | Per-layer active/spare 2K K/V buffers; prepare only committed new feature rows, evict expired rows, then publish by swapping banks |
| Rejection or failure | Active buffers and absolute frontier remain unchanged; partial spare writes cannot publish |
| Captured proposal integration | Stable per-bucket K/V inputs are allocated before capture and copied from the committed bank before replay; measured cached calls no longer copy the full feature history |
| Timing | Initialization and publication costs must stay visible; no free cache or amortization assumption |

Caching alone is **not enough for 200 TG**. At the measured 4K acceptance pattern
(121 tokens in 17 blocks), verification alone implies about115 TG even with all
other work removed. Reaching200 needs the complete mean cycle below35.59 ms,
versus122.22 ms measured. Target verifier work remains necessary alongside this
draft-side optimization; these bounds are calculations, not benchmark results.

## Gates

| Gate | State |
| --- | --- |
| Projection host tests | Four pass; validate geometry, precision, explicit positions and simulator-only entry |
| Transaction host tests | Five pass; multiple layers, prefixes1/7/8/32, short-window growth, eviction, abort, failed spare write, stale ticket rejection and full-history corruption detection |
| Learned numerical simulator | Passed `20260908T222847Z-416-draft-kv-projection-probe`: 20 bit-exact comparisons and four detected stale-history controls, pinned layer1 weights, both chips |
| Actual cached/uncached attention operands and cache publication/replay | Passed `20260908T230732Z-297-draft-kv-history-probe`; real learned layer1, native head layout and fused convolution |
| Full eager proposal comparison against uncached lead | Implemented as a separate hardware audit for every proposal; hardware evidence pending |
| Matched complete-request hardware ABBA | Running34291073085; no result yet |

The numerical probe compares full versus separate history/live projection at
CTX170 and4093, then seven-row eviction/append at4100. It also requires changed
input/position trace replay and a detected stale-history control. Its pass does
not qualify all learned layers or the complete transactional cache integration.
Weights are checked against the pinned checkpoint hashes before any device opens.

The simulator exits0 and closes both devices cleanly after1,356.6 s. Report
SHA256: `5cc6dc345e924d70c64ac75fa391218f23cd27abd332f0c094429ad9ebb5d7ef`.
Projection source SHA256:
`6c217c51738d7e3feb0cf7a0f911c82f5bd70850095019ed8c8ac14402d8f204`.
All six recorded source hashes matched the files in `78312fd`. Integration now
also forwards the already-created raw Q heads from the projection helper; its
new source hashes belong to the integrated gate, not this older report. The
original component commit passed838 CI tests and60 harness tests.

The integrated gate compares Q/K/V/mask inputs at the unchanged attention
boundary, avoiding a claim that simulated downstream attention or CCL ran when
they did not. It then uses the real `DFlashDevice` prepare/commit/discard methods
and `PreparedDFlashProposal` input buffers with a copy-only replay fixture.
Accepted and rejected feature inputs are already projected in that fixture;
the five-tap target feature projection is not replaced in the hardware runtime.
The gate also checks a separately live trace and a stale-cache negative control.

The integrated run exits0 and closes both devices cleanly after1,227.3 s:
eight exact attention-operand comparisons,36 replay comparisons,12 unchanged
state/live-trace checks, four committed-history comparisons and four detected
stale-cache controls. All11 recorded source hashes match this integration.
Report SHA256: `1608bee81863e51127a20c9c26dbaa5b81a9acc1532548bc949361ef716ceba9`.
Local validation passes846 CI tests,60 harness tests and shell syntax checks.

`full-dflash-kv-cache-request` keeps captured T8, commit-only GDN and fused
convolution in both arms at the same4K prompt. Two audits precede measured ABBA.
Candidate audits recompute every layer's complete committed K/V at initialization
and after each publication, compare cached and uncached eager proposals, then
compare eager and trace outputs. Measured requests perform none of those audits.

Hardware promotion requires unchanged committed target tokens/GDN/valid-KV/
inactive slots, exact audited eager/trace proposal outputs, accepted-prefix
publication and separate uninstrumented PP/CTX/TG. Keep the current uncached
lead as the paired control and retain four-link communication.
