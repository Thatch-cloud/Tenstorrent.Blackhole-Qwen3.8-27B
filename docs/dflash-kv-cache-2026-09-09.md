# Historical K/V cache for DFlash2

**Status: experimental components only; not connected to the request runtime.**
No cached-drafter hardware speed or correctness result exists yet. Serving
defaults and the measured lead are unchanged.

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
| Captured proposal integration | Still to implement: stable per-bucket K/V inputs, updated from the committed bank before replay |
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
| Transaction host tests | Four pass; multiple layers, prefixes1/7/8/32, short-window growth, eviction, abort, failed spare write and stale ticket rejection |
| Learned numerical simulator | Passed `20260908T222847Z-416-draft-kv-projection-probe`: 20 bit-exact comparisons and four detected stale-history controls, pinned layer1 weights, both chips |
| Device cache ownership/publication and stable replay inputs | Pending; host tests do not qualify device ownership |
| Full eager proposal comparison against uncached lead | Pending |
| Matched complete-request hardware ABBA | Pending |

The numerical probe compares full versus separate history/live projection at
CTX170 and4093, then seven-row eviction/append at4100. It also requires changed
input/position trace replay and a detected stale-history control. Its pass does
not qualify all learned layers or the complete transactional cache integration.
Weights are checked against the pinned checkpoint hashes before any device opens.

The simulator exits0 and closes both devices cleanly after1,356.6 s. Report
SHA256: `5cc6dc345e924d70c64ac75fa391218f23cd27abd332f0c094429ad9ebb5d7ef`.
Projection source SHA256:
`6c217c51738d7e3feb0cf7a0f911c82f5bd70850095019ed8c8ac14402d8f204`.
All six recorded source hashes match the tested files. The complete host suite
passes838 CI tests and60 harness tests. The transactional cache class still
requires its device ownership and integrated proposal gates.

Hardware promotion requires unchanged committed target tokens/GDN/valid-KV/
inactive slots, exact audited eager/trace proposal outputs, accepted-prefix
publication and separate uninstrumented PP/CTX/TG. Keep the current uncached
lead as the paired control and retain four-link communication.
