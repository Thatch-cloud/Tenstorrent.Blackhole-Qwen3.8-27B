# T16 fused MLP larger reduction blocks

Status: simulator and combined hardware correctness pass; performance regresses.
**Rejected for promotion. Keep the original K8 recipe.** No serving changes.

## Combined hardware result

[Run 35258782100](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35258782100)
at `d7f0d62a483c73ad32365b552c32a958be14e672` completes in 6m28s.
Two native audits and four complete ABBA requests pass at 4,096 context tokens,
one stream on both P150A cards. The independent report verifier recomputes:

| Arm | PP tok/s | Committed TG tok/s | Accepted / proposed |
| --- | ---: | ---: | ---: |
| Original K8 | 3,340.87 | 123.98 | 224 / 300 |
| Candidate K32 | 3,299.23 | 122.41 | 224 / 300 |

Both arms commit 242 timed tokens. TG regresses **1.26%** overall; paired
changes are **-1.73% and -0.79%**. Mean T16 verification/readback increases
from 66.63 to 67.14 ms. Fewer reduction blocks did not reduce combined verifier
latency; do not rerun the unchanged candidate as a performance improvement.

All 866 script, 1,520 native and seven candidate-source fingerprints remain
unchanged. The request report records clean closure and process exit is zero.
Report SHA-256:
`40a3d2de2acfd43f84be924f493cab6b7fb2184143362a17160f865f2d1aa0e5`.
This bounded correctness pass does not qualify held-out coding quality or serving.

## Simulator evidence

[Run 35257670281](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35257670281)
at `16559dae3b7fdaebaf3a64f652633076e69663ff` passes in 3m51s.
Independent artifact inspection validates two eager chip checks, twelve exact
native/fused changed-input replay checks, four stale-input negative controls
and four byte-exact gate/up packed-weight checks. All five staged source hashes
match the downloaded manifest; process exit and all container cleanup statuses
are zero. Report SHA-256:
`d7594fda2ade8ff6252fcb0a7d0bb8c450ed49a4695c6ecd5d64b72dcc05517d`.

This covers synthetic full-projection T16 inputs, not model-wide numerical
equivalence, coding quality or speed. Combined admission binds these sources
and compute hashes to the hardware candidate before comparing complete requests.

| Property | Frozen control | Candidate |
| --- | ---: | ---: |
| K tiles per block | 8 | 32 |
| Reduction blocks | 20 | 5 |
| Buffered blocks | 2 | 2 |
| Weight precision / layout | BF4 / unchanged | BF4 / unchanged |
| Extra L1 per active worker | — | 258 KiB |

The candidate reduces input multicast synchronization and intermediate packing
rounds. It retains the native compute source, LoFi arithmetic, FP32 destination,
BF16-rounded epilogue, weight traversal and independent K8 native reference.
**Reduction grouping changes, so numerical equivalence is not assumed.**

This differs from the rejected two-inflight weight-reader experiment, which
retained K8. The profile's 99-core fused MLP group includes multicast drainers:
only 91 workers perform matmuls with three output pairs each.

Admission: full-size synthetic T16 eager and changed-input trace replay on both
simulated chips, exact native output and packed-weight checks. The simulator
job is capped at 12 minutes and loads no model weights. Failure rejects this
candidate; success only permits a combined 4K ABBA hardware test. Promotion
requires native correctness and more than 2% TG improvement in both pairs;
neither a simulator pass nor an isolated kernel timing proves 200 TG.
