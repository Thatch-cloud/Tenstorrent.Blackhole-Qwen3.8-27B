# T16 fused MLP larger reduction blocks

Status: unqualified simulator candidate. No serving or winning-recipe changes.

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
