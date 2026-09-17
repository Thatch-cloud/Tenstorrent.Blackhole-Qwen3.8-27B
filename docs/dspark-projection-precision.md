# Drafter-only projection precision

**Query-projection simulator execution passes; combined correctness/performance unqualified.**

## First real-weight simulator screen

[Run 35226045287](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35226045287)
at `fbd3486a624ea160c1c59bcd91892ed0596e5cd8` completes in **1m47s**.
It uses layer-zero query weights, two synthetic activation patterns and both
simulated chips. All six poisoned changed-input replays reproduce HiFi2 eager
output exactly; five integrity checks preserve input/weight contents and buffer
bindings. Independent validation and container teardown pass.

| HiFi2 versus HiFi4, FP32 output | Observed range across chips/inputs |
| --- | ---: |
| Maximum absolute difference | 0.06569-0.08042 |
| RMS difference | 0.01147-0.01331 |
| Reference RMS magnitude | 2.851-3.323 |
| RMS difference / reference RMS | approximately 0.4% |

This is not a numerical-equivalence or model-quality pass: outputs differ.
Report SHA-256:
`361ace678704220f0ed138d2aa6878924187a92ef59d9b2dd4b0a6fd0d1e1e6c`.
Remaining projection weights/shapes require their own execution coverage before
enabling the broader drafter policy. Then measure changed proposal acceptance
and verify final tokens/state against the unchanged target in combined testing.
No throughput number or serving change is justified by this component result.

The winning combined runtime spends roughly 25 ms drafting and 66 ms verifying
each block. Small target-reader changes have not produced repeatable committed
TG gains. This candidate explores a separate contributor: DSpark layer linear
projections currently use BF16 weights, HiFi4, FP32 accumulation/output and
explicit BF16 casts. Try HiFi2 only in those projection matmuls.

The operations proxy leaves target kernels, weights, normalization, rotary,
attention, scoring, FP32 accumulation, packer policy and rounding boundaries
unchanged. It is explicitly passed to the existing drafter linear helper;
there is no global TT-NN patch or serving flag. Host tests cover routing,
ownership, policy rejection and failure without mutating shared operations.

## Required gates

1. Simulator: compare real-weight HiFi4/HiFi2 projection outputs; record numerical
   differences, require finite output and exact same-policy changed-input replay,
   poisoned-output replacement, input/weight integrity and stable trace bindings.
2. Combined audit: feed candidate proposals to the unchanged target verifier.
   Require final tokens and target state to match native greedy generation,
   including rejection/rollback. Compare candidate eager and replay proposals.
3. Matched timing: retain a native-precision control, record actual acceptance
   and complete-cycle committed TG. Candidate proposals may differ; do not
   silently reuse a comparator requiring identical proposals between arms.
4. Held-out coding and longer-context checks before any promotion. Lower drafter
   precision is not a claim of unchanged acceptance or a 200 TG result.

Existing native-precision numerical gates remain unchanged. A separate candidate
qualification must distinguish expected drafter approximation from target
correctness: no relaxing target checks to accommodate this experiment.
