# Drafter-only projection precision

**Host prototype only. Not installed, simulated or hardware-qualified.**

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
