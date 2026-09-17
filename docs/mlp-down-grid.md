# Native T16 MLP-down grid

**Simulator qualification passes; hardware performance is unqualified.**
Winning and serving defaults are unchanged.

## Simulator acceptance

[Run 35254182130](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35254182130)
at `b881571eb702fc3862a748e7fb42fdb0568097af` passes in **2m23s**. All 12
eager/replay comparisons, 14 activation/weight integrity checks and three
poisoned-replay negative controls pass. Exit and all cleanup statuses are zero.
Independent admission reproduces the complete matrix and verifies all four
measured source fingerprints, including the retained native projection factory.
Report SHA-256: `fb677f9c1dde8c1badd3f7a6f81378b8e6fa7fde06fd62e102d07a3a94d87404`.

The pending combined adapter is scoped to `FusedT16Arm.forward` at exactly
16 rows. It temporarily changes only that call's MLP-down program, restores
shared model arguments immediately and leaves native/tail calls alone. It checks
both chips' BF8 weight geometry and native LoFi/FP32/packer-accumulation policy.
Host tests cover normal and exceptional restoration, untouched smaller buckets,
source admission and rejection of incomplete numerical evidence. No hardware
request has yet qualified this adapter or measured its performance.

Combined staging now passes against the frozen winning recipe. The hardware
workflow runs two fresh native-reference audits and then four complete requests
in control/candidate/candidate/control order. Every candidate MLP call must match
the per-layer fused-arm hit counters; all 64 layers must be covered. The report
independently recomputes complete-cycle TG, rejects changed acceptance or output,
and keeps the existing norm-prefetch/history/weight-integrity gates. It does not
promote a candidate merely because its simulator or CI job passed.

The retained combined trace attributes approximately 7.804 ms to 64 MLP-down
calls per verifier replay. This experiment changes only native output-column
distribution for the local 8,704 x 5,120 projection.

| Parameter | Control | Candidate |
| --- | --- | --- |
| Grid rectangle | 11 x 3 | 11 x 8 |
| Output tiles per worker | 5 | 2 |
| Workers with output | 32 | 80 |
| Activation / weights | BF16 / BF8 | Unchanged |
| Reduction block / output subblock | 8 / 1 x 1 | Unchanged |
| Compute | LoFi, FP32 destination, packer L1 accumulation | Unchanged |
| Input/output placement | Interleaved L1 | Unchanged |

The rectangle also carries multicast bookkeeping: 88 grid positions is not a
claim of 88 productive matmul workers. More readers may worsen DRAM contention;
the current smaller grid was previously tuned for skinny decode. This candidate
must earn promotion in the combined T16 workload, not from core counts.

Unlike the rejected streamed MLP, there are no remote weight FIFOs, producer
cores or weight relayout. Unlike the rejected small-tile composition, there are
no input/output tile conversions. Gate/up fusion, down-projection precision and
the subsequent native reduce-scatter remain the required combined control.

The synthetic simulator probe uses independent weights/activations on both chips,
three eager inputs, three changed-input replays, poisoned output replacement and
activation/weight-value integrity checks. It does not claim raw packed-padding
bit preservation, coding quality, collective performance or TG. The bounded
runner reuses the legacy `gdn-output-grid` launcher filename; its report explicitly
identifies `projection=mlp_down`, 16 rows and the full 8,704 x 5,120 geometry.
It is a new MLP qualification, not reuse of the older GDN-grid acceptance.

Next gate: exact simulator results, then complete audited ABBA requests with only
the MLP-down program changed. Reject on numerical failures or non-repeatable
full-cycle gain. This component cannot alone close the entire 200-TG gap.
