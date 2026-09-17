# Native T16 MLP-down grid

**Unqualified experiment; winning and serving defaults are unchanged.**

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
