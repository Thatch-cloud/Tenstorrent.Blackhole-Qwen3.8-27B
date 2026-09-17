# Shared-Q/K recurrence input cache

The qualified shared-Q/K reader still gathers V, beta and gate tiles per token.
This candidate retains those three BF16 tiles in CB31 for the T16 block, adding
6 KiB per recurrence worker. Q/K normalization, recurrence arithmetic, writers,
prefix states and norm/gate stages remain unchanged. Reduced tile traffic is
not evidence of reduced device latency.

## Simulator qualification

Run **35081361623**, revision `4012e8b`, passes in **3m00s** without model weights.
Both chips pass all 24 exact output/state comparisons and 48 immutable-input
checks across eager and changed-input replay modes. Sources remain unchanged
through execution and cleanup passes. The report hash and all twelve selected
Python runtime dependency hashes were independently checked locally against the
historical sources plus the candidate helper. Native hashes are rechecked against
the actual runtime by the admission gate before hardware use.

Report SHA256:
`98b306ea4f495a5ad71aba3d8d631704893aae09a791dfe081a3813805da5150`.
Candidate helper SHA256:
`7f2c35cee636789a6068ef10f00324bdf6c82fb865f16a8296515c3e83fbca3a`.

## Matched hardware result

Run **35082112059**, revision `41fc678`, completes in **15m22s**. All six
requests pass exact output/state/inactive-slot checks and matched proposal
acceptance. Each candidate request constructs 96 cached three-stage pipelines;
the control constructs none. Shared-Q/K and two-block MLP buffering remain
enabled in both arms. Integrated capacity and correctness pass.

| CTX 32768, one stream | Control | Cached inputs |
| --- | ---: | ---: |
| Committed TG tok/s | 71.213 | 71.677 |
| Verifier/readback ms/block | 74.088 | 73.199 |
| Complete cycle ms/block | 149.292 | 148.327 |
| Committed tokens/block | 10.636 | 10.636 |

The verifier difference is 0.889 ms, while aggregate TG improves only 0.65%.
Individual control requests are 70.02/72.44 TG and candidate requests
72.05/71.30 TG. This small, overlapping sample does not establish repeatable
end-to-end improvement. Retain as a correctness-qualified candidate, not an
adopted throughput win. Serving defaults remain unchanged. It does not close
the roughly 95 ms cycle-budget gap to 200 TG.

Report SHA256:
`8042a8d2df5ede3b9b9b139869fd2342003969aa598bbf86d9c7236afd223e62`.
