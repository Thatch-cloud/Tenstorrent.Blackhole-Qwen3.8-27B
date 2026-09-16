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

Next is a matched full-request hardware comparison: shared-Q/K and two-block
MLP buffering in both arms; only V/beta/gate caching changes. Keep exact
output/state and proposal-acceptance checks. Hardware speed and integrated
capacity remain unqualified. Do not combine this with the rejected four-block
MLP buffering trial or change serving defaults.
