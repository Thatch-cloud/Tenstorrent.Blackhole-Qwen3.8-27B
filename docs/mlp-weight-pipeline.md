# Two in-flight MLP weight blocks

**Simulator qualified; combined hardware performance unqualified.**

[Simulator run 35223012166](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35223012166)
at `3f6f6cb582cfdbd9a72834d8a9ade85e7c004b5a` passes in **3m55s**.
Independent validation confirms both-chip native-exact T16 eager output,
all 12 changed-input replay comparisons, four stale-input negative controls,
four packed-weight comparisons, zero exit and clean container teardown.
Report SHA-256:
`abbb01d1ece1b625c19909263c9a45eab5fa859d46d27e265cd84e1535166371`.
Next admission must bind the generated reader and unchanged projection to this
report before a matched combined baseline/candidate hardware comparison.
No throughput claim, default change or hardware promotion follows from SIM.

Combined timing run `35221730435` shows the sampled MATH issue interval tracks
UNPACK input/weight readiness. Earlier read-order and larger-buffer trials did
not improve complete-request TG. This candidate changes a different boundary:
issue the next block before waiting for the current block's DMA completion.

| Property | Baseline | Candidate |
| --- | --- | --- |
| Weight buffer capacity | Two blocks | Two blocks, unchanged |
| Outstanding block reads | One before each barrier | Up to two, transaction IDs 1/2 |
| Compute, BF16 rounding, weights | Winning T16 | Unchanged |
| Input multicast and output writer | Winning T16 | Unchanged |

The producer reserves both slots before issuing ahead, waits for the current
transaction before publishing, and never reuses a slot until the consumer has
released it. Twenty blocks return the two-slot ring to its starting position on
every invocation. The final read barrier drains all reads and resets the read
transaction ID. Padding for the final worker retains explicit zero writes.

The pinned Blackhole API supports transaction-specific completion barriers and
stateful one-packet reads. Source references at runtime revision
`9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`:
- `tt_metal/hw/inc/api/dataflow/dataflow_api.h`
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h`

Host tests check exact restoration of the original reader, ordered publication,
slot ownership and page/padding geometry. They are not device proof. The
simulator route runs eager numerical checks, changed-input trace replay and
packed-weight controls without loading the full model. Only an exact passing
simulation can admit a matched combined hardware comparison. Extra command
register writes and later initial publication may outweigh overlap; reject the
candidate if complete-request TG does not improve reproducibly.
