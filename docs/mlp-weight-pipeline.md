# Two in-flight MLP weight blocks

**Numerically qualified; rejected for performance promotion.**

## Combined hardware result

[Run 35223968603](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35223968603)
at `d671a1b0389b67cdd6ee3011b6f6dbb2e962b54f` completes in **6m27s**.
Independent report validation passes both feature audits, exact native output
and state, packed weights, source identities and complete-cycle accounting.
The container exits zero without OOM. Report SHA-256:
`cfef8973baa3b1b1e0ba08a4eade81aa16ff0e941e15324f3f6c3a40300759b8`.

| 4K context, one stream | Original reader | Two-inflight pipeline |
| --- | ---: | ---: |
| Cold PP, tok/s | 3,302.32 | 3,378.78 |
| Complete committed TG, tok/s | 121.11 | 123.16 |
| Timed committed tokens | 242 | 242 |
| Accepted / proposed | 224 / 300 | 224 / 300 |
| Mean blocking verifier, ms | 65.779 | 66.160 |

Aggregate TG is +1.69%, but matched pairs are **-2.31% / +5.78%**, failing
the repeatability screen. More importantly, blocking verifier latency worsens
in both pairs (65.773 -> 66.162 ms; 65.785 -> 66.159 ms). The favorable aggregate
TG therefore does not demonstrate a kernel gain; drafting/selection variation
dominates that comparison. Cold PP differences are not attributable to this
decode-only reader change.

Keep the original reader. Do not rerun this unchanged two-inflight schedule or
promote it based on the aggregate rate. The next candidate needs a different
mechanism, not another capacity/ordering/barrier variation without evidence.

## Simulator admission

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

## Combined comparison

The opt-in `qwen-mlp-weight-pipeline-combined.yml` route loads the winning runtime
once, performs separate baseline/candidate feature audits, then times complete
requests in baseline/candidate/candidate/baseline order. Context is 4,096 and
stream count is one. Both arms retain shared-Q/K GDN, norm prefetch, incremental
feature publication and native weight/attention checks. Only the weight reader
changes. Source admission pins the simulator report, reader transformation,
unchanged projection, replay helper and compute manifest.

`QWEN_MLP_WEIGHT_PIPELINE=1` is experiment-only. Results report cold PP and full
committed TG per arm, including draft, verify, selection/publication and host
overheads. The screen requires >2% improvement in both matched pairs; that alone
does not qualify held-out coding quality or authorize serving promotion.

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
