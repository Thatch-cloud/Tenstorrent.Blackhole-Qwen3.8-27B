# T16 weight-read issue order

**Simulator qualified; combined hardware performance remains unqualified.**

Combined run 35210079674 found larger weight-read completion waits than the
isolated fixture. This experiment changes request ordering, not buffer capacity,
weight precision/layout, arithmetic, core count or NoC selection.

The 91-worker fused projection reads six columns per worker in the same order.
With eight-way interleaving, each synchronized issue slot initially addresses
four banks, with up to 23 requests to one bank. Rotating the six-column order
after each four-worker group spreads the idealized issue slot over all eight
banks, with at most 12 requests per bank. Workers are not actually synchronized
at each instruction: this is a scheduling hypothesis, not measured bank traffic
or a predicted throughput gain.

Each read keeps its original destination tile. The full block read barrier stays
before the same CB push. Host tests enumerate every worker and K block: all
87,040 packed pages and all padded destination tiles remain unchanged, exactly
once. A reverse source transformation reproduces the original reader verbatim.

Qualification order:
1. Bounded T16 simulator eager/replay and byte-exact packed-weight checks.
2. Same winning combined runtime, native output/state/feature audit and complete
   uninstrumented request timing against the unchanged reader.
3. Retain only a repeatable combined TG improvement; do not promote an isolated
   projection speedup or the analytical bank model.

The simulator fixture has a 510-second process cap and 12-minute whole-job cap;
it does not load the full target model. Serving defaults remain unchanged.

## Simulator result

Run **35211351632**, source `526bc70edac46dec956d8d09c39594273ba5eaa0`,
passes in **3m56s**. Both eager outputs, all 12 native/candidate replay outputs,
four stale-input controls and four complete packed-weight comparisons pass.
Independent admission checks clean process/container teardown, pinned runtime,
exact projection/replay sources and an otherwise unchanged T16 kernel manifest.
The candidate weight-reader SHA-256 is
`5150f285c437cd53bf8dab407e433ce096336c5c2aa6a4797c2dd820b4a3a222`.
Report SHA-256:
`3bac1b049aef7a12affd6463282aa773b199becfa1cddc605bb2f0074a41fbf8`.

The comparison wrapper retains shared Q/K, norm prefetch, incremental history,
fused T16 MLP, target attention and score layout in **both** arms. It plans two
fresh feature audits followed by complete unchanged/staggered/staggered/unchanged
timed requests in one loaded model. Output and proposal acceptance must match
across all six requests. Each arm reports cold PP and complete-cycle TG separately.
Both paired TG gains must exceed 2% to pass the improvement screen; passing that
screen is not automatic promotion or held-out coding-quality acceptance.

Fresh local staging of the complete frozen runtime passes, including the source
gate and simulator evidence. The dedicated `qwen-mlp-read-order-combined.yml`
loads the model once, keeps the normal disk-pressure and exclusive-card gates,
and schedules two audits plus four complete timed requests at 4K. Its launcher
cap is 600 seconds and whole-job cap is 12 minutes; the full context ladder is
unchanged. Hardware acceptance and throughput improvement remain unproven.

Host integration tests exercise all six calls, verify the candidate projection
and simulator identity are installed only for their intended arm, and check
restoration between requests. Per-request TG must agree with committed tokens
divided by complete decode wall time before the improvement screen is evaluated.
