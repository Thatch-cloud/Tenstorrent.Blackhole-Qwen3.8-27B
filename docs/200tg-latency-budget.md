# What must change to reach 200 TG

Recomputed from timed blocks in combined runs **35258782100** (unchanged T16
control) and **35204498094** (older T32). These are not matched width-only
experiments: T32 lacks shared-Q/K, norm-prefetch and incremental publication.

| Measured block quantity | T16 | Older T32 |
| --- | ---: | ---: |
| Mean committed tokens | 12.1 | 11.0 |
| Draft / verifier / selection ms | 24.67 / 66.63 / 5.34 | 42.40 / 90.11 / 13.31 |
| Whole cycle ms | 97.55 | 147.25 |
| Cycle budget for 200 TG at observed acceptance | 60.50 | 55.00 |
| Required cycle reduction | 37.98% | 62.65% |
| Hypothetical TG with drafting taking zero time | 166.04 | 104.91 |

The zero-draft calculation holds all other timings and acceptance fixed. It is
an optimistic block-only counterfactual, not measured full-request performance.
Whole cycles include time beyond the three displayed phases; the calculator does
not discard that remainder or add nested timing fields twice.

## Experiment priority

- **Do not retry longer drafts unchanged.** T32 committed fewer tokens per block,
  despite proposing 31 rather than 15. At its measured latency, 200 TG requires
  29.45 committed tokens per block, versus 11 observed.
- **Draft-only optimization cannot close the current gap.** Even perfect removal
  of T16 drafting leaves a 72.87 ms cycle at the observed acceptance.
- **Target verifier needs a substantial reduction.** With other T16 costs fixed,
  it must fall from 66.63 to about 29.58 ms. Smaller gains can contribute but
  cannot individually qualify the target.
- Next structural investigation: reduce verifier intermediate traffic and
  operation boundaries across adjacent projections/collectives, while retaining
  per-layer BF16 rounding and speculative rollback. Measure the complete request;
  do not substitute an isolated matmul result.

T32 parity requires explicit new simulator coverage, not changing a flag:
`gdn_shared_qk_scope.py`, `gdn_shared_qk_pipeline.py` and
`gdn_shared_qk_program.py` allocate and dispatch T16-only geometry, and
`full_dspark_request.py` rejects this combination. Those safeguards remain intact.

`scripts/ci/speculative_latency_budget.py` reproduces the block calculations;
its tests reject mixed widths, invalid counts and invalid timings. No serving
defaults or accepted recipe changed during this analysis.
