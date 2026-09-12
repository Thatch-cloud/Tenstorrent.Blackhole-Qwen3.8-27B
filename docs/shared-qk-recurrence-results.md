# Shared Q/K recurrence experiment

## Status

The complete synthetic pipeline passes numerical simulation and two matched
full-request hardware comparisons. The verifier saving repeats; full context
and coding-quality acceptance remain pending. Serving defaults are unchanged;
the target remains 200 committed tokens/s for one stream.

| Gate | Evidence | Result |
| --- | --- | --- |
| Host source checks | 12 tests | Pass |
| Complete pipeline simulation | CI 34701425373, revision `1d5cdb0` | Pass |
| Output, every state prefix, pre-normalization bridge | 24 exact comparisons across two chips | Bit-exact |
| Input preservation | 48 comparisons | Unchanged |
| Trace replay | Changed seeds 1, 2, then original seed 0 | Pass |
| Provenance | 752 source hashes checked against local sources and pinned runtime export | Match |
| Combined-request hardware | CI 34702027898, revision `4b3f90b` | Exact; small timing improvement |

## First hardware result

| Runtime | PP tokens/s | CTX | Committed TG tokens/s | Mean verifier trace ms |
| --- | ---: | ---: | ---: | ---: |
| Native combined runtime | 3291.48 | 4096 | 105.14 | 68.00 |
| Shared Q/K combined runtime | 3330.21 | 4096 | 107.15 | 66.81 |

Single stream, T16 verifier, 15 draft queries. Each arm has two timed requests
and one separate instrumented audit. Each timed response commits 121 tokens;
acceptance is 222/330 across each arm. This is not a long-context or held-out
coding-quality result.

TG improves 1.91%; verifier trace drops 1.19 ms. Draft time also drops from
28.43 to 27.37 ms, so not all TG improvement can be attributed to this kernel.
The candidate constructs 96 three-stage pipelines per request and releases
its retained preparation buffers. All six requests pass token/state/inactive
checks; 801 script hashes match. A repeat is required before promotion.

Hardware report SHA256:
`d2a4ee5c29b50accb741c0d18e4e08370d89e49252f35790153d220e90aee766`.

The candidate computes Q/K normalization once per shared head for a T16 block,
then feeds the existing recurrence and normalization/gating stages. The probe
uses synthetic tensors, not model weights. It includes preparation in the
captured pipeline and allocates persistent buffers before capture.

Report: `runner-evidence.local/34701425373/gdn-shared-recurrence.json`.
SHA256: `a155b893786f68ac36b4f040454892df683a0da0c3aa0b22b898697b22241ffe`.

## Next acceptance gate

Repeat CI 34702526963 uses the same revision and passes all six exact
token/state/inactive checks, 96 candidate pipeline constructions per request,
buffer release/restoration, and 801 script hashes. Native PP / CTX / TG is
3241.27 / 4096 / 105.85; candidate is 3279.29 / 4096 / 106.58. Verifier time is
68.01854 versus 66.81181 ms, confirming the first run's approximately 1.2 ms
saving. Draft time moves the other way: 27.13774 versus 28.10680 ms. The repeat
therefore shows only 0.70% end-to-end gain. Do not extrapolate to 200 TG.

Repeat report SHA256:
`73151825723aee8fa1fadb2e64754f797dff151adc0c7439c460b1a30129eacb`.

Integrate behind an experiment-only switch with explicit lifetime ownership of
the preparation buffers. Require the simulator report and matching source
hashes before hardware execution. Run matched complete-request native versus
candidate audits and timing, counting all preparation, drafting, verification,
and publication work. Reject token/state mismatches and report PP / CTX / TG
separately. Do not infer throughput from this simulator pass.
