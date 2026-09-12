# Shared Q/K recurrence experiment

## Status

The complete synthetic pipeline passes numerical simulation. It is not yet
qualified on hardware and has no measured performance gain. Serving defaults
are unchanged; the single-stream target remains 200 committed tokens/s.

| Gate | Evidence | Result |
| --- | --- | --- |
| Host source checks | 12 tests | Pass |
| Complete pipeline simulation | CI 34701425373, revision `1d5cdb0` | Pass |
| Output, every state prefix, pre-normalization bridge | 24 exact comparisons across two chips | Bit-exact |
| Input preservation | 48 comparisons | Unchanged |
| Trace replay | Changed seeds 1, 2, then original seed 0 | Pass |
| Provenance | 752 source hashes checked against local sources and pinned runtime export | Match |
| Combined-request hardware PP / CTX / TG | Not run for this candidate | Pending |

The candidate computes Q/K normalization once per shared head for a T16 block,
then feeds the existing recurrence and normalization/gating stages. The probe
uses synthetic tensors, not model weights. It includes preparation in the
captured pipeline and allocates persistent buffers before capture.

Report: `runner-evidence.local/34701425373/gdn-shared-recurrence.json`.
SHA256: `a155b893786f68ac36b4f040454892df683a0da0c3aa0b22b898697b22241ffe`.

## Next acceptance gate

Integrate behind an experiment-only switch with explicit lifetime ownership of
the preparation buffers. Require the simulator report and matching source
hashes before hardware execution. Run matched complete-request native versus
candidate audits and timing, counting all preparation, drafting, verification,
and publication work. Reject token/state mismatches and report PP / CTX / TG
separately. Do not infer throughput from this simulator pass.
