# T16 commit-only prerequisite: qualified

Run `20260910T041215Z-381-multitoken` completed with exit0 and clean mesh teardown.
All 17 accepted prefixes (0 through16) pass model-adapter and real T2 continuation
checks. All 17 precommit native-state invariants pass; the stale-state control
detects corruption.

Independent audit reconciles the exact prefix sequence in the raw log, all20
before/after/current source hashes against immutable commit `04e9071`, native and
handoff hashes, generated kernel hashes, flags and adapter/publication hashes.
The report SHA256 is
`fae61660a4718de7b8eaf39a42dee2cf20fbd7a7f667bdd6bd361d4271f681f8`.

The earlier run remains unqualified: its model-adapter completion path omitted
the post-run source fingerprint. The fixed harness has regression coverage for
both completion paths and rejection of changed sources; no numerical threshold
or prefix coverage was relaxed.

This qualifies simulator functional behavior against serial T1 of the same
kernel, not native-target correctness or throughput. The matched eager / captured
proposal / captured-plus-commit-only hardware comparison supplies those gates.
