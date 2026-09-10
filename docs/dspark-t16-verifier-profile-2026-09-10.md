# Current DSpark T16 verifier attribution

The repeat-confirmed native-attention request reaches87.10 committed TG at4K.
Verification still takes about76ms/block. Earlier device attribution measured
T8 DFlash2 and does not establish this T16 breakdown.

`dspark-verifier-profile` runs one complete audited native-attention DSpark
request with the current captured T16 verifier and commit-only GDN. The observer
surrounds real verification calls; it does not run an alternate fixture or change
draft proposals, verification, acceptance or publication. PP/TG are suppressed.

Required evidence:
- Exact target output/state/inactive slots and every committed feature check.
- A marker for every real block, position, width and trace invocation.
- Matching replay counts and stable operation/core coverage on both chips.
- At least three full T16 calls, with first replay excluded from steady medians.
- Incremental device-data dumps and retained cgroup peak/OOM evidence.
- Unchanged request/native source fingerprints and clean teardown.

Per-chip kernel envelopes and interval unions are observations, not a complete
dependency critical path. Summed kernel durations can overlap and must not be
added across chips or converted into TG. Host durations include device waits.
This profile is instrumentation only; serving defaults remain unchanged.
