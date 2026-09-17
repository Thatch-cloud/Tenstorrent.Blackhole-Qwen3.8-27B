# GDN norm bridge prefetch candidate

Simulator and matched combined hardware checks pass; serving remains unchanged.

## Matched hardware result

Run **35087582465**, revision `b126116`, passes in **13m45s**. Both arms retain
incremental publication. One audited request per arm precedes four A/B/B/A timed
requests; every response commits 117 tokens with identical output hashes and
exact target output/state/inactive-state checks.

| One stream | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: |
| Original norm reader | 2940.84 | 32768 | 87.41 |
| Prefetched norm reader | 2962.66 | 32768 | **89.01** |

The within-run gain is **1.83%**, not broad performance acceptance. Candidate
repeats are 89.45/88.58 TG; controls are 87.50/87.33. Verification/readback drops
74.26 -> 72.86 ms/block; drafting is 52.11 -> 51.47 ms and whole cycle
133.78 -> 131.38 ms. The observed 11.7 committed tokens/block still requires
58.5 ms for 200 TG. This small gain does not solve the dominant budget gap.

Every candidate request records 96 three-stage builds; controls record zero.
All six requests use bounded incremental publication. All 850 reported source
entries and 1,520 native entries match before/after; container exit is zero,
with no OOM. Pre-load full I/O stall is 0.0438%, not a throughout-run guarantee.
Held-out coding quality and sustained serving remain unqualified.

Hardware report SHA256:
`b8be862d143160de6f030d182dc2f2552b861cf5dee72fd54de91f79cc1a5132`.

## Simulator qualification

Run **35086789628**, revision `2d1b59e`, passes in **3m03s** with clean close
and exit zero. The full matrix contains 24 exact eager/changed-input replay
checks and 48 immutable-input checks across both chips. All 791 before/after
source entries are unchanged; the reported candidate helper hash matches git.
No model weights were loaded and no throughput claim follows from this result.
The hardware admission gate pins this report and checks the deployed helper,
shared pipeline dependencies and native source hashes before use.

Report SHA256:
`120eca72f4fde17cf79453fbfd535d393534828f1ef8ee272925125d9cf849f8`.

The current norm reader fetches 16 rows times four partitions through one
128-byte scratch read followed by a barrier for every partition. The proposal
issues all 64 reads into an 8 KiB staging buffer before one barrier, then uses
the unchanged FP32-to-tile copy and unchanged norm/gate compute and writer.
CB5 grows from one to two 4 KiB pages on each of 24 norm workers. No recurrence,
Q/K preparation, precision, weights, collective or serving setting changes.

The current verifier profile attributes 4.49 ms of summed kernel durations to
the 24-core group associated with normalization. This is not proof that reader
waits account for that time, nor an additive critical-path saving. This candidate
alone cannot close the gap to 200 committed TG.

Host checks cover all 24 heads and bridge words, unchanged computation/copy
source, buffer isolation and exception restoration. Simulator staging retains
the complete original eager/replay/immutable-input matrix. Device qualification
must pass before a matched combined test with incremental publication enabled
in both arms. Do not combine the earlier unproven V/beta/gate-cache candidate.
