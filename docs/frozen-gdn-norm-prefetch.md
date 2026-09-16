# GDN norm bridge prefetch candidate

Simulator-qualified; not hardware-qualified or selected by the combined runtime.

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
