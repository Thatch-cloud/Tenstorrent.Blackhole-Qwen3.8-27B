# GDN norm bridge prefetch candidate

Not device-qualified or selected by the combined runtime.

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
