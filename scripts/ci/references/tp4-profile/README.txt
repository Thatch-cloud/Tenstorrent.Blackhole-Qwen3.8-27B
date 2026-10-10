v138-trace0-slim.csv.gz: the positive control of test_tp4_profile_report.

A slim copy of the CPP device report of TP2's v138 profile run (the M3native packed-round op profile, 4 users x 32k, 64 rows,
before the T1/T2/K1/K5 levers): trace 0 (the 3,605-op verify), replay sessions 5 and 6 (complete on both chips) and session 8
(truncated: the per-core buffer filled), the ten columns the analysis reads. The analysis of it must reproduce the recorded table:
kernel sum 131.42 ms over a 134.24 ms span, GDN recurrence 25.24, GDN glue 24.29, MLP matmuls 21.08, SDPA 17.66, attention glue
12.92, conv gates 6.78, in-trace gaps 2.82, critical path 132.07, skew 0.55; gate 231 GB/s, 97 ns per tile per core.
It is TP2 (two chips): the test passes chips=2.

v676-multi-sdpa-slim.csv.gz: the fixture of test_tp4_profile_report's MultiSdpaBlockTests.

A slim copy of the CPP device report of the shipped multi-SDPA stack's M3native op profile (4 users x 4k, two 64-row blocks per
round, 4 chips): trace 0 (the 1,681-launch packed block: ONE SDPA launch per attention layer folded in and out by generic ops, the
F1 conv-gates as a generic op, so no named conv-gates launch), replay sessions 1 and 2, and the 1,599-launch 4-row lone step (four
SDPA launches per attention layer, one named conv-gates launch per GDN layer), replay session 1; the ten columns the analysis
reads, device cycles rebased to zero per chip. The block must come out as the packed verify (about 45.5 ms of kernels per chip over
a 46.7 ms span, 1.15 ms of in-trace gaps) and the 4-row step as the lone step (34.78 ms), not the other way round.
