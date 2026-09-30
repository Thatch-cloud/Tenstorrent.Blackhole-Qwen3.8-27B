v138-trace0-slim.csv.gz: the positive control of test_tp4_profile_report.

A slim copy of the CPP device report of TP2's v138 profile run (the M3native packed-round op profile, 4 users x 32k, 64 rows,
before the T1/T2/K1/K5 levers): trace 0 (the 3,605-op verify), replay sessions 5 and 6 (complete on both chips) and session 8
(truncated: the per-core buffer filled), the ten columns the analysis reads. The analysis of it must reproduce the recorded table:
kernel sum 131.42 ms over a 134.24 ms span, GDN recurrence 25.24, GDN glue 24.29, MLP matmuls 21.08, SDPA 17.66, attention glue
12.92, conv gates 6.78, in-trace gaps 2.82, critical path 132.07, skew 0.55; gate 231 GB/s, 97 ns per tile per core.
It is TP2 (two chips): the test passes chips=2.
