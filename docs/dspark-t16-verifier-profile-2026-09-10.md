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

## Measured result

[Run34448736208](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34448736208)
passes on revision`c97edc5`. The request commits121 tokens with exact target
outputs/state and11 actual T16 verifier calls on each chip. Ten steady replays
remain after retaining but excluding the first. Local re-analysis reproduces
the saved attribution exactly. Peak container memory is51.54GiB; no OOM events.

| Observed interval | Chip0 | Chip1 |
|---|---:|---:|
| Kernel envelope | 76.176ms | 76.171ms |
| Union of kernel intervals | 74.829ms | 74.812ms |
| Uncovered interval | 1.346ms | 1.358ms |

| Chip0 operation group | Cores | Calls/replay | Median summed kernel time |
|---|---:|---:|---:|
| Matmul, all groups | mixed | 321 | 32.060ms |
| Matmul subgroup | 39 | 128 | 11.795ms |
| Matmul subgroup | 32 | 128 | 10.801ms |
| Matmul subgroup | 43 | 48 | 5.743ms |
| Native decode SDPA | 110 | 256 | 11.648ms |
| Generic operation | 96 | 48 | 11.379ms |
| Generic operation | 24 | 48 | 4.489ms |
| Generic operation | 48 | 96 | 4.359ms |

Matmul subgroups are already included in the all-matmul row. These sums are
not additive critical-path estimates. The CSV labels generic operations by
operation/core count, not kernel source; do not assign their complete cost to
a guessed kernel. Source inspection finds an active-state snapshot copy plus
convolution-window DMA in each GDN layer. Eliminating a copy would require a
read-only native-state input path and every accepted-prefix/continuation gate;
it cannot be claimed to remove the entire48-core group.

The current76ms verifier cost is overwhelmingly device execution, not a large
unmeasured host gap. Prior T8 attribution had32.050ms summed matmul time and
5.827ms decode SDPA; current T16 matmul time is similar while256 serial attention
calls cost11.648ms. This points to row-serial work as well as the substantial
fixed projection cost. Changes must be measured on the full current request.

Request SHA256: `f5fa5cfbad7b5724d2c020cc4192c2306f7c120713b19fd9e8d56eae993d7ffc`.
Device CSV SHA256: `427339becfe6a62d84e10b22c3cc50a8e0e5b6481e571843b780b50e63e84424`.
