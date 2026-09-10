# Current DSpark T16 verifier attribution

## Updated profiling path (September 11)

The opt-in `dspark-verifier-profile` suite now selects folded T16 target attention,
matching the current target verifier rather than the older serial-attention path.
It retains the simulator-qualified attention sources and audited complete request;
the observer rejects an engine whose actual attention mode differs from its declared
mode. The report records and independently checks that mode. The request budget is
256 tokens, matching the current comparison. PP/TG remain suppressed.

This is instrumentation only, not a new kernel or a performance result. The drafter
still uses native score layout: its work lies outside verifier markers. Hardware
attribution for this updated path passed as recorded below. The older measurements remain
valid only for their original serial-attention configuration.

## Folded-T16 hardware result

[Run 34537396817](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34537396817)
passed on immutable revision `1148fb7`. Independent analysis reproduces the saved
device attribution and validates the declared folded-attention route and complete
request checks. Eleven real T16 calls per chip yield ten steady replays.

| Interval | Chip 0 | Chip 1 |
| --- | ---: | ---: |
| Kernel envelope | 69.301 ms | 69.302 ms |
| Union of kernel intervals | 68.074 ms | 68.068 ms |
| Uncovered interval | 1.224 ms | 1.235 ms |

| Chip 0 group | Cores | Calls/replay | Median summed kernel time |
| --- | ---: | ---: | ---: |
| Matmul subgroup | 39 | 128 | 11.789 ms |
| Generic operation | 96 | 48 | 11.734 ms |
| Matmul subgroup | 32 | 128 | 10.810 ms |
| Matmul subgroup | 43 | 48 | 5.739 ms |
| Generic operation | 48 | 112 | 4.607 ms |
| Generic operation | 24 | 48 | 4.495 ms |
| Native decode SDPA | 110 | 32 | 3.513 ms |

Compared with the older profile, attention calls fall from 256 to 32 and their
summed time from 11.648 to 3.513 ms. This confirms the folded path executes;
it does not establish additive critical-path savings. Projection and generic
operation costs remain substantial. Generic labels still do not identify kernel
sources, so they must not be presented as an exact per-kernel GDN attribution.

Request SHA-256: `c0e422bb06ac573a555fa9c5c44a0c11a985b93d5edd96981450a731acd88c48`.
Device CSV SHA-256: `376f41bd8789c678b7db3022d1072a7f4e0ad73e2dadb9514ed650e73af5c717`.
This is device attribution, not a new PP/TG result or held-out quality certification.

## Original serial-attention measurements

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
