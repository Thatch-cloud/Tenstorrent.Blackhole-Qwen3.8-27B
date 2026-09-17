# Two-slot normalized Q/K reader rings

**Unqualified candidate. Serving and the winning recipe are unchanged.**

The combined dependency diagnostic measures roughly 978 query-wait cycles and
1,666 key-wait cycles, versus 77 for previous-token state. Moving reads earlier
failed to improve TG. This candidate instead allows the next normalized Q/K rows
to be prepared before compute releases the current rows.

| Change | Scope |
| --- | --- |
| CB10 and CB11 | Four to eight FP32 pages each; two complete row slots |
| Extra L1 | 32 KiB per recurrence worker, not per card |
| Initialization | Zero both slots on their first use, including each replay |
| Unchanged | Input order, cached source tiles, math, state writes, norm gate |

The first two tokens initialize distinct slots. Later tokens overwrite only the
active row, retaining zero padding. The compute waits/pops still consume four
pages per token; changing the ring capacity must not change numerical results.
Larger rings may fail L1 admission or reduce combined-runtime performance;
neither local source tests nor simulator timing proves a speedup.

Five host tests check buffer isolation, both-slot initialization, exact source
scope, failure cleanup and retention of all replay comparisons. Next gate is the
existing synthetic eager plus three changed-input replays: 24 exact output/state/
bridge comparisons and 48 immutable-input checks. Only after that passes may
this enter a matched full-request hardware comparison with the winning T16
recipe. No throughput is published from the simulator.
