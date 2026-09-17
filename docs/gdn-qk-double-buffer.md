# Two-slot normalized Q/K reader rings

**Simulator-qualified candidate; hardware performance is unqualified.**
Serving and the winning recipe are unchanged.

## Simulator result

[Run 35250900553](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35250900553)
at `3ba2aadf9d9d4c57c747dc05373a06366256c4c6` passes in **3m4s**.
All 24 output/state/bridge comparisons and 48 immutable-input checks pass,
including eager and three changed-input replays. Both ring slots initialize
correctly in these fixtures. Exit and all cleanup statuses are zero.
Report SHA-256: `181173b7b6ab22e36f2068cce66255c621d5d30915ab7659ce9a5b6c755ea21c`.
Candidate reader SHA-256: `3adef6c2b0a46d5ee6ba5172ab279e00b0f37d4a8ffa6aae1b293e82f6c3d889`.
There is no hardware speed claim. Next is a matched complete-runtime comparison,
not another isolated performance test.

The combined adapter changes both the admitted reader and its two ring sizes,
and verifies one buffer construction per transformed recurrence. Norm prefetch
remains composed independently; both bindings restore between requests. The
hardware schedule is two fresh native-reference audits followed by control,
candidate, candidate, control timed requests. Acceptance, tokens and target
state must match, and TG includes the entire draft/verify/commit loop. Both
paired improvements must exceed 2% for the initial speed screen; that screen
alone is not held-out coding quality or 200 TG acceptance.

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
