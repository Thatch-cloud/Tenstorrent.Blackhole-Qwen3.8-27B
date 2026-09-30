# Two-slot normalized Q/K reader rings

**Combined correctness passes; the performance screen fails. Not promoted.**
Serving and the winning recipe are unchanged. Do not retry this candidate unchanged.

## Combined hardware result

[Run 35251919204](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35251919204)
at `9ac4461349dd6852b081491d145e196dfa7dff82` completes in **6m25s**.
Independent report validation passes both native-reference audits and all four
timed requests, including exact tokens, target state, inactive slots, feature and
proposal checks. Each arm commits 242 tokens with 224/300 proposals accepted.

| CTX / streams | Runtime | PP tok/s | Complete-cycle TG tok/s |
| --- | --- | ---: | ---: |
| 4,096 / 1 | Unchanged T16 | 3,331.52 | 122.26 |
| 4,096 / 1 | Double-buffered Q/K | 3,349.04 | 124.35 |

Aggregate TG rises **1.71%**, but paired changes are **+2.05% / +1.36%**,
below the required >2% in both pairs. More importantly, the changed verifier
gets slower in both pairs. Mean T16 block timings:

| ABBA request | Draft ms | Verify/readback ms | Commit ms | Whole cycle ms |
| --- | ---: | ---: | ---: | ---: |
| Control A | 27.142 | 66.514 | 5.126 | 99.711 |
| Candidate B | 24.845 | 67.016 | 4.966 | 97.697 |
| Candidate B | 24.519 | 66.913 | 4.715 | 96.824 |
| Control A | 25.817 | 66.522 | 4.961 | 98.145 |

The verifier penalty averages **0.446 ms**. Faster drafting accompanies the small
whole-cycle gain; these observations do not establish that reader buffering
improved the intended bottleneck. Extra L1 is not justified by this result.
Setup-inclusive latency also rises from 5,921.22 to 6,119.11 ms.

Exit is zero, no OOM, and all 864 script, 1,520 native and five adapter source
fingerprints remain unchanged. Report SHA-256:
`35700e0be291ac19a1b6b69a340d470658462400023cb0c98bc935c55e2d90a1`.
Held-out coding quality and 200 TG remain unqualified. Further work must address
larger full-cycle costs rather than infer a speed win from dependency clocks.

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
