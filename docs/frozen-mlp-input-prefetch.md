# Fused MLP activation prefetch

Correctness passes, but matched hardware performance regresses. **Not promoted.**

## Hardware decision

Run **35090533393**, revision `775fba4`, finishes in **13m56s**. Both arms retain
incremental publication and norm prefetch. The full-K activation candidate is
slower in both timed repetitions; keep the original streaming input reader.

| One coding stream | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: |
| Existing incremental/norm runtime | 2944.77 | 32768 | **88.95** |
| Full-K activation staging | 2964.77 | 32768 | 86.50 |

Candidate TG regresses **2.75%**. Its repeats are 87.06/85.95 versus control
88.89/89.01. Verification/readback grows 72.87 -> 73.66 ms/block; drafting
51.37 -> 53.01 ms, selection/commit 6.32 -> 6.95 ms, total 131.48 -> 135.19 ms.
These timings do not establish which internal wait or allocation causes the loss.
Do not increase buffers again without evidence of the limiting stage.

All six requests preserve exact output, target state and inactive state, with
identical output hashes. Both arms commit 234 timed tokens and accept 216/300
proposals. Candidate admission and all 64 MLP layer hit counters are verified.
All 852 reported source entries and 1,520 native entries match before/after;
candidate subdirectory files are separately checked by the admission gate and
per-layer kernel manifests, not covered by that top-level source count.
The container exits zero without OOM. Pre-load full I/O stall is 0.204%.
Serving is unchanged; no wider coding-quality claim is made.

Hardware report SHA256:
`ed334963ed57edc6b0d1ef21131fdca820995829b611d3c5b46498d83178a999`.

## Simulator qualification

Run **35089468990**, revision `7e13b88`, passes in **3m59s**, with wrapper exit
zero. Independent report checks confirm both-chip eager equality, all twelve
changed-input replay comparisons, four stale-input negative controls and four
43,520-page packed-weight comparisons. Kernel metadata differs from the retained
T16 control only in input-reader hash and explicit activation-prefetch metadata;
compute and weight-reader hashes remain identical. This does not establish speed.

| Retained artifact | SHA256 |
| --- | --- |
| Simulator report | `72ffca8dcf0af26d8f9743410142267aace4f065cb91c67c69ed868fb5efb5a3` |
| Projection source | `ac50a904fa4457bedce0504a694df2e43db781723fe96c6382005576cd1e518a` |
| Input reader | `41448b6c257f3b406fc38434032bab6ed5cb891fc984d27a61797f6f021022d4` |

The current fused gate/up reader repeats a receiver-ready handshake, eight-tile
activation read and multicast completion handshake for each of 20 K blocks.
The candidate stages all 160 activation tiles once, coordinates receivers once,
and sends the same twenty 16 KiB multicast chunks before signaling completion.
Compute consumes the same twenty eight-tile groups in the same order.

| Change | Candidate |
| --- | --- |
| Input CB0 | 16 -> 160 BF16 tiles |
| Additional L1 per multicast core | 288 KiB |
| Declared CB total per three-pair compute worker | 416 KiB |
| Weight reader / buffers / compressed weights | Unchanged |
| Compute and rounded epilogue | Unchanged |
| Multicast packet length | Unchanged, 16 KiB |

This is different from the rejected four-block input/weight buffer trial:
it changes activation coordination, not weight buffering. It may regress by
delaying compute until the whole input arrives. Declared CB usage excludes code,
runtime allocations and other L1 reservations; only the device can admit it.

Host tests cover exact input-page and multicast-byte coverage, preserved packet
size/order, and the projection source's capacity-only change. The simulator
matrix above retains eager equality, changed-input replays, stale-input controls
and complete packed-weight integrity without changing tolerances. The matched
complete-runtime comparison above retains incremental publication and norm prefetch
in both arms. A local source test is not a device qualification or speed claim.
