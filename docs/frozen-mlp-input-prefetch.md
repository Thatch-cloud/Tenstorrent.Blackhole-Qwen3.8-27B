# Fused MLP activation prefetch

Simulator-qualified candidate; not selected by hardware or serving yet.

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
and complete packed-weight integrity without changing tolerances. Next prepare a matched
complete-runtime comparison retain incremental publication and norm prefetch
in both arms. A local source test is not a device qualification or speed claim.
