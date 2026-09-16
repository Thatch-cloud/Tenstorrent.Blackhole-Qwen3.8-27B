# Fused MLP activation prefetch

Unqualified candidate; not selected by simulation, hardware or serving yet.

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
size/order, and the projection source's capacity-only change. Next use the
existing full-width T16 fused-MLP simulator matrix: eager equality, changed-input
replays, stale-input controls and complete packed-weight integrity. No tolerance
changes or correctness-check reductions. Only after that gate may a matched
complete-runtime comparison retain incremental publication and norm prefetch
in both arms. A local source test is not a device qualification or speed claim.
