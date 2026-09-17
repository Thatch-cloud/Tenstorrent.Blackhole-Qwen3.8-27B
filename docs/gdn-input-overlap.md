# GDN input overlap

**Simulator correctness passes. Combined throughput is not yet measured.**

The [combined phase diagnostic](gdn-recurrence-phase-diagnostic.md) found a
roughly 5,000-cycle UNPACK input wait at token 8. The candidate moves V, beta
and gate reads ahead of normalized Q/K gathering, so those reads can start
while the previous token still holds its Q buffer. It changes no arithmetic,
state snapshots, precision, writers or buffer sizes. This is not the earlier
input-cache experiment and adds **zero CB bytes**.

## Simulator result

[Run 35246637225](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35246637225)
at `3e8f47f94a501991dc70f0c86f171704e9f7ac71` passes in **3m8s**. Both chips
pass 24 exact output/state/bridge comparisons and 48 immutable-input checks,
covering eager and three changed-input replays. Exit and cleanup are zero;
generated source is reconstructed locally and matches the artifact.

| Source-bound evidence | SHA-256 |
| --- | --- |
| Report | `2dcd515b0236ebf35143406ee50b5c46a0567e617325324bf4307fdf84aac05b` |
| Control reader | `3a670f7e24dd53b8edcbaaac2934a9fc95cee1b46a9af8ccb5d0e90ad2f4df22` |
| Reordered reader | `71b58bd93eab0b5e11dfbd5fe72b49684105a80caf83c8535df51cf4e16609b8` |

## Combined acceptance

The prepared hardware comparison keeps the winning 4K T16/DSpark recipe,
including norm prefetch and incremental publication. Two fresh native-reference
audits precede complete control/candidate/candidate/control timed requests.
Every candidate GDN build must match simulator admission and the norm-prefetch
build count. Prompt, output and proposal acceptance must match between arms.

No diagnostic clocks or Tracy are enabled. PP and committed TG use the existing
complete-request accounting. Both matched TG pairs must improve by more than
2% to pass the screening gate; a pass is not held-out coding-quality acceptance
or permission to change serving defaults.
