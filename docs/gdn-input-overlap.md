# GDN input overlap

**Simulator and combined correctness pass. Performance screening fails; do not promote.**

## Combined result

[Run 35247597768](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35247597768)
at `956d2ebc01033e0b892edceb40a14aabbc8f2661` completes in **6m23s**. Both
fresh audits and all four timed requests pass exact token/state/inactive checks.
Each arm commits 242 timed tokens with identical 224/300 proposal acceptance.
The independent full-request validator passes; container exit is zero, no OOM
is reported, and source fingerprints are unchanged.

| CTX 4096, one stream | Control | Reordered reader |
| --- | ---: | ---: |
| Cold PP tok/s | 3,363.89 | 3,335.96 |
| Committed TG tok/s | **124.14** | **122.99** |
| Mean verifier/readback ms/block | 66.476 | 66.635 |
| Mean draft ms/block | 25.150 | 26.345 |
| Mean select/commit ms/block | 5.010 | 4.669 |
| Mean whole cycle ms/block | 97.431 | 98.343 |

TG changes by **-0.92%** overall; paired changes are **-2.67% / +0.90%**.
Verifier time is slightly worse in both matched pairs, not just hidden by draft
variance. Retain the original reader and do not rerun this candidate unchanged.
Report SHA-256:
`71e40e891c72a7d9806b6d8c217aa0e4a6fb418351812e9e7db8e3bb8858d271`.

The diagnostic's aggregated input-wait interval includes local state-feedback
and pipeline dependencies as well as reader readiness. This result contradicts
the hypothesis that simply moving the external reads earlier removes the
dominant wait. A subsequent GDN change needs individual wait attribution or a
larger arithmetic redesign, not another assumption that this interval is DRAM
latency. The complete GDN group is itself too small to close the whole 200 TG gap.

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
