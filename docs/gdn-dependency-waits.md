# Distinguish GDN input waits from state feedback

**Simulator and combined hardware attribution pass. No throughput improvement is claimed.**

## Combined hardware result

[Run 35249838905](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35249838905)
at `c8845ba6464bf4abaa5ecf3271c0cbfbf8a7b951` passes in about four minutes.
Independent report validation passes all 4,032 samples across 48 layers, two
verifier replays and both cards, with native-reference output, state, inactive-slot
and feature audits. Runtime fingerprints remain unchanged and cleanup completes.
Report SHA-256: `14894e276067fddae13fdd34ad415e99b54df2d3dc1d007afe07bdb36823cad6`.

Median UNPACK wait intervals, in cycles:

| Dependency | Card 0 | Card 1 |
| --- | ---: | ---: |
| Query | 978 | 978 |
| Key | 1,666 | 1,666 |
| Value | 867.5 | 863.5 |
| Gate | 1,371 | 1,357.5 |
| Beta | 54 | 54 |
| Previous-token state | 77 | 77 |
| Ones | 47 | 47 |

This rules against previous-token state feedback as the dominant **sampled
input-wait** dependency. It does not prove DRAM bandwidth saturation: query/key
gather from local cached tiles, and each interval includes synchronization and
instrumentation. Sequential waits cannot reveal independent producer-ready times.
MATH/PACK wait medians are only 14–21.5 cycles; do not sum processors or convert
these diagnostic clocks into TG.

Next reader candidate should reduce actual local gather/publication work or
overlap it using qualified buffering, not repeat the rejected read-order swap.
Any candidate still needs changed-input simulator replay followed by a matched
complete-runtime comparison. This GDN scope alone is insufficient for 200 TG:
the earlier full verifier profile places the entire recurrence group near 9.53 ms,
whereas the current approximately 97 ms cycle needs to reach approximately
60.5 ms at 12.1 committed tokens per block. Projection/MLP and drafting remain
part of the combined performance work; prefix-cache PP is a separate measurement.

## Simulator acceptance

[Run 35249071221](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35249071221)
at `6fc6563f2cacecd23493756538b61f8dcfb9066e` passes in **3m4s**. All 24
output/state/bridge comparisons and 48 immutable-input checks pass exactly,
with eager and three changed-input replays. All 168 wait samples validate;
unexecuted poisoned pages are rejected before and after replay. Exit and
cleanup are zero. Native-to-generated source reconstruction matches locally.

Report SHA-256:
`d1a9f1466adf0a764c4a655f94e699bca2939a54c17d5238b942d855f1c64628`.
Instrumented compute SHA-256:
`ce7a2541323de8e89dbd07e5e47b924d3d771bd7024a730a5a1abec0a52f9d95`.
Simulator cycles are not used as hardware attribution.

The [combined phase profile](gdn-recurrence-phase-diagnostic.md) recorded about
5,000 cycles in the aggregate UNPACK input-wait interval. Moving V/beta/gate
reads ahead of Q/K did not reduce complete verifier time. That interval also
contains the wait for rounded local state from the previous token; it cannot
be labelled DRAM latency from the existing evidence.

The refinement times the existing waits individually, in their original order:

| Wait | Producer / dependency |
| --- | --- |
| Q and K | Normalized vectors gathered by the reader |
| V, gate and beta | Reader input tiles |
| State feedback | Initial state for token zero; previous token's rounded state thereafter |
| Ones | Persistent broadcast constant |

No waits are removed or reordered. Arithmetic, readers, writers, CB geometry
and the winning norm-prefetch path stay unchanged. The sample remains token 8
on one worker per chip, with separate UNPACK/MATH/PACK pages. An interval
measures blocking **when that wait is reached**, not each producer's independent
ready time. Processor intervals overlap and must not be summed as active work.

Use the same gates as the earlier phase diagnostic: complete synthetic
eager/changed-input replay and poisoned-sample checks, then one complete
native-audited combined hardware request. Do not publish diagnostic cycles as
TG. Choose the next change only after distinguishing the actual blocking
dependency; do not retry the rejected reader reordering unchanged.

The combined adapter uses 48 preallocated layer-owned sample buffers and the
first two T16 replays of one complete native-audited 4K request. It retains the
winning reader order, norm prefetch, MLP, attention, drafting and publication.
Thirteen combined host checks pass, including source removal, failed-trace
ownership, full layer coverage and rejection of throughput claims. The original
component tests also pass. Fresh staging checks the retained simulator evidence
against every selected Python dependency; actual native APIs are rechecked in
the pinned hardware image.
