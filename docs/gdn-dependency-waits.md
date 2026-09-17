# Distinguish GDN input waits from state feedback

**Simulator qualification passes; combined hardware attribution is pending.**

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
