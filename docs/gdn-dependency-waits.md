# Distinguish GDN input waits from state feedback

**Host tests pass; simulator and combined hardware qualification are pending.**

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
