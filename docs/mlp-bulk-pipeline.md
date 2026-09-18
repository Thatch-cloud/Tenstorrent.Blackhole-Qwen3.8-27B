# Two-block bulk weight-read overlap

**Prepared for simulation; no performance or hardware qualification.**

The matched combined DFlash T16 result (35312812439) spends 60.27 ms per block
in verification. At 11 committed tokens per block, the entire request cycle
must fit 55 ms for 200 TG. This experiment addresses operand delivery in the
target MLP, not drafter width or lower precision.

## What is different

The rejected earlier pipeline issued individual 576-byte weight-tile reads.
The current combined recipe instead stores each worker's K block as a contiguous
27,648-byte page. Its reader still waits for one page before issuing the next.
The new candidate keeps two pages in flight, using separate transaction IDs and
publishing each only after its own reads finish.

- Same existing two-block circular buffer; no deeper buffering or extra memory.
- Same BF4 bytes, register epilogue, BF16 rounding, output writes and compute source.
- Each bulk page is split at the pinned runtime's `NOC_MAX_BURST_SIZE`.
- Slot reuse waits for consumer credit; the reader drains transactions and resets
  the transaction ID before returning to output handling.

The pinned runtime documents the transaction-ID state and barrier APIs in
[dataflow_api.h](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tt_metal/hw/inc/api/dataflow/dataflow_api.h).
CPU checks cannot establish device ordering or a speedup.

## Acceptance

1. Existing full T16 MLP simulator replay: exact eager and changed-input outputs,
   stale-input controls, raw packed weights and stream integrity on both chips.
2. Bind the new reader and helper hashes to the retained report; do not reuse the
   serial-reader qualification for the changed kernel.
3. Matched combined T16 ABBA on the same loaded model, with identical proposals,
   target tokens/state/features and component coverage. No global Tracy profiler.

Only the combined result can qualify a speedup. This change alone is not assumed
to close the remaining 200-TG gap. Serving and the measured control are unchanged.
