# Two-block bulk weight-read overlap

**Simulator and first combined hardware correctness checks passed; speedup remains unconfirmed.**

## First combined hardware result

Run **35315198247, attempt 1**, source `959309a`, tested the same T16 DFlash
recipe at **4,096 context tokens, one stream, two cards and four fabric links**.
Both arms shared the loaded model and packed weights. Target tokens, state,
features, proposals and acceptance matched; scope restoration and cleanup passed.

| Metric | Serial bulk reader | Two-block pipeline |
|---|---:|---:|
| PP (tokens/s) | 3,327.60 | 3,363.91 |
| Committed TG (tokens/s) | 111.82 | 118.70 |
| Verification/readback (ms/block) | 60.31 | 60.21 |
| Drafting (ms/block) | 22.26 | 20.38 |
| Select/commit (ms/block) | 12.32 | 10.57 |
| Complete cycle (ms/block) | 98.25 | 92.55 |

Each arm committed 242 timed tokens across 22 blocks, averaging 11 tokens/block.
The apparent **6.15% TG gain is not yet attributable to the kernel**: verification
improved by only 0.10 ms, while most savings came from drafting and host-side work.
An unchanged ABBA repeat is running as attempt 2. No promotion or serving change.
At this acceptance rate, 200 TG requires a complete cycle of 55 ms; even the
current verifier alone exceeds that budget.

Attempt-1 report SHA256:
`3edb1940f2d27fbb64de86a99012144e61f2cab7d8379862654abecec0a75bfe`.

## Simulator qualification

Run **35314071196** passed in **4m29s**: two exact eager outputs, 12 exact
changed-input comparisons, four stale-input controls, all four packed-weight
comparisons and unchanged stream hashes. Simulator exit and container cleanup
are zero. The executed reader reconstructs from the current source; the gate
also requires the unchanged serial-reader and register-arithmetic admissions.

Report SHA256:
`1bb9581a17de463bd7614ddf602a44c0db444776bdc58b9f0609fb8177a4b411`.
Reader SHA256:
`b197b129252cb81ff2e3afabf7e014e557b0ab5cba834c7b0be746dc2cd0279b`.

The combined adapter shares the same loaded model and 64-layer packed-weight
pool in both arms. Two correctness audits precede timed serial/pipeline/pipeline/
serial requests. It rejects changed proposals or acceptance as well as target
token/state/feature differences. The pipelined reader is installed only inside
the candidate request scope and must restore afterward.

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
