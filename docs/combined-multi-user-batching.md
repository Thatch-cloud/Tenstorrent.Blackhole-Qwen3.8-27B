# Combined-runtime multi-user batching

Status: planned after the DMA publication comparison. No multi-user runtime or
capacity qualification is claimed. The single-stream 200 committed-TG goal remains
separate; aggregate throughput cannot satisfy it. Serving defaults stay unchanged.

## Starting point

The T16 combined DFlash path batches speculative positions within **one request**.
`model_batch.py` takes one token sequence and start position; `verifier_engine.py`
owns one session and saves/restores its helper state. `full_dflash_request.py`
creates one drafter and proposal capture after fresh prefill. These interfaces
cannot become concurrent simply by increasing a batch flag or adding threads.

The first implementation component is the unselected `gdn_slot_copy.py/.cpp`:
explicit slot 0-7 snapshot save/restore, with whole recurrent-state pages and
face-row-only convolution writes. The existing `gdn_state_copy` remains unchanged.
CPU tests exercise every slot and preserve other convolution rows and padding;
they do not execute the kernel. Weight-free two-chip eager/replay qualification is
required next, before binding this transport into a two-user request harness.

`DraftKVHistory` has request-local position, active/spare banks and pending
publication. Its current tensor shapes assume one request. Combined scopes also
temporarily patch shared methods: overlapping independent scopes are unsafe.

## Implementation order

| Stage | Change | Required evidence |
|---|---|---|
| Inventory | Measure free/allocated memory after weights and trace capture; separate shared from request-local buffers | Per-card bytes, allocation peaks, fixed recipe and source |
| Two resident users | Share immutable weights; separate target KV pages, GDN/convolution state, draft banks, positions, checkpoints and trace buffers | Different prompts; interleave A/B/A; compare each output and state with independent single-user controls |
| Failure and reuse | Reject drafts, stop one user, release and reuse its slot while the other remains live | Other user's state unchanged; no stale pages, rollback or trace addresses |
| True batched execution | Batch compatible projection/MLP work across users; retain per-user attention masks, recurrence and accepted-prefix commits | Simulator-first changed kernels; combined two-user hardware correctness and throughput |
| Capacity sweep | Test 1/2/4/8 users, then contexts within measured memory | Per-user latency plus aggregate throughput, OOM headroom and fairness |

Two resident users executed serially are an **isolation test**, not proof of
batched speedup. Flattening two users into one 32-row sequence is invalid: it would
mix causal attention and recurrent state. One scheduler must own shared runtime
scopes and device execution; each trace must have explicit request bindings.

## Memory sizing

The inspected hardware report from run 35318783547 uses target BFLOAT8_B KV:
16 attention layers, two local KV heads per card, head dimension 256, K and V.
With 1088 bytes per 1024-element tile, target KV costs **17 KiB/token/card**.

| Context per user | Target KV per card |
|---:|---:|
| 4,096 | 68 MiB |
| 8,192 | 136 MiB |
| 16,384 | 272 MiB |
| 32,768 | 544 MiB |
| 65,536 | 1.0625 GiB |
| 131,072 | 2.125 GiB |
| 262,144 | 4.25 GiB |

These are calculated KV payload capacities, **not measured total request memory**
or proof that each window works. Add output headroom, page rounding, GDN state,
drafter banks, proposal/verification scratch and trace allocations. The current
short-context DFlash experiment reserves 1032 pages of 64 tokens per card, about
1.071 GiB of target KV, irrespective of its 4K live context. Avoid multiplying
that reservation per user when a correctly owned shared page pool can suffice.

## Measurement contract

Keep the current single-user control. Start with two distinct 4K coding prompts,
prefix caching off, unchanged precision and a fixed output budget. Report:

- PP and time-to-first-token per user, including queue delay separately.
- Committed TG per user and aggregate TG over one shared wall-clock interval.
- Median/p95 inter-token latency and maximum service gap per user.
- Accepted/proposed counts, draft/verify/publication time and peak memory per card.
- Exact committed outputs and state isolation; held-out coding quality remains a
  separate requirement from fixture correctness.

Do not add individual request rates to claim aggregate throughput, include
discarded draft tokens as output, or hide prefill stalls in a decode-only metric.
