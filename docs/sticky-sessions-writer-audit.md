# Sticky sessions: the device writer audit

Phase 1 of sticky sessions (`QWEN_FAST_STICKY_SESSIONS=1`, the `c2-packed-prefix` profiles) lets a new
request on the S2 fast path start from vLLM's cached KV prefix `[0, R)`. That prefix's blocks can be
shared. Another request of the same salted conversation may be reading them at the same time, and so
may a sibling that shares its system block.

A hit is exact only if nothing writes a shared block after it is published. This note lists every
device writer on the S2 path and where each gets its page indices. It then shows that each writer
lands in the request's own blocks at or above `R`, or outside vLLM's KV pool.
`scripts/ci/test_sticky_sessions.py` (`WriterAuditTests`, `KvGuardTests`) holds the checks that can
run on the CPU. The hardware gate adds a digest audit of the shared blocks across decode.

## The page tables

- **vLLM's allocation.** A request admitted at `num_computed_tokens == R` receives the cached blocks
  `[0, R / 64)` from the prefix cache. vLLM then allocates fresh blocks for `[R, P)` and for every
  decode block after that. The scheduler graft's cap publishes only whole blocks below `floor2048(P)`.
  vLLM keeps the first block cached under a given hash, so a later duplicate is never served.
- **The fast path's table.** The bridge factory (`serving_runtime.bridge_factory`) builds the engine's
  `(1, page_width)` table from the request's blocks and pads the rest with the request's **first**
  block. That block is finite, which the extent readers need past the allocation. Under sticky sessions
  the first block may be shared, but it is only ever read there:
  - the extent readers read the pad at masked positions;
  - the binding refresh (`serving_page_binding.VerifierPageBinding.refresh`) refuses a table that does
    not cover every scheduled verify row, so no row is written through the pad.

## The writers

| Writer | Where its pages come from | Why it never writes `[0, R)` or the pad |
|---|---|---|
| The prefill route's chunks and tail, `[R, P)` (`qwen_prefix_model_patch._qwen_prefix_prefill_slots` → the chunk loop from `R / 2048`) | The runner's page table row: block `pos // 64` for each position | Every position is at or above `R`, so every block index is at or above `R / 64`. vLLM allocated all of those blocks fresh. |
| The GDN checkpoint restore and the finished GDN state (`_qwen_prefix_restore`, `_write_gdn_slot`) | The B=1 prefill scratch and the GDN slot `empty_slots[0]` | Neither is in vLLM's KV pool. |
| Verify K/V: the packed block's rows, the K64j ordered cache (`ordered_cache.update`) and the per-request sequential engines | The engine's table as last refreshed; rows `[frontier, frontier + rows)` | The frontier is at or above `P`, which is above `R`. The refresh refuses a table that does not cover the rows, so every row maps to an allocated block and never to the pad. |
| Idle segments of a padded round (`PackedVerifierEngine.idle_inputs`) | An all-zero table: page 0 | Page 0 is vLLM's null block and never belongs to a request. The packed verifier also refuses page 0 anywhere in a live table (`packed_verifier.page_zero_index`). |
| The SDPA tree scratch (the decode factory's tree-reduction rounds) | An on-core circular buffer | Not a KV page. |
| The drafter's history (`kv_history`, the draft K/V and feature banks) | The engine's own device buffers and the buffer pool's slots | Not in vLLM's KV pool. |

## The proposal-time guard

`serving_packed_step.kv_shared_at_proposal` maps a round's 16 rows through the tables of the **last**
refresh, before vLLM has appended this round's block. A row past the bound blocks therefore maps to the
pad.

- **Prefix caching off:** the pad is the request's own first block, so it can never alias another user.
- **Sticky sessions:** two same-tenant requests may share their first block. If both cross a 64-token
  boundary in the same round, both would map to `(first block, tile row 0)`. That is a `KV_SHARED`
  conflict that no write makes, and it would send the round down the sequential path. So under the
  flag, `kv_guard` reads every entry past an engine's bound blocks as a sentinel page of that segment's
  own (`pad_sentinel_table`).

The pad is recognised without the binding. The bound blocks are unique, so the first entry after index
0 that equals entry 0 starts the pad. Two conflicts are still caught:
- a real one, where two tables name the same allocated block at a written row;
- the step-time check (`kv_shared`), which reads refreshed tables in which no row is past the bound.

With the flag off, `kv_guard` reads the tables exactly as before (`KvGuardTests` compares it with the
pre-sticky guard).

The four-card eight-seat levers (the fused commit in place on live banks, the GDN commit lanes, the draft K/V slide, the quad
drafts at two blocks, the verify-glue levers, the attention fold, 4,096-wide tables) are audited in
`docs/tp4-packed-prefix.md`; none of them writes the target K/V pool.
