# Sticky sessions on the four-card eight-seat packed path (tp4/packed-prefix, stage 1)

Base: `tp4/262k8`. This is stage 1 of the prefix-reuse design for the TP4 fast packed path: it switches the
machinery that already exists on, in gate-only profiles, and builds the audits, the harness and the job pack
that qualify it. It changes no serving code, so every serving path is byte-identical with the flags off (and
the flags are off in every traffic profile).

## Why every turn rebuilds the prefix today

None of the fast packed TP4 profiles switches reuse on. Every `c2-packed-tp4*`, `c2-packed-tp4-8*` and `8x262k*`
profile sets `no-enable-prefix-caching` and `no-enable-chunked-prefill`, and sets neither `QWEN_PREFIX_REUSE` nor
`QWEN_FAST_STICKY_SESSIONS`; the contract strips an inherited switch from a profile that does not own it. So
vLLM's prefix cache is off, every turn allocates fresh blocks and prefills its whole context again.

It is not just "turn on the KV cache": 48 of the 64 layers are GatedDeltaNet layers whose recurrent state vLLM
does not page. A KV hit at position Q is usable only together with the GDN state after exactly Q tokens, and the
only states byte-identical to what a cold prefill produces are the ones the prefill chunk program wrote at an
absolute 2048-token boundary. The end-of-turn state (written by the verify kernels, which sum in another order)
is not one of them. G1 (`general-prefix`) keeps those boundary checkpoints on the host and trims vLLM's hit to
them; sticky sessions (`QWEN_FAST_STICKY_SESSIONS=1`, merged into this base) grafts that onto the S2 fast path.
Until now it existed for the pair only (`c2-packed-prefix`) and never ran on hardware.

## What stage 1 adds

| Piece | What |
|---|---|
| Profiles | `c2-packed-tp4-8x262k-prefix-gate` (audited; the twin of `...-best`) and `c2-packed-tp4-8x262k-prefix-time-gate` (audits off; the twin of `...-best-time-gate`). Each is its parent plus exactly `QWEN_PREFIX_REUSE=1`, `QWEN_PREFIX_STORE_GIB=8`, `QWEN_FAST_STICKY_SESSIONS=1` and the argv's `enable-prefix-caching`, `enable-chunked-prefill`, `prefix-caching-hash-algo sha256` in place of the two `no-` flags. Gate-only, the 262k evidence waiver, every result UNQUALIFIED. The parents are the controls. |
| Harness | `prefix_replay.scenario_agent_turns` (the agent-turn replay), `SHARED_AGENT_TARGETS_8` (eight same-tenant agents across both packed blocks, chosen by `Driver.seats`), `CHAIN_HITS_262K` (the exactness chain on to 150k and 200k and the 262k prompt limit), plans `agent-turns`, `agent-turns-prefix`, `agent-turns-baseline` in `c2_prefix_gate.py` and `c2_serving_job.py`. |
| Per-turn logging | a `[TURN]` line per served turn: tokens reused (the model's own `[PREFIX]` row Q), tokens new, TTFT, decode tokens/s after the first token, the engine build. The server already logs `sticky admit` (Q, P, tail) and `sticky engine built` per request. |
| Per-arrival stall | a `[STALL]` line per turn that prefilled beside streaming seats: the other seats' longest gap between two streamed chunks while it prefilled (`prefix_agent_turns.stall_rows`, from the records' `chunk_times`), and a summary per arm in the comparator. |
| Smoke rules | engaged marker (`install sticky=1 lookahead=16 drop_last=True`, the model warm line), every continuation's reused tokens equal to the sticky oracle's Q (a miss with no eviction on record is a FAIL; a conversation that never reused is NOT_EXERCISED), continuations byte-identical to a cold full prefill (the bring-up's reference, `exactness-eager`'s strict pairs, and the agent-turn transcripts compared with the control's). |
| Comparator | `scripts/ci/prefix_agent_turns.py A B`: two arms' transcripts turn by turn (output token ids; one divergence is reported once, at its first turn, and later turns of that conversation have another prompt) and the paired continuation TTFT. |
| Audit | the TP4 section below and `test_tp4_packed_prefix_audit.py`. |
| Job pack | `scripts/ci/references/tp4-packed-prefix-jobs` (image `tp4-packed-prefix-1`), see its `ORDER.txt`. |

### Agent-turn replay

Eight busy coding agents (every seat), `--turns` turns each (eight by default), in the metering shape: first
contexts 8k to 40k tokens, about 2k-token tool results, exponential think gaps, 192-token answers, compaction
past 60k. The conversations are seeded by the phase and the agent, never by the arm, so the prefix arm and the
control send the same messages as long as the model answers the same: the comparison is exactness against a cold
prefill, end to end (the conversation feeds the model's own answers back, so one divergence propagates). The
control is the no-reuse parent profile; the ABAB jobs run one arm per job, each after its own all-four reset.

### Flags

No new serving flag. The existing ones, now set by the twins: `QWEN_PREFIX_REUSE`, `QWEN_FAST_STICKY_SESSIONS`,
`QWEN_PREFIX_STORE_GIB`; gate-only instruments (`QWEN_PREFIX_DIGESTS`, `QWEN_PREFIX_AUDIT`) are applied by the
gate to its arms. Kill switch: the `prefix-reuse.off` file under the image's `.qwen-c2` directory. Salt:
platform-minted `qps1.<tag>.<hmac>` only; an unsalted request neither hits nor publishes.

## Exactness (the claim the hardware gates test)

A turn resumes at Q, the largest 2048-multiple at or below the prompt minus 2048 whose GDN checkpoint and KV
prefix the grafts kept. From Q the same eager chunk program as a cold run prefills the rest from a restored,
byte-copied canonical state, so the last-position logits, the slot state, the KV of `[Q, P)` and the drafter
window features for `[P - 2048, P)` equal a cold prefill's. The drafter reads target features for the last 2048
positions only, and the trim keeps `Q <= floor2048(P - 2048)`, so the window is always produced by the tail's own
chunks: the drafter state after a hit equals a cold one's (acceptance after a hit equals acceptance after a cold
prefill; verify is greedy and lossless, so drafter state affects speed only, never tokens). No drafter state is
carried between turns. Nothing decode-produced is carried either: the end-of-turn GDN state, the answer's KV and
the prompt tail are not canonical, so a turn re-prefills the 2k-4k tokens since its boundary plus what is new.

## The TP4 writer audit (extends `sticky-sessions-writer-audit.md`)

The pair's audit shows every device writer lands in the request's own blocks at or above R, or outside vLLM's
KV pool. The four-card eight-seat levers add these writers; none of them writes the target K/V pool:

| Lever (flag) | What it writes | Where | Why it never writes a shared block |
|---|---|---|---|
| Fused commit, in place, live banks (`QWEN_FAST_FUSED_COMMIT`, `_INPLACE`, `_LIVE_BANKS`) | the drafter's K/V banks `(1, 2, 2048, 128)` per layer of its own slot | engine-owned device buffers | not in vLLM's pool |
| Draft K/V slide (`QWEN_FAST_TP_KV_SLIDE`) | the drafter's spare history bank | engine-owned | not in vLLM's pool |
| Quad drafts (`QWEN_FAST_QUAD_DRAFT`, `_BLOCKS=2`) | placeholder banks and draft outputs | engine-owned | not in vLLM's pool |
| GDN commit lanes (`QWEN_FAST_TP4_COMMIT_LANES`) | the GDN slot state records | the GDN slot buffers | not in vLLM's pool |
| Verify-glue levers, attention fold (`QWEN_FAST_TP4_GDN_GLUE`, `_GDN_BLOCK_CONV`, `_SHARD_VALUES`, `_ATTN_FOLD`) | query and result tensors, GDN rows | scratch | data movement; the SDPA reads the pool through the unchanged page tables |
| Extent reader modes (`QWEN_FAST_SDPA_MODES=tail,share`) | none: reads | the pool through the page tables | read-only |
| Verify K/V rows (ordered write, `packed_ordered_cache`) | rows `[frontier, frontier + 16)` of each live user, one chain per segment | the request's table | the frontier is at or above `P`, above `R`; the binding refresh refuses a table that does not cover every row; width 4,096 changes the table, not the rule (tested) |
| Idle segments of a padded round | page 0 | vLLM's null block | never in a request's table |
| Prefill route's chunks and tail | positions `[R, P)` | the request's table | every block is at or above `R / 64`, allocated fresh |

`test_tp4_packed_prefix_audit.py` holds the inventory (no lever module names a K/V pool write call, and a census read
from the syntax tree, not a name pattern, lists every device-write call site of the lever modules - a copy of any
kind, a host-to-device copy, a generic_op DMA kernel - against an audited table: a lever that grows a write fails
there and must be audited before it joins a sticky profile; a generic_op takes the tensors its caller built, so the
census fixes the site and E2's window digests are the check of which tensors a caller passes), page 0 as the idle
segment's page and refused in a live table, the attach-time page-table width check (below), a verify round at a 262k position
through a 4,096-wide table, and the guard across two packed blocks (every seat of both blocks crossing a 64-token
boundary together is not a conflict under sticky sessions; a real conflict inside a block still is). Across the
two blocks the guard runs per block and cross-block sharing is read-only. The hardware half is the
`exactness-shared` arm on eight agents (`E2`).

### The page-table width of a hit

A hit's prefill fits its page table to the model's chunk-input buffer (`_chunk_full_page_table_buf`) when one exists
and keeps the runner's width otherwise. The fast path never captures the chunked trace, so the buffer should not
exist and a hit replays the width the attach-time eager warm compiled (`runner.max_num_blocks_per_req`, 4,096 at 262k).
A buffer of another width would compile every hit at a new shape after the traces are parked (the second-request hang).
`serving_runtime.prefill_warm_before_traces` therefore refuses the attach, under `QWEN_PREFIX_REUSE=1` with sticky
sessions, when the buffer exists at any other width (nothing is checked with either switch off).

## The KV reservation with caching

The reservation rule is `r = ceil((P + max_tokens + 32) / 64) + 1` blocks over a request's life. A hit's shared
blocks are counted inside its own `r`, a shared block is one physical block, and cached-free blocks are evicted
on allocation, so sharing can only over-reserve. `test_tp4_packed_prefix_profiles.py` holds the arithmetic and
that the pair of profiles keeps the parent's pool and the rule's flag. The real-scheduler proof with prefix caching
on is `test_qwen_prefix_scheduler_vllm.StickyReservationOnRealVllmTests` (run by `qwen-fast-vllm-cpu.yml`): the reservation
wrapper installed over the plugin scheduler the graft is staged on, the graft's sticky sessions (DFlash lookahead 16, the
dropped block), a pool smaller than the traffic, conversations that continue (hits), two agents of one tenant sharing a
system block (siblings) and four arrivals at once; every step's preempted ids stay empty, no request holds more blocks
than it reserved, every block is free again at the end, holds and hits are both exercised, and a pool packed to two
requests preempts when the rule is off (the negative control). The cards' half is the lifecycle job `L1` (coupled eviction
under a flood, the kill switch, an abort during a hit's prefill, the restart): a re-admitted request is a FAIL there, and
the hold lines and hits restored are printed.

## What stage 1 does not cover (left)

- **Parking engines** (Stage E, `main`): every request still builds its engine after the prefill, 2.3 to 2.7 s
  at TP4, device-exclusive. Porting parked engines to eight slots on two blocks with the fused commit in place
  is the largest remaining term of a returning turn's first token.
- **The undrop** (`QWEN_PREFIX_STICKY_UNDROP`): vLLM drops a DFlash hit's last block, costing 2048 tokens of
  re-prefill a turn; neutralising it under sticky sessions is a separate stage.
- **The host-RAM warm tier** (spill on evict, restore on hit; a raw tile copy op) and the **cold NVMe tier**.
  Between turns a conversation holds only cached-free blocks and one host checkpoint (78 MB with the bf16 GDN state, `QWEN35_GDN_STATE_BF16=1`; twice that with fp32); when the pool
  overflows vLLM evicts the tail first and the next turn takes a partial hit at an older boundary.
- **The Lever N merge**: chunked prefill (`prefill/decode interleave`) and the sticky route both own the meaning
  of `start_pos > 0`; one route with three call kinds (cold-first, hit-first, continue) is the merge contract.
- **The host-warm job (PH2)** and the 8 x 131k ship-candidate family: the first belongs to the host tier above, which
  stage 1 does not build, and the second to a ship decision this branch does not make (the target is the 262k window).
- A traffic profile: the twins are gate-only; shipping needs the hardware ladder, the 262k evidence records
  (every result in the pack is UNQUALIFIED under the waiver), and the owner's decision on the node's session cap.
