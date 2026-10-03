# One SDPA launch for every user of a block (QWEN_FAST_TP4_SDPA=multi)

Branch `tp4/sdpa-multi` (from `tp4/262k8`, merged with `tp4/sdpa-long`). Default off. Gate-only until the audited window passes.

## What it changes

At four cards the packed verify makes one K64j SDPA decode launch per user per attention layer: geometry G8B2, flags `0x23`
(tail 0x1, KV share 0x2, extent 0x20), two entries of 8 tokens x 6 heads (48 rows, 2 row tiles), 16 cores per entry, 32 of 110 cores
active. Four users are four launches, one after another, each reading its own KV at about 150 GB/s.

`multi` makes ONE launch per layer for the whole 64-row block:

| | served | multi |
|---|---|---|
| query | (1, 2, 48, 256) per user | (1, U, 96, 256): one entry per user, 16 tokens x 6 heads, token-major |
| flags | 0x23 | 0x21 (tail, extent; no KV share: every entry reads its own KV once) |
| page table | the user's lent (2, W) table | one (U, W) table, row u = user u's row |
| cur_pos | the user's lent (2,) word (E - 1) | one (U,) tensor, word u = user u's E - 1 |
| mask | (2, 1, 48, 256) per user | (U, 1, 96, 256), row r of entry u masks the keys past start_u + r / 6 |
| cores | 16 per entry, 2 entries (32 active) | 16 per entry, U entries (16 x U active) |

Eight seats are two blocks of four users (`QWEN_FAST_M3_BLOCKS=2`), so each block is one launch of U = 4. The pinned block never holds
more than four users (`Packed replay segments must fill at most a 64-row block`) and only T16 users (the pinned reader qualifies one
bundle of two 8-row groups and nothing else), so the multi path sees one to four T16 users.

## Exactness (byte-identical to the per-user launches)

The accumulation order of the online softmax is fixed by three things: the chunk boundaries (256 keys), the per-core chunk ranges and
the reduction tree (cores per entry), and the per-op arithmetic (fidelity, exp mode, dest precision). `multi` keeps all three, and
rows never interact, so regrouping rows from two 48-row entries into one 96-row entry keeps every row's instruction sequence.

Factory fixture `optimisation/ttnn-op/k64j/fixtures/sdpa_decode_program_factory.3e0a69af.cpp` (the K64j edits do not touch these lines);
`rt_args_common` is `optimisation/ttnn-op/k64j_probe/fixtures/rt_args_common.1b52c60d.hpp`:

1. **Cores per entry stay 16.** `:117` B = the page table's batch (U). `:196` `max_cores_per_head` = the config's
   `max_cores_per_head_batch`, default 16. `:198` `max_num_cores_for_compute = max_cores_per_head * B * num_kv_heads`; `:199`
   `num_cores_per_batch_uncapped = min(num_cores_available, max_num_cores_for_compute) / B`; `:200` `num_cores_per_head = max(1,
   uncapped / num_kv_heads)`; `:206` `num_cores_per_batch`; `:209` `num_active_cores`. At the served 110-core grid and one KV head
   that is `min(110, 16 U) / U`:

   | users | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
   |---|---|---|---|---|---|---|---|---|
   | cores per entry | 16 | 16 | 16 | 16 | 16 | 16 | 15 | 13 |

   `sdpa_multi_tp.plan()` computes exactly this and REFUSES any user count where it is not 16 (7 and 8 here; a block never has them).
   Tree rounds `:240` = `32 - clz(15)` = 4 and the compact scratch gives `scratch_slots=4`, the number the factory line prints.
2. **Same partition and tree.** `get_workload_for_core` (`rt_args_common.1b52c60d.hpp:35-93`) depends on (cur_pos, the core's index in
   its entry, cores per entry, the 256-key chunk) only; the tree runs over the core index within an entry. Every user has the same
   inputs to both as in the per-user launch.
3. **Same per-row op sequence.** The QK and output subblocks are `min(PNHt, 8 / 8) = 1` high (`:398`, `:407`) and the granularity define
   is `MUL_BCAST_GRANULARITY = min(PNHt * 8, 8) = 8` (`:785`): none depends on PNHt at 2 (served) or 3 (multi). Compute config, chunk size
   (256) and `exp_approx_mode=False` are the served ones.
4. **Same data.** The fold-in and fold-out are the V3a fold kernel's row copies (`attention_block_fold_tp.cpp`, unchanged; its staging
   area grows with `rows`, so 16 rows fit); the mask is the pinned mask kernel's tile body with the position `word + head / 6` for the
   same token the served entry computed as `word + b * 8 + (head % 48) / 6`; the page-table gather and cur_pos assembly are page copies.

Tests (`test_sdpa_multi_tp.py`) hold each of these: the factory lines are read out of the fixtures, the kernels' loops are emulated
from the runtime arguments the planners write and compared with the served composition, and the mask is compared bit for bit with the
pinned `narrow_mask_host` for every start word class.

The cross-geometry equality (a G16 row against the G8B2 row of the same token) has never run on a card at TP4; the audit proves it.

## The new pieces

* `scripts/ci/sdpa_multi_tp.py`: the planners, the launch builders, `MultiBlock` (persistent stacked table, cur_pos and mask allocated at
  attach, before any capture), the audit. Imported lazily by `sdpa_long_tp.apply` when the value is `multi`.
* `extent_attention_fold_tp.PackedExtentReplayReader` (the unpinned twin): `multi` is None unless the flag is on; `__call__` hands the call
  to `multi.call` or runs `served_call` (the previous body, unchanged); `shared_masks` returns the pinned context manager itself when off.
  The pinned `extent_attention_replay_tp.py` is not edited (its sha256 is held by the four-card evidence).
* Three small JIT generic-op kernels (launched from Python, compiled by the runtime, no op build):
  * `sdpa_multi_mask_tp.cpp`: the (U, 1, 96, 256) narrow tail mask from each user's own positions word (the runtime arguments carry the
    users' positions-buffer addresses; a core walks a task list).
  * `sdpa_multi_gather_tp.cpp`: copies page 0 of each user's lent table into page u of the stacked table and word 0 of each user's lent
    cur_pos into word u of the stacked cur_pos, once per forward, in the trace. This is what lets the pool stay untouched: the lent
    per-bundle storage is still what `packed_values` writes every round.
  * `sdpa_multi_audit_tp.cpp`: the compare (below).
* The refresh runs where the pinned mask refresh ran (once per forward under `shared_masks`, else per call) and books
  `refresh_calls` the way `model_batch` checks (one per bundle per forward).

## The audit (QWEN_FAST_TP4_SDPA_AUDIT=1, gate profiles only)

Every layer runs the served per-user launches AND the multi launch on the same query, tables and cur_pos in one trace. A device kernel
compares the two block outputs word for word (32-bit words: -0 against +0 or any one bit counts), 64 cores each writing one counter page
(`differing`, `live` words of the served output, `compared`) into a persistent counter tensor, one slot per layer. After each replay
`packed_verifier` calls `replay_reader.sdpa_audit_round`, which reads the counters (never inside the replay) and logs
`[PINDIAG] tp4 sdpa audit <n> exact=True users=4 layers=16 chips=4 words=<w> live=<l>`, or `... audit MISMATCH ...` and raises. Two
all-zero outputs also fail: nothing was compared. The multi output is the one returned, so the audited run's text is the multi path's.

The smoke (`c2_smoke_check.sdpa_multi_problems`) requires, for `QWEN_FAST_TP4_SDPA=multi`: the engaged line with `config=multi` and
`flags=0x21`, the first-call line, and under the audit flag at least one passing audit line and none that is a mismatch or `exact=False`.
`QWEN_FAST_EXTENT_AUDIT=1` is refused without the multi audit (the extent audit reads the per-user masks, which only the audit keeps
refreshed).

## Profiles and window

* `c2-packed-tp4-8x262k-best-sdpamulti` (timed) = `c2-packed-tp4-8x262k-best-time-gate` plus exactly `QWEN_FAST_TP4_SDPA=multi`.
* `c2-packed-tp4-8x262k-best-sdpamulti-audit` = the timed twin plus exactly `QWEN_FAST_TP4_SDPA_AUDIT=1`; never a timed arm.
* Job pack `scripts/ci/references/tp4-sdpa-multi-jobs`, image `tp4-sdpa-multi-1`: X0 (status rescan reset), B0 (build), A1 (audited
  attach, stop), T1-T4 (ABAB at eight users, 32k and 128k prompts, control first), Z.

## Risks the card must answer

* The multi launch's CBs are about 1.05 MB per core on all 110 cores (3 row tiles); an in-trace L1 clash fails the capture loudly.
* One dominant long user among short ones: its layer gets up to 1.5x slower (3 row tiles on 16 cores instead of 2 + 2 on 32). The
  mixed-context sweep (`sdpa_tp4_long`) measures it; the fix is the factory share-groups work (design K4), not this lane.
* The three kernels have never run on a card; A1 builds them at attach (one eager refresh), so a kernel that does not compile fails the
  attach and not the first capture.
* `buffer_aligned_page_size` is the runtime call the gather and audit kernels' page strides come from; a runtime without it fails the
  attach by name.
