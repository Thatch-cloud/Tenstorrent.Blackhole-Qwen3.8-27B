# TP4 exact GDN memory: one per-token state history shared by both 64-row blocks (QWEN_FAST_GDN_SHARED_HISTORY)

Status: built behind a default-off, gate-only flag; proven on CPU; the card gates are in
`scripts/ci/references/tp4-gdn4e-jobs/` and are not yet run.

## Design (written before the code)

### What is being saved

Every GDN layer's verify launch writes, per packed user, a 16-token history of recurrence states:
`(16, 12, 128, 128)` bf16 per chip, 6 MiB per user per layer. The only reader is the deferred commit, which
copies the row for the accepted prefix into that user's carry. At eight seats (two 64-row blocks of four users)
both blocks keep their histories resident for the process lifetime:

    16 x 12 x 128 x 128 x 2 B x 48 layers = 288 MiB per user per chip
    x 4 users x 2 blocks                   = 2.25 GiB per chip

### The idea

The two blocks never verify at the same time. If block A's commits are finished before block B's verify replays,
B can write the same buffers A wrote, and A can write them again after B's commits are finished. One history set
(4 users x 48 layers = 192 tensors) then serves both blocks and 1.125 GiB per chip (1,207,959,552 B) is freed.
Nothing about the arithmetic changes: the launch, the kernels, the traces' op counts and the commit programs are
the served ones. It is a schedule change plus a change of which buffer a capture's states output is, so outputs and
committed states are byte-identical by construction (and proven so on the CPU path below).

### Where the buffers come from

The served GDN launch is K5-A (`QWEN_FAST_GDN_SEQ_BLOCK=1`, baked into the serving image), whose `(16, 12, 128, 128)` states are allocated
inside the pinned `gdn_seq_block.execute`, not in `gdn_user_batch_tp.execute` (the batched launch K5-A replaces). Both are plumbed:

- under K5-A, `tp_addresses.bound_twins` binds `gdn_shared_history.seq_block_execute` in place of `gdn_seq_block.execute` (only with
  `QWEN_FAST_GDN_SHARED_HISTORY=1`; production binds what it always did). The twin calls the pinned execute, with its bytes untouched, with an
  `operations` proxy: inside an open pool capture the one `(16, 12, 128, 128)` bf16 DRAM `empty` comes from the pool, and `deallocate` skips
  pooled tensors (so even the pinned function's failure path cannot free one); everything else is the real ttnn. Outside a capture the twin is
  the pinned function;
- the batched launch takes from the pool only when K5-A is off (`active_batched`). Under K5-A its only caller is
  `QWEN_FAST_GDN_SEQ_BLOCK_AUDIT`'s served audit launch, which allocates privately and takes nothing from the pool.

A process-wide
`SharedHistory` (new module `gdn_shared_history.py`) is opened around each block's verify capture only (never the
warm forward, never a sequential engine's capture, never the audits' served launches):

- block A's capture allocates the 192 tensors through the pool, in launch order (layer, then user);
- block B's capture is handed the same tensors in the same order (a cursor, reset per capture), after the pool has
  checked each requested shape, dtype and count against what A built;
- the pool, not the blocks, owns the tensors: `tp_addresses.release_owned` skips tensors the pool holds, and the
  pool frees them when the last attached block closes.

B's retained records, commit programs and commit traces simply name the same buffers. Per-user conv-window
histories, carries, checkpoints and taps stay per block.

### The ordering rule: commit before reuse

A block's deferred commits must have read its history before the other block's verify overwrites it. Three
schedules exist:

1. Blocking or pipelined commits without the deferral: `run_verified_block` commits every user right after the readback, on the one command
   queue, before the next block is verified. Already ordered; the cost is 0. (Not what ships.)
2. Deferred commits (`QWEN_FAST_GDN_AFTER_PAIRS`, round-fence plan H2, with early draft, round fences and pipelined commits): THE SHIPPED
   SCHEDULE. The image bakes all four, so all eight GDN commits are held until after the pairs, past block B's verify. Sharing would let B
   overwrite A's history first.
3. Anything undecided: a block that has verified but not committed every segment.

Enforcement is one guard in `PackedVerifierEngine.verify` (`SharedHistory.claim`): before a block verifies, the
block that last wrote the history must have nothing pending. Deferred commits are flushed there (the same
`flush_commits` the engine already uses, site `shared-history`, logged and counted); undecided segments raise and
fail the round closed (logged with the refused marker first). The guard sits in the engine, not in the step, so every caller is covered.

The shipped schedule therefore changes: in every eight-seat round with both blocks live, block B's claim enqueues block A's four commit traces
(site `shared-history`) ahead of B's verify and ahead of the drafts, and B's own commits go at the window or end site as before. The flush is
inside the `execute_model` that decided the commits, so the round-fence rule R1 holds, and `shared-history` is an in-step flush site for
`early_draft.IN_STEP_SITES`, the Lever N gate's `GDN_IN_STEP_SITES` and the serving gate's `FLUSH_SITES`. The cost is A's four commit traces (about
2.0 to 2.2 ms of device time) moving ahead of the drafts: 0 to +2.2 ms per round by the research, UNMEASURED: the pack's paired timing arms
(T1-T4) measure it, with a kill rule of a median round more than 1.5 ms slower, in which case the sharing is memory-only and not for traffic.

### Memory accounting and the opt-in KV growth

The ledger walks tensors by buffer address and counts a buffer once, so block B's `packed_block` line drops by
exactly the shared bytes (1,207,959,552 B per chip). That frees 2,168 KV blocks of 557,056 B. The freed bytes are
never spent automatically. A separate flag, `QWEN_FAST_GDN_SHARED_HISTORY_KV_GROW`, only marks a profile as one
whose pool was sized with the freed bytes; `gdn_shared_history.pool_problem` checks that such a profile (a) also
turns the sharing on, (b) is gate-only, (c) names a pool no larger than the production edge plus the freed blocks
(64-aligned), and (d) keeps `num-gpu-blocks-override` and `QWEN36_MAX_TOKENS_ALL_USERS` in agreement. The gate
profile built from it is 22,144 blocks (production: 19,968), which holds five full 262k reservations
(5 x 4,097 = 20,485) with room.

### Refusals (fail closed, at attach)

The flag is refused (a `ValueError` naming it) unless serving is four cards and two M3 blocks, and the split-V twin
(`QWEN_FAST_GDN_SPLIT_V=2`, which would replace the K5-A launch the pool is plumbed into, and is unqualified) is off. K5-A itself is the
served launch and is not a refusal. The refusal is raised from inside the constructor's try, so a refused attach closes the block (returns the
replay tables and stops the deadline watchdog) like every other construction failure. Joining is refused for any block that is not the 4-user 16-row-per-user 64-row
shape, for a third block, and for a capture that does not request exactly 192 tensors of the first block's
shape. The grow flag without the sharing flag is refused. With the flag unset nothing is imported on the serving
path beyond a constant-time flag read and no code path changes.

### What is proved on CPU, and what is left for the card window

CPU: the commit-before-reuse ordering under all three schedules with the real `PackedVerifierEngine` and a
byte-level device model (shared and private histories produce identical carries; a negative control with the guard
cut shows the hazard is real), also under the environment the cards run (the image's baked ENV with the profile's over it);
the capture plumbing through the real K5-A launch (`gdn_user_batch_conv.run_user_batched_projected` with the twins installed as the card
process installs them: 192 takes per block capture, none for the warm forward or the audit's served launch, no pooled tensor ever freed);
ownership and release (by object and by buffer address); the ledger arithmetic through the real `MemoryLedger`; the refusal paths; the
grown-profile arithmetic and the production profiles unchanged. The ModelBatch layer above `run_user_batched_projected` is not driven here
(its own tests cover it). Card window: audited attach, exactness against the control, the ledger before and
after, the five-reservation admission test and the hang shapes (`scripts/ci/references/tp4-gdn4e-jobs/ORDER.txt`).

## What was built

| Piece | Where |
|---|---|
| The pool, the guard, the accounting, the refusals, the growth-marker check | `scripts/ci/gdn_shared_history.py` (overlay list) |
| The K5-A twin: the pinned execute called with a pool-backed `operations` proxy | `scripts/ci/gdn_shared_history.py` (`seq_block_execute`, `PooledOperations`), bound by `scripts/ci/tp_addresses.py` (`bound_twins`) |
| The batched launch takes its states from the pool inside a verify capture (K5-A off only) | `scripts/ci/gdn_user_batch_tp.py` (`execute`) |
| The pool's tensors are never freed by a block's `owned` release | `scripts/ci/tp_addresses.py` (`release_owned`, by object and by address) |
| Join at construction (inside the try), capture window, claim in `verify`, detach in `close`, marker, `describe` | `scripts/ci/packed_verifier.py` |
| The flush site counts as in-step; the smoke check's rule for the markers | `early_draft.py`, `lever_n_m3native_gate.py`, `c2_serving_gate.py`, `c2_smoke_check.py` (`shared_history_problems`) |
| Boot check of the growth marker | `scripts/ci/serving_c2_contract.py` (`shared_history_problem`) |
| Four gate-only profiles | `scripts/ci/qwen_c2_profiles.json`: `c2-packed-tp4-8x262k-ship-prefix-4e-control-audit`, `-4e-audit`, `-4e`, `-4e-grow` |
| Card-window pack | `scripts/ci/references/tp4-gdn4e-jobs/` |

The ledger attributes a block's buffers by walking the objects reachable from it. The pool is reachable from both blocks, so its
references to the blocks live in a slots-only holder the walker cannot enter; otherwise the first block's line would absorb the second's.
In a read of the P6 lines, the second block drops by exactly the shared bytes and the first is unchanged.

## What the CPU tests prove, and the limit of that

- `test_gdn_shared_history_schedule`: two real `PackedVerifierEngine` blocks built through `complete_blocks_two_phase`, a device model that
  keeps values, twelve random rounds of mixed accepted prefixes (0 to 16, including blocks that sit out). Private and shared histories give
  identical carries after every schedule (blocking, pipelined, deferred), equal to a pure-Python recurrence. The deferred schedule shows the
  commit order the lever needs (each block's commits run before the other block's verify) and, without the guard, wrong carries for block A's
  users only. Refusals: a verify while the other block has undecided segments, deferred commits that cannot be enqueued, a failed or closed
  writer, a flag the environment cannot honour. Ownership: nothing is freed until the last block closes, each tensor once; a failed second
  capture frees none of the first block's tensors.
- `test_gdn_shared_history`: the real four-card launch inside and outside a capture (192 tensors built once and reused, a wrong count or width
  refused, a failing launch never frees a pooled tensor), the K5-A served path with its audit (twin binding, 192 takes per capture, the audit's
  launch allocating privately), the effective environment of every 4e profile, `release_owned`, the real memory ledger, the accounting, the profile arithmetic
  and its negative cases, and that no production profile names either flag.
- What it cannot prove: that the second capture's trace, which allocates no history, still replays correctly on the device (its intermediates sit
  in different holes), that the freed bytes reach the KV pool the way the arithmetic says, and the commit's behaviour on real buffers. Those are
  the card window.

## Open items

- The split-V twin (`QWEN_FAST_GDN_SPLIT_V=2`) allocates its own states too; the same proxy would plumb it, but it is unqualified, so the flag is
  refused with it.
- Round time under the shipped (deferred) schedule is 0 to +2.2 ms by the research and unmeasured; the pack's timing arms T1-T4 measure it.
- Whether the worker's KV pool really grows into the freed bytes (the pool is sized from `QWEN36_MAX_TOKENS_ALL_USERS` before or after the packed blocks
  are built) is what A5 reads; if the attach fails at 22,144 blocks the ledger's free DRAM at P7 says how many fit.
- Under one 128-row block (M8) the lever disappears; this is a memory bridge for the two-block shape.

