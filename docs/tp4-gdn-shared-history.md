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

The states output of the batched GDN launch is allocated inside the verify capture. A process-wide
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
schedules exist today:

1. Blocking or pipelined commits (the shipped profile): `run_verified_block` commits every user right after the
   readback, on the one command queue, before the next block is verified. Already ordered; the cost is 0.
2. Deferred commits (`QWEN_FAST_GDN_AFTER_PAIRS`, round-fence plan H2): block A's commit traces are enqueued after
   the next drafts' fence, which is after block B's verify. Sharing would let B overwrite A's history first.
3. Anything undecided: a block that has verified but not committed every segment.

Enforcement is one guard in `PackedVerifierEngine.verify` (`SharedHistory.claim`): before a block verifies, the
block that last wrote the history must have nothing pending. Deferred commits are flushed there (the same
`flush_commits` the engine already uses, site `shared-history`, logged and counted); undecided segments raise and
fail the round closed. The guard sits in the engine, not in the step, so every caller is covered. The cost under
schedule 2 is A's four commit traces (about 2.0 to 2.2 ms of device time) moving between verify A and verify B:
0 to +2.2 ms per round, none under schedule 1.

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

The flag is refused (a `ValueError` naming it) unless serving is four cards, two M3 blocks, and the K5-A launch
(`QWEN_FAST_GDN_SEQ_BLOCK`, and with it its split twin) is off: K5-A allocates its own states inside pinned
sources and is not plumbed. Joining is refused for any block that is not the 4-user 16-row-per-user 64-row
shape, for a third block, and for a capture that does not request exactly 192 tensors of the first block's
shape. The grow flag without the sharing flag is refused. With the flag unset nothing is imported on the serving
path beyond a constant-time flag read and no code path changes.

### What is proved on CPU, and what is left for the card window

CPU: the commit-before-reuse ordering under all three schedules with the real `PackedVerifierEngine` and a
byte-level device model (shared and private histories produce identical carries; a negative control with the guard
cut shows the hazard is real); the capture plumbing through the real `gdn_user_batch_tp.execute`; ownership and
release; the ledger arithmetic through the real `MemoryLedger`; the refusal paths; the grown-profile arithmetic and
the production profiles unchanged. Card window: audited attach, exactness against the control, the ledger before and
after, the five-reservation admission test and the hang shapes (`scripts/ci/references/tp4-gdn4e-jobs/ORDER.txt`).

## What was built

| Piece | Where |
|---|---|
| The pool, the guard, the accounting, the refusals, the growth-marker check | `scripts/ci/gdn_shared_history.py` (overlay list) |
| The launch takes its states from the pool inside a verify capture | `scripts/ci/gdn_user_batch_tp.py` (`execute`) |
| The pool's tensors are never freed by a block's `owned` release | `scripts/ci/tp_addresses.py` (`release_owned`) |
| Join at construction, capture window, claim in `verify`, detach in `close`, marker, `describe` | `scripts/ci/packed_verifier.py` |
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
  refused, a failing launch never frees a pooled tensor), `release_owned`, the real memory ledger, the accounting, the profile arithmetic
  and its negative cases, and that no production profile names either flag.
- What it cannot prove: that the second capture's trace, which allocates no history, still replays correctly on the device (its intermediates sit
  in different holes), that the freed bytes reach the KV pool the way the arithmetic says, and the commit's behaviour on real buffers. Those are
  the card window.

## Open items

- K5-A (`QWEN_FAST_GDN_SEQ_BLOCK`, and its split twin) allocates its own states inside sources whose bytes are pinned; sharing there needs its own
  plumbing and re-qualification. The flag is refused with it.
- Round time under the deferred schedule (0 to +2.2 ms) is estimated, not measured; the shipped profile does not defer, so its cost is 0 by
  construction. No timing arm is in the pack.
- Whether the worker's KV pool really grows into the freed bytes (the pool is sized from `QWEN36_MAX_TOKENS_ALL_USERS` before or after the packed blocks
  are built) is what A5 reads; if the attach fails at 22,144 blocks the ledger's free DRAM at P7 says how many fit.
- Under one 128-row block (M8) the lever disappears; this is a memory bridge for the two-block shape.

