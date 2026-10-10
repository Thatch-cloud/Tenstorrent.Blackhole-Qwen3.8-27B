# tp4/octo-2 Lever 3: the octo block's attention in two launches per layer (`QWEN_FAST_OCTO_ATTN_BUNDLE`)

Status: built and tested on CPU (`test_octo_attn_bundle`, 62 tests); nothing here has run on a card. Default off, gate only, byte-identical off. **No K64j kernel or factory change, no
graft, no op build**: every device piece is a `generic_op` JIT source that already ships for `QWEN_FAST_TP4_SDPA=multi`.

## 1. What it changes

The octo block (docs/tp4-octo.md section 5) makes one K64j decode launch per user per attention layer: G8B1, flags `0x21` (tail | extent), ONE entry of 8 tokens x 6 heads (48 rows, 2 row
tiles) on 16 cores. A layer is eight launches one after another, a pass is 128. The two M3 blocks make the same 128, so the octo pass wins nothing on attention (docs/tp4-octo.md,
"Attention is eight launches of one entry per layer").

With `QWEN_FAST_OCTO_ATTN_BUNDLE=N` (2 to 6 users per launch) the users' entries go in ONE launch each: query `(1, n, 48, 256)` (entry u is user u's entry exactly), page table `(n, width)`
(row u = user u's row), cur_pos `(n,)` (word u = user u's E - 1), mask `(n, 1, 48, 256)` (entry u masks past user u's own start), flags `0x21` again. The n users of a launch run on
16 x n cores at the same time. Eight users are split evenly: N = 4 (recommended), 5 and 6 give 4 + 4, N = 2 gives 2 + 2 + 2 + 2, and N = 3 (3 + 3 + 2) is refused by name because the fold
launch has never run over stacked queries of two shapes. Per layer: fold-in (1 launch, both stacked queries), 2 SDPA launches, fold-out (1 launch), where V3a runs fold-in, 8 SDPA, fold-out.

| | served (V3a on) | bundled, N = 4 |
|---|---|---|
| query | 8 x `(1, 1, 48, 256)` | 2 x `(1, 4, 48, 256)` |
| page table / cur_pos / mask | the user's lent `(1, W)` / `(1,)` / `(1, 1, 48, 256)` | `(4, W)` / `(4,)` / `(4, 1, 48, 256)`, gathered and written once per forward, in the trace |
| flags, cores per entry | `0x21`, 16 | `0x21`, 16 (the factory gives `min(110, 16 B) / B`: 16 up to B = 6) |
| SDPA launches per layer | 8 | 2 |
| launches per forward besides the layers | 8 mask refreshes | 2 gathers + 2 mask writes |

## 2. Why it is byte-identical (the argument; the audit proves it on the card)

Each entry of a bundled launch IS the entry of the single launch: the same 48 query rows in the same tile rows, the same page-table row, the same cur_pos word, the same mask bits, the same
program constants (PNHt = 2). What the K64j program computes for an entry depends on exactly four things, none of which depends on how many other entries share the launch:

1. **Cores per entry** `= min(110, max_cores_per_head_batch * B * kv_heads) / B` with the config's default 16 (`sdpa_decode_program_factory.3e0a69af.cpp:196-209`, B = the page table's batch,
   `:117`): 16 for B <= 6, 15 at B = 7, 13 at B = 8. `sdpa_multi_tp.plan_problem` refuses any group where it is not 16; this is why eight users are two launches and not one.
2. **The partition and the tree**: `get_workload_for_core` (`rt_args_common.1b52c60d.hpp:35-93`) takes (cur_pos, the core's index inside its entry, cores per entry, the 256-key chunk); the
   tree runs over the core index inside the entry (`:240`: 4 rounds at 16 cores). Core PLACEMENT differs between a bundle and a single launch; the arithmetic order does not.
3. **The per-row op sequence**: subblock heights `:398`/`:407` and `MUL_BCAST_GRANULARITY` `:785` are functions of PNHt, and PNHt is the single launch's 2. (`sdpa-multi` (T16 users) had to
   argue this across 2 -> 3 row tiles; here the entry does not change at all.)
4. **The data**: fold-in, fold-out, the page-table gather and the mask are row and page copies. Held on CPU by emulating each kernel's loops from the runtime arguments its planner writes:
   the stacked query is the served entry byte for byte for 2, 3, 4 and 6 users; the mask of entry u equals the pinned narrow mask of the single launch for 11 start-word classes
   (0, 1, 7, 127, 128, 200, 239, 240, 247, 248, 255); the gather assembles every user's row and cur_pos.

`QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT=1` (gate profiles only) runs the eight single launches beside the bundled ones in every layer and compares the two block outputs word for word in the trace
(`sdpa_multi_audit_tp.cpp`, 32-bit words, -0 against +0 counts); `[OCTO-ATTN-BUNDLE] audit <n> exact=True ...` per replay, a mismatch raises. The timed arm never carries it.

## 3. What is reused, what is new

Reused unchanged: `sdpa_multi_gather_tp.cpp`, `sdpa_multi_audit_tp.cpp`, `sdpa_multi_mask_tp.cpp` (its `rows` is a runtime argument and its position is `word + head / 6` for the entry's own
word), `attention_block_fold_tp.{py,cpp}` (V3a already runs it at eight one-group chunks; here a chunk is n consecutive users' groups, and `fold_in` / `fold_out` take any chunk list),
`extent_attention_octo_tp` (the readers, their words, lent tables and cur_pos stay exactly as they are: the per-forward gather copies them).

New: `scripts/ci/octo_attn_bundle.py` (flags, planners, the rows-parameterised mask launch builder, `OctoBundle`, `bundle_problems`) and a six-line hook in
`extent_attention_fold_tp.PackedExtentReplayReader.__init__` that asks the two flags BEFORE importing the module: a process with them unset never imports it, sets nothing and reads two
environment variables. `OctoBundle` is assigned to the reader's existing `multi` slot (the same duck type as `sdpa_multi_tp.MultiBlock`: `call`, `scope`, `rebound`, `audit_round`,
`close`), so `shared_masks`, `rebound_reason` (checked before every replay), `sdpa_audit_round` and the captured-reader identity check all work as they do for `multi`.

## 4. Flags and markers

| | |
|---|---|
| `QWEN_FAST_OCTO_ATTN_BUNDLE` | unset, `''`, `0` off; `2` to `6` users per launch; anything else raises. Needs `QWEN_FAST_OCTO=live|alternate`, `QWEN_FAST_TP=4`, no `QWEN_FAST_TP4_SDPA`; `admission_problems(environ)` is the list serving_octo's admission can refuse with. |
| `QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT` | `1` audits; needs the bundle flag. |

The M3 blocks (sixteen-row segments) are untouched with the flag on (`apply` returns None for them, silently).

Markers (namespace `[OCTO-ATTN-BUNDLE]`, which does not contain `[OCTO]`, so the octo judge's "a flag-off arm logs no `[OCTO]` line" rule never sees it):

* `engaged users=8 launches=2 per_launch=4,4 flags=0x21 cores_per_entry=16 active_cores=64,64 grid=11x10 audit=0` once at the attach, after one eager refresh of every launch (a kernel that does
  not build fails the attach, not the first capture);
* `UNQUALIFIED (gate only): K64j 0x21 with B > 1 entries of eight rows ...` once at the attach;
* `call users=8 launches=2 ...` once, at the first layer that ran;
* `audit <n> exact=True users=8 launches=2 layers=16 chips=4 words=<w> live=<l>` per audited replay, or `audit MISMATCH ...`.

The smoke rule is the pure function `octo_attn_bundle.bundle_problems(container_log_text, profile_env)`: flag off, any marker line is a leak; flag on, exactly one engaged line saying the split
the value gives, `flags=0x21`, `cores_per_entry=16`; the call line; the UNQUALIFIED line; no mismatch; and under the audit flag at least one `exact=True` line.

## 5. The estimate (per 8-live octo pass), and the reasoning

Measured basis (the O3 container log, octo block at 8 live: `trace_ms` 85-87, mean context 3.8-4.6k; the TP4 device profile of the served recipe: the SDPA is `53 us + 6.09 us per 1k keys` per
launch, about 164 GB/s per entry, against about 405 GB/s of DRAM per chip). Model: a single launch costs `53 + 6.09 K` us (K in thousands of keys); a bundle of 4 entries costs
`53 + 9.9 K` us, because four concurrent entries are DRAM-bound at about 100 GB/s each; 16 layers.

| context per user | served, 8 launches | bundled, 2 launches of 4 | saved per pass |
|---|---|---|---|
| 4k | 8 x 77 = 619 us a layer | 2 x 93 = 185 us | 434 us x 16 = **6.9 ms** |
| 6k | 8 x 90 = 720 us | 2 x 112 = 224 us | 496 us x 16 = **7.9 ms** |
| 32k | 8 x 248 = 1984 us | 2 x 370 = 740 us | 1244 us x 16 = **19.9 ms** |
| 120k | 8 x 784 = 6272 us | 2 x 1241 = 2482 us | 3790 us x 16 = **60 ms** |

About 5 ms of the 4k figure is the six launches' fixed cost; the rest is the DRAM overlap, which is what grows with the context. Against the 8-live octo round of 140 ms (step 94.7 +
gap 45.0) that is 5% at the O3 steady shape and 14% of the same 140 ms at 32k (the 32k round is longer, so less of it). The estimate is a model on a fitted launch cost, not a measurement: the slope of `trace_ms` against the mean context in the O3
log is about 1.0 ms per 0.8k (noisy, 142 points in 2k-6k), the same order as the model's 0.78 ms per 1k. The gather and mask launches add about 0.1 ms a forward and replace the 8 mask
refreshes. Cost the model does not carry: the L1 of 64 active cores instead of 16 (the CBs are the single entry's, so unchanged per core), and any skew from mixed contexts (a bundle waits for its longest
user; the sum over two launches is never more than the sum over eight, bandwidth permitting).

## 6. Card jobs (templates for the parent's pack; nothing is pushed from this branch)

Both use a gate-only twin of `c2-packed-tp4-8x262k-ship-prefix-levern-octo-live` (`QWEN_FAST_OCTO=live` isolates the lever: every eligible round is octo, so no restage per switch) plus
exactly `QWEN_FAST_OCTO_ATTN_BUNDLE=4`; the audited twin adds `QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT=1`. The bundle audit is independent of the T1/T2/extent audits, so the audited job needs no
audited base and stays cheap (it doubles the attention only).

1. **OA1** (needs O1: the octo attach itself): the audited twin, tests `warmup,concurrent8_steady`. READ: `bundle_problems` clean with the audit arm, at least 16 octo rounds, zero audit mismatch,
   `exact=True` lines with `words` and `live` non-zero.
2. **OA2** ABAB (needs OA1): A = `-octo-live`, B = the timed twin, A, B; tests `warmup,concurrent8_steady,concurrent8_code_equal,concurrent8_code_32k`; CI paused; judged PAIRED per round
   on `trace_ms` of the octo rounds and on committed tokens per second per seat; texts equal user by user (`octo_compare.py` A against B). GO: the paired `trace_ms` gain at least 5 ms at the
   steady shape and at least 15 ms at 32k, every text equal.

NO-GO: any audit difference, any text difference, a program compiled after the attach, a stall.

## 7. Files

`scripts/ci/octo_attn_bundle.py` (new; to ship: `docker/qwen-c2-overlay.txt` and the plain-import allowlist in `test_serving_image_copy_closure.py`, the precedent of `sdpa_long_tp`),
`scripts/ci/extent_attention_fold_tp.py` (the hook), `scripts/ci/test_octo_attn_bundle.py`, `scripts/ci/test_serving_image_copy_closure.py` (one allowlist entry), this document.
