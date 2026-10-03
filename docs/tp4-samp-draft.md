# TP4 sampler tail and drafter op fixes (branch tp4/samp-draft)

Two exact levers from the wave-3 design, each behind default-off strict flags (`0`/`1`, anything else raises, any flag at two cards raises),
gate-only profiles only. Nothing has run on a card: every saving below is an estimate from the production device profile, every exactness
claim is either by construction (held on the CPU by transliterations of the kernels) or waits on the in-trace audits.

| Lever | Flag | What | Estimate | Exactness |
|---|---|---|---|---|
| S1 | `QWEN_FAST_TP4_SHARD_ARGMAX` | the per-shard argmax of every verify is one scan launch on 110 cores (`tp4_shard_argmax_scan.cpp`, two data-movement RISC-Vs a core, straight from the tiled logits) and a one-core fold (`tp4_shard_argmax_fold.cpp`), replacing untilize + `ArgMax` (940 us) + the V4a gather | -0.9 to -1.8 ms per verify | ids identical to today's for every row without a NaN (first index holding the maximum, the rule of `torch.argmax` and `combine_shards`); NaN rows follow torch |
| D2a | `QWEN_FAST_TP4_DRAFT_CONV` | the drafter's fused convolution (20 launches a pass) keeps the served compute kernel but builds each page's seven-tile input with word copies into a double-buffered CB and writes the output from the second data-movement core (`draft_conv_io_fast.cpp`, `draft_conv_out.cpp`) | -4.7 ms (quad) / -6.3 ms (pairs) per 4-user round | byte-identical tiles into the served compute kernel, so identical proposals and tau |
| D2c | `QWEN_FAST_TP4_DRAFT_HEADS` | the drafter's `nlp_create_qkv_heads` and `nlp_concat_heads` (one core each) are one multi-core tile-copy launch each (`draft_heads_copy.cpp`): with head_dim 128 both are tile permutations | -0.3 ms per round | byte copy, identical proposals |

Audit flags (a correctness arm only, each needs its lever, name ends in `_AUDIT`): `QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT` runs today's path beside the
kernels in the same capture and compares every row of every chip every round (ids exactly, values as numbers); `QWEN_FAST_TP4_DRAFT_CONV_AUDIT`
runs the served I/O kernel beside the new one and byte-compares every conv output on every chip for the first rounds.

Not here (design routes S2 and S3, D2b exists, D2d): the hang-fix diet (the in-trace pinned sampler, 4.3 ms a verify, output never read), one
collective and one readback instead of eight, the wide hidden norms (`QWEN_FAST_TP4_DRAFT_WIDE`, built on tp4/next-3-cheap), the quad's double LM head.

## Where things are

* Flags, markers, the one table of run-time files: `scripts/ci/tp4_sampdraft.py` (`RUNTIME_FILES`; every entry is in the overlay list, the Dockerfile and the image workflow).
* S1: `tp4_shard_argmax.py` (plan, launch builder, audit), reached from `verify_trace_t1.sample_shards` (the packed block, the lone-user lane and the request engines all
  call it). It returns ids `uint32` and values `bf16`, ROW_MAJOR, one page each per chip, shaped (1, 1, 1, 64) (a 4-byte row page of the old (1, 1, rows, 1) shape is
  below a DRAM alignment unit); every consumer reads `to_torch(part).reshape(-1)[:rows]`. With `QWEN_FAST_TP4_VGLUE_AUDIT` the V4a value audit leaves kernel-made maxima to S1's audit.
* D2a: `tp4_draft_conv.py`, reached from `draft_convolution_fused_tp.fused_convolution` (pair, 80 workers x 2 pages) and `quad_draft_tp.quad_fused_convolution` (110 x 3 or 80 x 4).
  The pinned `quad_conv_io.cpp` (sha256 in `quad_draft.py`) and the pair's kernels are untouched; the flag-off path is `served_fused_convolution` / `served_quad_fused_convolution`.
* D2c: `tp4_draft_heads.py`, reached from `draft_head_layout_tp` and `quad_draft_tp` (`served_*` keep today's bodies).
* Markers (`c2_smoke_check.sampdraft_problems` fails a run on any `fell back` line, on a lever the profile sets with no `engaged` line, and on an audit flag with no passing audit line):
  `[PINDIAG] tp4 shard argmax engaged|fell back|audit N exact=True|audit mismatch`, `[PINDIAG] tp4 draft conv ...`, `[PINDIAG] tp4 draft heads engaged|fell back`.
* Tests (`py -3.11`, targeted): `test_tp4_sampdraft`, `test_tp4_shard_argmax`, `test_tp4_draft_conv`, `test_tp4_draft_heads`, `test_tp4_samp_draft_report`, `test_tp4_samp_draft_window`.
  The kernels cannot run here: the tests hold line-for-line Python transliterations of them (the scan and fold over the tile-face word layout against `torch.argmax` on ties at every
  boundary, all-negative rows, +-0, +-inf and NaNs; the served per-element conv loop against the word-copy image for every page, live count and seam mask; the tile permutation against
  the torch reshape/permute of both head ops) and the launches the builders describe.

## Profiles (gate only, UNVERIFIED on hardware)

| Profile | = | plus |
|---|---|---|
| `c2-packed-tp4-best-samp` | `c2-packed-tp4-best-strace` | `QWEN_FAST_TP4_SHARD_ARGMAX=1` |
| `c2-packed-tp4-best-d2` | `c2-packed-tp4-best-strace` | `QWEN_FAST_TP4_DRAFT_CONV=1`, `QWEN_FAST_TP4_DRAFT_HEADS=1` |
| `c2-packed-tp4-best-gate-samp` | `c2-packed-tp4-best-gate` | `QWEN_FAST_TP4_SHARD_ARGMAX=1`, `QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT=1` |
| `c2-packed-tp4-best-gate-d2` | `c2-packed-tp4-best-gate` | `QWEN_FAST_TP4_DRAFT_CONV=1`, `QWEN_FAST_TP4_DRAFT_HEADS=1`, `QWEN_FAST_TP4_DRAFT_CONV_AUDIT=1` |

## The window

`scripts/ci/references/tp4-samp-draft-jobs/` (image tag `tp4-samp-1`): B0 build, A0 agentstop and unserve, the audited sampler smoke S1 then the ABAB TS1-TS4 against
`c2-packed-tp4-best-strace`, the audited drafter smoke S2 then TD1-TD4. No agentstart and no hand-back: the driver closes the window. The tau report after the TD (and TS) pair:

    python scripts/ci/tp4_samp_draft_report.py --control <A1 log> <A2 log> --lever <B1 log> <B2 log>

Exit 0: the lever's pooled full-draft tau is within 1% of the controls' over at least 2,000 rounds each, no per-position acceptance rate is down more than 2 points and the two
controls reproduce each other; 1: it is not; 2: too few rounds to judge. Beside it `speed_window_compare.py ... --position-keyed --strict-concurrent-prefixes` exit 0 means every verified
proposal is identical (what byte-identical tiles must give).

## Open until a card runs it

* The scan kernel's real cycles per element. The design allows 80-150 us for scan + fold; the one-core fold over 110 tasks at 64 rows may be the larger half. If the kernel is slower than
  about 150 us, a two-level fold (eight partial folds, then one) or an FPU-assisted scan (reduce each tile row's maximum first, then scan only the winning tile) is the follow-up.
* `get_write_ptr(0)` before the first reserve (the conv reader writes the constant zero tile into both CB slots there) and a DM-only program with plain scratch CBs are standard
  patterns, but no card has run these programs; the audited smokes S1 and S2 are the first proof.
