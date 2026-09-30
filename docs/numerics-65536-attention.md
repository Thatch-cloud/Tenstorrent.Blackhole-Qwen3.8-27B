# Why the 65536 frozen-recipe attention probe fails its audit

Run 35585249041 (tag `experiment/frozen-64k-reciprocal-replay-v2`, workflow
`qwen-frozen-32k-numerical.yml`) fails `scripts/ci/dspark-native-8k-attention-probe.py`
at stage `eager_0`, line 255 (`dspark-native-8k-attention-probe.py:255`), audit()
raising `'Fixed-storage attention fails retained FP32 accuracy or exact replay'`.
Not a hardware run; this is `backend: simulator`.

## 1. Failure pattern (from `dspark-native-8k-attention.json`)

Only one `numerical_failures` entry: `mode=eager, case=0, chip=0`, shape
`[1,16,32,128]`, all-finite. Of the 240 populated (head, proposal-row) pairs
(16 heads x rows 0-14; rows 15-31 are padding, per `fixtures()` at
`dspark-native-8k-attention-probe.py:105`), only **3 breach the strict
`torch.isclose(rtol=.01, atol=.01)` test outright, each failing all 128
channels**: head 10/row 2, head 11/row 12, head 12/row 12
(`failed_by_head_row`). But `max_abs_by_head_row` shows a **nonzero deviation
in essentially every one of the 240 pairs**, ranging ~0.002-0.64, with no
concentration by row index (row 0 and row 14 are both large for some heads,
small for others) and no head standing out as categorically worse. That
pattern - small-to-moderate error nearly everywhere, a handful of outliers
crossing the line - is the signature of a **systematic rounding effect that
compounds with iteration count**, not an isolated address/mask/golden bug
(those tend to produce a clean block of exact zeros next to a clean block of
large, uniform errors).

## 2. What the probe computes

`reference()` (`dspark-native-8k-attention-probe.py:117-123`) builds the FP32
golden via `torch.nn.functional.scaled_dot_product_attention` over the
*exact* concatenated history+proposal keys/values (`joined()`, line 110-114:
`cat(history, query[:PROPOSALS])`, padded to `geometry(CAPACITY, PROPOSALS)`'s
last chunk boundary), `attn_mask=values['mask'].float()`, `is_causal=False`.
`fixed_mask` (imported, not in this file) supplies the per-position causal +
padding mask. Tolerance is `rtol=atol=0.01` (`numerical_tolerances`, line 172)
- unchanged from the qualified 32768 recipe; nothing in `geometry.patch`
touches this constant.

`CAPACITY`/`POSITIONS` are **not** the file's own defaults (8448 / `(8192,
8433)`, still visible at `dspark-native-8k-attention-probe.py:31-32` in this
worktree) - `geometry.patch` (hunk at `geometry.patch:14-38`) replaces them
with `frozen_context_geometry.selected_geometry()`, driven by
`QWEN_DSPARK_REQUEST_CONTEXT=65536` (workflow: grep hit at
`.github/workflows/qwen-frozen-32k-numerical.yml:75`). `frozen_context_geometry.py:9-19`
gives `capacity=65792, storage_keys=65856, padded_keys=66048, key_chunk=256` -
**every JSON field matches this formula exactly** (`positions`, `capacity`,
`native_padded_keys=66048`), so the probe's own geometry math checks out; this
is not a geometry.patch arithmetic bug. "Fixed storage" = the KV physically
occupies a capacity-sized buffer regardless of how many tokens are logically
valid; `key_chunk_size=256` is the online-softmax merge granularity.

## 3. Kernel side: the fix that *would* apply isn't the one that ran

**The Skt selector is correct.** `dflash_combined_sim_runtime.py:18-25`
(`COMBINED_FACTORY`) and `dspark_fp32_intermediates.py:11-18` (post-patch)
both gate `qwen_draft_fp32_intermediates` on an OR-list of `Skt` values from
`frozen_context_geometry.factory_selector()` (`frozen_context_geometry.py:29-31`),
which enumerates every context in `CONTEXTS` (`frozen_context_geometry.py:6`).
For context 65536: `padded_keys=66048`, `Skt=66048/32=2064` - present in both
lists. `docs/t16-recipe-rung-65k.md` ("Not a pin, and no rebuild needed"
section) independently confirms this: "65536 maps to `Skt == 2064`, already
present... the binary already carries the `Skt == 2064` branch." **This run's
`dspark-fp32-build.json` confirms `factory_enabled: true,
precision_variant: 'stats-only'`, and `source_after` hash
`7c5f32b5...` matches `dflash_combined_sim_runtime.py:14`'s `COMBINED_FACTORY`
exactly** - the fp32 branch is compiled in and selected for this geometry.

**But "stats-only" is the caveat, not a label.** In `dspark_fp32_intermediates.py:17-18`
(and identically in `dflash_combined_sim_runtime.py:24-25`), only `stats_df`
becomes `Float32`; `im_df` - the output/accumulator intermediate format - stays
`tt::DataFormat::Float16_b` **unconditionally**. `docs/draft-8k-numerical-investigation.md:238-239`
names this directly: "FP32 accumulation is enabled, but keeps `im_df` and
`stats_df` in BF16" [sic - stats_df is the one that flips]. So the online-softmax
output accumulator round-trips through bf16 on every chunk merge regardless of
this flag.

**This exact failure mode was already root-caused and fixed for 65536, in a
different code path this lane doesn't use.** `docs/context-ladder-investigation.md`
runs an extensive 64K investigation (`~line 340-935`) that: rejects "FP32
output intermediates only" (worse: 4131 failing vs 129 baseline, `:378-390`);
rejects a reciprocal-reload-rounding candidate (worse: 257 vs 129, `:424-430`);
then isolates the real cause via `:821-895` ("Normalization format audit" /
"One-tile normalization reproduction"): tt-metal's format-family-B rule
silently narrows an FP32-typed CB operand to **TF32** on unpack whenever it
shares a CB with a BF16 operand - so the final normalization broadcast
multiply (numerator x reciprocal) truncates the reciprocal to TF32 right
before the multiply, independent of how precisely the reciprocal itself was
computed. The validated fix, `dspark_ladder_normalization.py` (`:8-25`,
`sfpu-column-scratch`), gives the reciprocal a **dedicated scratch CB** with
`UnpackToDestMode::UnpackToDestFp32` set only on that one CB, avoiding the
shared-CB downgrade. Combined with `dspark_ladder_factory.py`'s geometry
(`:11,16`: `Skt == 2112`, sourced from `dspark_ladder_geometry.py:14`'s
`key_chunk={65536: 1024}` - 67,584 keys / 1024-key chunks = **66** merge
iterations, half the 258 this run does at the standard 256-key chunk), this
passed hardware with **zero** failing elements: "Hardware 64K correctness
passes: 34797353681" (`docs/context-ladder-investigation.md:915-933`).

**None of that fix is wired into this lane.** `grep -rl dspark_ladder_normalization
scripts/ci/*.py` shows it consumed only by `dspark-ladder-attention-probe.py`
and its own tests - never by `dspark-native-8k-attention-probe.py`,
`frozen_recipe_context.py`, or `frozen_context_geometry.py`. This lane's
`--scalar-reciprocal` flag (workflow line: `frozen-64k-reciprocal-replay-*`
tags -> `--scalar-reciprocal --probe-seconds 1020`,
`.github/workflows/qwen-frozen-32k-numerical.yml:52-53`) applies
`dspark_ladder_scalar_reciprocal.py` instead - its own docstring
(`dspark_ladder_scalar_reciprocal.py:1`) calls it "Unqualified TR0 scalar
reciprocal diagnostic; **no shared CB format changes**." It recomputes the
reciprocal precisely on TR0 (`1.0f / values[offset]`, line 20) but leaves the
downstream unpack-to-TF32 truncation on read exactly where the investigation
found it. `reciprocal_variant: 'scalar-fp32'` in the JSON names this
diagnostic, not the qualified `sfpu-column-scratch` one. The `frozen-recipe`
checkout is additionally pinned to commit `8c102b20`
(`qwen-frozen-32k-numerical.yml:26`), predating/excluding the normalization
module's integration into any frozen-recipe path.

So: the geometry, tag matrix and Skt selector are all correct (ruling out
4d, a probe/patch bug) - **this is a real kernel-precision gap**, but it is a
*known, already-fixed* gap being hit by a *different, weaker* variant
(scalar-reciprocal-only, standard 258-iteration chunking) than the one
(`sfpu-column-scratch`, 66-iteration chunking) already proven sufficient at
this exact context.

## 4. Options, ranked

**(a) Port `dspark_ladder_normalization.py` into the frozen-recipe path -
recommended.** Change: extend `frozen_context_geometry.factory_selector()`
(or a 65536-specific override) to also apply the dedicated-scratch-CB
normalization patch, gated on `Skt==2064` instead of the ladder's `2112`
(the scratch-CB technique doesn't depend on the specific Skt value; only the
geometry differs). Files: `frozen_context_geometry.py`,
`dspark_fp32_intermediates.py` (REPLACEMENT wiring),
`dspark_ladder_normalization.py` (reuse as-is or copy the substitution
triplet). Requalification: one simulator run (fast) plus one hardware run on
`thatch-qwen-p150a-pair` (`docs/t16-recipe-rung-65k.md` estimates 0.5-1 day
for this class of change). Risk: low - this exact patch is already
hardware-validated at 65536, just at a different Skt gate; the main work is
confirming the scratch-CB substitution composes cleanly against the pinned
`8c102b20` factory source (same `SOURCE_SHA256` family, should be fine) and
against `Skt==2064`'s specific tile counts rather than `2112`'s.

**(b) Adopt the ladder's 1024-key chunk geometry (66 iterations) for 65536,
without the scratch-CB fix.** Cheaper to try, but the investigation already
showed this reduces but does not eliminate the error (129 failing / 0.514 max
abs at 66 iterations, still a numerical failure) - not sufficient alone.
Only useful as a magnitude check, not a fix.

**(c) Widen tolerance for 65536 only.** Not recommended and stated plainly:
this would mask a real, already-diagnosed bf16-accumulation drift rather than
fix it, and needs explicit authorisation since it weakens the recipe's
accuracy guarantee specifically at the rung the team is trying to qualify.
No effort estimate given because this isn't proposed as the fix.

**(d) Probe/geometry bug - ruled out.** Checked first, per instructions:
`positions`, `capacity`, `native_padded_keys`, `key_chunk_size` in the JSON
all match `frozen_context_geometry.geometry(65536)` exactly; the Skt=2064
branch is confirmed compiled and selected (`dspark-fp32-build.json`); the
golden is built the same way (`reference()`) as the passing 32768 case. No
evidence of a geometry.patch or golden-construction error.

## 5. Update: the scratch-CB + 1024-key-chunk port, built and run (run
35593521063, `experiment/frozen-64k-reciprocal-replay-v3`)

Option (a) was implemented as `scripts/ci/frozen_wide_chunk_scratch.py` /
`scripts/ci/frozen_wide_chunk_normalization.py`, gated on Skt==2080 (the
frozen recipe's own context+256 capacity basis with 1024-key chunking, not
the ladder's Skt==2112 - see those files' docstrings for the arithmetic). It
compiled and ran. Result: **improved but still failing.**
`dspark-native-8k-attention.json`: `failed_elements` 384 -> **9**, `max_abs`
0.638 -> **0.455**. Artifact:
`C:/Users/liamb/.claude/jobs/8376c877/tmp/64k-recip3-art/qwen-frozen-32k-numerical-35593521063-1/`.

**Residual location and pattern.** All 9 failures sit at a single (head=0,
row=8), across 9 of 128 channels (indices 28,45,50,51,56,65,67,107,126) -
`numerical_failures[0].failed_by_head_row`. Every one of the 9 `actual`
values is **bit-identical at -43.5**, while `expected` varies narrowly
(-43.949 to -43.955). That flatness is the tell: at this magnitude bf16's
ulp is 0.25, so -43.5 is exactly 2 ulps off the correctly-rounded -44.0 -
several *different* per-channel numerators are landing on the *same* bf16-
representable value before the final normalise, so the output loses its
per-channel variation entirely at this one row. This is a **different**
(head, row) than any of the three that failed outright in the pre-port run
(head10/row2, head11/row12, head12/row12) - not the same rows with less
error, a smaller and relocated residual. The widespread sub-threshold
pattern is still present and similarly shaped (`max_abs_by_head_row`: 441
nonzero (head,row) pairs, next-highest 0.435, 0.435, 0.402, 0.394...), just
uniformly smaller than the pre-port run's - consistent with the scratch-CB
fix and the ~4x iteration-count drop (258 -> 65) both reducing, not
eliminating, the same underlying compounding bf16-rounding mechanism.

**Build evidence, confirming the patched branch executed exactly as
designed** (not just staged): `dspark-native-8k-attention.json` itself
reports `key_chunk_size: 1024`, `native_padded_keys: 66560`,
`added_masked_poison_rows: 704` - the literals `_patch_probe` writes, an
exact match. `geometry.patch` contains the generated text verbatim:
`KEY_CHUNK = 1024`, `PADDED_KEYS = 66560`,
`(Skt == 2080 && Sk_chunk_t == 32))`, and both `validate_manifest` imports
redirected to `frozen_wide_chunk_scratch`. `dspark-fp32-build.json`'s
`builders.dspark_fp32_intermediates.py` hash
(`0ee87770add7508f2ef2a35950b447049fcb0627d7dab4da1bc2b90c53dbcaa2`) matches
the staged file's hash from this port's own dry-run verification exactly;
`binaries_after` (`274907a9d5a8...`) matches `frozen-build-cache.json`'s
`binary_sha256` and the binary team lead cited, with `cache_hit: true`
(reused the pre-warmed context-build-65536-v3 entry) and `import_passed:
true`. Golden-vs-kernel key range: the golden (`reference()` /
`dspark_full_attention.geometry`) pads to `storage_keys` (65856), a formula
untouched by this port and independent of `key_chunk`/`padded_keys`; the
kernel operates over `padded_keys` (66560), with the extra 704 keys masked
to `-inf` in both the key/value pad and the mask pad
(`dspark_attention_chunk_trial.py`, both padding amounts derived from the
same two overridden literals, so they can't disagree with each other). This
run fails inside `audit()` before reaching `layout_checks`, so there's no
direct re-confirmation the mask pad landed correctly on hardware this time,
but the failure's own shape - a handful of channels quantising to an
identical bf16 value, not a wrong magnitude, NaN, or a wide swath of
elements - is the signature of a precision effect, not a range/masking bug.

**Comparison against the ladder's zero-failure pass (run 34797353681) -
correction to this doc's earlier assumption.** `dspark_ladder_factory.py`
defines `output_precision()`, which *would* widen `im_df` (the output
accumulator) to FP32 at Skt==2112 - but grep confirms it is never called:
`dspark_ladder_build.factory_scope()` only composes `dspark_ladder_factory.
transform` (the Skt/Sk_chunk_t widening) and `dspark_ladder_normalization.
factory_transform` (the scratch-CB patch), never `output_precision`. So the
ladder's own validated, zero-failure configuration *also* runs with `im_df`
staying bf16 unconditionally - identical to this port on that axis. "Output
accumulator still bf16" is therefore **not** the differentiator between the
two runs; both share it. Real differences: (1) capacity basis - the ladder
reserves `output_tokens=1024` (`dspark_ladder_geometry.py:7,12`), giving
`storage_keys=66624`, `native_keys=67584`, Skt=2112, 66 iterations; this port
uses `context+256` (`frozen_context_geometry.py:12`), giving
`storage_keys=65856`, `padded_keys=66560`, Skt=2080, 65 iterations - a
768-key/one-iteration gap; (2) a different probe/fixture harness entirely
(`dspark-ladder-attention-probe.py` vs `dspark-native-8k-attention-probe.py`)
- a quick grep found no `manual_seed`/`torch.rand` calls directly in the
ladder probe or `dspark_ladder_fixtures.py`, suggesting it may import the
same fixture-building code this port's probe uses, but this was not fully
traced and is left unconfirmed. Scale, exp-approximation and mask
construction were not directly re-diffed line-by-line against the ladder
probe in this pass, given the report deadline; flagging as unverified rather
than asserting they match.

**Recommendation.** Single next change with the best odds: reproduce the
ladder's exact geometry (Skt=2112, 67584 padded keys, 1024-key chunk, 66
iterations - i.e. `dspark_ladder_geometry.py`'s own numbers) instead of this
port's independently-derived 2080/66560/65, since 2112 is the *only*
combination with an actual zero-failure hardware result behind it. This is
a pure constant swap in `frozen_wide_chunk_normalization.py` (`SKT`,
`PADDED_KEYS`, `KEY_CHUNK` stay 1024, `CAPACITY`/`STORAGE_KEYS` change to
match the ladder's `output_tokens=1024` formula) - cheap, no new kernel
logic - but confidence is moderate, not high: the fixture harness differs
too, and the ladder pass has never been shown to hold for *this* probe's
specific random values. If that still fails, the honest reading is that this
residual is a genuine, narrow bf16-boundary case - a handful of per-channel
numerators closer together than one bf16 ulp at this magnitude - that no
reciprocal-precision or chunk-count lever closes for every possible input,
only an FP32 output accumulator would. That specific candidate
(`dspark_ladder_factory.output_precision()`, currently dead code) was tried
*alone* earlier in this investigation and made things worse (4131 failures,
`docs/context-ladder-investigation.md`, "FP32 output-intermediate result:
worse, rejected") - it has never been tried *combined* with the scratch-CB
reciprocal fix now in place, which changes the starting point enough that
the old rejection may not transfer. Trying it would cost a fresh
build+requalification cycle and roughly doubles the stats+output CB
footprint at this rung, which matters given the tight CB budget already
flagged separately for target-replay.

## 6. Update: the Skt selectable knob (v4) - and why the swap alone probably
isn't sufficient

Implemented as requested: `frozen_wide_chunk_normalization.py` now resolves
Skt from `QWEN_FROZEN_65536_SKT` (accepted `'2080'`/`'2112'`, default
`'2112'` as of this lane) at staging time, and generates exactly one Skt
branch for 65536 accordingly - `padded_keys`/`iterations`/masked-poison-row
count all derive from the resolved Skt (`geometry_for_skt()`), and
`frozen_wide_chunk_scratch.py`'s own `SKT` constant is patched to match
before staging, so the scratch-CB substitutions gate on the same value. The
`frozen-geometry.json` manifest's `padded_keys`/`key_chunk` fields are now
also overridden per the resolved Skt (`manifest_geometry_override()`,
wired into `frozen_recipe_context.main()`'s manifest write) - the cosmetic
gap from section 1 is fixed for this field; `storage_keys`/`positions`/
`capacity` are untouched, which is correct (they don't change under either
Skt choice - this stays a kernel-internal padding change).

**Why Skt==2112 alone is unlikely to close the residual.** Read
`dspark-ladder-attention-probe.py:77-85` (the probe behind run 34797353681)
in full this round. Its passing configuration applies FOUR separate
precision fixes simultaneously, not one:

1. `scalar_reciprocal()` (`dspark_ladder_scalar_reciprocal.py`) - already
   present in this port's lane via the existing `--scalar-reciprocal` flag,
   unrelated to this port specifically.
2. `scalar_sum_update()` (`dspark_ladder_sum_update.py`) - a TR0 scalar-FP32
   correction to the running-sum update (`current[offset] = current[offset]
   + previous[offset] * correction`, replacing a bf16 `add_block_inplace`),
   gated on `get_compile_time_arg_val(3) == 2112`. **Not staged or applied by
   this port at all.**
3. `scalar_score_center(key_tiles=2112)` (`dspark_ladder_score_center.py`) -
   a TR0 scalar-FP32 mask-add and max-centering pass on the raw QK scores
   before the exponential, replacing `sub_bcast_cols_init`/
   `sub_tiles_bcast_cols`, also gated on `get_compile_time_arg_val(3) ==
   2112`. **Not staged or applied by this port at all.**
4. `scratch_normalization()` (`dspark_ladder_normalization.py`) - the
   dedicated-scratch-CB reciprocal fix this port already carries (as
   `frozen_wide_chunk_scratch.py`), also gated on Skt==2112 in the original.

Both (2) and (3) are themselves hardcoded to `Skt == 2112` in their own C++
text, exactly like (4) - but they live in modules this port has never
staged, imported, or applied. Selecting `QWEN_FROZEN_65536_SKT=2112` makes
*this port's* scratch-CB condition match the validated value; it changes
nothing about (2) or (3), which simply don't run. Given
`docs/context-ladder-investigation.md`'s own history shows the sum-update
fix alone got the failure count from 129 down to "one, still unqualified"
and the score-centering work continued from there before scratch-CB
normalization was even introduced, these look like fixes for a *different*
stage of the computation (denominator/running-sum accumulation, and
raw-score masking/centering) than the final normalization reciprocal
scratch-CB addresses - not redundant with it. This is the most likely
explanation for why 9 elements remain at Skt==2080: not exclusively a
generic "which Skt" question, but three additive, independently-gated fixes
of which this port carries only the last.

**Fixture trace (best-effort, CPU-only).** `dspark-ladder-attention-probe.py`
does not build its own fixture directly - it calls `fixture_probe(context)`
(`dspark_ladder_fixtures.py`) and `adapter(context)`
(`dspark_ladder_attention.py`), neither of which was read in this pass (time
did not allow it); a grep for `manual_seed`/`torch.rand`/`torch.randn` found
no direct hits in either file or in the ladder probe itself, suggesting they
likely wrap/reuse `dspark-native-8k-attention-probe.py`'s own `fixtures()`
(same seed 383928, same query/key/value construction) rather than
generating independent random data, but this was not traced through
`fixture_probe`'s and `adapter`'s actual bodies to confirm, and scale/mask
construction were not diffed line-by-line either. Left as an open item
rather than asserted either way - if it turns out the fixtures differ, that
would be an *additional*, independent reason Skt==2112 alone might not
transfer, on top of the missing sum-update/score-center fixes above.

**Revised recommendation:** the geometry swap was worth doing as the
requested cheap experiment and is in place, defaulted to 2112. But given the
above, the single next change with the best odds is no longer "just try
2112" - it's porting `scalar_sum_update()` and `scalar_score_center()`
alongside the existing scratch-CB fix, the same way this port already
ported `scratch_normalization()` (own module, gated on the frozen recipe's
resolved Skt, staged only for 65536). That is real, unstarted work - two
more C++ text substitutions to port and parameterize away from the ladder's
hardcoded 2112, plus their own build/requalify cycle - not a knob flip.

## 7. Critical correction: the scratch-CB port was never actually active, and
the two missing ladder fixes are now ported too

**The scratch-CB fix has never actually run, in any prior evidence run of
this port (v2, v3).** `dspark_ladder_normalization.py` has two independent
halves. `factory_transform()` patches `sdpa_program_factory.cpp` (the
FACTORY) to allocate the dedicated `recip_scratch` CB and set its unpack
mode - this port carried that from the start. `scratch_normalization()`
patches `compute_common.hpp` (the KERNEL) via `native_draft_sdpa.replacements`,
inserting the `qwen_normalize_scratch` helper AND - the part this port was
missing entirely - replacing the actual call site
(`mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(...)`) that performs
the final normalization multiply, so it routes through the dedicated CB
instead of the original shared one. Without that second half, the factory
allocates `recip_scratch` but nothing in the kernel ever reads or writes it:
the original, shared, TF32-truncating reciprocal path stayed in use the
entire time, unconditionally, regardless of Skt or the factory-level patch.
Found this while tracing dspark-ladder-attention-probe.py:77-85 for the
sum-update/score-center port below - `frozen_wide_chunk_scratch.py` only
ever exported `factory_transform`/`factory_scope`/`validate_manifest`,
nothing touching `native_draft_sdpa.replacements` at all.

**Consequence:** every improvement this port showed before now (384 -> 9
failed elements, run 35593521063) is attributable entirely to the widened
fp32-stats condition (`stats_df` now `Float32` at `Sk_chunk_t==32`, which
never matched anything before this port existed - the base
`factory_selector()`-driven clause only ever paired with `Sk_chunk_t==8`)
and the reduced online-softmax iteration count (258 -> 65/66), not to the
scratch-CB reciprocal fix this port is named for. That fix has now actually
been ported, for the first time, in this round.

**Fixed, and the two remaining ladder fixes ported alongside it.**
`frozen_wide_chunk_scratch.py` gained `kernel_scope()` - the missing
kernel-level half, parameterized on the module's own `SKT` constant (patched
at staging time, same mechanism as before). Two new modules,
`frozen_wide_chunk_sum_update.py` (`sum_update_scope()`, a parallel,
independent port of `dspark_ladder_sum_update.scalar_sum_update()` - that
function takes no parameter at all, so this is a from-scratch parameterized
copy, not a call-through) and `frozen_wide_chunk_score_center.py`
(`score_center_scope()`, ported from `dspark_ladder_score_center.
scalar_score_center()`, which IS parameterized by `key_tiles` but refuses
anything outside `(40, 2112)` via its own validation guard - reproduced
without that restriction rather than modifying the ladder's own module),
generated for both accepted Skt values in every case (2080 and 2112 - their
underlying substitution text is a clean literal parameter in both, so
neither needed to refuse 2080).

All four fixes (`scalar_reciprocal()`, already active via
`--scalar-reciprocal`; the two new ones; `kernel_scope()`) are now composed
into `dspark-native-8k-attention-probe.py`'s `main()`, in the ladder's own
order (`dspark-ladder-attention-probe.py:77-85`). A synthetic-fixture test
(`test_frozen_wide_chunk_kernel_composition.py`) confirms every anchor is
found exactly once at the point its own patch runs - no collision with an
earlier patch's output - built from each module's own anchor constants, not
hand-retyped text.

**A real bug this surfaced and fixed:** the first attempt at the
composition substitution targeted `with scalar_reciprocal(),
scoped_stats_pack(),` (the post-`--scalar-reciprocal` text) directly. That
anchor does not exist when `--scalar-reciprocal` is not passed - and the
real `experiment/frozen-64k-target-replay-*` lanes do not pass it
(`qwen-frozen-32k-numerical.yml`'s candidate-array logic is a sequence of
independent, non-`elif` `if`-blocks; only the `reciprocal-eager`/
`reciprocal-replay`/`reciprocal-diagnostics` tag patterns add
`--scalar-reciprocal` - the `target-replay`/`target-replay-scratch`
patterns do not). Since context==65536 staging runs unconditionally for any
flag combination, this would have made every `target-replay`-only staging
fail outright with "anchor missing", even though
`dspark-native-8k-attention-probe.py` is not even the probe those lanes
execute (`target-t16-attention-8k-probe.py` is, per `QWEN_SIM_CASE`) - the
staging-time substitution still has to succeed regardless. Fixed by
targeting the bare `scoped_stats_pack(),` call site instead, which survives
`adapt_scalar_reciprocal()`'s edit unchanged either way. Re-verified with a
real dry run of `--context 65536 --target-replay --probe-seconds 1020`
(no `--scalar-reciprocal`) against the pinned checkout: generates cleanly,
`git diff --check` passes, and the `with` line correctly omits
`scalar_reciprocal()` when it was never requested.

Re-verified end to end: 32768 byte-identical (plain and `--target-replay`),
65536 clean under both Skt knob values and both with/without
`--scalar-reciprocal`, `git diff --check` passing throughout. Still
UNVALIDATED beyond that: none of this has been built or run on hardware or
in the simulator yet.

## 8. Update: 3000 s probe budget, and the target-replay-scratch factory pin
fix (commit `93069ed2`)

Two real failures came back from the full four-fix precision set (section 7)
running in the simulator. `reciprocal-replay-v4` (run 35598117412) timed out
at its 1020 s budget inside `eager_0` - the full composed set is
substantially slower to simulate than the earlier two-fix set the 1020 s
figure was sized for. `target-replay-scratch-v6` (run 35598585759) died at
probe start with `ValueError: Exact pinned SDPA factory required`, unrelated
to timing.

**Probe budget.** `PROBE_SECONDS_CHOICES` (`frozen_recipe_context.py`) gained
`3000` alongside the existing `510`/`1020`, threaded through the same
`--probe-seconds` argparse choice and `adapt_cache_launcher()` path that
already handled the other two values - no new plumbing needed, since that
path already parameterized the `frozen_sim_phase.py --phase probe --seconds
N` literal in `simulator-suite.sh` on this same value. `prepare_seconds`
(the `--phase prepare` step's own budget, fixed at 510) is a separate,
independent phase and does not need to scale with the probe budget - real
dry run confirms it stays `prepare_seconds=510` regardless of the chosen
`--probe-seconds`. Existing values (510, 1020, and the default) produce
byte-identical output to before.

**Target-replay-scratch factory pin.** `frozen_target_replay.py`'s
`adapt_target_probe()` (lines 68-87 at the pinned revision) injects, into
`target-t16-attention-8k-probe.py`'s compact-scratch branch
(`QWEN_FROZEN_TARGET_SCRATCH=1`), a plain `from dspark_fp32_build import
validate_manifest` call used to audit the build that produced the binary
under test. That plain, unwrapped `validate_manifest` cannot reconstruct a
scratch-CB-patched `sdpa_program_factory.cpp`: its own reconstruction check
(undo the condition-text swap, re-transform, compare against
`SOURCE_SHA256`) leaves the scratch-CB layer in place, so the intermediate
"original" never matches, and `dspark_fp32_intermediates.transform()`'s
guard raises exactly the observed error - the same class of bug
`frozen_wide_chunk_scratch.factory_scope()` was built to fix for the
numerical probe (section 3/7), just never applied to the target probe's own,
separate import of the same function.

Confirmed safe to fix with the same redirect rather than a per-context
factory-hash pin: `sdpa_tree_scratch.py`'s own pins (`HASHES`,
`PATCHED_FACTORY_SHA256`) are entirely about
`sdpa_decode_program_factory.cpp` - the DECODE kernel family
`attention_replay.py`'s `ReplayAttentionReader` actually computes through
(`paged_scaled_dot_product_attention_decode`), a completely different file
from `sdpa_program_factory.cpp`, which this port's wide-chunk patch touches.
The compact-scratch target probe's own numerics and scratch layout (chunk
sizes, CB ids) are not coupled to `sdpa_program_factory.cpp` at all; its
`validate_manifest` call is a build-provenance audit of a binary the target
probe doesn't otherwise depend on, not a source of correctness assumptions.
`_patch_target_probe()` (`frozen_wide_chunk_normalization.py`) redirects the
import through `frozen_wide_chunk_scratch.validate_manifest` - the same
already-tested redirect used for `dspark-native-8k-attention-probe.py` -
applied only when `target-t16-attention-8k-probe.py` is present in `sources`
(i.e. only when `--target-replay` was passed; a no-op omission otherwise,
and a no-op for every context other than 65536 regardless).

**A pre-existing test gap found and fixed while re-running the full suite:**
`test_frozen_recipe_context.py`'s `test_deployment_preserves_historical_
hardware_runtime` asserted an incomplete expected file set for 65536 -
missing `attention_replay.py` (staged by `adapt_replay_k_chunk`), the four
files the unconditional verbatim-copy loop added for the fused MLP input
reader (`fused_1d_input.cpp`, `fused_t16_admission.py`,
`fused-t16-target-simulator.json`, `fused-t16-target-simulator.exit-status`),
and the three `frozen_wide_chunk_*` fix modules the `KERNEL_FIX_MODULES`
loop stages. A latent gap from the section-7 round, never caught because
that round's own new tests didn't touch this older assertion. Fixed by
adding the missing names; re-verified.

**Re-verified via real dry run** against the pinned checkout (fresh clone,
`git checkout 8c102b20...`, comparing the last-committed adapter code
against the code with this round's changes): 65536 + `--target-replay`
staging produces exactly one changed file
(`target-t16-attention-8k-probe.py`, the single import-line redirect shown
above) versus before; 32768 stays byte-identical, both plain and under
`--target-replay` (`diff -rq` empty, manifest `before`/`after`/`geometry`
hashes equal); `git diff --check` clean throughout; 65536 +
`--scalar-reciprocal --probe-seconds 3000` (the `reciprocal-replay-v6`
lane's flags) stages cleanly with `--seconds 3000` correctly threaded into
`simulator-suite.sh` and `prepare_seconds` unchanged at 510. 95 tests pass
locally across `test_frozen_wide_chunk_normalization.py`,
`test_frozen_wide_chunk_scratch.py`, `test_frozen_wide_chunk_sum_update.py`,
`test_frozen_wide_chunk_score_center.py`,
`test_frozen_wide_chunk_kernel_composition.py`,
`test_frozen_wide_chunk_replay.py`, `test_frozen_recipe_context.py`,
`test_frozen_combined_adapters.py` and `test_frozen_target_replay.py`.

**Since committed and iterated on the workflow side (no further adapter
changes needed):** `reciprocal-replay-v6`/`v7` and
`target-replay-scratch-v7`/`v8` retry tags were added
(`qwen-frozen-32k-numerical.yml`), the job timeout raised to 65 minutes and
the run-simulator wrapper's `limit` raised to 3400 s for 64k tags
(`ea7d2d63`, `3222bcfd`, `fb6ceb60`). `target-replay-scratch-v9`/`v10` tags
followed with a further retry at `--probe-seconds 3000` after `v7`
progressed past two native references before hitting the then-620 s
target-probe wrapper limit (`710148c7`); a bash array-editing bug in the
`v9`/`v10` candidate construction (`"${candidate[@]/--probe-seconds/}"`
against an array that no longer contained that literal) was found and fixed
by building the `--target-replay --probe-seconds 3000` array directly,
`bash -n` checked (`c43ca1cb`). All of this is workflow-YAML-only; none of
it implicates `frozen_wide_chunk_normalization.py` or the fixes above. Lane
outcomes for `v9`/`v10` and `reciprocal-replay-v6`/`v7` are not yet known
from this side - pending CI results.
