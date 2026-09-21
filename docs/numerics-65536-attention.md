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
