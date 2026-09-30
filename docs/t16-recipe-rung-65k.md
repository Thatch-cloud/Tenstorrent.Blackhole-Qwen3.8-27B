# Staging the T16 fast-path recipe at context 65536 (four users)

Scope: what it takes to move the m3native gate (`scripts/ci/lever_n_m3native_gate.py`,
`lever_n_m3native_run_arm.sh`, `.github/workflows/qwen-lever-n-m3native-gate.yml`) off its
current 32768-only staged recipe onto a 65536 rung, for four users, without touching the
32768 default. Corrects and extends `docs/lever-n-131k-attach-arm.md`.

## 1. Every place the pin actually lives

The "CLI-locked to 32768" framing undersells how many independent pins there are. Five,
not one, all need lifting together:

1. **`frozen_recipe_context.py:196-198`** - the argparse guard on the offline staging tool:
   `options.combined_runtime and (options.context != 32768 or ...)`. Blocks invoking
   `--combined-runtime --context 65536` at all.
2. **`frozen_combined_runtime.py:76-83`** (`qualify()`) - `selected_geometry()['context'] !=
   32768`, gated by `QWEN_DSPARK_REQUEST_CONTEXT` at runtime. Also line 88 hardcodes
   `context=32768` in the call to `qualify_components`.
3. **`frozen_combined_gate.py:29-30`** (`qualify()`) - `if type(context) is not int or context
   != 32768: raise ValueError('Only 32768 has both retained candidate qualifications')`. This
   is the evidence-hash gate: `REPORTS` (lines 11-15) pins `draft-numerical.json`,
   `draft-diagnostics.json`, `target-replay.json` to three fixed SHA256 values that are
   qualification **evidence from the actual 32768 hardware/simulator runs** - not derivable
   from geometry, and not something a code edit alone can produce for 65536.
4. **`frozen_combined_runtime.py:58-68`** (`validate_target_option`) - `position != 32768`
   hardcoded, independent of (2)'s env-driven guard.
5. **`frozen_combined_adapters.py`** - `adapt_combined_sources` (lines 10-91) and
   `adapt_admission` (94-110) hardcode `8192`/`8448`->`32768`/`33024` literal replacements into
   six files (`dspark_8k_admission.py`, `dspark_8k_entry.py`, `target_t16_attention_gate.py`,
   `dspark_8k_build.py`, `dspark_runtime_cache.py`, `coding_context_request.py`), never
   referencing `selected_geometry()`/`geometry()` - contrast `frozen_runtime_context.py:10-52`,
   which already generalized the sibling files (`dspark_context_selection.py`,
   `dspark_8k_scope.py`, `dspark-target-hardware.py`) this same way. Needs the same treatment.
6. **`target_t16_attention_gate.py`'s base clause** (`rows != 16 or position != 4096 or ...`,
   historical-checkout shape, confirmed identical via `git show
   8c102b20:scripts/ci/target_t16_attention_gate.py`) - the fallback any `request_context()`
   other than 32768 falls into, since `frozen_combined_adapters.py:53-57` only rewrites the
   `== 8192` dispatch line to `== 32768` and never touches the base clause. At 65536 this base
   clause fires and raises before `frozen_combined_runtime.validate_target_option` is even
   reached - so this file needs its own `request_context() == 65536:` branch (or a
   geometry-driven rewrite), staged the same way the 32768 branch is today. Easy to mistake
   for "not a pin" since it looks unrelated to the combined-runtime work, but it is one.

**Not a pin, and no rebuild needed:** the hashed hardware runtime
(`dflash_combined_sim_runtime.py`) patches the SDPA factory with a literal C++ condition
(`REPLACEMENT`, lines 18-25) gated on `Skt == 144 || 272 || 528 || 1040 || 2064 || 4112 ||
8208`. Computing `geometry(context)['padded_keys'] // 32` from `frozen_context_geometry.py:9-19`
for every rung in `CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072, 261888, 262144)`
(`frozen_context_geometry.py:6`) reproduces that exact list - **65536 maps to `Skt == 2064`,
already present**, and 131072 to `4112`, also present. `factory_selector()`
(`frozen_context_geometry.py:29-31`) generates precisely this string; whoever built
`REPLACEMENT` used the full-ladder output, not just the 32768 term. So `BINARY_SHA256`
(`dflash_combined_sim_runtime.py:10`) and `COMBINED_FACTORY`
(`dflash_combined_sim_runtime.py:14`) need no change, and the K64 graft's binary-override pin
in `lever_n_m3native_run_arm.sh:89-98` is unaffected. This is the single biggest de-risking
fact for this rung: the previously-assumed "multi-day kernel rebuild" is not required for 65536.

**Also not blockers, confirmed context-independent, no change needed:**
- `serving_fast_policy.py:68-109`/`118-145` - bounds (`MINIMUM_MODEL_LEN=4352`,
  `OUTPUT_BUDGET=256`), not equalities; `model_len % cache.block_size` is the only
  context-shaped check and 65792 (65536+256) % 64 == 0.
- `attention_request_plan.capture_plan` / `verifier_engine.capture_widths`
  (`verifier_engine.py:64-75`) - driven by `position`/`capacity` arguments, not literals.
- GDN native slot allocation (`gdn_snapshot.py:4-20`, `ActiveSnapshot.allocate`) - fixed-size
  recurrent/conv state per layer, context-independent by construction (why GDN was chosen over
  full attention for those layers).
- `dspark_context_selection.py`, once staged through `frozen_runtime_context.py:31-36`,
  already accepts every `CONTEXTS` string for `QWEN_DSPARK_REQUEST_CONTEXT` - load-bearing
  once (5) is fixed, not cosmetic.
- `dflash_prefill_window.MAX_POSITION` (`dflash_prefill_window.py:15`) - already
  `QWEN_FAST_MAX_POSITION`-driven, and `lever_n_m3native_run_arm.sh:43-49` already sets it to
  `$context` for any context other than 33024. No script change needed.
- `longctx_cycle_bench.py:125-126` (`--context` default 163840) - already fully generic; only
  the frozen-recipe chain above is context-locked.
- `len(prompt) != 4096` in `dflash_combined_request.py:80-81` (`measure_combined_dflash`) is
  real but scoped to an **offline hardware-probe harness** (`run-dspark-hardware.sh`) -
  `dflash_request_runtime.py`, the file actually shipped per the Dockerfile, never imports it.

## 2. What needs new evidence, and which lane produces it

Widening (1)/(2)/(4)/(5)/(6) is a mechanical refactor - follow the `frozen_runtime_context.py`
pattern (`selected_geometry()`/`geometry()` instead of literals) and update the tests that
assert the literals (`test_frozen_recipe_context.py`, `test_frozen_combined_adapters.py`,
`test_frozen_combined_history.py`, `test_target_t16_context_cli.py`, and others in the
33024-literal file list). That is not the long pole.

The long pole is **new qualification evidence**, because `frozen_combined_gate.REPORTS`
(`frozen_combined_gate.py:11-15`) pins content hashes of actual run outputs, not formulas:

1. Rerun the offline historical-checkout staging + hardware probe at the new context, mirroring
   `.github/workflows/qwen-frozen-combined.yml:76-79` and `:128-146` exactly but with
   `--context 65536` / `QWEN_DSPARK_REQUEST_CONTEXT: '65536'` instead of `32768` (that value is
   hardcoded at lines 77 and 134 today, both need to become workflow inputs or a new tag
   branch). This produces new `draft-numerical.json` / `draft-diagnostics.json` evidence.
2. The target-replay counterpart (`target-replay.json`,
   `frozen_target_replay.validate_target_report`) needs the equivalent hardware run - the
   pipeline downloads pre-existing run artifacts by hardcoded run ID
   (`qwen-frozen-combined.yml:41-43`); a 65536 rerun needs fresh run IDs substituted throughout.
3. Update `frozen_combined_gate.REPORTS` (three new SHA256s) and widen its `context != 32768`
   guard to accept 65536 too (a set, not a second literal swap, so 32768 keeps working).
4. Single-user references at 65536: run `qwen-lever-n-m3native-gate.yml` with
   `M3NATIVE_USERS=1`, `M3NATIVE_CONTEXT=65792`, `M3NATIVE_PROMPT_TOKENS=65536`,
   `M3NATIVE_ALLOW_MISSING_REFERENCES=1` - the same pattern as the existing `-v15`/`-v16` 131k
   attach arms (`.github/workflows/qwen-lever-n-m3native-gate.yml:155-167`), just at 65536.
   Harvest the report JSON into `scripts/ci/references/packed-gate/single-user-<base>-<run>.json`
   (see `lever_n_m3native_gate.py:80-107`) for each of the four prompt bases (1000-1003), same
   as the existing 32768 references (`single-user-1001-35493236124.json` etc.).
5. Once all four references exist, rerun the four-user gate *without*
   `--allow-missing-references` as the real acceptance run - turning "attach and report" into
   a pass/fail token-exact gate, per `evaluate_gate` (`lever_n_m3native_gate.py:216-234`).

Tag lanes: `qwen-frozen-combined.yml` triggers on `experiment/frozen-32k-combined-v*` (and
sibling profile tags, hardcoded at line 4) - a 65536 rung needs new tag names (e.g.
`experiment/frozen-65k-combined-v1`) added, since tags are allowlisted by exact string.
`qwen-lever-n-m3native-gate.yml` triggers on `experiment/lever-n-m3native-v*` (line 4) and is
already generic enough to reuse via a new `-vN` case arm (mirroring `-v15`/`-v16` at
lines 166-167), no workflow-file rename needed.

## 3. Where the staged tree lives in the serving image, without touching the 32768 default

The 32768 tree is not baked from source at image-build time. `docker/qwen-fast-serving.Dockerfile`
does `COPY bundle/experiment-scripts /experiment-scripts` (line 6) - `bundle/` is a pre-built
artifact assembled by `scripts/ci/serving_bundle.py` (invoked from
`.github/workflows/qwen-serving-bundle.yml:30-32`), gated on a `staged_files` inventory
(`serving_bundle.py:32-41`) whose `required` set is exactly the staged/adapted 32768 files from
section 1 (`model_batch.py`, `verifier_engine.py`, `dflash_combined_request.py`,
`draft_kv_slide.cpp`, `draft_kv_slide_gate.py`, `frozen_combined_runtime.py`,
`target_t16_attention_gate.py`, `dflash_t16_native_scope.py`), fingerprint-locked against a
prior inventory run. The Dockerfile then overlays ~30 more `scripts/ci/*.py` files individually
(lines 8-30) from the checked-out repo, on top of the bundled tree, at the same path.

To add 65536 **without changing the default**: once section 1's literals are generalized to
`selected_geometry()`, there is only ever *one* staged tree, parameterized at runtime by
`QWEN_DSPARK_REQUEST_CONTEXT` - the same env var already selects rung-specific behavior via
`factory_selector()`/`geometry()` elsewhere. No second bundle directory, no `serving_bundle.py`
or Dockerfile COPY-line change is needed; risk is confined to the adapter files themselves,
following the `QWEN_FAST_*`/`QWEN_DSPARK_*` measurement-gate convention already used for
`QWEN_FAST_MAX_POSITION`.

Verify, don't assume: `qwen-serving-bundle.yml:31` reads `--staged
"$GITHUB_WORKSPACE/read-combined"`, but no step in that workflow checks out or downloads a
`read-combined` directory - it must rely on self-hosted-runner workspace persistence from a
prior job on `thatch-build-amd64-02-cp-temp`. Confirm that dependency before adding a 65536 lane
on top of it - the same class of implicit cross-workflow coupling as the `--served-model-name`
false negative at `lever_n_m3native_gate.py:116-120`.

## 4. Trace region and DRAM budget at 65536 x 4

Source numbers: `docs/batch-spec-tasks-2026-09-19.md:2004` (KV formula: `(pages, 2, 64, 256)
bfloat8_b`, 34,816 B/page, 516 pages -> 574.9 MB/user at context 32768), `:2638` (29.99 GB
allocated after attach at 33k x 4, 3.12 GB free, of `:2062`'s 33.10 GB usable DRAM/chip), and
`docs/concurrent-fast-path-programme-2026-09-19.md:218` (131k x 4 projected ~36.8 GB).

Pages/user = `ceil((context+256)/64)`. KV/user = `34816 * pages * 32` bytes (32 = full-attention
layers with a KV cache; GDN layers' state is context-independent, section 1). At 32768: 516
pages -> 574.9 MB/user -> 4 users = 2.30 GB, matching the measured figure exactly. At 131072:
2052 pages -> 2.286 GB/user -> 4 users = 9.15 GB, and 27.69 (measured 29.99 - 2.30 non-KV
baseline) + 9.15 = 36.84 GB - matches the doc's independently-stated "~36.8 GB", validating the
"non-KV overhead is ~constant across context" assumption this projection depends on.

At **65536**: 1028 pages -> 1.1455 GB/user -> 4 users = **4.58 GB KV** (not the naive "~2x
32768" you'd get from context alone - it's 1.99x, since pages scale with context but the +256
padding term does not shrink the ratio much at this size). Projected total = 27.69 (non-KV,
assumed constant, includes the existing 1 GiB default trace region already) + 4.58 = **32.27 GB
used, ~0.83 GB free against the 33.10 GB ceiling.**

That is thin - about a quarter of the 3.12 GB headroom measured at 32768 - and "non-KV is
constant" is the load-bearing, unverified assumption (no rung between 32768 and 131072 has been
measured). Take it as a plan, not a guarantee. The default 1 GiB trace region
(`lever_n_m3native_gate.py:264-265`) is already inside that 0.83 GB projection, so the plain
(eager-proposal) arm needs no separate check. The 512 MiB traced-proposal arm
(`M3NATIVE_TRACED_PROPOSAL=1`, `lever_n_m3native_run_arm.sh:160-167`) trades that region for
larger per-engine proposal uploads (0.80 GB vs 0.58 GB/engine) - close to a wash, needs its own
on-hardware number.

## 5. Effort and the single riskiest step

- Guard/adapter generalization (section 1, items 1/2/4/5/6) plus test updates: **0.5-1 day**,
  mechanical once the `frozen_runtime_context.py` pattern is followed.
- New offline numerical/hardware qualification run at 65536 for `frozen_combined_gate.REPORTS`
  (section 2, items 1-3): CI/hardware wall time on the exclusive `qwen-two-p150a-exclusive`
  group every lane here shares - **0.5-1 day**, contingent on runner availability.
- Single-user + four-user 65536 references and the real acceptance gate (section 2, items 4-5):
  **0.5-1 day**.
- DRAM verification and any trim if the projected 0.83 GB margin (section 4) doesn't hold on
  hardware: **0-2 days**, unknown until measured.

**Total: roughly 2-4 days if the DRAM margin holds, more if not** - substantially less than
131072 (which additionally needs the already-tracked 4-6 GB DRAM trim, task #35).

**Single riskiest step: the new draft-numerical/target-replay evidence in section 2.** Everything
else is either already generic (section 1's "not blockers") or a mechanical literal swap with a
known-good pattern. Requalifying the fp32-intermediates draft path and the T16 target replay at
65536 is a genuine unknown - the binary already carries the `Skt == 2064` branch (the *kernel*
is not new), but nothing has exercised it end-to-end, and `frozen_combined_gate.qualify`
(lines 28-42) refuses until real runs produce reports that hash-match new pins. Pass/fail
unknown, not a code change with a known outcome - the same risk class the team's plan already
flags as "multi-day, long pole" for 131072 (`concurrent-fast-path-programme-2026-09-19.md:219`),
just with a much smaller, binary-de-risked blast radius at 65536.
