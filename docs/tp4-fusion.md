# The op-fusion programme (tp4/fusion-1)

Branch `tp4/fusion-1` is `tp4/next2-1` (the octo-2 and prod-fixes stack) plus the integration scaffolding below and, as each lands with a green CPU suite, the work
packages. Every package fuses ops of the packed speculative path behind a default-off strict flag, byte-identical off, with an `_AUDIT` twin flag that compares the lever
against the served composition on the device and logs `exact=True` or a mismatch, a `<pkg>_smoke.py` that logs `[PINDIAG] tp4 <lever> engaged|fell back|audit N exact=True`,
and a kill path back to the served ops. Drafter-side levers cannot change a token (lossless greedy verification commits only the target's argmax); they can change tau.

| Package | Lever |
| --- | --- |
| WP1 | S1 shard argmax (the sampler's untilize, argmax and gather as one scan) |
| WP2 | page-parallel K/V cache writer |
| WP5 | per-call collective options |
| WP6 | drafter local reduce, residual and SwiGLU kernels, fused gate and up |
| WP7 | drafter permutation kernels (K/V assembly, query fold and unfold), fused q, k and v, one 64-row head |

## The integrator's files

A package edits only its own modules and writes `scripts/ci/fusion-wp/<WP>.json` (schema: `scripts/ci/fusion-wp/SCHEMA.md`). The integrator's generators place the manifest in the
shared files, inside fenced blocks that nothing else edits:

| Generator | Writes |
| --- | --- |
| `make_fusion_profiles.py` | per lever two gate-only twins of the production profile (`...-w2-er-fx-<id>` and `...-fx-<id>-audit`) in `qwen_c2_profiles.json`; the marker tables in `c2_smoke_check.py` (`FUSION_LEVERS`, `FUSION_AUDITS`, read by `fusion_problems`); the image copy lines in `docker/qwen-c2-overlay.txt` (the one list); the CPU allowlist in `.github/workflows/qwen-integration-cpu.yml`; twin rows in `tp_addresses.py` |
| `make_fusion_jobs.py` | the card pack `scripts/ci/references/fusion-jobs`: B0 build (`tp4-fusion-1`), X0 status, rescan and reset, per lever an audited attach and a timed control/lever ABAB at eight live (`concurrent8_steady`), then Z |

After a merge, run both with `--write` (`--check` fails when a block is stale, `make_fusion_profiles.py --report` lists where each manifest entry landed). `test_fusion_wp`
holds that the output is deterministic and idempotent, that every manifest entry lands and that a manifest it cannot place is refused by name; `test_tp4_fusion_jobs` holds the
pack. The generated twins are exempt from the profile-enumerating tests through `profile_twins.py`.

## Cross-package hooks

Two packages each need a line in a file the other owns. The integrator's hooks (one commit, `test_fusion_hooks` holds them): WP6's three drafter levers in
`draft_attention_branch.execute_attention_branch` (`QWEN_FAST_DRAFT_MM_GRID` widens the wo projection's grid, `QWEN_FAST_DRAFT_REDUCE` hands the gather to
`draft_reduce_tp.choose`, `QWEN_FAST_DRAFT_TAIL` replaces the residual tail), and WP7's permutation levers at the octo shapes (`octo_draft_tp.octo_fold_query` and
`octo_unfold_output`, `QWEN_FAST_DRAFT_PERMUTE`, `site='octo'`). Both are lazy, strict and flag-guarded; flags off, the op sequence is the one the files had before.

## The trace-capture gate

Two audits of the programme broke a rule of the device on a card that no CPU test could see: one synchronized inside the fused-commit capture, the other launched an audit
program that had never run eagerly ("Cannot load new binaries during trace capture"). `scripts/ci/capture_rules.py` is the CPU model of exactly those two rules, a wrapper any
fake ttnn can be put behind: while a capture is open (between `begin_trace_capture` and `end_trace_capture`) it refuses (a) every host round trip (`synchronize_device`,
`to_torch`, a host write to a device tensor) and (b) every op or program launch whose key (op, tensor shapes / dtypes / layouts, configs and, for `generic_op`, the kernels'
sources, compile-time arguments, defines, circular buffers and core ranges, never the runtime arguments or addresses) did not run eagerly before it. Every violation is also
recorded, so a lever that catches the exception is still found.

`test_fusion_capture_rules` drives each lever's lifecycle through it with its audit on, in the order the stack gives it (the eager warm forward, the capture, replays, the audit after
a replay, the release): the K/V page writer (warm ordered writer, captured per-user writer), the MLP configuration, the shard argmax with the tree fold, the CCL options (reduce-scatter
and the norms' gather), WP6's four drafter levers in the single and quad MLP branch and in the fused commit's segment-by-segment order, WP7's permutation sites (K/V, fold, unfold at the
pair, quad and octo shapes), the fused q|k|v projection and the 64-row head. A census lists every synchronize / readback / host write in the lever modules with the reason it cannot
run in a capture, and a coverage test requires a scenario for every audit flag of the audited combined twin. Run against the pre-fix heads, the gate fails exactly where the cards
did: WP6's audits at 9175d883 (a synchronize inside the capture) and WP2's audited writer at 96d674c7 (its audit programs inside the capture).

## The pack

`scripts/ci/references/fusion-jobs/ORDER.txt` runs B0 (the image `tp4-fusion-1`), X0, every lever's audited attach, every lever's timed control/lever ABAB at eight live,
then Z. A lever that cannot engage yet (WP2 until its card-M record lands) runs last in each section (`pack_last`). The packages' own card-M and detail templates are in their
folders beside the pack.

## Reading a card run

`tp4_profile_report.py` classifies a replay by what it holds: the multi-SDPA 64-row block (one SDPA launch per attention layer, no named conv-gates launch) is the packed
verify, and the 4-row lone step (four SDPA launches per attention layer, one conv-gates launch) is not. Timed pairs are read PAIRED per round with
`w2ln_timing_compare.py pair`, never by unpaired medians. The cards expose a 13x10 compute grid since the firmware unlock: read it from the run, never assume 11x10.

### The parked engines' DRAM band on a lever arm

The engine-reuse smoke rule bounds what eight parked engines take of a chip: 0.46-0.51 GB each (`parked_judge.ENGINE_GB`, measured 0.468-0.495). An arm that turns on
levers of this programme (`c2_smoke_check.fusion_arm_flags`) is judged against `parked_judge.ENGINE_GB_LEVERS` instead; production and every other profile keep the old band.

1. A parked engine's DRAM is the allocator growth of its captured verify graph. The levers fuse or remove ops of that graph and keep intermediates in L1, and none of the
   levers in the combined arm adds a per-engine buffer, so an engine of any subset of them is no larger than the production engine and no smaller than the engine of the whole set.
2. The ceiling therefore stays at 0.51 GB: a lever that adds per-engine DRAM (the fused MLP gate|up op, if it ever joins) fails it and gets its own measured figure. The K/V page
   writer removes a DRAM copy (it reads the prepared K/V from L1, its circular buffers are L1) and adds none.
3. The first combined set (no page writer, MLP config l1, the served CCL set) measured 3.346 GB for eight engines, 0.418 GB each, identically on all four chips in two runs, which is 0.050-0.077 GB (10-16 percent) under the production engine
   and failed the production floor of 3.68 GB.
4. The floor is that measurement less 9 percent, 0.38 GB, because the set that ships differs from the measured one (MLP config g3u4d3, CCL rs-c1, the page writer, the capture
   fix). Tighten it to the measurement less 2 percent once the final set has run.
5. An audited lever arm holds the served composition beside each audited lever, which no run has measured: it has no per-engine ceiling, the audited free-DRAM floor still
   binds (3.6 GB free after eight engines), and the smoke prints `engine_gb` per chip so the first completed audited run sets the ceiling.
