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

## The pack

`scripts/ci/references/fusion-jobs/ORDER.txt` runs B0 (the image `tp4-fusion-1`), X0, every lever's audited attach, every lever's timed control/lever ABAB at eight live,
then Z. A lever that cannot engage yet (WP2 until its card-M record lands) runs last in each section (`pack_last`). The packages' own card-M and detail templates are in their
folders beside the pack.

## Reading a card run

`tp4_profile_report.py` classifies a replay by what it holds: the multi-SDPA 64-row block (one SDPA launch per attention layer, no named conv-gates launch) is the packed
verify, and the 4-row lone step (four SDPA launches per attention layer, one conv-gates launch) is not. Timed pairs are read PAIRED per round with
`w2ln_timing_compare.py pair`, never by unpaired medians. The cards expose a 13x10 compute grid since the firmware unlock: read it from the run, never assume 11x10.
