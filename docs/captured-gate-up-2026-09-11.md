# Captured gate/up fusion: hardware result

## Combined T16 runtime

[Run 34585332201](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34585332201)
passed at `f301272`, after guarded fabric recovery. Independent validation checks
743 source files, complete request recomputation and correctness/weight gates.

| One coding stream, merge intervals | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: |
| Combined control | 3344.93 | 4096 | 100.45 |
| T16 gate/up fusion | 3302.00 | 4096 | 102.44 |

Both arms commit 242 timed tokens, accepting 222/330 drafts. The measured TG
gain is 1.98%, not a route to 200 TG by itself. Mean complete cycle decreases
109.459 to 107.340 ms, but verifier/readback only decreases 69.157 to 68.512 ms;
draft and commit variation contributes to the overall result. Setup-inclusive
mean increases 5988.16 to 6274.35 ms. No serving or held-out quality qualification.

A same-revision repeat is [34586017906](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34586017906).
It passed independent validation of 743 source files and complete request
recomputation. Control PP/CTX/TG is 3325.70 / 4096 / 100.47; fusion is
3354.98 / 4096 / 101.88 (+1.40% TG). Acceptance remains 222/330, with 242
timed committed tokens per arm. Verifier/readback falls 69.211 to 68.441 ms;
complete cycle falls 109.441 to 107.926 ms. Setup-inclusive mean is 6020.81
versus 6074.33 ms.

The two paired runs support a small repeatable improvement on this prompt.
Keep it as an opt-in combined-runtime candidate, not a serving default or a
broad coding-quality claim. At 11 committed tokens per block, 200 TG requires
a 55 ms cycle; the observed roughly 108 ms cycle still needs a major reduction.
Further work must target larger verifier/drafter costs rather than repeat this
same small projection experiment.

Repeat report SHA256: `2fc8ce17545ba1aa6defb5c323428c54b7d662c921a0667a758c07643ce55d32`.

Report SHA256: `74696c102735f26f9191a955dc5c3df7287afd5ff0f1309b31b10b4b2469b7d1`.

## Earlier component timing

### Untuned coding screen

[Stable-unique run 34586808967](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34586808967)
passes independent source/request validation and four isolated functional cases,
including unchanged inputs. At CTX4096 / one stream, control PP/TG is
3298.23 / 118.02; fusion PP/TG is 3298.34 / 119.57. Both accept 98/120 drafts
and commit 104 timed tokens. This task-specific result does not replace the
merge-intervals baseline or establish broad held-out coding quality.

Report SHA256: `fd93f1c3c9fb01ace53a3551f528976907190f5f7a52b7dca00da9c170dd20ae`.

[Run-length encoding 34587483522](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34587483522)
also passes independent source/request validation and four isolated functional
cases. At CTX4096 / one stream, control PP/TG is 3332.30 / 117.72; fusion is
3311.14 / 121.68. Both accept 196/240 drafts and commit 210 timed tokens.
Setup-inclusive mean is 5571.42 versus 5843.45 ms. This is a different coding
task, not a higher result for the merge-intervals benchmark.

Report SHA256: `a47e3e80ccd7836f177a0577e4cf2a359d442847deab623b73d63a7ac1350988`.

[Rotate-right 34588114930](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34588114930)
passes independent validation of 743 sources, complete request recomputation and
five isolated functional cases. Control PP/CTX/TG is 3341.64 / 4096 / 82.58;
fusion is 3369.42 / 4096 / 84.71. Each arm commits 128 timed tokens and accepts
116/210 drafts (55.24%). Setup-inclusive mean is 5645.17 versus 5649.14 ms.
The three-task screen now passes all 13 functional cases, but its task-dependent
84.71–121.68 TG does not qualify the 200-TG objective or broad coding quality.

Report SHA256: `a13d5b6f73246506095a27bed405fd32b0114ce432964c0b227b733a70f1d99d`.

[Run 34553944965](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34553944965)
completed the previously blocked captured-fusion test on revision
`f4db588c85d3a99d08c8d141c6bb50ec52f11c67`.

| Projection rows | Native median block ms | Fused median block ms | Latency change |
| --- | ---: | ---: | ---: |
| 1 | 0.206101 | 0.197591 | -4.13% |
| 8 | 0.207376 | 0.197311 | -4.85% |
| 32 | 0.206181 | 0.195991 | -4.94% |

Each width has three A/B/B/A blocks. Timings cover blocking trace execution,
not graph construction, uploads, validation or a complete MLP/request. The
geometry-matched weights are from the pinned DFlash fixture, not all 64 target
layers. T16 is checked eagerly but not timed in this trace matrix.

Independent local checks reproduce every reported mean from all 36 raw samples,
verify the four byte-exact packed-weight checks, six eager widths on both chips,
36 replay comparisons and 12 stale-input controls. Generated kernel manifests
match the retained simulator report; current reader and trace-helper hashes match
the hardware report. The trace helper differs from its earlier simulator version
only by the reviewed opt-in hardware timing loop and associated result fields.

The earlier eager regression remains valid: eager execution included constructing
program descriptors and arguments. This trace result supports further integration
testing, not a claim that host overhead explains every earlier regression.

**Decision:** modest component win, not a solution to the 200 TG gap. Reuse the
existing rounded epilogue rather than inventing a concatenated/BF16-post-SiLU
replacement. Any adoption requires target-weight packing checks on all layers,
T16 captured correctness, and a matched combined-runtime PP / CTX / TG result.
Keep both no-copy candidates disabled. Do not multiply a projection delta by 64
and present it as measured end-to-end improvement.

## Target integration admission

The T16 captured simulator extension retains the original T1/T8/T32 checks
and adds changing-input T16 replay. The local run passed T16 replay but was
deliberately stopped before full completion to free the developer PC.
The full test moved to [CPU-only CI](ttsim-ci.md).

CI run [34558325869](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34558325869)
passed on `db4c7c0`: exit 0, four byte-exact packed-weight checks, all six eager
widths on both chips, and changing-input replay at T1/T8/T16/T32. Independent
artifact validation checks all 48 replay combinations and 16 stale controls,
plus the trace-helper hash. This remains the non-approximate component gate,
not target-mode or complete-runtime qualification.

Report SHA256: `3b417d24dc125b35d336cf570794a3c41c1718ada39042eeb9078f9f165b2d07`.

The separate target-math run is
[34559222228](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34559222228),
revision `3b3de38`; it passed with exit 0. Independent artifact checks confirm
`math_approx_mode=True` in the report and all six kernel manifests, 48 exact
replay comparisons at T1/T8/T16/T32, 16 stale-input controls, 12 eager checks,
four byte-exact packed-weight checks and the trace-helper hash.

Target-math report SHA256:
`f155635711e3617cbdb9566a3fdd19e8ab62d9dc7d9eacbf684b9343ab8bd9bb`.
This closes the synthetic/geometry-matched target-mode simulator gap, not the
all-layer target-weight or combined-runtime hardware gates. The earlier 4-5%
hardware component gain was measured in non-approximate mode; do not transfer
that timing claim to this newly qualified mode.

Source inspection found an additional numerical-mode admission gap. The probe's
native control and `FusedProjection` explicitly use `math_approx_mode=False`.
The retained target `mlp.py` constructs its decode compute configuration without
that argument; querying the local TTNN constructor confirms its default is
`True`. The existing streamed-verifier guard also requires `True`.

The measured component result therefore compares matched non-approximate arms,
not yet the target's exact compute configuration. Before runtime integration,
compare against the native target mode and record the actual hardware model
configuration. Do not silently change the target control to make fusion pass.
If results differ, preserve the target mode in a separately validated candidate.
Existing packed target weights still need per-layer byte-exact qualification.

Report SHA256: `b25cf1baa36e3b6aca6ec35a4ca0e27fd0a64e9f4050cfa76c1b378ecd61b154`.
