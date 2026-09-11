# Captured gate/up fusion: hardware result

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

Report SHA256: `b25cf1baa36e3b6aca6ec35a4ca0e27fd0a64e9f4050cfa76c1b378ecd61b154`.
