# Learned DSpark projection and normalization

**Next step toward the complete learned drafter, not a new speed result.**
The selector, rotary and attention components already have separate simulator
evidence. This test starts connecting the actual backbone weights to the frozen,
upstream-checked CPU reference.

| Item | Scope |
| --- | --- |
| Learned FC | All 5,120 outputs and all 25,600 inputs; no neuron or K-dimension slice |
| Inputs | Taps 5, 19, 33, 47, 61; each hidden dimension split across two chips |
| Fixtures | Two original CPU-reference patterns, each with 32 feature rows |
| Parameters | Complete pinned BF16 `fc.weight` and `hidden_norm.weight` |
| Projection | Native HiFi4 matmul with FP32 partials; no weight quantization |
| Normalization | FP32 sum, BF16 projection, unweighted BF16 RMSNorm, separate BF16 gamma product |
| Comparator | Frozen learned CPU backbone, independently matched to upstream at 48 stages |

The 32 rows are historical feature rows, not concurrent requests. DSpark rounds
the normalized values **before** applying its learned RMS weight. Reusing the
existing fused weighted-normalization helper would change that arithmetic.
Host tests preserve the separate casts and distinguish the two policies.

## Simulator boundary

Projection and normalization have **separate** changed-input traces. The host
stages the actual measured FP32 partials from both chips into the normalization
test; it does not substitute CPU-golden projection values. Each chip then performs
the sum and normalization on device.

This is not a fabric test or a complete captured projection pipeline. The report
and qualifier require `fabric_tested=false`, `full_pipeline_captured=false`,
`target_integrated=false` and `eligible_for_hardware=false`. A later real collective
and learned-layer integration gate remain mandatory.

| Required checks | Count |
| --- | ---: |
| Learned eager stages, including frozen backbone output | 32 |
| Exact changed-input replays and stable bindings | 36 |
| Borrowed input checks | 80 |
| Complete learned parameter checks before/after | 8 |
| Stale-input controls | 4 |
| Intermediate-rounding controls | 2 |
| **Total** | **162** |

Concatenation, summation, narrowing and the gamma product require exact bits.
Projection and normalization comparisons use a declared 0.01 relative plus 0.01
absolute threshold. Numerical failures still fail the final gate even if replay
and ownership checks pass. Every physical output row is included.

## Current status

- 1,133 host tests and 59 simulator-harness tests pass, plus wrapper syntax.
- The saved CPU inputs, projected-context tensors and all seven frozen reference
  sources match their recorded hashes. The learned gamma is not the identity.
- Run `20260909T162015Z-395` completes all 162 checks but **fails two final-output
  comparisons**: pattern 1 has four out-of-threshold elements on each chip.
  All 130 replay/input/parameter/control checks and 30 of 32 eager comparisons
  pass. The original native runtime and packer remain unchanged; mesh and
  checkpoint close cleanly, with outer exit 1. Independent qualification rejects it.
- The failed JSON is `scripts/ci/dspark-projection-simulator-failed.json`, SHA256
  `7babee248db2d25c3e6561bb0ee83159fa25a9301a472cae2d494ade3761aa36`.
  No tolerance, frozen reference or hardware eligibility is changed.
- Eager-only diagnostic `20260909T163428Z-430` reproduces the same two failures
  and saves the actual intermediate tensors. It skips replay deliberately and
  cannot qualify the complete matrix. All 1,137 host and 59 harness tests pass
  at this checkpoint, including capture integrity and mode rejection.
- Setup attempt `20260909T161941Z-387` failed before opening a mesh because the
  wrapper changes directory to `/opt/ttsim`; the retry supplies absolute paths.

Next, fix the numerical mismatch and requalify the complete matrix, connect one learned attention/MLP layer,
then all five layers and the real target/collective path. A component pass does
not establish coding acceptance or progress to 200 TG by itself. The retained
4K hardware result remains PP 3,324.52 / CTX 4,096 / TG 74.27 for one stream.

## Numerical attribution

The captured values reproduce all original final-output comparisons. On both
patterns, CPU normalization of the **actual device projection** passes the
unchanged backbone threshold. Native normalization compared with its own CPU
input reference also passes individually. The combined errors cross the limit
at four pattern-1 elements; this is accumulated error, not corrupt weights or
stale replay. No reference values are substituted into device execution.

| Pattern-1 coordinate (row, feature) | Frozen CPU | Native result | Allowed error |
| --- | ---: | ---: | ---: |
| 5, 1268 | 1.273438 | 1.250000 | 0.022734 |
| 14, 4682 | 1.148438 | 1.125000 | 0.021484 |
| 17, 2451 | -4.218750 | -4.156250 | 0.052188 |
| 24, 1969 | 2.750000 | 2.703125 | 0.037500 |

The attribution report is `scripts/ci/dspark-projection-attribution.json`, SHA256
`e5231cdcb460c43262a4ae4770d322226f9363d507238aea1a5cb9c3034fae31`.
Captured operand SHA256:
`896140c440c4a84552a1ef303ee79d7ae970ba354a4ef14ae036da6a0528b3bd`.

| Normalization diagnostic | Outcome |
| --- | --- |
| FP32 native RMS input/output, explicit BF16 cast | Same four failures per chip; rejected, clean exit 1 |
| Explicit square / sum / rsqrt / product | API setup initially rejected an unsupported sum keyword; no numerical claim |
| Same composition using the installed sum signature | Run `20260909T165149Z-403` active |

These short diagnostics reuse the hash-pinned **observed** FC output, avoiding
another full weight upload. They still require exact native-control reproduction
and unchanged borrowed inputs. Even a pass must be followed by the complete
learned FC/norm matrix with changed-input replay; no hardware or speed claim follows.
