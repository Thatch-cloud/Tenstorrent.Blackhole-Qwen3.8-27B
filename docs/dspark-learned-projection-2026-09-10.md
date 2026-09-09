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
- Run `20260909T162015Z-395` is active using the original native runtime and packer.
  There is no device qualification yet.
- Setup attempt `20260909T161941Z-387` failed before opening a mesh because the
  wrapper changes directory to `/opt/ttsim`; the retry supplies absolute paths.

Next, reconcile this complete matrix, connect one learned attention/MLP layer,
then all five layers and the real target/collective path. A component pass does
not establish coding acceptance or progress to 200 TG by itself. The retained
4K hardware result remains PP 3,324.52 / CTX 4,096 / TG 74.27 for one stream.
