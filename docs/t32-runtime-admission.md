# T32 simulator runtime admission

The experimental 31-query Markov gate uses the immutable CPU CI image
`sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
It does not inherit the old local Markov probe's runtime qualification.

Run 34589120590 failed before simulation because Python 3.10 lacks
`hashlib.file_digest`; streaming SHA-256 replaces that call.
Run 34589267015 then rejected the old binary pin before opening the mesh.
Audit run 34589455129 reports these exact image contents:

| File | SHA256 |
| --- | --- |
| `build_Release/lib/_ttnncpp.so` | `f65ac9e332d34ff462a051a021221fc12377b05711dc67d1faa5aa6fe37858c3` |
| `build_Release/ttnn/_ttnncpp.so` | `d6c53113a104719a442b4d4a9ec2b344cdd0e00daa1e4d907afb9c13d1e531d9` |
| Original packer | `87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181` |

The new probe pins both paths separately and retains before/after hashes of
embedding, matmul and argmax sources. No binary is copied over another and no
hash check is disabled. These pins establish reproducibility, not numerical
correctness: the new eager oracle and changed-input replay gates still must pass.
The initial gate uses 64 synthetic vocabulary entries. Full-vocabulary learned
weights, wider attention, every-prefix target state and combined PP/CTX/TG
remain unqualified. No serving defaults change.

## Completed component gates

| Run | Gate | Independently checked result |
| --- | --- | --- |
| 34589904935 | 31-query Markov, synthetic vocabulary 64 | 186 eager and 248 replay query/chip comparisons; input/weight/stale controls pass |
| 34590151457 | T32 folded target attention at CTX4096 | 8 replay, 24 distinct mask-bundle and 4 KV-integrity checks pass against native B1 |
| 34590621276 | Learned full-vocabulary, 31-query Markov | 124 eager and 186 replay comparisons pass; immutable probe sources verified |
| 34653659471 | T32 drafter attention with FP32 SFPU denominator reduction | 4 eager, 4 exact replay, 48 input and 16 layout checks pass; 46 source hashes verified |

Both exit cleanly with zero status; recorded probe sources match their immutable
CI revisions (`cab5126` and `8e90139`). Markov report SHA256:
`ce24fcf9924258daa703f74e08cf6570f1e5d872102dc5a952eebd4f05831065`.
Attention report SHA256:
`774a53bf54fcbb7fe5c02fb1f358be63c9fcc0254679f19543232d4079429b1b`.

The learned Markov report SHA256 is
`d93147bfa2e0223ad479d6bffba7416e5e41e663e1a6463080a1172ccb9bec3a`.
It uses learned full-vocabulary matrices with synthetic base logits, not a
complete learned proposal or a throughput measurement.

### Drafter attention investigation

| Run | Change | Result |
| --- | --- | --- |
| 34591332709 | Initial CI packaging | Failed before simulation: missing repository-relative support path |
| 34599714782 | Restore support path | Three numerical failures on chip 1; chip 0 passes first eager case |
| 34642483772 | Retain exact error coordinates | All three failures are head 11, live query row 13, columns 53/70/106 |
| 34643311170 | 32-key chunks instead of 64 | Same three failing coordinates and values |
| 34644020684 | Scoped precise reciprocal with 32-key chunks | Same three failing coordinates and values; clean teardown |
| 34644680369 | Retain failed tensor and audit failing-chip inputs | Both assembled KV tensors and all six inputs exact on both chips |
| 34645764708 | Rebuild with FP32 partial-output and statistics buffers | 227 numerical failures on chip 0; rejected |
| 34647839458 | Matched rebuild without FP32 buffers | Both first-eager output hashes exactly match original baseline; original three chip-1 failures remain |
| 34648932728 | FP32 partial outputs, BF16 statistics | 63,488 failures on chip 0 (all 31 live rows across 16 heads); max absolute error 16.7873; rejected |
| 34649985604 | Same output-only variant plus explicit pack format in max-difference correction | Failures fall to 106 on chip 0, max absolute error 0.5837; still unqualified |
| 34651033542 | FP32 statistics, BF16 partial outputs, explicit correction packing | First eager chip 0 passes; three chip-1 failures remain; not qualified |

The observed values are -44.75 versus approximately -44.295 in the FP32
reference, narrowly outside the unchanged `rtol=.01, atol=.01` bound. Neither
chunk-size reduction nor the precise reciprocal is a demonstrated fix.
Simple CPU rounding diagnostics also do not reproduce the simulator output;
they are not a bit-accurate native-kernel model and do not rule out intermediate
precision loss. The drafter attention gate remains failed, with no hardware
promotion or T32 throughput claim.

The matched rebuild isolates the FP32-intermediate regression from build effects.
Its runtime binary SHA256 is
`9abba20f930dd47c0c4dcab25a95b67dec6394a7bba558992ab51f4d680a02c8`;
the FP32 variant is `761e78b2b951d34435696de20b1de2a05316f7cc1165de5f2055e3d84c37de89`.
Neither is numerically qualified. A preliminary CPU truncation model reproduces
the failing scalar but is not proof of the native rounding cause. Simply widening
both buffer classes is demonstrably not a fix for this fixture.
Widening partial outputs alone also fails substantially. This is evidence against
changing circular-buffer formats without auditing the intervening mixed-format
math and unpack/pack transitions; it is not a reason to relax numerical bounds.
Adding `pack_reconfig_data_format(out_cb)` in the scoped `sub_exp_block` path
removes most of that mixed-format corruption, but does not meet the numerical
gate. Retain this correction when investigating mixed-format variants; do not
describe it as a qualified T32 attention implementation or throughput gain.
Statistics-only widening changes output hashes but does not remove the original
three failing comparisons. No tested precision-buffer combination is admitted.

### Denominator reduction fix

Run 34653659471 (`f2bba8d`) replaces the final denominator matmul reduction with
the FP32 SFPU row-reduction pattern already used by `draft_row_sum_compute.cpp`.
It keeps the original buffer formats, 64-key chunks and `.01/.01` tolerances.
No diagnostic taps or disposable FP32 build are enabled. Both eager cases and
changed-input exact replay pass on both simulated chips, including negative
fixture controls, input preservation, stable bindings and clean teardown.

Independent validation checks zero exit status, report coverage and all 46 probe
source hashes against the immutable CI revision. Report SHA256:
`9bb6b5c8f65627efddccfa88e0c9453f9719bb7b0c65714217050257a8032941`.
This qualifies the tested attention component only. Complete captured drafting,
every-prefix target state, coding quality and combined PP/CTX/TG remain pending.

Dispatch uses `simulator_t32` with values `none`, `t32-markov`,
`t32-markov-learned`, `t32-attention` or `t32-draft-attention`, avoiding GitHub's
25-input limit. These CPU-only jobs have a 16-CPU quota and 64 GiB memory limit;
they do not mount the cards or measure hardware speed.

Before a combined hardware comparison, integrate the validated SFPU reduction,
complete captured T32 proposal integration and every-prefix target-state gates.
The prepared T32 Markov path currently uses native score layout, not the fused
score layout used in the T16 comparison: that difference must be explicit in
the eventual matched baseline. Component passes and host tests do not qualify
complete T32 drafting, coding quality or combined PP/CTX/TG.
