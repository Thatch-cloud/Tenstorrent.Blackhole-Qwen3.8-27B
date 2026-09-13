# Full-context ladder investigation

## First synthetic ladder: 2026-09-13

CI run 34737225647, attempt 2, revision 335a11c. Synthetic attention only;
no target-model weights, full-request throughput or serving qualification.
Each context reserves 1024 output rows. The retained FP32 reference uses
rtol=0.01 and atol=0.01; changed-input replay must match eager output exactly.

| Prompt tokens | History capacity | Native padded keys | Result |
|---:|---:|---:|---|
| 128 | 1152 | 1280 | 4 eager and 4 replay comparisons pass |
| 4096 | 5120 | 5376 | 4 eager and 4 replay comparisons pass |
| 8192 | 9216 | 9472 | 4 eager and 4 replay comparisons pass |
| 32768 | 33792 | 34048 | First eager case: chip 0 passes, chip 1 fails |
| 65536 | 66560 | 66816 | Not executed: stopped at 32K failure |

At 32K, chip 1 has 271 failing elements and maximum absolute error
0.504302978515625. The report records numerical failure, not replay failure:
replay had not started. All four executed context processes close cleanly.

Do not infer the error mechanism from its magnitude alone. The next diagnostic
must locate errors by head/row and separate accumulated output rounding from
masking, padding or state errors, while preserving the same fixture and tolerance.
The new capacity needs its own evidence; the earlier qualified 8192+256
configuration is not interchangeable with this 8192+1024 test.

## Decision

No 32K hardware admission and no tolerance relaxation. Retain the uncached
qualified hardware baseline. Finish numerical diagnosis, then resume the full
hardware PP/CTX/TG ladder. These component passes do not finish that ladder.

## Next candidate: 512-key long-context chunks

The existing value diagnostics show constant-value error up to 0.0195312
on chip 1. The first failing output coordinates include head 4, row 14:
actual -47 versus reference approximately -46.50. This is evidence of
normalization/output numerical drift, not proof of a specific faulty stage.

Test 512-key chunks only at 32K/64K. Keep 256 at smaller contexts. At 32K,
native padded storage changes from 34048 to 34304 keys, reducing online chunk
iterations from 133 to 67. The extra keys remain poisoned and masked; logical
history, queries, precision flags and tolerance are unchanged. Factory and
compute selectors admit exact key-count/chunk-size pairs, not a broad range.
Fourteen host tests pass. This is an unqualified hypothesis until CI numerical
and replay results pass; it is not a TG improvement claim.

CI run 34740697532 tests this candidate at immutable revision af94fb5, tag
`ci-qwen-ladder-chunk512-20260913`. It repeats all five synthetic contexts;
the previous lower-context passes do not qualify a changed native build.
Sixteen host checks now also lock down the exact long-context padding and
key-count/chunk-size pairs. These additional test-only changes do not change
the running candidate. Hardware admission remains closed pending the reports.

## 512-key result

Run 34740697532 completed with numerical failure, not timeout. At 32K the
first eager case passes on chip 0 but fails on chip 1: 17 elements, maximum
absolute error 0.4752197265625. The previous 256-key candidate failed 271
elements with maximum error 0.504302978515625. This is an improvement in the
same fixture, not qualification. Replay was not reached and 64K was not run.
The process closed cleanly. Constant-value diagnostic error on chip 1 remains
0.019531190395355225; fewer chunks alone have not eliminated the drift.

Retain the tolerance and hardware block. Investigate normalization and partial
output precision before assuming a further chunk-size increase is sufficient.
Future CI revisions run 32K first, then 64K and all smaller regressions, with
per-stage elapsed times; this completed run used the old ascending order.

### Remaining-error localization

The first 16 recorded failures are chip 1, query row 2, heads 12/13:
output -46.25 versus reference about -45.78. At those same head/row coordinates,
the constant-value diagnostic returns 0.9921875 rather than 1. A uniform
normalization gain alone cannot explain both: it would shrink the negative
output magnitude, whereas the observed failing output has larger magnitude.
This rules out treating constant-value drift as a sufficient root-cause diagnosis.

The pinned `compute_common.hpp` packs each `QK @ V` chunk into `out_im_A/B`,
then rescales and accumulates through `mul_block_bcast_cols<..., false, true>`
using explicit L1 pack accumulation. The ladder factory still allocates those
buffers in BF16; QK/sums and the patched statistics are FP32. Next isolate that
output recurrence from the denominator, preserving the actual mixed-value fixture.
Do not repeat the previously rejected blanket FP32-intermediate switch without
auditing its unpack, pack and accumulation paths.

## Explicit output recurrence diagnostic

Run 34741921352, revision c211e67, replaces output L1 pack accumulation with
BF16 multiply-in-place followed by explicit addition. The same 32K case still
fails 17 elements on chip 1, with the same maximum error 0.4752197265625.
Both output hashes change, so the candidate does affect arithmetic, but neither
the worst error nor constant-value drift improves. Chip 0 passes; cleanup passes.
No replay, 64K or smaller regression cases were reached in this fail-first run.

Reject this as a fix and restore the native output recurrence in the ladder
entrypoint. Retain the isolated diagnostic module and immutable tag for reproduction.
This rules out this replacement as sufficient; it does not prove BF16 output
storage is harmless or that all affected intermediate values are identical.

Measured cycle breakdown: native build 261 seconds, 32K probe 492 seconds.
The numerical failure is available after roughly 12.6 minutes of these stages,
without spending time on the three smaller contexts first. Simulator build caching
can address the 4.35-minute build component, not eliminate the 8.2-minute probe.

## CPU rounding check at the failing geometry

Extend `dspark_attention_rounding_control.online_attention` with explicit
64/256/512-key chunks (default 64 unchanged). Five CPU unit tests pass.
Use `fixture_probe(32768)`, fixture 0, chip 1, 15 live query rows, joined
full-history keys/values with fourfold KV-head expansion, and the unchanged mask.
Compare to `probe.reference` at rtol=atol=0.01. These are CPU hypotheses, not
Tensix simulation or admission evidence.

| Key chunk | Partial/running output storage model | Failing elements | Maximum absolute error |
|---:|---|---:|---:|
| 256 | FP32 | 0 | 0.1250114 |
| 256 | BF16 nearest | 0 | 0.2220345 |
| 256 | BF16 truncation | 30720 | 1.9038239 |
| 512 | FP32 | 0 | 0.1250000 |
| 512 | BF16 nearest | 0 | 0.2220345 |
| 512 | BF16 truncation | 18146 | 1.1017494 |

At head 12 / row 2 / channel 5, 512-key truncation reproduces -46.25,
while nearest produces -45.75 versus reference -45.7820702. However, the
truncation model fails 18146 elements, not the device's 17: matching one
coordinate does not establish the mechanism. Ordinary nearest BF16 output
storage alone does not reproduce this failure. Do not promote blanket FP32
storage on this evidence; inspect the actual per-operation rounding/unpack paths.

## Native unpack audit

The pinned SDPA `ComputeConfigDescriptor` sets FP32 destination accumulation,
but does not set `unpack_to_dest_mode`. In `tt_metal/jit_build/genfiles.cpp`,
`compute_data_formats` selects **Tf32** as the conditional unpack destination
when FP32 accumulation is enabled with exponent-B buffers. In `data_format.cpp`,
`get_single_unpack_dst_format` maps Float32 storage to that conditional format;
only explicit non-default per-buffer unpack modes select Float32 directly.

Thus FP32 QK, sum and statistics storage does not imply full FP32 precision
through every reload in this path. The previous CPU output-rounding control
does not model these TF32 reloads. Next diagnostic must cover intermediate
reload precision, not mislabel the native route as BF16 unpack or assume a
missing mode is automatically a bug. Direct-to-destination unpack also changes
operand routing: audit compatibility for copy/SFPU versus binary/matmul users
before enabling it for shared circular buffers.
