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

### CPU reload hypotheses

The CPU diagnostic now supports `reload_mode='nearest'` or `'truncate'` for
10-bit-mantissa intermediate reloads; `'none'` retains the existing control.
Six unit tests cover controls plus signed ties and nonfinite preservation.
For fixture 0/chip 1 at 32K, 512-key chunks and nearest BF16 output storage:

| Reload model | Failing elements | Maximum absolute error | Head 12 / row 2 / channel 5 |
|---|---:|---:|---:|
| None | 0 | 0.2220345 | -45.75 |
| Nearest | 0 | 0.3749504 | -45.75 |
| Truncate | 470 | 0.5313950 | -46.00 |
| Native simulator | 17 | 0.4752197 | -46.25 |

Neither hypothesis reproduces the native result. The model uses CPU matmul,
exponentials and flat row sums rather than Tensix tile-wise reduction/SFPU
instructions, so this is evidence against a simple rounding-only explanation,
not exoneration of the reload path. Do not choose a device rounding mode by
assuming its name implies either CPU hypothesis. Further qualification requires
native stage-level evidence, particularly partial denominator reduction and
probability/output matmul, rather than additional aggregate CPU curve fitting.

## Native stage snapshot run

Run 34742988039, revision e43a8ec, adds read-only TRISC0 tile slices immediately
after final denominator reduction. Capture row 2, numerator channel 5 and
denominator/maximum column 0, with chip/core prefixes and global query index.
The native BF16 output recurrence is unchanged. Instrumented timings are not
performance evidence. Separate per-context print logs are retained in CI artifacts.

CPU FP32 reference for the mixed-value fixture 0 (unscaled QK maximum):

| Chip | Head | Maximum | Numerator | Denominator | Quotient |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 164.951171875 | -56.029891968 | 1.226281285 | -45.690898895 |
| 0 | 13 | 169.750000000 | -56.498088837 | 1.193859696 | -47.323894501 |
| 1 | 12 | 165.258789063 | -56.044185638 | 1.224151254 | -45.782077789 |
| 1 | 13 | 165.251953125 | -56.043537140 | 1.224238634 | -45.778278351 |

Computed from `fixture_probe(32768).fixtures()[0]`, local KV head `head // 4`,
query row 2, joined full-history keys and value channel 5. Add the original mask
to unscaled QK, subtract its maximum, exponentiate with `128 ** -0.5`, then sum
probabilities and their value products. These are reference calculations, not
device observations. The constant-value fixture has numerator equal to denominator;
the last-proposal-only fixture has numerator 1 at these coordinates. Both retain
the same maximum and denominator, helping distinguish print decoding from math drift.

The initial snapshot run failed before kernel execution: the runtime rejects
`TT_METAL_DPRINT_RISCVS=TRISC0`; the accepted selector is `TR0`. The outdated
example in `rtoptions.cpp` is not authoritative over the HAL parser/runtime.
Build completed in 266 seconds, then initialization failed after 7 seconds.
Correct the selector and retain a CI regression assertion. No stage readings
or new numerical evidence were produced by run 34742988039.

Retry 34743320392 accepts `TR0` but fails the 15000-ms fabric-router startup
handshake before kernel execution. Its stage log is empty; no arithmetic
conclusion follows. Restrict debug polling to logical cores (4,1)/(5,1), the
global-query 12/13 workers under the probe's 8x8 grid and one query tile per
head, instead of all workers. Retain the existing startup timeout for this
comparison. Debug-server overhead is a hypothesis, not a proven cause of the
handshake failure; do not change physical fabric configuration or reset cards.

Run 34743843204 passes fabric startup with restricted polling, then fails JIT
compilation: the TR0 `TileSlice` constructor takes five arguments, whereas the
seven-argument form includes CB/pointer selectors only available on BR/NC.
Correct all three slices to the compute-thread signature confirmed in
`api/debug/dprint_tile.h` and `api/debug/dump.h`. The migration guide example
was insufficiently thread-specific. No stage readings were obtained; this is
an instrumentation compile error, not attention numerical evidence.

## First usable native stage readings

Run 34744366341 subsequently failed snapshot statement syntax; revision 379314b
replaces the invalid multi-statement `UNPACK((...))` expression with a TR0
preprocessor guard and adds host C++ syntax checks for all three compute threads.
Run **34744884673** compiles and produces the snapshots. Its eager output hashes
match the uninstrumented 512-key run exactly on both chips; the same 17 elements
fail, and cleanup passes. Build: 260 seconds; instrumented probe: 494 seconds.

Mixed fixture, row 2 / channel 5:

| Chip | Head | Native maximum | Native numerator | Native denominator |
|---:|---:|---:|---:|---:|
| 0 | 12 | 164 | -60.5 | 1.327724457 |
| 0 | 13 | 169 | -60.5 | 1.277560234 |
| 1 | 12 | 165 | -57.5 | 1.247028351 |
| 1 | 13 | 165 | -57.5 | 1.247047424 |

Compare numerator and denominator at the **same maximum**, since changing the
softmax reference maximum rescales both. At chip 1/head 12, converting the CPU
reference to maximum 165 gives numerator -57.340910913 and denominator
1.252475117. Both native quantities therefore differ from the reference; this
is not only denominator drift. Their native quotient is -46.109617278, while
the final BF16 output is -46.25. Inspect the stored reciprocal and final
normalization stage next before assigning that additional discrepancy to a
particular rounding instruction. A shifted maximum alone cancels in the quotient.

Reciprocal-snapshot run 34745650412 fails the 15000-ms simulator fabric startup
handshake before kernel execution. This recurrence with the restricted print
cores means the earlier successful startup did not prove polling restriction
fully resolves it. Test a bounded 60000-ms startup timeout in the CPU-only
ladder branch; keep router topology, kernel arithmetic, numerical tolerances
and the 1800-second per-context process bound unchanged. A longer timeout is
not a fabric correctness fix and supplies no hardware performance evidence.

## Reciprocal reload boundary identified

Run **34746136745** (revision 8ac4e12) reaches the kernel and preserves both
uninstrumented eager output hashes, the same 17 failures and clean teardown.
Build takes 261 seconds; the instrumented 32K probe takes 500 seconds.
For chip 1/head 12, the stored denominator is 1.247028351 and the captured
reciprocal is 0.802507818. Truncating that denominator to a 10-bit mantissa
gives 1.24609375, whose reciprocal is 0.80250783699, matching the native value
to float precision. The full-denominator reciprocal would be 0.80190638745.

Multiplying the native numerator -57.5 by the captured reciprocal gives
-46.144199535, which rounds to BF16 -46.25. Dividing by the stored denominator
instead gives -46.109617278, which rounds to BF16 -46.0. A CPU regression check
locks down this observed boundary. This identifies a concrete contribution
from the reciprocal reload precision at this coordinate, not proof that fixing
it resolves all contexts or the earlier numerator/denominator drift.

Next candidate should preserve the denominator's FP32 bits during the reciprocal
copy/SFPU path. Audit shared-buffer binary/matmul consumers before enabling a
per-CB direct-unpack mode; do not globally reroute every FP32 operand or relax
the existing full numerical/replay matrix.

### Direct-unpack implementation constraint

The pinned `copy_tile_to_dst_init_short`/`copy_tile` APIs select `UnpackToDestEn`
and generated per-operand formats internally; they have no per-call FP32 mode
argument. `llk_unpack_A_api.h` reads `unpack_dst_format[operand_id]` for both
initialization and execution. The binary `llk_unpack_AB_api.h` also reads that
same format table. Therefore setting the sum buffers' descriptor mode is not
a reciprocal-only change: correction, sum addition, final row reduction and
normalization consumers must be included in validation.

Do not force the fourth math-datacopy-init template parameter to true as a
shortcut. The pinned `tile_move_copy.h` explicitly documents that parameter
as integer-FPU mode on Blackhole/Wormhole, unlike Quasar's unpack-to-destination
meaning. Use an audited descriptor/kernel combination or an isolated compatible
operand route; the observed error does not justify bypassing these API contracts.

The direct-sum candidate builds successfully in run 34747044277, but fabric
startup fails even with the 60000-ms bound, before attention executes. Do not
attribute this to sum-buffer arithmetic. Remove stage-print instrumentation
and its debug-server environment from the next candidate run, retaining the
same direct-sum transform, startup bound and numerical tests. Prior snapshots
remain useful evidence; the print path need not stay enabled for qualification.

Run **34747542923** reaches execution without debug printing, then the simulator
rejects `tensix_unpacr: unpack_to_dst=0 in_data_format=0 out_data_format=0`
as undefined behavior. The globally changed sum-buffer format is incompatible
with a source-register unpack consumer. This is not a numerical pass/fail and
must not be retried on hardware. Build: 259 seconds; failed probe: 146 seconds.

Remove direct sum routing from the active ladder builder and keep its transform
and immutable CI tag as a rejected experiment. An isolated reciprocal operand
must preserve the existing formats for binary/matmul sum consumers. Do not
suppress simulator undefined-behavior detection or mark the direct-sum gate passed.

## Isolated scalar reciprocal: 32K passes, 64K remains open

Run **34748217636**, revision b0f3a77, evaluates the reciprocal directly from
stored FP32 first-column values on TR0. Shared-buffer formats stay unchanged.
The native toolchain links and executes this diagnostic; host object compilation
and two-tile address tests were prerequisites, not the device proof.

| Prompt / output headroom | Numerical result | Replay | Cleanup | Simulator probe time |
|---|---|---|---|---:|
| 32768 / 1024 | All four eager comparisons pass | All four changed-input comparisons exact to eager | Clean | 940 s |
| 65536 / 1024 | First eager comparison: chip 0, 256 failing elements, max abs 0.638324738 | Not reached | Clean | 916 s |

32K report SHA256:
`8d05be0a5853e763e9a0b475f5a01f119ffa4923d517f4f5d62aba580549029b`.
At 64K the first failures include head 9, row 10: -46.75 versus reference
approximately -46.125. The remaining 128/4K/8K regressions were not executed
because this run stops at the first failing context.

This confirms the isolated reciprocal change is sufficient for the tested 32K
component matrix, not for 64K, full-model correctness, coding quality or TG.
Do not compare simulator times as hardware performance: the passing 32K probe
executes additional eager and replay work that earlier failed probes never reached.
Retain this candidate as a diagnostic baseline while isolating 64K output/history
accumulation drift; hardware admission and the full PP/CTX/TG ladder remain open.

### 64K candidate: reduce accumulation iterations

The 64K constant-value diagnostics show maximum drift 0.0312497 on chip 0 and
0.0351562 on chip 1 despite the full-precision reciprocal. Test 1024-key chunks
only at 64K, retaining the 32K/512 configuration and scalar reciprocal. This
reduces online accumulation iterations from 131 to 66, with full logical
history unchanged; masked poisoned padding grows from 448 to 960 keys.
Native storage is 67584 keys and the selector admits only the exact
2112-key-tile/32-chunk-tile pair. This is a bounded accumulation hypothesis,
not a general chunk sweep or assumed speed improvement. Run 64K first, then
all four smaller regressions if it passes. Retain unchanged tolerances.

### Faster subsequent probes

Run 34749444881 tests revision 50a185b with diagnostics unchanged. Do not
cancel or replace that in-flight numerical experiment.

For subsequent ladder probes, the three non-qualifying value-diagnostic
executions are opt-in with `QWEN_LADDER_VALUE_DIAGNOSTICS=1`; default `0`
goes directly to the existing eager and replay acceptance matrix. Reports
explicitly record `value_diagnostics_enabled`. Numerical tolerances, fixture
controls, input/layout checks, replay checks and cleanup are unchanged.
Use diagnostics when investigating a failure, not on every regression context.
CPU wrapper/fixture/compiler/CI tests pass; elapsed CI savings remain unmeasured.

### 64K/1024 result: improved but rejected

Run **34749444881**, revision **50a185b**, completed with a numerical failure,
not a timeout. The first eager case on chip 0 has **129 failing elements**,
maximum absolute error **0.514190674**, versus 256 and 0.638324738 at chunk
512. Probe duration was 907 seconds; teardown was clean. Replay and smaller
context regressions were not reached. No hardware admission follows.

The constant-value diagnostic still drifts by about 0.03125 on both chips
(3200/2816 failed elements). Oldest-token and last-proposal diagnostics pass.
Fewer accumulation iterations reduce the error but do not establish adequate
precision; avoid treating another chunk-size sweep as a demonstrated fix.
The next precision investigation should isolate BF16 output recurrence and
its operand conversions, preserving the passing 32K reference and full history.

Report SHA256:
`38050b207858adcf3ad10a80fc04aeff41a7502275242fb8f2afa3476940929f`.
Committed single-stream hardware throughput is unchanged by this experiment.

### Next candidate: FP32 output intermediates only at 64K

Keep chunk 1024, full history, scalar reciprocal and all tolerances fixed.
Within the guarded 64K selector only, change `im_df` to Float32. This changes
the two output intermediate buffers and their tile sizes together; the final
output retains its native output dtype. The factory also uses `im_df` for
reciprocal scratch, so this is not claimed to be a single-buffer change.
All smaller context configurations retain BF16 output intermediates.

Do not change global unpack routing: the previous direct-sum experiment
demonstrated that shared source-register consumers cannot safely use that
shortcut. This candidate uses normal format-derived unpacking. The simulator
must establish native compatibility, numerical accuracy, replay and cleanup;
24 CPU preflight tests alone do not establish those properties. The three
non-qualifying diagnostics are omitted on this run to shorten feedback time.
