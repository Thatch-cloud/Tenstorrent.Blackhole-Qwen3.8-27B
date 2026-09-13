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

### FP32 output-intermediate result: worse, rejected

Run **34750342186**, revision **474e6ad**, builds and executes but fails the
first eager comparison on chip 0: **4131 failing elements**, maximum absolute
error **0.737335205**. Teardown is clean; replay is not reached. Restore BF16
output intermediates in the active candidate; retain the immutable CI tag
as evidence. FP32 storage alone is not an accuracy fix: normal format-derived
unpacking still converts operands, so this does not test an all-FP32 recurrence.

The probe took **402 seconds** without optional diagnostics, versus 907 seconds
for the prior diagnostic-enabled probe. These are different numerical candidates,
so the 505-second difference is not a controlled timing attribution or TG result.

Report SHA256:
`5c5e8fd4c535390f017794114d36c63fb6be5701344a648f55b740dfa5318d43`.
No hardware qualification or serving changes follow from this run.

### Reciprocal reload rounding candidate

With BF16 output recurrence restored, isolate one remaining conversion: the
scalar FP32 reciprocal is subsequently unpacked as TF32 by the final broadcast
multiply. At the exact 64K geometry only, round that reciprocal to nearest-even
TF32 before normal unpacking, rather than letting unpack truncate it. This is
not an all-FP32 normalization and does not alter sum-buffer formats or routing.
Smaller contexts retain their previous reciprocal arithmetic. Reports name the
reload mode separately from reciprocal computation. CPU tests check both
geometries on all three thread branches and preserve untouched tile lanes;
native numerical/replay qualification remains required.

### Reciprocal reload rounding result: rejected

Run **34751094661**, revision **87be2c3**, completes with a numerical failure:
257 failing elements, maximum absolute error 0.521884918, first eager case
on chip 0. Probe duration is 400 seconds; cleanup is clean, replay is not
reached. This is worse than native truncation's 129 / 0.514190674 at the same
geometry. Restore native truncation; the immutable tag retains the experiment.

Report SHA256:
`e49c84f200c4e85e0961edfe6403e26694244f39104ab69c09e2f4eaa0d53516`.
Neither FP32 intermediate storage nor nearest reciprocal reload fixes 64K.
The next step needs numerator/denominator stage evidence for the failing 64K
coordinates, not another unsupported precision-toggle sweep. The 32K component
pass remains distinct from unimplemented longer-context full-runtime admission.

### Targeted 64K stage measurement

Instrument the restored BF16/1024 baseline at proposal row 5, channel 0.
The recorded failure includes all 128 channels of chip 0 head 5 at that row
(for channel 0, -46.25 versus -45.756687164). Enable TR0 prints on logical
core (5,0), retaining the q/head label in each snapshot so mapping is checked
from output rather than assumed. Capture numerator, denominator, unscaled
maximum and scalar reciprocal, without consuming or writing their buffers.
Smaller contexts remain uninstrumented. Compare final eager hashes against
run 34749444881 before treating snapshots as representative of that failure.
No simulator timing from this instrumented run qualifies hardware performance.

### 64K stage evidence: denominator drift precedes reciprocal

Run **34751818673**, revision **5fe0eba**, reproduces the baseline's 129
failing elements / 0.514190674 max error and exact chip-0 eager SHA256
`c020277846d324c70116bf8629f1d9f4dd0dcfe5c204f18ace0595560bea2c07`.
The logical core reports q=5, confirming the selected head. Clean teardown;
391-second probe, no replay. Report SHA256:
`c077d9255328bcb3d43f2743cf53d8633298440e328f92501c961d3dd502613a`.

Row 5 / channel 0, fixture 0:

| Chip | Native maximum | Native numerator | Native denominator | Scalar reciprocal |
|---|---:|---:|---:|---:|
| 0 | 166 | -59 | 1.278387070 | 0.782235682 |
| 1 | 171 | -58.5 | 1.228707314 | 0.813863456 |

CPU Torch FP32 recomputation from the same deterministic full-history fixture
uses unscaled `K @ query`, masked maximum, and `exp((score-max)/sqrt(128))`.
Compare numerator/denominator only after rescaling the reference to the native
maximum with `exp((reference_max-native_max)/sqrt(128))`:

| Chip | Reference max | Reference numerator at native max | Reference denominator at native max | Reference quotient |
|---|---:|---:|---:|---:|
| 0 | 166.5234375 | -58.851352031 | 1.286179920 | -45.756702571 |
| 1 | 171.486328125 | -59.136698852 | 1.244706821 | -47.510544551 |

Chip 0 numerator magnitude is 0.253% high and denominator 0.606% low. Their
native quotient is already -46.151906089 before final reciprocal reload and
BF16 rounding (reported output -46.25). Thus reciprocal-only changes cannot
be assumed to fix the upstream error. Next isolate the denominator's final
row reduction from its per-chunk accumulation; do not infer that all drift
comes from one stage or that FP32 storage implies FP32 operand arithmetic.

### Separate final reduction from accumulated partials

Extend the same read-only row-5 snapshot to print all 32 stored FP32 partial
denominator values immediately before `matmul_reduce`. Sum those values on
the host and compare against both the observed native post-reduction value
and reference 1.286179920 (chip 0, at native maximum). This distinguishes
error already in the partials from extra error in the final native reduction.
Arithmetic, output buffers, chunk geometry and numerical tolerances remain
unchanged. Sixteen preflight tests cover patch composition, C++ slice syntax,
read-only behavior, scalar reciprocal and CI isolation.

Run **34752550619** (658a596) preserves the baseline eager hash and clean
teardown, but the debug API truncates each TileSlice to 16 values. Only half
of each partial-sum row was emitted, so this run cannot attribute the complete
reduction error. Fix the snapshot as two 16-value slices; retain the same
arithmetic and coordinate. Add a slice-size regression check before rerunning.
Report SHA256:
`7ed67f8fd64778ec795579532421924b6a9eb0c4bb35710ffcdff023e399f77e`.

### Complete partial sums and next isolated update

Run **34753183554** (6c76a6f) captures both halves and exactly reproduces the
baseline eager hash, 129 failures and clean teardown. Report SHA256:
`a42adf96addfa5e864f613374fab368dfa59301ede7b6b4c387b4c9ac43eb849`.

| Chip | Sum of 32 printed partials | Native final reduction | FP32 reference at native maximum |
|---|---:|---:|---:|
| 0 | 1.277565000 | 1.278387070 | 1.286179920 |
| 1 | 1.230964668 | 1.228707314 | 1.244706821 |

Printed values have nine decimal places. Most denominator drift is already
upstream of final reduction; on chip 0 that reduction actually reduces the
shortfall. Isolate the recurring sum update next: at 64K only, use a TR0 scalar
FP32 `current += previous * correction` over all partial lanes, retaining CB
formats and leaving the correction available to native output accumulation.
Consume only the previous sum, as the original update does. This is a
diagnostic, not a production-speed implementation. Two-tile face/address and
ownership tests plus native patch composition and Blackhole object compilation
pass; simulator execution must still establish synchronization and correctness.

### Scalar sum update: 129 failures reduced to one, still unqualified

Run **34781446816**, revision **7c098b1**, executes successfully with clean
teardown but fails numerical acceptance in the first eager case on chip 0.
The failing element count decreases from 129 to **1**. The remaining element
is head 0, row 8, channel 116: **-43.5** versus **-43.949645996**, absolute
error **0.449645996**. Do not widen the unchanged tolerance to admit it.
Chip 1 eager acceptance and changed-input replay are not reached.

At the previously failing head 5 / row 5 / channel 0, numerator stays -59
while the denominator improves from 1.278387070 to **1.284729004**, versus
the reference 1.286179920 at native maximum. This supports repeated sum-update
conversion as a material source of that row's drift, not a complete diagnosis
of the remaining element. Preserve this diagnostic for the next targeted
head-0 investigation; it is not yet hardware-admitted or performance-qualified.
The probe takes 530 seconds, not a hardware speed measurement.

Report SHA256:
`52f610a6623e4a53fca20ab018614ea15226ef5ef0e3331f7205952ecb2c0689`.

Next measurement retains scalar sum update and moves read-only snapshots to
head 0 / row 8 / channel 116. Channel 116 maps to output tile 3, lane 20;
denominator and maximum remain in tile 0. Print only logical core (0,0),
checking q=0 in the resulting log. Test the cross-tile coordinate mapping
explicitly. No arithmetic or tolerance changes accompany this measurement.

### Remaining element: numerator drift dominates

Run **34782414632** (c2a62c0) reproduces the scalar-sum candidate's exact
chip-0 eager hash `1e6891b6e1a90951b420b562b4504d91d05bc79ec396b32705ab86f4813fe5d6`,
one failure and clean teardown. Report SHA256:
`c0486e391b460b048fc5a9e86018e1fba53a4dc7723ddbbccda8a2bc71a701ba`.

At chip 0 / head 0 / row 8 / channel 116 the native maximum is 162,
numerator -56, denominator 1.285524368, scalar reciprocal 0.777892709.
Their pre-rounding quotient is -43.561990262, versus FP32 reference
-43.949645996. Rescaling the FP32 reference from maximum 162.189453125 to
162 gives numerator -56.703657611 and denominator 1.290195957. Numerator
magnitude is 1.241% low; denominator is 0.362% low. Final output is -43.5.

This isolates the remaining error differently from the earlier head-5 failure:
numerator precision now needs investigation while retaining the improved sum
update. Earlier FP32-output-storage rejection used the old native sum update,
so it does not prove how FP32 output storage interacts with the corrected sum.
Any combined precision experiment must retain full-history, unchanged tolerance,
both fixtures/chips and exact replay rather than targeting only this element.

### Combined sum/output precision candidate

Retain the simulator-executed scalar FP32 sum update and scalar reciprocal;
change only the exact 64K factory's output intermediate storage to FP32.
The earlier standalone output-storage candidate lacked this sum update.
Keep the row-8 snapshots to compare numerator and denominator directly, with
the full eager/replay gate unchanged. Other context configurations remain
unchanged. Normal unpack routing still applies; do not call this all-FP32
arithmetic or assume it improves performance. Twenty-six CPU preflight tests
pass, including factory rebuild provenance and scalar-update buffer ownership.

### Combined output-storage result: rejected on complete first-case comparison

Run **34783345209**, revision **3db0cbf**, fails the first eager comparison
on chip 0 with **343 failing elements**, maximum error **0.617980957**.
Cleanup is clean; later cases/replay are not reached. At the inspected row,
numerator improves from -56 to -56.609519958 (reference -56.703657611 at
native maximum); denominator stays 1.285524368. Improvement at one coordinate
does not qualify the broader candidate. Restore BF16 output intermediates
while retaining the scalar sum update, which had only one first-case failure.
Report SHA256:
`b57744cc5cb70dc9aefc2c9c98be1782d9fd8f95e7025bfdaea20ab2db4bfb43`.

### CPU screen before another native candidate

Extend the existing non-qualifying rounding model to the actual 1024-key
chunk and test chip 0 / head 0 / row 8 / channel 116. Seven CPU unit tests
pass, including FP32-reference preservation at that chunk size.

| CPU reload hypothesis | FP32 output intermediates | Nearest BF16 storage | Truncated BF16 storage |
|---|---:|---:|---:|
| None | -44 | -44 | -44.75 |
| Nearest TF32 | -44 | -44 | -44.75 |
| Truncated TF32 | -44.25 | -44.25 | -45 |

None reproduces native -43.5. This cheap screen does not justify selecting a
rounding flag or assuming a numerical fix; it models neither native partial
PV matmul nor the exact pack/unpack recurrence. Retain it as a CPU diagnostic,
not a simulator replacement or admission gate. The next native isolation
must separate output recurrence from the partial PV results while retaining
the improved denominator, rather than extrapolating from this surrogate.

### Native output recurrence capture

Keep scalar sum update and BF16 output storage. At the existing head-0,
row-8, channel-116 coordinate, print previous output, current partial PV
output and maximum-rescaling factor before each native output update.
Chunk labels order the records; the next record's previous output supplies
the prior update result, with the existing final numerator snapshot closing
the last update. This separates native partial PV values from accumulated
rounding without changing arithmetic or widening the acceptance criterion.
Eighteen CPU preflight tests pass. Require matching eager hashes before
using the new snapshots as evidence about the one-failure baseline.

### Output operand evidence points upstream of recurrence

Run **34784419980** (08ef82a) preserves the one-failure baseline's exact
eager hash and clean teardown. Report SHA256:
`38dbe46337d79a4354829664bc98b997eb92f5e66080eca30199024c3964cf7c`.
Chip 1 emits all 65 update records. Chip 0 emits 62: labels 25-27 are
absent, so its log is not a complete recurrence trace and must not be used
to claim a full high-precision reconstruction.

All observed next-state comparisons agree with nearest-BF16 rounding of
`previous * correction + partial`; the chip-0 gap remains unobserved.
The final recorded update is:

| Chip | Previous | Partial PV | Correction | Final numerator |
|---|---:|---:|---:|---:|
| 0 | 64 | -64.5 | 0.130979255 | -56 |
| 1 | 68.5 | -64 | 0.109758809 | -56.5 |

CPU FP32 recomputation of the final proposal chunk at the native maximum
gives partial PV -65.095664978 (chip 0) and -64.677909851 (chip 1).
Thus the remaining numerator error is already materially present in partial
PV, not just output accumulation. Next inspect the final chunk's QK/exp
inputs and probability values before proposing another recurrence change.
This is coordinate-level diagnosis, not whole-model correctness or speed.

Next capture limits output-update logs to final processed chunk 65 and adds
the 15 proposal scores immediately before exponentiation plus the 15
exponentiated values immediately after it. Each slice stays below the
16-value debug limit. This reduces log pressure and separates QK/exp error
from partial PV matmul error at row 8. Retain identical arithmetic and
compare eager hashes again. Eighteen CPU preflight tests pass.

### QK/exp evidence

Run **34785336947** (20aa0ce) retains the exact one-failure eager hash and
clean teardown. Report SHA256:
`fc42c85c9a927d0b1fed05528dc6f0f103d8a3e6261b45b4d0f6df01c90d0a16`.
At chip 0 / row 8, final proposal score is **162.125**, versus CPU FP32
**162.189453125**. With native maximum 162, its exponentiated value is
**1.011108637**, versus reference **1.016886473** at that maximum.
Chip 1 records score 170 and exponent 1, versus reference score
170.119140625 and exponent 1.010586262 at native maximum 170.

The logged scores are after provided-mask addition and max reduction, not
raw matmul output. Native `add_block_inplace(cb_qk_im, cb_mask_in, ...)`
reloads scores through source registers and repacks them; default FP32
unpacking uses TF32. That is a candidate source of score truncation, but this
capture does not separate matmul packing from mask-add reload. The exp
function also reloads scores, so changing mask addition alone cannot be
assumed to preserve FP32 arithmetic through normalization. Avoid another
output-recurrence change as a remedy for this observed upstream score loss.

Next read-only capture brackets the provided-mask addition with the same
15-element score slice, retaining post-reduction/pre-exp and post-exp slices.
This directly tests whether the mask-add operation changes unmasked proposal
scores or whether precision was already lost in QK matmul packing. Only the
final chunk is logged. Eighteen preflight tests pass; no arithmetic changes.

### Confirmed zero-mask score truncation

Run **34786241488** (c95eaf1) matches the one-failure eager hash and closes
cleanly. Report SHA256:
`8452b36e39aad8141a02fd4bb4d96d46fd65f9f05aa2a66d5432757ea9cbd6df`.
For the unmasked final proposal, QK matmul produces the exact reference
score; adding its zero mask changes it:

| Chip | Before mask addition | After mask addition | CPU FP32 reference |
|---|---:|---:|---:|
| 0 | 162.189453125 | 162.125 | 162.189453125 |
| 1 | 170.119140625 | 170 | 170.119140625 |

The other 14 proposal scores also truncate at mask addition. This directly
identifies source-register reload during mask addition as a precision-loss
boundary, rather than QK matmul packing at these coordinates. A fix must
retain zero-mask scores and apply blocked entries exactly, while accounting
for the subsequent score reload in exponentiation. Merely storing FP32,
changing output recurrence, or loosening tolerance does not fix that boundary.
No hardware performance is established by this diagnostic result.

### FP32 mask and centering diagnostic

At exact 64K geometry only, apply the BF16 additive mask directly to stored
FP32 scores on TR0, preserving zero-mask values and blocked negative infinity.
Then subtract the stored row maximum in FP32 before the native exponential
reload. Replace that path's broadcast subtraction with a tile copy: native
exponentiation and partial-sum packing remain, but they now reload the small
centered difference instead of independently truncating large score/max values.
This still uses native lower-precision reload, not an all-FP32 softmax.

Retain scalar sum update, scalar reciprocal, BF16 output intermediates and
all numerical/replay gates. Smaller contexts retain original arithmetic.
Twenty-eight CPU tests pass, including four score tiles, two maximum rows,
all tile faces, zero/blocked masks, TR0-only writes, buffer consumption,
combined patch composition and Blackhole helper compilation. Native scheduling
and full numerical acceptance still require CI; scalar loops are diagnostic
and must not be presented as the final production-speed implementation.
