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

### Score-centering timeout: run 34787283613

The 64K simulator probe exited with status 124 after exactly 1800 seconds.
Fabric initialization completed on both simulated devices; the final progress
record was `eager_0`. There is no numerical result or clean-close evidence,
so this candidate is not admitted to hardware. The log does not distinguish
a synchronization stall from excessive scalar execution cost.

Do not extend the timeout or rerun the full ladder unchanged. First isolate
this path at small workload size while retaining its mask, centering and
native exponential sequence. The existing host tests do not model TRISC
synchronization. Zero masks now bypass scalar addition entirely, preserving
stored scores without a software floating-point operation; this is a local
cost reduction, not a demonstrated fix for the timeout. Context-ladder
acceptance and the 200 committed-token/s objective remain outstanding.

### Small probe completed; 64K cancelled: run 34789440914

The completed GitHub job log resolves the ambiguous `eager_0` messages:

| UTC time | Event |
| --- | --- |
| 23:27:26 | 128-token score smoke started |
| 23:30:14 / 23:31:19 | Replay stages 0 / 1 reached |
| 23:32:27 | Smoke exited 0 after 301 seconds; 64K started |
| 23:32:50 | 64K first eager invocation entered |
| 23:45:52 | User cancellation terminated the job |

The ten-minute smoke timeout did not fail: the smoke had already finished.
The 64K probe was cancelled before its thirty-minute deadline. This rules
out an unconditional hang of the score-centering sequence at small geometry,
but does not prove that the 64K invocation was progressing. Scalar execution
cost and geometry-dependent synchronization remain distinct possibilities.
Only the asset checksum file survived in the uploaded artifact; the detailed
smoke JSON and container log were not retained. Exit 0 and replay progress
are useful evidence, not a substitute for a retained acceptance report.
Do not rerun the same small probe merely to rediscover this outcome.

Progress records now include context and smoke mode. Cancellation-safe result
retention still needs improvement before another costly full-context run.

### Cancellation retention verified: run 34791208534

After user cancellation, the artifact retained the streamed container log,
write-preflight record and partial `dspark-ladder-score-smoke.json`.
Both simulated chips reached `QWEN_SCORE_DONE` for masking and centering.
The labelled 128-token probe reached eager stages 0/1, capture and replay 0;
its partial report correctly remained `passed=false`, `closed_cleanly=false`.
There is no completed acceptance or 64K result from this run.

The permission preflight recorded the image's default UID as 0; dropping
capabilities does not change UID. The group-access fix allowed writes without
adding capabilities or changing that existing container identity. Result
retention now has cancellation evidence, not only local source assertions.
No further full-context simulation is planned; prepare a separately guarded
hardware correctness entrypoint rather than spoofing simulator environment
flags on physical cards.

### Hardware padding fix and remaining live-row error

Run 34793296815 (5afb298) completed in 23 seconds with clean device closure.
The fully masked-chunk centering fix removed the padded-row failures. The
first case passed tensor shard 0, then failed 65 elements in shard 1, head 4,
row 3 (maximum absolute error 0.5759735107421875). Later cases and replay
remain unqualified. Run 34793805954 reproduced those failures with snapshots.

Do not equate the report's `chip` (tensor-list index) to the DPRINT device ID.
For this fixture, shard 1's independently recomputed final proposal score is
166.0517578125, matching DPRINT device 0, not device 1 (166.837890625).
This identifies the matching diagnostic stream by data; explicit mesh-to-device
identity reporting is still needed. Earlier comparisons against device 1's
numerator and reciprocal were invalid.

At maximum 166, the matching device-0 diagnostics give numerator -56 and
denominator 1.240598679; the CPU reference gives -56.38211441040039 and
1.2374624013900757. The raw QK score matches; investigate accumulated output,
denominator precision and final output rounding. Do not infer corrupted QK
or a normalization race from the mismatched device-1 diagnostic stream.

### Rejected final-output rounding: run 34794628077

Candidate 6656de4 inserted native FP32-to-BF16 round-to-nearest in DST only
after the final normalization multiply at 64K. Hardware completed in 22 seconds
and closed cleanly, but retained the same 65 failing elements in shard 1,
head 4, row 3. The first actual/reference pair remained -45 / -45.56269454956055.
This does not establish that the normalization operands retain full precision.
It does show that this final rounding change is insufficient. Remove it from
the active probe; retain the experiment module and test for provenance.

### Rejected FP32 output accumulation with corrected masking

Run 34795359685 tested 340a350: 64K FP32 output intermediates combined with
scalar masking/centering, empty-chunk handling and scalar denominator updates.
It failed 146 elements on first-case shard 0, which passed with BF16 output
intermediates. First failing coordinate was head 0, row 3, channel 1:
-46.75 versus -46.277095794677734. Devices closed cleanly. This is not an
aggregate comparison against the prior 65 errors: execution stopped on the
earlier shard. Reject the candidate and restore BF16 output intermediates.

### Normalization format audit

The pinned native source selects TF32 for FP32 CB operands when FP32 destination
accumulation is enabled and the format family is B (`jit_build/genfiles.cpp`,
`compute_data_formats`). `jit_build/data_format.cpp` preserves that selection
per operand; it does not automatically narrow an FP32 reciprocal to BF16 just
because the numerator is BF16. The broadcast multiply passes both operand IDs
through the generated unpack-format arrays.

Therefore, BF16 reciprocal truncation is not established by the observed -45
output. The read-only `QWEN_NORMALIZATION_FORMATS` snapshot now records the
actual numerator and reciprocal CB IDs and generated source/destination formats
alongside the reciprocal value. Five local source-composition and C++ syntax
tests pass. This is instrumentation, not a numerical fix or hardware result;
no additional hardware run was launched solely for this change.

### One-tile normalization reproduction

`scripts/ci/dspark_normalization_microprobe.py` now reproduces the normalization
multiply without weights, attention history or a native library rebuild. It
uses the same column-broadcast multiply, HiFi4, FP32 destination accumulation,
BF16 numerator -56 and FP32 reciprocal 0.8060624599456787. One simulated core,
one tile; invocation was externally bounded to 120 seconds.

| Output CB | All 1024 output elements | FP32 reference product |
| --- | ---: | ---: |
| BF16 | -45 | -45.13949775695801 |
| FP32 | -45.1171875 | -45.13949775695801 |

Both simulator invocations finished successfully and closed the device. The
FP32 output equals -56 multiplied by 0.8056640625, the reciprocal truncated to
TF32. That product rounds to BF16 -45, whereas the full-precision product rounds
to -45.25. This explains why rounding only after the multiply did not change
the observed coordinate. It does not establish all 65 errors have this cause,
nor qualify a fix: output accumulation and denominator already differ from the
reference before normalization. Next isolate a normalization implementation
that avoids this source-register precision loss, without changing shared
intermediate formats or globally overriding unpack behavior.

The same microprobe now has `--sfpu-scalar`, which copies the BF16 numerator
into DST and calls native `mul_unary_tile` with an FP32 runtime scalar. It
returns -45.25 with BF16 output and -45.139495849609375 with `--fp32-output`.
The FP32 variant passes exact equality to the FP32-rounded reference for all
1024 elements; both executions close cleanly. The scalar is deliberately
supplied as a runtime argument to isolate arithmetic, not read from the
attention denominator CB. This is not yet a drop-in SDPA normalization: the
real operation needs independent per-row reciprocals, correct synchronization
and no shared-CB format changes. No full-context or hardware retry is justified
by the constant-scalar test alone.

`--sfpu-column --vary-rows` now tests the native `sfpu_mul_bcast_col` operation
with 32 independent FP32 reciprocals in a dedicated CB. Only that CB has
`UnpackToDestFp32` enabled (the descriptor requires 64 mode entries); the BF16
numerator retains its default format. Values outside reciprocal column zero
are deliberately set to 7, not duplicated from column zero. Both BF16 and
`--fp32-output` simulator executions pass exact equality at every one of 1024
coordinates and close cleanly. The cached BF16 invocation completed in about
eight seconds wall-clock, including simulator startup; this is not device
performance evidence.

This establishes a per-row primitive candidate, not integration. In SDPA the
existing denominator CB is also consumed by matrix/binary operations, so do not
enable full-FP32 unpack globally on that shared CB. Integration needs a dedicated
normalization scratch CB with explicit copy/publication and lifetime ownership,
followed by the unchanged 64K correctness gate before any model ladder claim.

The `--scratch-copy` microprobe now exercises that handoff: publish an FP32
scratch tile through the existing unpack/math/pack handshake, wait for it on
TR0, replace its bits from the still-owned denominator tile, then unpack only
the scratch at full FP32 for the SFPU operation. The original denominator CB
keeps its default TF32 unpack format. With 32 different reciprocals and poisoned
non-column-zero values, all 1024 FP32 outputs match exactly in simulation.

`dspark_ladder_normalization.py` contains the shared helper and narrowly scoped
factory/kernel transformations. Source composition against the pinned factory
and kernels passes. These transformations are not activated in the ladder yet:
the next checks must cover four numerator tiles (D=128), repeated scratch reuse,
and the composed kernel build before dispatching hardware correctness.

D=128 and scratch reuse are now covered by `--tiles 4 --repeats 3`: each
numerator tile/cycle has a different value, reciprocals differ by row, and
unused reciprocal columns remain poisoned. Both BF16 and FP32 simulator runs
pass exact equality across 12,288 outputs and close cleanly. Seventeen focused
tests cover source composition, report selection, factory restoration and
existing baseline selectors. The first broader test pass exposed stale report
expectations and the need for an explicit reversible factory transform; those
were corrected without relaxing source-hash validation.

The candidate is now selected only for the experimental 64K ladder probe.
Other context kernels retain native normalization. The factory creates one
dedicated FP32 scratch tile per compute core and routes only that operand
through direct FP32 unpack. Full combined-kernel compilation and 64K hardware
correctness remain pending; this is not a performance-qualified change.

### Hardware 64K correctness passes: 34797353681

Revision 770bc5a passed on the two-card hardware runner. The device probe took
28 seconds (excluding native rebuild). All four eager and four replay checks
pass; replay is exact against eager, with zero out-of-tolerance elements.
All 48 input-integrity checks, 16 layout checks, eight fixture controls and two
stale-data controls pass. Devices closed cleanly. Tolerances remain rtol=0.01,
atol=0.01, including padded query rows. Report context is 65,536, capacity
66,560, and normalization mode is `sfpu-column-scratch`.

Report SHA256: `bcf40ffe3834298526dee40daefe1dee88bdd8d05f76c5c3f2fab8885d8a7c20`.
Artifact: `qwen-hardware-inventory-34797353681`,
`dspark-ladder-hardware-65536.json`.

This qualifies the synthetic draft-attention correctness fixture, not the full
Qwen request or coding quality. It provides no PP/TG measurement. Next move
this exact candidate through full-history runtime admission and the combined
context ladder; do not substitute more isolated arithmetic sweeps for that
integration. The 200 committed-token/s objective remains unachieved.

### Combined-request integration status after the passing component gate

| Area | Implemented | Remaining hardware/runtime gate |
| --- | --- | --- |
| Component evidence | Pinned report, all matrices and dependency hashes | No full-request inference from this gate |
| Request admission | Exact 65536 prompt / 256 output, request-local state and combined build validation | Execute the admitted hardware request |
| Draft history | Scoped capacity 66560, transactional publication and cleanup | Execute real learned prefill projection |
| Target headroom | Scoped 66560 positions, 1040 mapped pages, 1048 cache blocks | Verify model initialization and rotary limits |
| Prompt selection | Explicit `QWEN_DSPARK_64K_TRIAL=1`, bounded 12-file corpus | Validate exact length with the real tokenizer |
| Combined entry/CI | Explicit trial input, pinned artifact staging, scoped cache build and CLI guards | Hardware build, real-tokenizer preflight and dispatch |

The folded T16 target-attention gate is separately limited to 4K/8K. Do not
claim the draft-attention pass qualifies that target kernel at 64K, or silently
widen its existing gate. The first 64K comparison uses native target attention
in both arms, captured publication, and the norm reader/scatter comparison.
The combined CCL/GDN cache now creates its own 64K build manifest; the component
ladder manifest is not substituted for it.

Dispatch configuration is `suite=dspark-64k-request`,
`dspark_captured_publication=true`,
`cards_allocated=true`, and `simulator_only=false`. Other candidate flags remain
false. The hardware script rejects incompatible options before downloading
evidence or opening devices. Serving defaults are unchanged. Local admission,
scope, build-fixture and policy tests pass; no 64K full-model PP/TG is recorded yet.

### Combined 64K attempt 34799480361: shared-header compilation failure

The real tokenizer produced exactly 65536 prompt tokens and preflight passed.
The combined native build took 308 seconds. Model/draft upload reached the first
audited request at 196.24 seconds of device-probe elapsed time; the run failed
at 229.88 seconds during the first prefill and closed devices cleanly.

Native target `sdpa_flash_decode.cpp` includes the patched shared attention
header without the draft selector macro. The new reciprocal and centering code
used `QWEN_DRAFT_EXP_APPROX` before the existing late fallback defined it.
This is a combined-runtime include-order bug, not a 64K numerical failure or
device hang. PP and committed TG remain unavailable.

The correction puts an `#ifndef`-guarded native default at the shared header
start within the 64K runtime scope. Draft SDPA still defines its qualified
selector before including that header. Nine local scope/entry tests pass,
including application to the pinned native sources and definition-order checks;
these are not a replacement for a hardware compiler/execution check.

### Retry 34800350550: KV audit frontier excluded decode headroom

The native decode compiler passed the previous failure point. The request then
failed in the gold-output KV digest: its hardcoded 65536 upper bound rejected
the valid prefix after generation beyond the 64K prompt. Devices closed cleanly
and no combined PP/TG result was produced.

The 64K-only correction validates the audit frontier against the admitted 66560
capacity and verifies that physical cache pages cover the requested prefix.
The existing non-64K bound is unchanged; no audit is skipped or truncated.
Boundary tests cover 65537, the final admitted position, overflow, undersized
storage, and calls outside the admitted request scope.

### Retry 34800917218: ordered writer's page-table validation

The request passed the prior KV-digest failure and reached verifier warmup.
The ordered BF8 cache writer then rejected the 1040-column page table because
its fixture validator permits at most 1024 columns. Its buffer allocation and
reader/writer compile arguments already use the actual padded page-table width.

The 64K request scope now installs an exact 1040-page validator for this writer,
retaining row pairing, supported tile geometry, cache capacity and active
admission checks. Outside that scope the original validator remains unchanged.
Twelve local scope and ordered-cache tests pass. Physical 1040-page writer
execution still requires the combined hardware retry; no PP/TG is accepted.

### Run 34801544880: two audits pass, timed control trace shape fails

Both 64K audit arms completed 20 blocks and emitted 135 identical-to-oracle
tokens with exact target-state checks. This is full-request audit evidence,
not held-out coding quality or a completed timed benchmark. Instrumented draft
time averaged 27.04 seconds/block for control and 25.40 seconds/block for scatter;
these include audit work and cannot be reported as production TG.

The native build cache hit reduced build time to three seconds. Device-probe
failure occurred at 1891.74 seconds, after the two expensive audits, when the
first timed gold decode tried copying a 1040-page table into the native control
trace captured by hardcoded `num_blocks=1024` warmup.

Warmup now uses the admitted 1040-page count and records it in the report.
For the 64K experiment it executes before the first audit, still before fresh
prefill resets model state. This surfaces warmup failures earlier without
removing either audit or changing the A/B/B/A timed comparison. Twelve local
request tests pass; a hardware retry must validate the corrected trace reuse.

### Run 34803729259: complete measurements recovered after summary schema error

All six requests completed. CI failed only when the 64K summary compared
`prompt_tokens` (the token-ID list) with integer 65536. The corrected check
requires a 65536-element integer list and matching `length`. Local reprocessing
of the retained report passes all existing publication, output/state, proposal
repeatability, norm-loader and A/B/B/A checks. The original CI result remains
failed; the raw artifact is not modified or relabelled.

| CTX | Mode | PP input tok/s | Committed TG tok/s | Timed requests |
| --- | --- | ---: | ---: | ---: |
| 65536 | Single-stream control, native target attention | 2591.52 | 4.38 | 2 |
| 65536 | Single-stream norm-scatter, native target attention | 2589.66 | 4.39 | 2 |

Each timed request has 135 committed decode tokens. Acceptance is 38.67%; the
scatter change is +0.395%, not a meaningful demonstrated speedup from this small
sample. This runtime is far below the 200 tok/s target. Timed log samples show
roughly 1.38 seconds drafting versus 120 ms verification/readback and 40 ms
selection/commit per block. Unlike the expensive audit timings, these establish
that the draft path itself is slow. No held-out quality or serving qualification
is claimed. Next optimisation must address the 64K draft execution path rather
than repeat the norm-reader comparison or attribute all delay to audit overhead.

### Fast diagnostic cycle

- Local regression tests use fixtures, not model weights; aim for seconds.
- The `dspark-64k-phase-probe` hardware suite loads once and stops after three
  fixed-input proposal replays. It separates input/history updates, blocking
  trace replay, and token readback. It does not measure committed TG.
- From revision `7ee6e29`, the third replay drains queued updates before starting
  the trace timer. The first two retain normal scheduling. Without that fence,
  replay wall time can include pending history copies; it is not kernel time.
- Its host script has a 465-second timeout plus 15-second kill grace, including
  setup. Queueing, checkout and artifact upload are outside that budget.
- Phase JSON is checkpointed to a host bind mount after each replay so completed
  measurements survive timeout. A timeout is not a successful benchmark.
- Full correctness audits and context ladders follow a demonstrated candidate
  improvement, rather than running on every diagnostic edit.

Run `34807274468` (`74add75`) passed all three fixed-input replays and closed
cleanly. The hardware step took 246 seconds including setup. At context 65536,
input/history submission took 2.02–8.49 ms, blocking replay 1372.82–1372.87 ms,
and token readback 0.51–0.55 ms. These unfenced measurements exclude token
readback as the main bottleneck, but do not distinguish queued copies from
compute. No committed tokens or new TG result are claimed.

Run `34807645660` (`e9071aa`) passed the fenced comparison. Normal replay was
1372.73-1372.83 ms. After draining input/history updates, replay was still
1369.62 ms; the fenced update itself took 5.61 ms and readback 1.14 ms.
External queued history updates therefore do not explain the 1.37-second
draft delay. Work inside the captured trace remains the target; these timings
do not separate its internal copies from attention or other compute kernels.

### Scalar score candidate: simulator status

The isolated bitwise infinity check removes two software floating-point
comparison call sites from Blackhole compiler output. Subtraction and addition
are unchanged and still compile to software floating-point calls. This is not
a measured runtime speedup; moving the arithmetic to SFPU remains separate work.

| Run | Outcome | Next action |
| --- | --- | --- |
| `34808847890` | Child launcher lost candidate scope; source audit rejected it | Re-enter through candidate wrapper; keep audit |
| `34809473922` | First eager case passed on both simulated chips; timed out before completion, CI simulator step 395 seconds | Reduce synthetic history, not timeout |
| `34810126959` | Both eager cases and first replay passed; timeout during final replay, CI step 371 seconds | Reuse populated build cache; assign simulation 165 seconds plus 15-second kill grace, within unchanged 480-second outer limit |
| `34810827485` | Passed, clean close, build-cache hit; complete CI simulator step 122 seconds | Bounded 64K hardware diagnostic, not full-request qualification |

The completed simulator run has four eager and four replay checks, 48 input
checks, 16 layout checks, eight fixture controls and two stale-cache controls.
None of these simulator runs admits the candidate into a 64K production request
or proves committed TG. The smaller fixture changes only simulator geometry;
the qualified hardware baseline and serving defaults remain unchanged.

Hardware diagnostic `34811230273` (`b9fb240`) passed all three fixed-input
proposal replays with the bitwise infinity check. Fenced replay decreased from
1369.62 ms in `34807645660` to 977.42 ms, about 28.6% less time (1.40x speed).
The two unfenced samples were 977.75 and 980.20 ms. This is a captured draft-path
measurement at context 65536, not committed TG, a full-request correctness gate,
or coding-quality acceptance. It motivated the SFPU score-centering candidate below.

### SFPU score-centering: simulator and hardware diagnostic

The candidate stages exact FP32 score bits in a dedicated scratch tile and uses
SFPU subtraction, without changing the original score buffer's unpack format.
Tiny simulator run `34815143244` passed numerical and replay checks with clean
close before hardware execution. Serving defaults remain unchanged.

| 64K draft candidate | Fenced replay | Change from original |
| --- | ---: | ---: |
| Original scalar path (`34807645660`) | 1369.62 ms | Baseline |
| Bitwise infinity checks (`34811230273`) | 977.42 ms | 28.6% less time |
| SFPU score centering (`34815969358`, attempt 2, `ac70763`) | 573.56 ms | 58.1% less time |

SFPU is 41.3% lower latency than the bitwise candidate. All three fixed-input
replays passed, with clean device close; unfenced samples were 576.74 and
576.61 ms. Input/history synchronization took 5.71 ms on the fenced sample.
This measures 15 draft proposals at context 65536: **not committed TG**, full
request correctness, or coding-quality acceptance.

Attempt 1 reached prefill but exhausted the eight-minute budget after a cold
runtime build. The compiled library was cached. Attempt 2 reused it and its
hardware step finished in **222 seconds**, with the same timeout and code.
Artifacts are retained under `runner-evidence.local/34815969358-attempt2`.
Next admission requires 64K numerical and full-request output/state validation
before a bounded combined PP/CTX/TG comparison; no throughput result is inferred
from these draft-only timings.

The next numerical gate must validate the cached combined-runtime manifest,
not reuse the standalone ladder build manifest unchanged. The numerical probe
currently expects `binaries_before`/`binaries_after`, while the combined build
validates `binaries` plus exact factory inputs and builder hashes. Its adapter
must retain those checks and the child-process candidate scope; synthesizing
legacy provenance fields would not establish build provenance. Reuse the
existing full 64K fixtures, including poisoned padding and stale-cache controls,
without loading model weights or rebuilding the already qualified library.

Hardware numerical run `34817498912` (`0d225f1`) completed in **37 seconds**
using that combined-build validator. Context is 65536, capacity 66560, tested
positions 65536 and 66545. All four eager, four exact replay, 48 unchanged-input,
16 layout, eight fixture-control and two stale-cache checks passed; clean close
and source restoration passed. Numerical tolerances remain rtol=0.01,
atol=0.01, not a claim of bitwise equality to the reference calculation.
The immutable report SHA256 is
`04d6333660a92e9eeb717e39915f0fb5f6ac954b6a855818e303890c472858e9`.
`dspark_score_sfpu_request_gate.py` verifies this report and current source/build
dependencies before admitting request experiments. Full-request correctness,
committed throughput and held-out coding quality remain separate open gates.

The initial 32-token combined correctness screen (`34818606754`) reached
audited commits but timed out at the unchanged eight-minute limit. Per-block
instrumented draft time was approximately 22 seconds, unlike the 574 ms
uninstrumented replay. These include independent eager execution and state
readbacks; they are not throughput measurements. No correctness assertion was
reported before termination, but final output/state acceptance was not reached.
Use a 16-token fully audited screen to keep this prerequisite bounded. It is
explicitly not full-request acceptance: the subsequent uninstrumented request
must retain the original 256-token budget, exact output and final-state checks.

The 16-token screen (`34819480314`, `203f1ba`) passed in **359 seconds**.
All 15 committed tokens matched the independent target; final active and
inactive state checks, per-block audits and clean close passed. This is still
only a bounded correctness screen. Its report hash is
`800f211f5258a4bccd1dbacd4641f3f67d025709e68c951d667f475f1878ecee`.
The next measurement module requires two 256-budget EOS requests, matching
audited prefixes, exact output/final state, and repeatable proposal histories.
It reports measured PP/CTX/TG separately from sustained or held-out acceptance.

### Combined SFPU result: full requests

Run `34820424369` (`9b9ad85`) passed in **331 seconds** with clean close.
Both requests retained a 256-token budget and reached EOS after 135 committed
tokens. Output, active state and inactive state matched the target; the two
requests reproduced proposal histories and the earlier audited prefix.

| Runtime | PP tok/s | CTX | Committed TG tok/s |
| --- | ---: | ---: | ---: |
| Previous 64K native + scatter | 2589.66 | 65536 | 4.3939 |
| SFPU centering + bitwise checks + scatter | 2621.39 | 65536 | 8.9792 |

This is about 2.04x committed throughput, not 200 tok/s acceptance. It is a
historical comparison, not a fresh matched A/B. The 270 total committed tokens
take 30.069 seconds across both measured decode loops. Acceptance is unchanged
at 116/300 proposed tokens per request (38.67%). Mean block timings over 40
speculative blocks: draft 580.61 ms, verification/readback 117.89 ms,
selection/commit 51.51 ms, complete cycle 751.67 ms. Drafting accounts for
approximately 77% of measured block time and remains the next bottleneck.

Next kernel target: remaining scalar online-softmax sum update, preserving
exact FP32 staging and the established numerical/replay/request gates. Current
SFPU runtime remains the combined control; no serving defaults change. These
two EOS requests do not establish sustained long-output or held-out coding
quality acceptance.

### Online-sum SFPU result

Tiny simulator `34821692776`, full 64K hardware numerical gate `34822563217`,
and fully audited bounded request `34823325684` all passed. Full-budget run
`34824124447` (`03bb2b9`) then completed two exact 135-token EOS requests in
329 seconds, including setup and final-state checks. PP is **2617.59**, CTX
**65536**, committed TG **9.3315**; both requests retain a 256-token budget.
Acceptance remains 116/300 per request. This is 3.92% above the previous
8.9792 TG in a historical comparison, not a fresh matched A/B.

Mean block timing: draft 559.80 ms, verification/readback 117.87 ms,
selection/commit 43.93 ms, cycle 723.29 ms. The modest draft improvement
does not explain away the remaining bottleneck. Next inspect scalar mask
addition for negative-infinity masks; retain zero-mask behavior and the
arithmetic fallback for arbitrary biases, positive infinities and NaNs.

### Exact mask result and remaining latency budget

Mask simulator `34825080088` passed in 66 seconds; full 64K numerical run
`34825617040` passed in 39 seconds, both using cached libraries. Combined
per-block audit `34825996484` passed in 364 seconds. Full-budget measurement
`34826937360` (`53b3d6d`) passed in **328 seconds** with clean close.

| PP tok/s | CTX | Committed TG tok/s | Requests / output |
| ---: | ---: | ---: | --- |
| 2622.59 | 65536 | 11.1557 | Two exact 135-token EOS responses; 256-token budget each |

Output, active state, inactive state and repeated proposal histories pass;
acceptance remains 116/300 per request. This is approximately 19.55% above
9.3315 TG and 2.54x the original 4.3939 TG, using historical comparisons.
It is not held-out coding, sustained long-output or serving acceptance.

Mean block costs: draft **441.08 ms**, verification/readback **117.78 ms**,
selection/commit **44.54 ms**, total **605.01 ms**. At 6.75 committed tokens
per block, 200 TG requires about **33.75 ms per complete block**. Even removing
all current draft time leaves approximately 164 ms: optimizing drafting alone
cannot meet the target while verification and commit stay unchanged.

Next priorities are exact score-staging/packing overhead and the 64K target
verifier path, rather than another small scalar cleanup. Keep this combined
runtime as the measured control. Do not extrapolate its performance to the
different 4K/8K T16 runtime, or promote a component timing to committed TG.

### Folded T16 verifier: component qualified, combined audit pending

Tiny simulator `34828115400` passes in 66 seconds. Full-context hardware run
`34828634864` (`51e1e61`) passes in 35 seconds: eight exact attention comparisons,
16 mask checks, unchanged KV on both chips, stale-input and mask-poison controls,
and clean shutdown. The hardware reader covers a 65792-token capture family
starting at context 65536. Neither run measures committed TG.

Combined audit `34829739931` (`b49430b`) adds this reader to the measured mask/SFPU
runtime, with no serving changes. It reserves the existing 256-token output
capacity but generates at most **17 tokens**, including the prefill seed.
The previous 16-token screen leaves only 15 decode positions and cannot capture
a T16 verifier bucket. The new gate requires five capture buckets and an actual
16-row block, plus all output, state, proposal and publication audits.

The execution cap remains 465 seconds plus 15 seconds for forced cleanup;
queue time is separate. Only after this combined audit passes should the
candidate proceed to repeated full-response PP/CTX/TG measurement. The current
accepted 64K control remains **11.1557 committed TG**.
