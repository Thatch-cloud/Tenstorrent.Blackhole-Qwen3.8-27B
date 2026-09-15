# Split-K draft attention: 64K accuracy repair

The objective remains 200 committed tokens/s for one coding stream on two P150A cards. These weight-free kernel experiments are not committed-token throughput measurements.

**Retained simulator milestone:** run 34911012256 passes original-order 16-worker-per-KV-lane
simulation with FP32 local denominators and tree payloads, and round-indexed scratch.
Four eager numerical checks, four exact replay checks and the native target gate pass;
both scopes close cleanly. `dspark_splitk_sim_gate.py` pins the report and verifies
matching source dependencies before hardware admission. The pinned report SHA256 is
`90eb2538f9f97eab84d727b5d4e662fe20d0a2594e7b647dd27bbee385594d96`.

**64K hardware result:** run 34912625173 built and imported successfully, then
failed the first eager numerical check: 2,502 elements outside tolerance, maximum
absolute error 0.727303 on chip 0. Devices closed cleanly. Constant-value, oldest-key
and last-proposal diagnostics pass their existing tolerances on both chips; that
does not qualify the mixed-value attention output. No replay qualification was reached.

| Current work | Status |
| --- | --- |
| Hardware build | Verified cache reuse; latest hardware job took 34 seconds |
| Hardware numerical execution | Diagnostics and first eager check took about five seconds |
| Local denominator recurrence | Small simulator passes; full 64K still fails |
| Next isolated change | 128-key chunks instead of 32; unchanged FP32 tree candidate, tolerances and full history |
| Combined requests / TG | Blocked on numerical qualification; no new throughput result |

### Subsequent 64K results

| Run | Local recurrence | Failed elements (chip 0, first eager) | Maximum error | Outcome |
| --- | --- | ---: | ---: | --- |
| 34913947170 | FP32 denominator, native numerator; eight workers | 1,976 | 0.643814 | Fails; clean close |
| 34915054290 | FP32 denominator and fused FP32 numerator; eight workers | 22,956 | 1.161568 | Rejected; clean close |
| 34915414445 | Denominator-only control with value-linearity diagnostics | 1,976 | 0.643814 | Sign reversal and power-of-two scaling are exact; mixed output still fails |
| 34916226351 | Explicit BF16 correction-factor rounding | 1,976 | 0.643814 | Identical output hash to control; ineffective |
| 34917227924 | FP32 tree denominator arithmetic and transport | 1,624 | 0.636242 | Partial improvement, still fails; clean close, 36-second job |

The fused numerator passed the small simulator gate (34914738395), but worsened
the matched 64K hardware result. It is removed from the active candidate; its
helper and tests remain as experiment history. Denominator-only is still not
hardware-qualified. Do not rerun that unchanged baseline merely to reconfirm it.

The latest hardware job completed in **34 seconds** with a verified compiled-binary
cache hit, versus 5m17s for the preceding build-heavy job. Admission reports still
undergo current source checks; only identical compiled factory inputs are reused.

The FP32 tree candidate passes simulator run 34916906739 in 2m17s, but still fails
the full 64K hardware screen. The next candidate changes only the key chunk from
32 to 128: four times fewer chunks at fixed history and worker limit. This targets
repeated arithmetic/packing overhead and accumulation error, not a shorter context.
Its small simulator gate exercises the larger chunk and tree; it cannot establish
long-recurrence accuracy or performance. A fresh 64K hardware check remains required.

Default FP32 FPU unpack
is TF32 in the pinned source: FP32 buffer storage alone does not preserve all bits.
This is a hypothesis test, not a demonstrated explanation for the full error.
The small simulator must pass before a fresh, source-matched 64K hardware test.

A CPU ablation on chip-0/head-6/row-2 predicts a last-proposal weighting error
equivalent to -0.340 output units when denominator reloads are truncated, versus
-0.000244 without that truncation. This is evidence for a contributor, not an exact
device model or proof of the complete cause. Keep tolerances and full-history checks.

## Early experiment history

| CI run | Candidate | Observed result |
|---|---|---|
| 34847175693 | Initial wrapper | Checkout failed on root-owned profiler output; simulator did not start. |
| 34847477310 | Checkout ownership repair | Green, but child process ran the old candidate. Not split-K evidence. |
| 34847925528 | Observed split-K execution, 256-key tiles | L1 allocation rejected: 5,131,136 bytes against 1,572,864. |
| 34848216379 | 32-key tiles | L1 allocation still rejected: 4,443,008 bytes. |
| 34848579371 | Four KV groups mapped to batch lanes | Allocation succeeded; first eager numerical gate failed. |
| 34849128703 | Exact input layout audit | All eight query/KV/mask checks passed across two chips; attention still failed. |
| 34849417468 | One core, one 512-key partition per lane | 8,023 elements failed; maximum absolute error 1.15783. |
| 34849801082 | One core, sixteen 32-key chunks per lane | 7,767 elements failed; maximum absolute error 1.15783. |
| 34850289534 | Sixteen cores with redistributed keys | 54,942 elements failed; maximum absolute error 93,699.83. Reordered device layouts require a separate audit before interpreting this control. |

The batch-lane representation preserves all 32 padded query rows, all four query heads per KV group, full history and the capacity-gap mask. KV tensors are reshaped, not replicated. CPU equivalence tests pass; this does not establish correct device execution.

An early simulator report had 65,188 failing elements, maximum absolute error 149.92046, and finite outputs. Sample outputs were around -179 to -190 where the FP32 reference was around -49.76. That failure is superseded by the milestone above; tolerances remain rtol=0.01 and atol=0.01.

## Historical diagnostic progression

Run 34910369212 (artifact 10374307550) fails before numerical execution: static buffers require 2,175,872 bytes versus 1,572,864 bytes L1. The pinned factory at 9f9cd4 allocates c19 for `num_cores_per_head - 1` payloads; the probe's `QWEN_SDPA_TREE_SCRATCH_ROUNDS` environment variable had no implementation in that factory. Writer offsets are indexed by `round`/`send_at_round`, bounded by `num_tree_reduction_rounds`. Allocate c19 by round count only inside the experimental FP32 branch and remove the ineffective environment flag. For 16 workers this reduces scratch from 15 to four payloads, saving 1,081,344 bytes at this geometry; native allocation remains unchanged. Device allocation and correctness must still be tested.

Run 34909852460 attempt 2 (artifact 10374276665) launches the 16-worker configuration. Chip 0 passes case 0; chip 1 fails 247 values in head 3/row 7 and head 9/row 4, max error 0.521545. The earlier attempt failed fabric startup and is not this numerical result. Next promote the tree payload and receiving workspace together: c6/c7, c16/c17/c18/c19 and c21 use FP32 under the experimental guard. Writer source derives all L/M/O byte lengths from a shared tile size, so changing only sum-transfer buffers would corrupt packet offsets. Local maximum/correction storage remains BF16; wire maxima use FP32 only to keep that existing uniform packet contract. Local FP32 denominators and native numerator addition remain unchanged. This requires a bounded factory rebuild; no relaxed tolerance or hardware promotion.

Run 34909491876 (artifact 10374081933) passes the two-worker gate with FP32 local denominators: four eager comparisons, four exact replay checks, native target gate, and clean close in both scopes. Case-0 max errors are 0.362774/0.340893. Next test the intended 16-worker-per-KV-lane configuration on the same 512-key fixture using 32-key partitions and original ordering (up to 64 workers per chip). This exercises deeper tree reduction and masked partitions directly instead of spending separate CI cycles on every intermediate worker count. Requested worker counts are not proof of actual placement or performance; inspect runtime scheduling before hardware timing.

Run 34908895727 (artifact 10373254866) passes after the local FP32-denominator change: four eager checks with zero failing elements, four exact replay checks, three adapter calls, and the native target gate; both scopes close cleanly. Case-0 maximum errors are 0.362774/0.326508; case-1 errors are 0.015106/0.015526. This establishes the single-worker/two-chunk correction, not parallel or hardware performance. Next restore two workers per KV lane with the same 256-key chunks, FP32 local denominators, BF16 transfer statistics, original key order and unchanged gates. The built factory can be reused.

The next candidate changes only local denominator buffers c29/c30 to FP32 under the existing experimental factory guard. Maxima, correction factors and cross-worker transfer buffers remain BF16; native target allocations remain unchanged. It adds explicit recurrence-multiply format configuration and an exact FP32 reciprocal-input copy. The rejected SFPU numerator add stays disabled. Twenty-eight local tests and composed compute/factory transformations pass. This requires a cached-build-key change and bounded factory rebuild; single-worker/two-chunk simulator accuracy must pass before attempting parallel precision changes.

Run 34908303954 (artifact 10373043752) rejects the SFPU numerator-add candidate: 773 failing elements, max 0.689598, versus 258/0.532555 for the local native-add control. Revert its integration; retain the isolated helper as rejected experiment history. The CPU ablation is not a reliable predictor of this device change. Failures remain finite and predominantly biased toward zero, with newly affected head/row groups. Source audit identifies local denominator ping-pong buffers c29/c30 as BF16. A subsequent denominator-precision experiment must account for explicit format reconfiguration before recurrence multiplication and reciprocal copying; changing `stats_df` globally would also alter maxima and cross-worker transfer formats and confound the test. No new CI run is warranted merely to re-prove this rollback.

Run 34907788905 (artifact 10373017962) retains exactly the same 258-element failure and output hash after precise local correction. CPU ablation combining BF16 statistics with truncated TF32 intermediate reloads produces a chunked-only failure (63 elements, max 0.535652); keeping numerator addition in FP32 reduces this model to five failures, not a pass. This model is not an exact device emulator. Next replace only the local numerator FPU add with explicit FP32 unpack and SFPU add, retaining FIFO ownership and all other numerical choices. No factory rebuild or tolerance change is needed. Twenty-seven local tests and the composed native-source transform pass; device correctness remains unproven.

Extended CPU controls with BF16 maxima and truncated correction scaling also pass (no device acceptance); these simplified effects alone still do not reproduce the failure. Source inspection shows the local multi-chunk recurrence retains the original first-column exponential helper, separate from the patched tree merge and local score exponential. Route that single local correction call through the same full-scale precise helper, retaining the single-worker/two-chunk diagnostic and all output checks. This tests the remaining exponential implementation difference rather than asserting it is the root cause.

Run 34907259411 (artifact 10373565837) fails the single-worker/two-chunk control with 258 elements and maximum error 0.532555, versus 773/0.689598 for two workers. Local multi-chunk arithmetic already fails; cross-worker transfer is not the sole cause. A CPU chunk-statistics ablation on the same fixture passes in FP32 and with BF16-rounded denominators/corrections (maximum errors 0.125 and 0.343472 for two chunks). It handles empty masked chunks explicitly and does not emulate all device score/max/PV rounding. BF16 denominator rounding alone therefore does not reproduce the device failure in this simplified model. Inspect local recurrence scaling, rounded maxima and numerator accumulation next rather than claiming a diagnosed transfer defect or blindly rebuilding all buffers as FP32.

Run 34906756708 attempt 1 fails during simulator fabric startup. Attempt 2 launches the reduced-diagnostic kernel and fails accuracy with exactly the same 773 elements, 0.689598 maximum error and output hash as run 34906073385. The FP32 merge scale change therefore has no observable output benefit on this fixture. Attempt-2 artifact ID is 10372368988; the run also contains an older same-named artifact, so unqualified `gh run download` can retrieve stale attempt-1 evidence. Next keep 256-key chunks but use one worker per KV lane: this is a diagnostic control separating local multi-chunk accumulation from cross-worker transfer/merge, not a substitute for the required parallel implementation.

Run 34906539929 cannot launch the new merge kernel: program size 70,752 bytes exceeds the 70,656-byte kernel-config buffer by 96 bytes. No numerical result exists for the merge-scale change yet. Remove the temporary read-only per-row probability-sum audit from the generated kernel; keep the precise merge candidate, layouts, numerical comparisons, replay checks and native target gate unchanged. Do not enlarge hardware limits.

Run 34906073385 fails the first two-worker eager comparison on chip 0: 773 elements outside the retained tolerance, maximum absolute error 0.689598. Original-order single-worker run 34905622466 passed. The change introduced partitioning and cross-core reduction together; this does not yet distinguish partial-statistics rounding, transfer precision, or merge arithmetic. Do not promote or relax the tolerance.

Source inspection finds that the cross-worker `sub_exp_block` still truncates the scale to BF16 (`scale_fp32 >> 16`), whereas local softmax now multiplies by the original FP32 scale and uses the precise exponential. Next replace only the two tree-merge exponent calls with explicit FP32 scaling and precise exponentiation. Keep statistics/transfer formats, worker count, key order, tolerances and target gates unchanged; this is a tested hypothesis, not an established root cause.

Run 34905622466 passes in original key order: four eager checks, four exact replay checks, three adapter calls, and the native target gate; both scopes close cleanly. Next use two 256-key partitions and up to two workers per KV lane to exercise cross-core softmax merging on the same 512-key fixture. No arithmetic, tolerance, target-gate or timeout changes. This is still simulator correctness, not a TG measurement.

Run 34905137219 is green: four eager numerical checks, four exact replay checks, exactly three Python adapter calls, and the native target-attention gate all pass; both scopes close cleanly. This validates the isolated striped single-partition diagnostic only. Next remove key striping while retaining the 512-key partition, one worker per KV lane, FP32 arithmetic changes and all correctness gates. Original-order correctness must pass before restoring parallel split-K and measuring the combined runtime on hardware.

Run 34904225195 passes draft numerical/replay checks and the native target gate with clean close after the isolation fix. The final wrapper fails because it incorrectly requires at least four Python adapter calls: the fixture performs two eager calls plus one capture; replay executes the device trace without a Python adapter call. Require exactly three calls, retaining all per-chip numerical and replay checks. This remains a striped, single-partition simulator diagnostic, not multicore or performance acceptance.

Run 34903134274 attempt 2 again passes the draft scope, then hangs in the native target subprocess even with draft DPRINT removed. Source audit identifies a scope defect in our wrapper: the transformed decode translation unit was installed unconditionally, while only flagged draft geometries allocate c32/c33. Native target calls therefore compiled experimental buffer accesses without allocating those buffers. Add `QWEN_SPLITK_NATIVE_EXPERIMENT` only for the already-guarded factory geometry and retain the original decode translation unit under the opposite preprocessor branch. Both draft and target gates must pass with this boundary; no timeout increase or skipped target check.

Run 34902627956 passes the draft numerical scope: four eager checks, four exact replay checks, 48 input checks, 16 layout checks, eight fixture controls and two stale controls; draft close is clean. First eager maximum errors are 0.446648/0.352352 on the two chips, and second eager 0.490570/0.489685. The overall CI run still fails: the subsequent independent target-attention process stalls after mesh/program-cache setup and the existing outer probe deadline exits 124. No combined admission or hardware promotion. Next prevent the target subprocess from inheriting draft DPRINT settings while retaining the same active headers, target assertions and deadlines; the timeout cause is not yet proven.

Run 34902252785 reaches lane 2 with single-core logging. On chip 0, failing rows 12/55 have stored-probability sums 1.167067885/1.167143703 but reduced denominators 1.171875. This is direct evidence of denominator error, not proof it explains every remaining output difference. Adapt the repo's `draft_row_sum_compute.cpp` SFPU accumulation/reduce sequence to consume the original FP32 probabilities directly, eliminating the BF16 temporary-copy reduction; keep BF16 sum output and all other arithmetic unchanged. The temporary CB allocation remains for this kernel-only diagnostic to avoid another factory rebuild.

Run 34901934714 also times out during simulator fabric initialization with four-core DPRINT enabled. No attention arithmetic ran. Retry the unchanged candidate with only core (2,0), the failing lane for the current single-worker geometry, restoring the one-core logging footprint used by successful-startup controls. Do not extend the fabric timeout or treat these startup failures as numerical regressions.

Run 34901550212 fails in simulator fabric startup (60-second local-router handshake timeout), before attention arithmetic. It supplies no accuracy evidence. Narrow DPRINT from all worker cores to (0,0), (1,0), (2,0), (3,0): source assigns the current DRAM-query, four-batch, one-worker-per-batch geometry linearly across these four cores. This includes failing lane 2 without enabling diagnostics on idle/dispatch cores. The logging change is a bounded mitigation, not proof of what caused the startup timeout.

Run 34901206746 successfully emits read-only probability sums, but the inherited DPRINT selection only includes core (0,0). It therefore does not expose the failing folded lane 2; do not attribute lane-0 row-55 values to that lane. The next diagnostic enables all worker-core DPRINT for the split-K simulator suite only, retaining TR0-only printing, six audited rows and the existing 165-second probe limit. Arithmetic remains unchanged.

Run 34900711534 worsens the first-fixture result to 64 failing elements and maximum error 0.535671. Reject the probability-rounding candidate; restore the prior precise FP32 exponent path (25 failures). The CPU ablation did not predict native behavior closely enough to establish a fix. Next compare the actual device probability sums against the native reduced sums before changing arithmetic again; retain intermediate snapshots and unchanged reference tolerances. No runtime or hardware qualification.

CPU probability-consistency controls on the first fixture show 138 failures (maximum 0.54427) when the denominator uses BF16-truncated probabilities but the numerator uses TF32 probabilities; using identical truncated probabilities in both yields zero failures (maximum 0.35075). RNE variants pass both ways, so this does not prove native truncation is the cause. The next simulator candidate rounds exponent results once to BF16-representable values in FP32 storage, before either consumer, preserving FP32 raw scores. Exact fixture gates still decide acceptance. Exponent initialization is repeated before each tile because the following typecast may change SFPU initialization state.

Run 34899937594 successfully allocates the separate FP32 reciprocal buffer but still fails the same 25 elements with maximum error 0.511822, closing cleanly. Keeping the reciprocal in FP32 is not sufficient to close the gap. Next examine the mismatch between the BF16 probability copy used for the denominator and the FP32 probabilities used for the numerator; neither final rounding nor reciprocal storage improved this fixture. Keep these negative controls distinct from the successful precise-exponent improvement. No hardware promotion.

Run 34899159381 (precise exponent with explicit FP32 scale) reduces first-fixture failures from 3,575 to 25 elements, maximum error 0.511822. Run 34899515768 adds explicit final BF16 typecast rounding but produces the same 25 failures and maximum error; final rounding is not sufficient. Failing indices include head 8/query row 3 and head 11/query row 13 (folded lane 2, rows 12 and 55). Next isolate BF16 reciprocal storage from the sum input: keep the already-corrected reciprocal calculation but retain its result in FP32 through final normalization. This must use a separately allocated, hash-gated CB and preserve producer/consumer counts. No replay or hardware gate has been reached by either run.

Run 34898672073 retains the same 0.770657 maximum error with FP32 numerator normalization (3,575 failing elements, clean close). A CPU ablation on the exact first fixture passes when adding TF32-truncated scores, BF16-truncated maxima, BF16-rounded sums and BF16-rounded reciprocals sequentially (maximum errors 0.2492, 0.2492, 0.3650 and 0.4466 respectively). This is not a native-kernel emulator: exponent approximation, FPU arithmetic and packing differences remain unmodeled. It rules out those isolated idealized rounding stages as a sufficient explanation. Next investigate native exponent approximation and sum-copy rounding rather than changing final normalization again. Native `sub_exp_block_bcast_cols_inplace` hardcodes approximate exponent initialization and calculation; the precise alternative must explicitly preserve scale, not merely toggle its approximation template.

Run 34898339559 confirms the reciprocal initialization repair: chip 0 reciprocal is now 0.83203125 rather than 29.125. The first eager gate remains finite but fails 2,299 elements, maximum absolute error 0.770657. Next remove the BF16 numerator staging introduced to investigate final normalization; retain the corrected reciprocal and BF16 sum input. The prior gross normalization failure was not evidence that the FP32 numerator itself needed conversion. Accuracy tolerances remain unchanged.

Run 34898037968 locates the remaining gross error in reciprocal calculation: chip 0 denominator 1.203125 becomes reciprocal 29.125 (expected about 0.83117), then the BF16 numerator -59.5 correctly multiplies to about -1736. Source inspection finds the diagnostic paired `recip_tile_first_column<false>` with default `recip_tile_init()` (legacy mode true). The next candidate changes initialization to `<false>` to match calculation; it does not change buffers, geometry, tolerances or downstream multiplication. This is a concrete mismatch in our diagnostic, not a validated native-runtime defect.

Run 34897663062 remains finite but inaccurate after BF16 numerator staging (first output -1736 versus -49.758987). That conversion is not sufficient. Next instrument the post-reciprocal denominator, copied numerator and post-multiply output independently, keeping arithmetic unchanged, to distinguish reciprocal corruption from multiplication/output transfer. No hardware admission.

Run 34896898132 fits L1 and removes the observed alternating zero sums. Chip 0 sums are now 1.203125, 1.1171875, 1.234375, 1.234375; first numerator remains about -59.47361. All reported outputs are finite, but the first output is -1728 rather than -49.758987. The final normalization/output path is therefore still incorrect even relative to its own numerator and denominator. The next candidate copies the numerator into existing BF16 c16 scratch before final broadcast normalization, with no extra allocation and no changes to score or value matmul. This remains an accuracy diagnostic, not runtime qualification.

Run 34896117815 did not execute the new sum arithmetic: its extra BF16 probability copy exceeded L1 by 11,136 bytes (1,584,000 required versus 1,572,864 available). The revised diagnostic streams one 32-query tile-row through the scratch buffer, reducing allocation from 128 KiB to 32 KiB at the same 512-key geometry. It retains all four tile-rows and the FP32 input used by value matmul. Local transform tests pass; allocation and correctness still require simulator verification.

Run 34895661245 provides the first stage-value localization (single worker, single partition; failed, not hardware-admitted):

| Chip 0, first folded row | Observed | Interpretation |
|---|---|---|
| Raw QK, first four striped keys | 146.37890625, -72.838562012, -73.574417114, -72.647857666 | Matches the independently calculated CPU fixture samples. |
| After mask addition | 146.375, -72.8125, -73.5625, -72.625 | Arithmetic reload rounds scores despite FP32 storage. |
| Maximum, first four rows | 170, 170, 163, 164 | BF16-rounded; reference maxima are about 170.7754, 170.1055, 163.9053, 164.9717. |
| First exponent per row | 0.121582031, 0.116455078, 0.133789062, 0.133789062 | Positive on every sampled row. Not a full-exponent accuracy check. |
| Reduced sum per row | 0.243164062, 0, 0.234375, 0 | Zero sums contradict positive sampled exponents; the sum path is demonstrably incorrect. |

Next isolate mixed-format reduction with a separate BF16 exponent buffer, retaining FP32 raw scores. Do not assume fixing sum alone repairs final output: weighted-value accumulation and normalization still require the same retained reference gate. This is a producer/consumer format investigation, not evidence for weakening precision or reducing context. Artifacts are retained under `runner-evidence.local/34895661245/qwen-hardware-inventory-34895661245/`.

Run 34895417818 fails even with one 512-key partition: first outputs -8768, nonfinite outputs elsewhere, clean close. Neither cross-core correction nor online chunk recurrence is required for the hybrid error. The next run retains that geometry and adds read-only first-chunk tile snapshots at score, masked-score, maximum, exponent and sum stages. This locates the earliest divergent arithmetic stage rather than making another format guess; no accuracy threshold or serving path changes.

Run 34895149662 retains the single-worker failure after explicit copy-format transitions: first outputs remain -248, and nonfinite outputs remain elsewhere. Before normalization chip 0 shows sum 0.240234375 and numerator -59.5625. Copy-format state was not sufficient to fix the local error. The next control uses one 512-key partition, keeping key ordering, geometry and tolerances unchanged, to remove online recurrence as well as cross-core reduction. Simulator execution reported 15.7 seconds; no hardware admission.

Run 34894814221 also fails with one worker per KV lane (first output -248 versus reference -49.758987), closing cleanly. Cross-core reduction is therefore not required to reproduce the hybrid failure. Source audit finds that tree transfer CBs c16-c19 all retain BF16, so their shared byte size is not itself a mismatch. Separately, `move_block` does not reconfigure unpack/pack formats; hybrid callers cross BF16/FP32 boundaries. The next controlled candidate explicitly configures both formats at all eleven moves, retaining the single-worker setup. This addresses a concrete format-state hazard but does not establish the complete cause of the numerical failure.

Run 34893952832 (f21d6be, FP32 intermediates with BF16 statistics) also fails the first eager accuracy gate, with nonfinite outputs; it closes cleanly. The first output is -2.71875 against -49.758987 in the FP32 reference. Restoring BF16 statistics therefore does not repair the candidate. CI reached this result in approximately five minutes, including its factory build; the simulator reported 15.2 seconds of execution. No replay or hardware admission follows. Evidence is retained under `runner-evidence.local/34893952832/qwen-hardware-inventory-34893952832/dspark-splitk.json`.

Next isolate a single-worker FP32-intermediate path before further cross-core changes, and inspect the decode reader/writer byte strides and reduction format transitions against the factory's changed CB formats. A format change must cover the complete producer/consumer path, not only allocation and compute metadata. This is a diagnostic hypothesis, not an established root cause; retain the same fixture and tolerances.

Run 34893131735 clears the unsupported FP32 unpack trap using the JIT's default TF32 arithmetic inputs, and reaches all instrumented stages. It still fails: 65,408 elements, infinite maximum error. Before normalization, statistic rows alternate between nonzero values and zero, while numerators remain nonzero. The next hybrid diagnostic keeps scores/output intermediates in FP32 but restores native BF16 statistics. This isolates a statistics-format mismatch; it is not an accuracy or performance qualification.

The CPU-only `dspark_splitk_precision_oracle.py` reproduces the exact 128-context fixture. Keeping all arithmetic FP32 passes; rounding only QK scores to BF16 produces 7,936 failing elements and maximum error 1.06045 (truncation: 7,808 and 1.00553). This establishes that BF16 score storage alone is insufficient for the retained tolerance on this fixture. It is not a complete emulator or proof that all device error comes from that operation. Next preserve FP32 scores through centering/exponentiation rather than trying more final-normalization changes.

Run 34852917157 snapshots show the remaining error already exists before final normalization. For chip 0, lane 0, first row, device numerator is -55.75 and denominator 1.109375 (ratio about -50.2535); the FP32 reference ratio is -49.7590. Individual numerator/sum magnitudes can depend on the chosen softmax maximum, so their ratio is the relevant comparison. Replacing the reciprocal alone did not reduce the 9,939 failing elements or 1.15783 maximum error.

Explicit cross-core correction (34851806253) reduced the redistributed-key maximum error from 93,699.83 to 1.15783 without changing FP32 destination mode. The proposed doubled register stride (34851366019) worsened error to about 1.08e37 and was rejected. An SFPU final-normalization replacement (34852498907) produced infinities and was removed. Next inspect score packing, exponent/sum precision and weighted-value accumulation rather than continuing to change the final division.

The one-core chunked control retains the same maximum error as one full-history chunk, while sixteen cores produce a much larger error. Cross-core reduction or independently initialized masked partitions therefore account for the large additional failure in this fixture; ordinary serial chunk recurrence alone does not reproduce it. A separate smaller numerical error remains even without cross-core reduction. Neither control qualifies the runtime.

Next isolate empty partitions by distributing the same valid and masked keys across partitions without dropping or duplicating any key. Retain a separate precision investigation for the single-core error. These controls must not be promoted as the final low-parallelism runtime.

Inspect device layout round trips before changing arithmetic. Then isolate fully masked key partitions from cross-core reduction. Native decode computes exp((score - maximum) * scale); fully masked partitions introduce negative-infinity subtraction, making empty-partition handling a hypothesis worth testing, not a proven cause. Mask reader batch offsets appear consistent with four lanes on source inspection, but require device evidence.

No hardware admission, serving-default change, or speedup claim follows from these runs. Retain poisoned padding, exact replay, input/layout checks, and the mixed target-attention gate. A green workflow must also report the native split-K candidate and observed adapter executions.
