# Split-K draft attention: experimental, not admitted

The objective remains 200 committed tokens/s for one coding stream on two P150A cards. These are weight-free simulator experiments, not throughput measurements.

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

The last simulator report has 65,188 failing elements, maximum absolute error 149.92046, and finite outputs. Sample outputs are around -179 to -190 where the FP32 reference is around -49.76. Tolerances remain rtol=0.01 and atol=0.01.

## Next diagnostic

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
