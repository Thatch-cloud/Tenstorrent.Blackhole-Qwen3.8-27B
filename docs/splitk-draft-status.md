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
