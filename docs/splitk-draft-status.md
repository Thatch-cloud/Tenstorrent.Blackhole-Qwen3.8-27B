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

The CPU-only `dspark_splitk_precision_oracle.py` reproduces the exact 128-context fixture. Keeping all arithmetic FP32 passes; rounding only QK scores to BF16 produces 7,936 failing elements and maximum error 1.06045 (truncation: 7,808 and 1.00553). This establishes that BF16 score storage alone is insufficient for the retained tolerance on this fixture. It is not a complete emulator or proof that all device error comes from that operation. Next preserve FP32 scores through centering/exponentiation rather than trying more final-normalization changes.

Run 34852917157 snapshots show the remaining error already exists before final normalization. For chip 0, lane 0, first row, device numerator is -55.75 and denominator 1.109375 (ratio about -50.2535); the FP32 reference ratio is -49.7590. Individual numerator/sum magnitudes can depend on the chosen softmax maximum, so their ratio is the relevant comparison. Replacing the reciprocal alone did not reduce the 9,939 failing elements or 1.15783 maximum error.

Explicit cross-core correction (34851806253) reduced the redistributed-key maximum error from 93,699.83 to 1.15783 without changing FP32 destination mode. The proposed doubled register stride (34851366019) worsened error to about 1.08e37 and was rejected. An SFPU final-normalization replacement (34852498907) produced infinities and was removed. Next inspect score packing, exponent/sum precision and weighted-value accumulation rather than continuing to change the final division.

The one-core chunked control retains the same maximum error as one full-history chunk, while sixteen cores produce a much larger error. Cross-core reduction or independently initialized masked partitions therefore account for the large additional failure in this fixture; ordinary serial chunk recurrence alone does not reproduce it. A separate smaller numerical error remains even without cross-core reduction. Neither control qualifies the runtime.

Next isolate empty partitions by distributing the same valid and masked keys across partitions without dropping or duplicating any key. Retain a separate precision investigation for the single-core error. These controls must not be promoted as the final low-parallelism runtime.

Inspect device layout round trips before changing arithmetic. Then isolate fully masked key partitions from cross-core reduction. Native decode computes exp((score - maximum) * scale); fully masked partitions introduce negative-infinity subtraction, making empty-partition handling a hypothesis worth testing, not a proven cause. Mask reader batch offsets appear consistent with four lanes on source inspection, but require device evidence.

No hardware admission, serving-default change, or speedup claim follows from these runs. Retain poisoned padding, exact replay, input/layout checks, and the mixed target-attention gate. A green workflow must also report the native split-K candidate and observed adapter executions.
