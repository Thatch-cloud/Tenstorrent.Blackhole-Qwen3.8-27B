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

The one-core chunked control retains the same maximum error as one full-history chunk, while sixteen cores produce a much larger error. Cross-core reduction or independently initialized masked partitions therefore account for the large additional failure in this fixture; ordinary serial chunk recurrence alone does not reproduce it. A separate smaller numerical error remains even without cross-core reduction. Neither control qualifies the runtime.

Next isolate empty partitions by distributing the same valid and masked keys across partitions without dropping or duplicating any key. Retain a separate precision investigation for the single-core error. These controls must not be promoted as the final low-parallelism runtime.

Inspect device layout round trips before changing arithmetic. Then isolate fully masked key partitions from cross-core reduction. Native decode computes exp((score - maximum) * scale); fully masked partitions introduce negative-infinity subtraction, making empty-partition handling a hypothesis worth testing, not a proven cause. Mask reader batch offsets appear consistent with four lanes on source inspection, but require device evidence.

No hardware admission, serving-default change, or speedup claim follows from these runs. Retain poisoned padding, exact replay, input/layout checks, and the mixed target-attention gate. A green workflow must also report the native split-K candidate and observed adapter executions.
