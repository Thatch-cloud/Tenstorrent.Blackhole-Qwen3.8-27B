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
