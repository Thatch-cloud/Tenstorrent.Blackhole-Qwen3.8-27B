# 8K draft attention: not qualified

The 8K target-attention replay gate passes, but draft attention does not yet
pass its unchanged FP32-reference tolerance. There is no 8K PP/TG result.

| Draft configuration | CI run | First failing comparison |
| --- | --- | --- |
| 64-key chunks, precise exponential | 34705245937 | Chip 1: 256 values, heads 11/13 at query rows 13/11 |
| 32-key chunks, precise exponential | 34705535416 | Chip 0: 512 values, maximum absolute error 0.54798 |

Both use capacity 8448, logical positions 8192 and 8433, 15 live queries,
the same synthetic fixtures, and rtol/atol 0.01. Both stop at eager case 0;
neither has complete replay coverage. The 64-key diagnostic has finite output
near -46 versus FP32 reference near -45.46 in an affected row. These are live
queries, not padding. Fixture negative controls pass on both host shards.

## Precision experiments

| Configuration | CI run | Numerical outcome |
| --- | --- | --- |
| FP32 statistics and output intermediates | 34723541474 | Chip 0: 15,843 failures; max error 0.92117 |
| Rebuilt native control | 34723984465 | Exact original output hashes; same 256 failures |
| FP32 statistics only | 34724444926 | Chip 0: 30,720 failures; max error 17.85616 |
| Statistics plus explicit pack-format transition | 34724923953 | Chip 0 passes; chip 1 still has 256 failures |
| Above plus nonlegacy reciprocal | 34725385866 | Same 256 failures; max error 0.54706 |

These are failed numerical trials, not accepted optimizations. The rebuilt
control reproduces the prebuilt baseline exactly. Adding the missing explicit
pack-format transition removes the large mixed-format regression, but does not
resolve the original error. The reciprocal change is not a solution.

## CPU rounding control

`scripts/ci/dspark_attention_rounding_control.py` runs online softmax with 64-key
chunks against the same FP32 reference, two fixtures and both host shards.
It tests FP32 intermediates, BF16 partial/running output rounding, BF16
probability rounding, and both rounding options together. The expanded control
also follows native ordering (unscaled maximum, then scaled subtraction), BF16
maximum/correction storage, and truncation of the scale to its upper 16 bits.
All 32 comparisons pass the unchanged tolerance. On the failing simulator shard
(case 0, chip 1), even combined output/statistics rounding and truncated scaling
has maximum absolute error 0.38304, below that row's relative tolerance.

This CPU-only check takes seconds and loads no model weights or device. It
does **not** emulate Tensix instructions or qualify a kernel. The tested
rounding mechanisms alone do not reproduce the simulator failure. Next isolate
native score scaling, maximum subtraction and exponential arithmetic rather
than dispatch another reciprocal or chunk-size trial.

## Correction exponential trial

The native first-column correction uses the degree-four polynomial when
`EXP_APPROX_MODE` is false. The already-grafted probability-tile path instead
uses `_ckernel_sfpu_exp_accurate_` with FP32 destination accumulation. The next
synthetic simulator trial selects that accurate helper for the correction too,
only for the existing draft signature. Counterintuitively, `true` in
`exp_tile_first_column` selects this helper; it is not the fast approximate
probability-tile path. This difference is a hypothesis, not a demonstrated bug.

Keep FP32 statistics and explicit pack-format repair; restore the native
reciprocal so the comparator is run 34724923953, not the failed reciprocal
trial. Record both exponential headers in the runtime fingerprint. Host tests
verify scope restoration and the transformation matches pinned native sources;
only the unchanged simulator numerical/replay matrix can qualify the change.

## Source-backed precision scope

The pinned SDPA factory uses FP32 QK and sum buffers when FP32 destination
accumulation is enabled, but keeps `im_df` and `stats_df` in BF16. Thus the
FP32 accumulation flag does not make every attention intermediate FP32.
This is a plausible contributor, not a proven explanation of the failures.

Local factory source matches the failed CI report exactly:
`a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc`.
Relevant definitions are around lines 704-718 of
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`.

Stop blind chunk-size changes. The isolated draft-only FP32 variants above
failed; do not promote them. Keep target attention and serving defaults unchanged. Require
the existing negative controls, full accuracy matrix, exact changed-input trace
replay and clean resource teardown before any full-request hardware trial.

32-key report SHA256:
`3cd1178e1888cf669a637c4af5f69aaa2f793ce225bb4e81fd3a32ae2f9e4331`.
