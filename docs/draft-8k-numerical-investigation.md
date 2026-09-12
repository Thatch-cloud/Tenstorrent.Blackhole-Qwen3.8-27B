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

## Source-backed next experiment

The pinned SDPA factory uses FP32 QK and sum buffers when FP32 destination
accumulation is enabled, but keeps `im_df` and `stats_df` in BF16. Thus the
FP32 accumulation flag does not make every attention intermediate FP32.
This is a plausible contributor, not a proven explanation of the failures.

Local factory source matches the failed CI report exactly:
`a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc`.
Relevant definitions are around lines 704-718 of
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`.

Stop blind chunk-size changes. Next test an isolated draft-only FP32 statistics
and intermediate-buffer factory variant, checking L1 footprint and unpack-format
assumptions first. Keep target attention and serving defaults unchanged. Require
the existing negative controls, full accuracy matrix, exact changed-input trace
replay and clean resource teardown before any full-request hardware trial.

32-key report SHA256:
`3cd1178e1888cf669a637c4af5f69aaa2f793ce225bb4e81fd3a32ae2f9e4331`.
