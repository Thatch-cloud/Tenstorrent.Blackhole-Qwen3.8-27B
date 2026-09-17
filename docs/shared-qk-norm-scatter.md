# Shared-Q/K with direct-scatter normalization

**Shared-Q/K composition passes simulation; combined performance is unqualified.**

## Simulator result

Run **35214644611**, source `1b91e469e8ae92fba888446b1c8b09d02a5bac41`,
passes in **3m02s**; the actual probe takes 165.66 seconds. All 24 exact
output/bridge/state comparisons and 48 input-integrity checks pass across eager
execution and three changed-input replays on both chips. Exit status and all
four container-cleanup statuses are zero; sources remain unchanged.

The independent validator checks the full matrix, retained report digest,
staged probe identity and current numerical Python dependencies. Runtime
admission additionally requires the native source files to match that report.
Report SHA-256:
`33f2a5dc51918aa34dbda412e07d7595105ef1edd5010e4de20a4ede1dfe9602`.

This is a correctness result, not a throughput measurement. The next gate is
the matched combined comparison against today's prefetched reader.

## Candidate

The earlier direct-scatter norm reader passed exact simulator and combined
hardware checks in run 34451473973. It reduced verification/readback by 1.92 ms
on that older recipe, but its 1.39% TG gain was not repeat-confirmed. The current
winning runtime instead uses shared-Q/K and the prefetched norm reader.

This candidate reuses `gdn_norm_scatter.py` unchanged inside the current
three-stage shared-Q/K pipeline. It does not revert shared normalization or
modify recurrence, norm arithmetic, precision, output publication or weights.
The new scoped wrapper changes only the norm reader and restores it on failure.

| Current reader | Candidate |
| --- | --- |
| 64 full 128-byte reads into scratch | 128 direct 64-byte face-row reads |
| RISC-V copies scratch into four FP32 tiles | DMA lands in those same four tiles |
| One bridge read barrier | One bridge read barrier |
| Zero-filled inactive tile rows | Same zero-filled inactive rows |

The historical gain was against a serial reader, **not today's prefetch**.
Doubling packet count may erase the copy saving. Do not add historical gains
or infer this candidate reaches 200 TG.

## Gates

1. Reuse the bounded synthetic shared-Q/K eager and changed-input replay matrix
   on both simulated chips, including bridge/state/output and input integrity.
   Tag: `experiment/shared-qk-norm-scatter-sim-v1`; 570-second launcher and
   12-minute whole-job cap. No target weight loading.
2. Bind the resulting source hashes and report to a fresh admission check.
3. Compare against the unchanged winning **prefetched** T16 reader in the same
   loaded model, with native token/state/feature checks and complete-cycle TG.
   Keep incremental publication and every other winning option in both arms.

The source-bound simulator admission helper is now implemented. Hardware
integration and combined performance remain outstanding. Serving defaults and
the accepted recipe remain unchanged.
