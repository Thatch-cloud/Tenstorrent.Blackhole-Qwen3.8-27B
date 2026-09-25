# Shared-Q/K with direct-scatter normalization

**Combined correctness passes; no whole-request TG gain. Not promoted.**

## Matched combined outcome

Run **35216035465**, source `d79f9de119c9d2ebaa44b23190fcb00b88e105a7`,
finishes in **6m26s**, including loading, two audits and four ABBA timed requests.
The independent validator checks the complete report and recomputes the summary.

| Norm reader | CTX / streams | Cold PP | Complete-cycle TG |
| --- | --- | ---: | ---: |
| Winning prefetch | 4,096 / 1 | 3,348.80 | **122.3352** |
| Direct scatter | 4,096 / 1 | 3,324.58 | **122.3076** |

Aggregate TG changes **-0.0225%**. Paired changes are **+0.9818%** and
**-1.0120%**, failing the repeatable-improvement screen. Both arms commit 242
timed tokens and accept 224 of 300 proposals, with identical target responses.

| Mean timed phase per block | Prefetch | Scatter |
| --- | ---: | ---: |
| Draft | 26.44 ms | 26.96 ms |
| Verification/readback | 66.60 ms | 65.75 ms |
| Blocking verifier replay (nested) | 65.78 ms | 64.96 ms |
| Selection/commit | 5.04 ms | 5.25 ms |
| Complete cycle | 98.86 ms | 98.88 ms |

The approximately 0.82 ms verifier reduction appears in both repetitions, but
does not establish a whole-request win: drafting/commit variation offsets it.
Keep this as a correctness-qualified component candidate, not a new default or
an added claimed speedup. Do not repeat norm-reader sweeps to chase the much
larger 200-TG gap. Cold PP variation is not caused by a decode-only reader change.

All token/state/inactive-slot, feature/proposal, packed-weight, executed shared-Q/K,
draft-tail and incremental-publication checks pass. Sources remain unchanged,
the report closes cleanly and process exit is zero. Report SHA-256:
`28c49a453fd82a7cf093c55d844a71b272eac1a2cf8bb8bb3fcb0168ce1571db`.
Held-out coding quality and serving remain unqualified.

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

The source-bound simulator admission helper and combined comparison are now
implemented. Fresh frozen-runtime staging passes against the retained evidence.
The scoped comparison replaces only the norm scope inside the existing draft-tail
scope, retaining tail assembly and incremental history in both arms. It records
the actual reader, report hash and executed three-stage builds per request; it
does not label direct scatter as prefetch. Host tests exercise all six requests,
reader selection, admission identities and restoration after an injected failure.

`qwen-shared-qk-norm-combined.yml` loads the model once for two fresh audits and
four complete ABBA requests at 4K. It retains the exclusive-card and disk-pressure
gates, a 600-second launcher cap and 12-minute whole-job cap. The context ladder
is unchanged. Both paired TG changes must exceed 2% for the improvement screen;
neither a green job nor that screen automatically promotes the candidate.
The completed combined result above supersedes the implementation gates.
Serving defaults and the accepted recipe remain unchanged.
