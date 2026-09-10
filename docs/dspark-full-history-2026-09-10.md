# DSpark: full history and wider verification

**4K attention and fifteen-query layouts pass TTsim. Full-request hardware measurement is next.**
Best repeat-confirmed single-stream hardware TG remains **74.27 at CTX 4,096**.

## What is ready

| Component | Change | Evidence |
| --- | --- | --- |
| Full attention | One global softmax over 2,048 / 2,048 / 64-key chunks | 58 simulator checks pass; clean exit 0 |
| Native prefill capture | Retain all five taps from every chunk, starting at position zero | Host tests; no 2,048-row sliding window |
| Learned history cache | Project each 32-row block once; cache all five layers' K/V | Host orchestration tests; native integration pending |
| Cache publication | Prepare only the verified input prefix; commit after target publication | Host tests cover rejection, discard, failure and ragged tails |
| Cached layer | Seven or 15 query rows attend to all cached history | Host tests; no historical re-projection inside the layer |
| Request bridge | Explicit `dspark` route with 15 proposals and a T16 target verifier | Host request-loop test; DFlash2 defaults unchanged |
| Wider device plumbing | Target embeddings, five cached layers, full target head and 15-step Markov feedback | Host tests; complete native execution pending |
| Full-request adapter | Native token/state controls plus all-tap committed-feature audit; timed requests exclude that audit | Connected to the explicit `dspark-request` CI suite; hardware result pending |

The cache uses existing 32-row learned projection, normalization and rotary
operations. Host tests do **not** qualify their new native composition. Initial
prefill setup and per-publication concatenation costs still need measurement.
The wider layout probe `20260910T004253Z-387`, source `218fe82`, passes all
126 checks and closes cleanly with exit 0. This includes full-vocabulary
transport, all fifteen packed tokens, wide prefill snapshots, and ragged history
at 4,093 / 4,096 / 4,108 rows. Both eager cases and changing-input replays are exact.

## Simulator evidence

Run `20260909T235443Z-398`, source revision `04f520e`, checks CTX 4,096 plus
15 query rows on both simulated chips. No model weights are loaded.

| Check | Passed |
| --- | ---: |
| Eager output versus FP32 full-attention reference | 4 |
| Changing-input replay, also bitwise equal to its eager output | 6 |
| Borrowed input immutability | 40 |
| Oldest-history, final-proposal and padding fixture controls | 6 |
| Stale-output controls | 2 |

The independent audit checks exact matrix coordinates, regenerates reference
hashes, reconciles all 35 source and 1,517 native fingerprints, and verifies clean
teardown. The original `rtol=atol=0.01` is unchanged. Peak cgroup memory is
1,631,444,992 bytes; no swap or OOM. This is not a physical-fabric or TG result,
and it does not clear earlier learned-backbone numerical failures.

Retained report: `scripts/ci/dspark-full-attention-simulator.json`.
SHA-256: `7fa3290673df7b77aaed954ab55d683caedbcc51e32da4eb10df8eb292829499`.
The wider report is `scripts/ci/dspark-wide-layout-simulator.json`, SHA-256
`3746601ab4c6b45b5287b1e41e2bc99d6a74cc25cd9764c92c24d710510dc6ef`.
An independent audit reconciles all 126 checks, 36 source files and 1,517 native
fingerprints. Peak cgroup memory is 2,396,958,720 bytes, with no swap or OOM.
Simulator single-link discovery messages do not qualify physical fabric links;
the hardware lane uses four explicit proposal/sampling links and rejects fallback.
All **1,275 CI host tests** pass; the unchanged simulator harness passes **61 tests**.

## Next hardware path

1. Run the explicit `dspark-request` suite using the audited simulator reports
   and cached native build. Load the real target and learned drafter once.
2. Run the complete `DSparkDevice` through the request bridge. Its initial eager
   implementation performs one packed token read per chip, not fifteen separate
   scalar readbacks. It has not yet been measured or captured as a proposal trace.
3. Run full 4K coding requests through the batched verifier and committed-feature
   cache. Compare every committed token and target state against native controls.
4. Report PP / CTX / committed TG, acceptance, setup and complete cycle timings.
   Reuse the native build cache and reject losers before the wider context ladder.

The lane executes one full feature-audited request and two timed requests at a
requested 4K context, each with a 257-token generation cap. It preserves all
native tokens, full recurrent state, valid KV and inactive slots; reports pooled
PP / actual CTX / committed TG, and keeps all stalls and setup costs visible.
Serving defaults remain unchanged; this is not permission to deploy the drafter.
