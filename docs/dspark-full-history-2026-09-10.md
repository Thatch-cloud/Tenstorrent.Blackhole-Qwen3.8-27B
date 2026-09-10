# DSpark: full history and wider verification

**4K attention passes TTsim. The full 15-proposal request is not yet hardware-tested.**
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

The cache uses existing 32-row learned projection, normalization and rotary
operations. Host tests do **not** qualify their new native composition. Initial
prefill setup and per-publication concatenation costs still need measurement.

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
All **1,256 CI host tests** pass after the cache and request-bridge changes.

## Next hardware path

1. Connect actual target embeddings/head and Markov feedback at 15 queries.
2. Run a focused layout simulation for ragged history append and the widened
   target boundaries, not another whole learned-model simulation.
3. Run full 4K coding requests through the batched verifier and committed-feature
   cache. Compare every committed token and target state against native controls.
4. Report PP / CTX / committed TG, acceptance, setup and complete cycle timings.
   Reuse the native build cache and reject losers before the wider context ladder.

No hardware run has been dispatched for this new cache. Serving defaults remain
unchanged; passing component tests is not permission to deploy the drafter.
