# DSpark: full history and wider verification

**4K attention and fifteen-query layouts pass TTsim. The first full-request hardware screen fails.**
Best repeat-confirmed single-stream hardware TG remains **74.27 at CTX 4,096**.

## What is ready

| Component | Change | Evidence |
| --- | --- | --- |
| Full attention | One global softmax over 2,048 / 2,048 / 64-key chunks | 58 simulator checks pass; clean exit 0 |
| Native prefill capture | Retain all five taps from every chunk, starting at position zero | Host tests; no 2,048-row sliding window |
| Learned history cache | Project each 32-row block once; cache all five layers' K/V | Full request executes; cache lifetime is under investigation |
| Cache publication | Prepare only the verified input prefix; commit after target publication | Host tests cover rejection, discard, failure and ragged tails |
| Cached layer | Seven or 15 query rows attend to all cached history | Host tests; no historical re-projection inside the layer |
| Request bridge | Explicit `dspark` route with 15 proposals and a T16 target verifier | Host request-loop test; DFlash2 defaults unchanged |
| Wider device plumbing | Target embeddings, five cached layers, full target head and 15-step Markov feedback | Full request executes; acceptance collapses after the first two blocks |
| Full-request adapter | Native token/state controls plus all-tap committed-feature audit; timed requests exclude that audit | Audited request passes; first timed request fails |

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

First full-request run: [34424354652](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34424354652),
source `7f52347fb14ee3c42ebd213dc6e9ce9b9a625dea`. It fails at token index 121
in the first timed request; the second timed request never runs. No new hardware
PP/TG is qualified.

## First full-request result

| Check | Result |
| --- | --- |
| Context / streams / draft-verifier rows | 4,096 / 1 / 15-16 |
| Instrumented native token, recurrent-state, valid-KV and inactive-state comparison | Pass; 121 committed tokens through EOS |
| Committed five-tap features, both chips | All 970 comparisons pass |
| Draft acceptance | 24/1,455 (1.65%); first two blocks accept 12 and 6, then collapse |
| First uninstrumented request | Fails against native at token index 121 |
| Physical link-discovery fallback | None in the hardware log |
| Complete timed PP / TG | Not qualified |

The audited request is correctness evidence, not a throughput sample. Its history
frontier reaches 4,217. The failed report is retained as
`scripts/ci/dspark-request-hardware-failed.json` (SHA-256
`adb3e59b8e3c5b19ec70ceb228d34d2658918e38417270e400b04138df3be789`).
Independent reconciliation checks all 568 source fingerprints against the tagged
revision and unchanged before/after sets for 1,517 native files. Do not replace
the failed gate with the narrower audited success.

The pinned native generator performs decode while compiling and capturing a cold
trace. Unlike the existing full-prefix harness, the new request lane omitted its
warmup before fresh prefill. The next run fixes that reference-state preparation.
It also hashes every learned K/V shard around proposal, verifier and publication,
checks finite values and preservation of the old prefix, and stops at the first
mutation. This tests suspected trace-buffer overlap rather than assuming poor
acceptance is an inherent property of the checkpoint. Failed-token and cache
evidence are now retained in the JSON instead of only an exception string.
The diagnostic run [34425893520](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34425893520)
now **confirms cache corruption**: after the first 13 committed rows, the next
target verifier changes layer 0 K on chip 0 at position 4,109. It passes the
checks before and after drafting, then fails immediately after verifier replay.
The warm-native-control correction is not reached in this run and remains to be
validated; this failure is independent of that reference-harness bug.
The retained corruption report is `scripts/ci/dspark-history-corruption-hardware.json`,
SHA-256 `ccfb08ce46377b4126997342e29d21ce1ee63520fe70577bd7c28ad7bd4dc17f`.
Independent reconciliation verifies all 570 tagged source files, unchanged sets
of 1,517 native fingerprints, the exact failure boundary and clean teardown.

The fix preallocates two complete K/V banks before verifier capture. Publication
copies the new full history into the spare bank and releases every temporary
before target replay; commit swaps the banks without allocating or freeing them.
Only the valid prefix enters proposal attention. Full history and learned math
are unchanged. A dedicated TTsim probe checks all five K/V pairs on both chips,
ragged publication, discard, changing-input scratch-writer traces, fixed addresses
and preservation of every old row before another hardware run. No serving changes.
All 1,292 host tests pass. Simulator run `20260910T014520Z-382` now passes all
372 bank-layout and replay-lifetime checks and exits cleanly with status 0.

| Fixed-bank simulator check | Passed |
| --- | ---: |
| Entire active/prepared banks against CPU references | 260 |
| Complete logical prefix views, excluding capacity padding | 80 |
| Changing-input scratch-writer trace output | 8 |
| Unchanged physical bank addresses | 4 |
| Missing-publication controls | 20 |

All five layers' K/V and both chips are checked at capacity 4,384, with frontiers
4,093 / 4,096 / 4,111 / 4,112. A discarded publication does not advance history.
Independent verification regenerates every full-bank and logical-view hash,
checks all matrix coordinates, and reconciles 41 source and 1,517 native files
before/after/current. Peak memory is 2,524,934,144 bytes; no swap or OOM.
The 1,102.7-second simulator duration is not hardware latency.

Retained report: `scripts/ci/dspark-history-bank-simulator.json`, SHA-256
`72870b5018ab69fdbc5f86f4ceed6fa334a98bd76931d41536a0379b757523e0`.
Hardware preflight now requires this exact report and unchanged qualified source.
The repair must still pass full target token/state/cache audits and two timed
requests before any PP/TG result is accepted.
