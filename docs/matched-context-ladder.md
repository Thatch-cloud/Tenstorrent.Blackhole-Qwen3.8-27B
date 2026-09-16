# Matched context ladder

Use the combined split-K draft, fused T16 MLP, shared Q/K and incremental-history
runtime. Do not join the older 4K/8K runtime results to the new 64K measurement
and call that a scaling curve. Serving defaults remain unchanged.

## Current hardware evidence

| CTX | FP32-maxima draft attention | Folded verifier vs native B1 | Combined PP / TG |
| ---: | --- | --- | --- |
| 4096 | Pass | Exact pass | Not measured on this candidate |
| 8192 | Pass | Exact pass | Not measured on this candidate |
| 16384 | Pass | Exact pass | Not measured on this candidate |
| 32768 | Pass | Exact pass | Not measured on this candidate |
| 65536 | Pass | Exact pass | PP 2309.96 / TG 37.94; two complete repeated requests |
| 131072 | Numerical failure: 59 elements | Not run | Not qualified |
| 262144 | Cancelled after 131K failure | Not run | Not qualified |

Draft evidence: runs 35034936748 (32K) and 35035520826 (remaining rows).
Verifier evidence: run 35035966582; all five contexts close cleanly with eight
exact output comparisons, 16 mask checks, four source checks, two unpoisoned
replay checks, two stale controls and eight poison controls. Exact reports and
current source hashes are checked by `matched_draft_gate.py` and
`matched_target_gate.py`.

The verifier component reserves CTX+256 positions; the proposed full runtime
reserves CTX+1024. Integration must account for this explicitly, not pretend
the component fixture validates every runtime allocation. The previous 64K
incremental-history runtime reached 37.95/38.32 committed TG, but did not use
this FP32-maxima factory. Those are not measurements of the new candidate.

## Combined 64K baseline, run 35039344557

The new maxima runtime passes both complete responses with exact output/state,
all 120 learned-parameter checks and clean shutdown. The 5m49s job uses a cache
hit. Each response commits 135 tokens to EOS in 20 blocks; all publications use
the incremental writer. This is an offline short coding fixture, not sustained
serving or held-out coding-quality acceptance.

| Repetition | PP tok/s | CTX | Committed TG tok/s | Decode ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2106.06 | 65536 | 37.87 | 3564.79 |
| 2 | 2557.57 | 65536 | 38.00 | 3552.32 |
| Pooled identical runtime | 2309.96 | 65536 | 37.94 | 7117.11 total |

Mean block costs: 85.51 ms draft, 82.32 ms verification/readback, 8.07 ms
selection/commit, 177.85 ms whole cycle. At the measured 6.75 committed tokens
per block, 200 TG requires at most 33.75 ms per whole cycle. Neither removing
host overhead nor draft work alone reaches the target. Both sides need work.

Next controlled experiment: raise the split-K worker limit from 8 to 16 per
KV lane, keeping maxima precision, 256-key chunks and all history unchanged.
Run a weight-free simulator check, then full-context numerical validation and
combined-request measurement; do not claim a speedup from configured core count.
Simulator run 35040143195 passes in 2m53s: four numerical eager checks, four
exact replay checks, 48 input checks, 16 layout checks, poison/frontier controls
and the unchanged native target gate. The factory is identical to the baseline;
only the worker limit changes. The 64K hardware screen requires that cached
binary and has a four-minute outer deadline. Model speed remains unmeasured.
Combined timing report SHA: `aa5604cb21ccad7d8d920e91e5c065ab1829ce2ca8cad039d7fdcecddd6ad79e`.

## Proposed allocation contract

Keep one stream, 15 draft proposals, T16 verification, a 256-token output budget,
64-row target pages, 256-key split-K chunks and eight workers per KV lane.
Reserve 1024 history rows beyond the prompt, matching the current 64K allocation.
Pad attention storage to whole eight-worker chunk groups; all padding is masked.

| CTX | History capacity | Padded attention keys | Draft history banks GiB/chip |
| ---: | ---: | ---: | ---: |
| 4096 | 5120 | 6144 | 0.098 |
| 8192 | 9216 | 10240 | 0.176 |
| 16384 | 17408 | 18432 | 0.332 |
| 32768 | 33792 | 34816 | 0.645 |
| 65536 | 66560 | 67584 | 1.270 |
| 131072 | 132096 | 133120 | 2.520 |
| 262144 | 263168 | 264192 | 5.020 |

These memory figures cover only the two BF16 draft-history banks. They exclude
model weights, target KV, recurrent state, traces and scratch. They do **not**
prove the larger contexts fit. CPU tests enforce complete history coverage and
exact agreement with the existing 64K padded geometry.

## Remaining implementation gates

The next 64K run uses `qwen-matched-combined.yml`: one cache-hit model load,
the new maxima factory, fused T16 MLP, shared Q/K, score-layout fusion and the
incremental history writer. It retains the existing 17-output correctness
screen (256-token allocation), including native output/state and trace checks.
This is not a TG measurement or a full-response qualification. The run has a
590-second outer deadline; previous timing admission is deliberately rejected.
After this screen passes, measure complete responses on the exact new build.

Run 35037299095 timed out during the second prefill, before decode: the cold
build used 305 seconds and the unchanged KV-reader comparison used about 116
seconds. It provides no request qualification or TG result. The retry requires
an exact content-addressed build-cache hit (fails rather than compiling) and
retains the original full-prefix KV digest at every audit boundary without
re-benchmarking the alternative reader. Numerical/state checks are unchanged.

Retry **35038298030 passed in 6m13s**, with a two-second cached build. All 120
learned-parameter checks, native output/state checks and clean shutdown passed.
All three bounded decode commits used the incremental writer (maximum 32 rows
touched). Loading hit all 320 shard and 64 packed caches without preprocessing.
The report SHA is `e9129dcb65a87d7bc25463d38eca83c4a83a276e1a6521fab0dec3b47cb4f329`.
This qualifies only the bounded correctness screen, not full-response speed.

`qwen-matched-timed.yml` now measures two complete repeated 256-budget/EOS
requests on that exact build: native score path, maxima split-K, fused MLP,
shared Q/K and incremental history. It rejects old audit identities and any
build-cache miss. PP and committed TG cover complete prefill/decode work;
loading, compilation and setup remain separately reported. No serving change.

1. Completed: repeat the previous combined 64K incremental-history candidate.
2. Completed through 64K: context-specific draft and folded-target evidence.
   131K draft accuracy remains open; 262K still needs execution. Component
   evidence is not a blanket full-request admission.
3. Add a separate request scope for target allocation, prefill capture, history
   banks, attention tickets and context-specific correctness summaries. Current
   `dspark_64k_*` gates intentionally reject other contexts; do not relax those
   historical gates or claim their artifacts cover a new shape.
4. Run one bounded hardware job per admitted context. Preserve actual prompt
   length, all valid KV/history, exact target output/state checks, clean shutdown
   and separate loading/setup costs. Report PP/CTX/committed TG and acceptance.
5. Admit 131K/262K only after actual allocation and correctness checks. Report
   failure or unsupported status rather than truncating history or relabeling
   a padded 64K request as a different context.

`scripts/ci/matched_context_geometry.py` currently supplies planning and CPU
validation only. It does not enable runtime admission or dispatch ladder jobs.

The 64K repeat now passes (35027433446). The new context-attention probe reuses
the unchanged, source-pinned simulator-qualified split-K factory/kernel. It tests
two frontiers, complete history and poisoned padding against the retained FP32
reference tolerance, changed-input exact replay, stable addresses and unchanged
inputs. It starts at 4K with no model weights and a 120-second probe cap.
This is the draft-attention shape gate, not a full-model context result; folded
target attention, allocation and request-scope admission remain required.

The 4K draft gate passes in run **35028892841**: 72 eager/replay/input/layout
checks, eight fixture controls and clean shutdown; the runtime build is a cache
hit. Report SHA256:
`250c7bc6e46a44c1eedb94523a48e37a5dc065d71ed2ca5c01702f9fb7cf0d11`.
The unchanged probe now runs sequentially at 8K/16K/32K/64K/131K/262K, with one
120-second probe per job and fail-fast scheduling. These are draft-attention
component checks, not model-memory fit, folded-verifier or PP/TG ladder results.
