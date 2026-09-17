# 262K total-window qualification

This is a pending combined-runtime test, not a throughput result.

| Setting | Tokens |
| --- | ---: |
| Prompt (CTX reported in results) | 261,888 |
| Generation budget | 256 |
| Total positional capacity | 262,144 |

The original 262,144-token **prompt** case remains distinct and is rejected
when prompt plus generation exceeds the model's positional limit. No implicit
truncation, RoPE extension, precision change or serving-default change is made.

The new explicit context uses the same geometry calculation: 4,096 addressable
64-row KV pages, eight spare cache blocks, and a 262,144-row target sequence.
Padding for scratch storage is not extra usable model positions.

The captured drafter always evaluates 15 query positions, even when fewer
tokens are requested. Above position 262,129 that would exceed the fixed RoPE
limit. The full-window-only staging scope therefore selects the existing
singleton target path for those final positions. Both request and verifier
plans use the same guard; the original planner is restored on exit. No draft
positions are clamped and no extra RoPE positions are generated. This tail
policy still needs complete hardware output/state acceptance and is included
in measured TG rather than excluded as overhead.

## Acceptance order

1. Finish the existing 131,072-token combined hardware request, including its
   fresh feature/state audit and two timed requests. Run 35173979225 attempt 2
   has reached the complete same-recipe context step on the AMD runner.
2. Qualify the new page-table extent with the weight-free simulator lane:
   eager output, changed-input replay and unchanged-input checks on both chips.
3. Bind that new report and exact source hashes into the hardware admission
   path, then run the complete 261,888-token request with the same T16 recipe.
4. Report PP, exact prompt CTX and committed TG; label the total window
   separately. Allocation failure or a correctness failure is not a TG result.

Adding the context changes the geometry source hash. Old cache qualifications
must not be silently relabeled or reused against that changed source. Existing
131K CI uses its immutable prior tag and is unaffected by this preparation.
The simulator alone does not prove full-model memory capacity or performance.

## First cache-check attempt

Run **35186984447** timed out at the inner 360-second probe limit (exit 124),
not the nine-minute job limit. Allocation completed in about 33 seconds;
both native-reference seeds completed and both-chip eager comparison was exact.
Replay began at about 348 seconds, leaving only twelve seconds for two replay
checks and shutdown. The partial report is rejected: `passed=false` and
`closed_cleanly=false`. This is not a hardware OOM or a numerical-failure result.

The bounded retry keeps the exact kernel/probe sources and all ten checks.
Only the 261,888-token case receives a 480-second probe and 510-second launcher;
the whole job remains capped at nine minutes. Smaller cases retain their old
budgets. No incomplete evidence is admitted to the hardware lane.

The retry **35189341242** also times out, after the first replay and while
checking the changed-input replay. Do not extend the budget again. The next
fixture removes two full native-reference readbacks: native writes touch only
two physical pages of a zero-initialized cache. It reads those pages, including
the cumulative written-page set across seeds, and reconstructs the full expected
zero-plus-written-pages tensor on the host. Every candidate comparison still
reads and checks the **complete allocated cache on both chips**, including all
unwritten pages. Kernel math, page-table width, eager/replay sequence and all ten
checks remain unchanged. The new helper is included in source-bound admission.
Host tests cover expected pages, writes outside the expected region, invalid
initial state and cleanup after failed readback. Simulator acceptance is pending.

Run **35190488209** exits after about two minutes with a Python `TypeError`,
not a timeout: native TT-NN `Shape` supports integer indexing, not slices.
The reference helper now converts the shape to a tuple before slicing. Host
fixtures reproduce the native indexing restriction so this error is covered.
No kernel or admission criteria changed; full-window qualification remains pending.
