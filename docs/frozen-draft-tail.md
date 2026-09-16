# Draft KV assembly: pad the tail, not the history

Status: host prototype only. No model integration, hardware timing or serving changes.

## Why test this

The current native draft layer slices 7/15 proposal rows, concatenates them onto
the entire fixed history, then pads that large result. The history capacity is
tile aligned, but the intermediate concatenation is not.

The candidate pads only the small proposal tail to 32/64 rows first, then joins
two tile-aligned tensors. It preserves every historical row, every valid query
row and every zero-padding row. Attention, masks, precision and acceptance rules
are unchanged. Poisoned unused query rows must never reach the result.

The incomplete capture **35151664646** motivates this test, but does not qualify
it: inferred draft trace 2, chip 0, replay 4 contains roughly 5.0 ms concat,
3.8 ms padding and 6.6 ms untilize/tilize kernel-duration sums. These groups are
not all necessarily attributable to KV assembly; they are not additive critical
path savings or a throughput prediction. The clean combined capture is separate.

## Gates

| Gate | Scope | State |
| --- | --- | --- |
| Host tests | Original versus reordered row assembly; both tail sizes; poisoned padding; alignment and layout rejection | Three tests pass |
| Weight-free two-chip simulator | BF16 bitwise eager output and changed-input traces; 64/96 history rows, 7/15 proposals, both chips | Prepared; not yet qualified |
| Hardware component | Exact full-size output and replay; measure original versus candidate | Not started |
| Combined 32K runtime | Same qualified recipe, candidate changes assembly only; exact proposals/output/state; PP/CTX/TG | Not started |

The simulator has a six-minute whole-job cap, uses no model weights and has no
physical device access. Its 112 checks cover original and candidate eager output,
three changed-input replays and unchanged inputs on both chips. Simulator time
is never reported as hardware throughput. Neither the prototype nor its test
adds a runtime hook or modifies serving defaults.

Simulator run **35155862154** did not execute the probe: root checkout encountered
root-owned artifacts from an older marker workflow and failed cleaning the shared
workspace. The retry uses `draft-tail-source/` as an isolated checkout and results
directory, leaving unrelated artifacts untouched. This is not a numerical failure.
