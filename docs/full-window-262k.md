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
