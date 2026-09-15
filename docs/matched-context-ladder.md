# Matched context ladder

Use the combined split-K draft, fused T16 MLP, shared Q/K and incremental-history
runtime. Do not join the older 4K/8K runtime results to the new 64K measurement
and call that a scaling curve. Serving defaults remain unchanged.

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

1. Repeat the exact combined 64K candidate before changing context.
2. Add context-specific numerical/replay evidence for draft attention and folded
   target attention. The decode factory is dimension-generic, but its existing
   hardware evidence is not a blanket context admission.
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
