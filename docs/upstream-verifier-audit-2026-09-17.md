# Upstream verifier audit: 17 September 2026

**No drop-in 200-TG fix found. Keep the validated runtime unchanged.**

## Source refresh

| Upstream | Live source finding | Decision |
| --- | --- | --- |
| [ctxbot #53482](https://github.com/tenstorrent/tt-metal/pull/53482) | Head remains `a7100468ec7a7df7b9c7699719a66779fc52d8d0` | Same head as the earlier source audit; do not repeat the port |
| [ctxbot #53587](https://github.com/tenstorrent/tt-metal/pull/53587) | Head remains `c802b8c54034c78e0c8f5aeeb0828e58a9238c3a` | Same audited demo head, not a missing new decode kernel |
| [MTP #55548](https://github.com/tenstorrent/tt-metal/pull/55548) | Head `2eab47685ea3dedbe8552824cf80fb586a2db543`; fused multi-token recurrence plus grouped speculative SDPA | Inspect algorithms individually; do not replace the combined DSpark runtime |
| [Scan multicast #53790](https://github.com/tenstorrent/tt-metal/pull/53790) | Prefill scan shares inputs between value partitions; reported TP4 scan gain is not a TP2 decode result | Separate PP candidate, not evidence for faster T16 verification |
| [Profiler #55654](https://github.com/tenstorrent/tt-metal/pull/55654) | MiniMax-M3 intra-galaxy prefill profiling harness | Useful measurement discipline; no Qwen decode kernel to import |

The MTP PR reports 50.45 tok/s at 4K on its workload. That is not a matched
comparison with our coding fixture. Its stated greedy equivalence excludes
near-tie positions, so it does not substitute for our exact native-token/state
gate. No upstream reported number is treated as local hardware acceptance.

## Kernel-level distinction

The MTP recurrent compute source at the pinned head has separate CB stages for
state decay, key/state matmul, subtraction, beta scaling, key transpose,
outer-product matmul, state addition, query/state matmul and state snapshots.
It retains recurrent state on-core across tokens, but is not an all-in-register
fused recurrence. Our native-derived kernel already uses the broadcast outer
update, shared Q/K preparation and a separate whole-head norm stage. Replacing
it wholesale is not justified by the word "fused".

## Avoid the next false optimisation

The actual fused MLP uses the native matmul kernel with `PACKER_L1_ACC=1`:

- 20 K blocks of eight tiles; three two-output subblocks per worker.
- The first 19 blocks pack FP32 partials; the packer accumulates them in L1.
- **Only the final K block reloads accumulated partials into DST.** It does
  not reload them on every block. The source sets `enable_reload` after block 18.
- The final epilogue preserves BF16 gate/up rounding before multiplication.

Therefore a proposal to remove "20 FP32 reloads" starts from a false cost
model. Holding accumulators across all K blocks also changes the accumulation
schedule; it must not be assumed bit-exact or deployed without simulation.

The pinned source used for this inspection is the retained runtime export
35092212895, `bmm_large_block_zm_fused_bias_activation.cpp`, together with
`fused_1d.py`. The active numerical kernel is unchanged by this audit.

## Next measurement boundary

The bounded MLP reader clocks identify waits, not arithmetic or packer
utilisation. The bank-order trial gives no TG gain; direct norm scatter saves
about 0.82 ms of verifier replay but no whole-request TG. Do not keep sweeping
these small reader variants.

Before designing a register-resident or different accumulation kernel, extend
the bounded diagnostic to distinguish input/weight wait, matmul issue/compute,
partial packing and final reload/epilogue on the same actual T16 verifier.
Record the processor and selected block/subblock for every sample; do not add
overlapping processor intervals or extrapolate selected samples to total time.
Require poison/missing-write controls and unchanged full-request numerics.
Do not re-enable the overflowing global profiler or change the production path.

This diagnostic is not implemented yet. It must select a substantive kernel
change, not become another isolated throughput claim. Any candidate still needs
the simulator gate followed by complete, uninstrumented matched requests.
