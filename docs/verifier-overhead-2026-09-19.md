# The verifier streams weights at 37% of measured DRAM bandwidth

The 200 tok/s gap is not a memory-system limit and not host overhead. It is the
verifier's traced device program using 2.7x more time than the bytes it moves require.

Derived from run 35258782100 (T16 control, 4096 context, one user) and the bandwidth
measurement in run 35416870419. No new instrumentation: the phase timing this needs has
been emitted all along by `serving_fast_request.step` under `QWEN_FAST_PHASE_TIMING=1`.

## The decomposition

Median over the 40 post-warmup blocks. `cycle_ms` 97.57 reproduces the 97.55 in
`200tg-latency-budget.md`, so this is the same regime that doc measured.

| Phase | ms |
| --- | ---: |
| cycle | 97.57 |
| draft | 24.62 |
| **verify + readback** | **66.89** |
| select / commit | 5.15 |

and inside the verifier:

| Component | ms | |
| --- | ---: | --- |
| `blocking_trace_host_ms` | **66.09** | device execution |
| `input_ms` | 0.44 | host |
| `replay_checks_sync_ms` | 0.69 | host |
| `binding_validation_ms` | 0.40 | host |
| `output_readback_host_ms` | 0.13 | host |

Host-side overhead totals **1.66 ms**. Dispatch, staging and readback are not the
problem; 98.8% of the verifier is time the host spends blocked inside the device trace.

## What that time is worth in bytes

Each card must read its 9.96 GB half of the weights once per verification.

```
effective bandwidth = 9.96 GB / 66.09 ms = 150.7 GB/s
measured capability                      = 405.0 GB/s   (run 35416870419)
                                           37.2%, a 2.7x headroom
```

If the verifier streamed at the rate the same memory system delivers to a plain matmul:

| | now | at 405 GB/s |
| --- | ---: | ---: |
| verifier | 66.09 ms | 24.59 ms |
| cycle | 97.57 ms | 56.07 ms |
| per user | 123 tok/s | **214 tok/s** |

That is above the 200 tok/s target **at the current single-user, 4096-context
configuration**, before any concurrency work.

## The draft is the control that makes this credible

The drafter runs 15 sequential steps in 24.62 ms, 1.64 ms per step. At 405 GB/s that is
about 0.66 GB per card per step, a plausible size for five DFlash2 layers. So the draft
appears to run near the bandwidth limit on the same hardware, in the same process, in
the same cycle. The verifier does not. Whatever costs the verifier 2.7x is specific to
its traced program, not to the device, the memory system or the runtime.

## What the 41.5 ms is not, and what it might be

Not host overhead (1.66 ms, measured). Not arithmetic: 16 rows over a 27B model is
roughly 0.9 TFLOP, under 3 ms even at pessimistic utilisation. That leaves, in order of
suspicion:

1. **TP2 collectives.** An all-gather or all-reduce per layer across 64 layers, serialised
   against the matmuls rather than overlapped with the next layer's weight fetch.
2. **Non-overlapped weight fetch.** If each layer's DRAM read waits on the previous
   layer's compute, the pipeline is idle for much of every layer.
3. **Matmul shape efficiency at 16 rows.** A 16-row activation against a large weight
   matrix may tile poorly and under-use the DRAM burst.

These are distinguishable. The next step is a device-profiler run
(`TT_METAL_DEVICE_PROFILER`) over one verification, giving per-op device times inside
the trace, which separates collectives from matmuls from waits.

## Why this is the right thing to work on

- It is the largest single term: 41.5 ms of a 97.57 ms cycle.
- It is the only lever that helps *every* configuration, including any future
  multi-session work, which needs verifier overhead under 12.1 ms to reach the target at
  4 users with bf4 KV.
- It needs no architectural change to attempt, unlike multi-session support, which
  requires a batch axis through a single-sequence speculative engine.
- The draft demonstrates the hardware already achieves the rate elsewhere.

Ruled out by measurement: DRAM bandwidth itself. 405 GB/s is 79-80% of the 512 GB/s
spec, which is a normal sustained figure for GDDR6, consistent across access sizes and
across read-only and read-write regimes. There is no meaningful win there; a physically
impossible 100% of spec would save 5.1 ms against the 41.5 ms above.
