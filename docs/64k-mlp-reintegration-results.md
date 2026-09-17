# 64K fused-MLP reintegration

The combined runtime passes correctness and two clean EOS requests. This is
not a matched A/B result, sustained-output test, or held-out coding evaluation.

| Runtime | Run | PP tok/s | CTX | Committed TG |
| --- | --- | ---: | ---: | ---: |
| Split-K control | 34923389309 | 2575.67 | 65536 | 30.23 |
| Split-K plus fused T16 MLP | 34925118588 | 2584.54 | 65536 | 31.40 |
| Split-K plus fused MLP and shared Q/K | 34926705012 | 2594.11 | 65536 | 31.33 |

Both timed requests commit 135 tokens to EOS. Exact outputs, final state and
inactive slots pass, as do clean kernel restoration and device shutdown.
All 64 MLP instances record two construction-time executions per request;
trace replay does not increment Python counters. Weight bindings remain
unchanged. The prior instrumented audit is run 34924260072.

| Mean block cost | Control ms | MLP candidate ms |
| --- | ---: | ---: |
| Draft | 86.18 | 85.64 |
| Verify/readback | 81.43 | 80.97 |
| Selection/commit | 54.07 | 46.70 |
| Complete cycle | 223.18 | 214.92 |

The aggregate TG difference is about 3.8%, but verification changes by less
than 0.5 ms. Most observed cycle difference is selection/commit. Do not claim
that MLP fusion caused the total gain: these are separate runs and request
timings vary. Candidate decode times are 4.41707 and 4.18299 seconds.

Shared-Q/K audit **34925963042** and clean timing **34926705012** now pass.
Each timed request constructs 96 three-stage shared-Q/K pipelines, restores
the executor and releases preparation buffers. Exact output/state/inactive-slot
checks and clean device closure pass. The two decode times are 4.49141 and
4.12680 seconds; each request commits 135 tokens to EOS.

Mean block costs with shared Q/K are 85.46 ms draft, 81.89 ms verify/readback,
46.42 ms selection/commit and 215.38 ms total. **There is no demonstrated
throughput improvement over MLP alone.** Do not promote shared Q/K on speed
grounds or describe these separate runs as a paired comparison.

Next: investigate the remaining score-layout/materialization path and obtain
matched device attribution before changing more verifier kernels. At the
observed 6.75 committed tokens per block, 200 TG requires a 33.75 ms complete
cycle, versus 215.38 ms measured. Small isolated savings are insufficient;
both execution cost and speculative acceptance/yield need attention.

Report: artifact `qwen-splitk-combined-34925118588`, file
`dspark-64k-request-hardware.json`, SHA256
`519954b1bc8b7bb2389d87a7f24dfde8df43f09f40567681800371a32bc4bdac`.

Shared-Q/K report: artifact `qwen-splitk-combined-34926705012`, file
`dspark-64k-request-hardware.json`, SHA256
`57ff9920f105b281583892ec80bfcc2b2e0da62348b1eed08b1c2d3c6ed88fbc`.
