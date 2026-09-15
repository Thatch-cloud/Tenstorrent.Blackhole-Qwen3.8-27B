# 64K fused-MLP reintegration

The combined runtime passes correctness and two clean EOS requests. This is
not a matched A/B result, sustained-output test, or held-out coding evaluation.

| Runtime | Run | PP tok/s | CTX | Committed TG |
| --- | --- | ---: | ---: | ---: |
| Split-K control | 34923389309 | 2575.67 | 65536 | 30.23 |
| Split-K plus fused T16 MLP | 34925118588 | 2584.54 | 65536 | 31.40 |

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

Next: integrate the existing shared-Q/K recurrence into the same audited
runtime, then measure paired control/candidate requests. More substantial
verifier/commit and draft improvements are still needed for 200 TG.

Report: artifact `qwen-splitk-combined-34925118588`, file
`dspark-64k-request-hardware.json`, SHA256
`519954b1bc8b7bb2389d87a7f24dfde8df43f09f40567681800371a32bc4bdac`.
