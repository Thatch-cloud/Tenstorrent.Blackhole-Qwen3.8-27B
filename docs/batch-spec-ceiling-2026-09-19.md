# Batched speculative decode at 131k: reachable on paper, on an unmeasured number

Reproduce with `scripts/ci/batch_spec_ceiling.py`. The ceiling is the rate at the DRAM
floor with **every** non-bandwidth overhead set to zero - not achievable, but it bounds
what weeks of architectural work could ever buy.

| context | users | KV | floor ms | ceiling tok/s | vs 150 |
| ---: | ---: | --- | ---: | ---: | ---: |
| **131072** | **4** | **bf8** | **75.8** | **159.6** | **+9.6** |
| 131072 | 4 | bf4 | 65.2 | 185.6 | +35.6 |
| 131072 | 8 | bf8 | 97.0 | 124.7 | -25.3 |
| 163840 | 4 | bf8 | 81.1 | 149.2 | -0.8 |
| 163840 | 8 | bf8 | 107.6 | 112.4 | -37.6 |

The rescope from 161k to 131k is what makes this arithmetically possible at all. At
161k the ceiling is **below** the target, so no amount of work reaches it; at 131k there
is 9.6 tok/s of daylight.

## The two things that daylight is made of

**1. An 85% cut in all non-weight verifier work.**

| | ms |
| --- | ---: |
| cycle floor, irreducible | 75.8 |
| budget at 150 tok/s | 80.7 |
| **overhead allowed** | **4.9** |
| overhead today | 33.0 |

Batching does not reduce that overhead - it amortises the *weight* pass, which is
already in the floor. The 33 ms of GDN machinery, selection, SDPA and residue is per
cycle regardless of user count. So batched speculation gets to the ceiling only if
something else removes 28 ms of verifier overhead first, and today's session closed
four candidate levers without finding it.

**2. Acceptance holding at long context, which has never been measured.**

| committed/cycle | ceiling tok/s |
| ---: | ---: |
| 14.0 | 184.7 |
| 13.0 | 171.5 |
| **12.1 (measured, at 4096)** | **159.6** |
| **11.37 (break-even)** | **150.0** |
| 11.0 | 145.1 |
| 10.0 | 131.9 |

**Break-even is 11.37 tokens per cycle against 12.1 measured - a 6.4% margin, on a
quantity only ever measured at 4096 context.** A 9% fall in acceptance at 131k puts the
target out of reach at the physical floor, with zero overhead, however good the
implementation.

## Recommendation: measure acceptance at 131k before building anything

One hardware run against weeks of architectural work through the draft device,
verifier, masks, bridge and hook. It is the cheapest experiment that can invalidate the
whole plan, and this session has three examples of a cheap probe killing a large build
before it started - the shift-matmul kernel most recently.

The standing verdict already recommended this and it was not done. The ceiling
arithmetic now says exactly what would invalidate the plan: **below 11.37 committed
tokens per cycle, stop.**

If acceptance holds, the second-cheapest move is qualifying **bf4 KV**, which lifts the
131k/4-user ceiling from 159.6 to 185.6 and turns a 6.4% margin into 63%. That is the
difference between a target that survives ordinary variance and one that does not.
