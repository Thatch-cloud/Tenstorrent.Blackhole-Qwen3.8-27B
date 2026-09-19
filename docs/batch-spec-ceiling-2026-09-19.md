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

## The blocker: the number that decides this cannot currently be measured

The recommendation above is to measure acceptance at 131k before building. Checking
what that would take says it is not reachable today, and the reason is the same
mechanism the programme has been working around all along.

Acceptance is a property of speculation, and speculation is **fast-path only** - the
plain path's `DFlash2DraftModel` raises on every method. The fast path asserts a
**4096/256 request profile** at four independent layers, of which two remain unlifted by
standing instruction:

| layer | state |
| --- | --- |
| `serving_fast_policy.validate_fast_config` | lifted |
| `serving_fast_policy.validate_request_sampling` | lifted |
| `dflash_device.__init__` line 40 | **not lifted** |
| `dflash_combined_request` line 80 | **not lifted** |

The two lifted ones are what allowed the fast path to *start and allocate* at 65,536
positions (runs 35412244363, 35412592950). The two unlifted ones gate actual **requests**
to the 4096 profile. So the fast path can be brought up at long context but cannot serve
a long request, and 12.1 tokens per cycle at 4096 is the only acceptance figure
obtainable without lifting a pin.

That means the 6.4% margin sits on a number that **cannot be checked** under the current
constraints, and the check is the cheapest thing that could invalidate weeks of work.

Three ways forward, and the choice is not the author's to make:

1. **Lift a context pin for a measurement-only run.** It is a serving-default change and
   needs authorisation. It is also the only route to the actual number.
2. **Build the batch axis anyway** and discover acceptance at 131k afterwards, when the
   work is already spent.
3. **Qualify bf4 KV first.** It lifts the ceiling from 159.6 to 185.6, which turns the
   6.4% margin into 63% and makes the acceptance question much less load-bearing. It does
   not need a pin lifted, and it is on the path to every surviving configuration in the
   verdict.

Option 3 is the only one that reduces risk without spending either a pin or the
architectural work, which is why it is the recommendation.
