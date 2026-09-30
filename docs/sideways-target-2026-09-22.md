# Four concurrent single-streams: the reachable target, with the arithmetic

2026-09-22. Written after the user reframed the goal: "the single stream was fast, we
are basically extending sideways to get to four concurrent single streams on this
hardware."

That is a different target from the one the programme has been measuring against, and
the difference matters: one is arithmetically impossible on this hardware and the other
is not. This note fixes both numbers so the distinction stops being rhetorical.

## Single stream, measured

From `runner-evidence.local/packed-gate/single-user-*.json`, the references the four-user
gate compares byte-exactly against. 64 completion tokens each, 32,768-token prompt.

| reference | TTFT | wall | decode | rounds | round | tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| single-user-1001-35493236124 | 14.56 s | 16.15 s | 1.59 s | 11 | 144 ms | 40.4 |
| single-user-1002-35498370154 | 14.68 s | 16.17 s | 1.49 s | 11 | 135 ms | 43.0 |
| single-user-1003-35498374589 | 14.17 s | 15.56 s | 1.40 s | 10 | 140 ms | 45.8 |
| single-user-35492921706      | 14.52 s | 15.87 s | 1.35 s | 10 | 135 ms | 47.4 |

**Single stream is ~44 tok/s at a ~138 ms round.** The `chunks` field doubles as a round
count, and 64 tokens over 10-11 rounds is 5.8 accepted tokens per round - which
independently reproduces the 5.89 figure the four-user attribution measured, from a
different artifact. The two agree, so the round arithmetic below rests on two sources.

## The sideways cost today

| | single stream | four users | ratio |
| --- | --- | --- | --- |
| per user | 44 tok/s | 23.0 tok/s | **0.52x** |
| total | 44 tok/s | 92 tok/s | 2.1x |
| round | 138 ms | 256 ms | 1.85x |

Batching is amortising something - total throughput doubles - but not four ways.

## Why one target is reachable and the other is not

**200 tok/s per user** needs a 29.4 ms round at 5.89 accepted tokens. The measured floor,
from deleting every SdpaDecode plus its cc=64 companion plus all Matmul (58.2% of op
time, docs/200tps-verdict-2026-09-22.md), still leaves **66.0 ms of verify trace alone**.
That is 2.2x outside the envelope at its most generous, at a fifth of the target context.
Unreachable, and the verdict stands.

**Four concurrent single-streams** needs a **138 ms** round - 1.85x from 256 ms. The same
58.2% of op time is headroom of the right order. Working it through honestly rather than
optimistically:

    verify trace floor (all attention and matmul free)   66 ms
    proposals, commits, publish, scheduling (measured)  ~98 ms
                                                       -------
    round                                              ~164 ms  ->  ~36 tok/s/user

So the aggressive floor lands at roughly **36 tok/s per user, ~145 tok/s total**: 82% of
single-stream per user, 3.3x single-stream throughput. Not a clean 4x, and the 98 ms
non-trace half would itself have to hold still while the trace shrinks. But it is inside
the envelope the measurements bound, which 200 tok/s/user is not.

This is the same number task #39 arrived at independently - its estimated ceiling was
~35-55 tok/s - from the mechanism rather than from the budget. Two routes to one figure
is the useful kind of agreement.

**#39 is therefore the central lever, not a side quest.** "One weight pass for four
users" is precisely the mechanism that turns a 1.85x round penalty into an amortised one.

## What the reframing makes MORE important

The single-stream table is dominated by prefill: **TTFT 14.5 s against 1.4 s of decode.**
For a 64-token completion at 32k context, prefill is 91% of the wall clock. Four users
admitted serially turn that into 13.5 / 26.5 / 39.6 / **52.5 s** (run 35658854824), of
which 79.4 s of 257.4 s wall - 31% - is decode stall explained by admission order.

So "four concurrent single streams" as a *user experience* is gated by the prefill ramp
far more than by the decode round. A user waiting 52.5 s for a first token does not care
that the round is 256 ms rather than 138 ms.

That is task #37 and Lever N, and this reframing raises its priority rather than lowering
it. It remains the thing with no hardware measurement, after five arms died on delivery
plumbing (docs/dead-flag-verdict-2026-09-22.md,
docs/lever-n-fastpath-scope-2026-09-22.md addendum).

## The reporting correction

Every report in this programme has led with "23.0 against 200, 8.7x short". Under the
sideways framing the operative comparison is **23.0 against 44, 1.9x short**. Both are
true; only one of them describes a problem worth working on. The 200 figure should be
cited as a closed verdict on the stated target, not as the live gap.
