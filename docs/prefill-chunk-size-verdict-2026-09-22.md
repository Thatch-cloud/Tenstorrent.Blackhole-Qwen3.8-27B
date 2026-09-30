# Raising the prefill chunk size makes everything worse: the cheap TTFT lever is closed

**Date:** 2026-09-22. **Runs:** v34 `35658854824` (chunk 4), v35 `35667198294` (chunk 32).
Same pair, same graft, same image, same four-user 32,768-token workload, both
`gate_passed: true` and token-exact. The only difference is
`MAX_PREFILL_CHUNK_SIZE`.

## 1. The hypothesis

`docs/ttft-decode-stall-2026-09-22.md` established that prefill is serial at 13.1 s per
user and that every user's stall is exactly the remaining prefill time, so anything that
shrinks 13.1 s shrinks the 79.4 s stall proportionally. Probe `35665853903` then read
`get_max_prefill_chunk_size` out of the image and found the value in force is a
**compatibility fallback**: neither `Qwen3.8-27B` nor `P300` is a key in
`MAX_PREFILL_CHUNK_SIZES_DIV1024`, so the lookup raises `KeyError` and the except branch
logs *"setting MAX_PREFILL_CHUNK_SIZE to 4 for compatibility"* with a second line
inviting *"larger powers of 2 up to e.g. 128 for faster performance"*.

The units are 1024 tokens (`return max_prefill_chunk_size_div1024 * 1024`), so 4 means
4,096-token chunks — eight of them per prompt — while every comparable model on
P150-class hardware sits at 128. It is a plain environment variable read before the
table, so the experiment cost one `-e`.

**v35 set it to 32**: the whole prompt in one chunk, still 4x below the 32B peers.

## 2. The result

| | v34, chunk 4 | v35, chunk 32 | |
|---|---:|---:|---|
| prefill rate | 2,423 tok/s | **2,294 tok/s** | −5.3% |
| TTFT, four users | 13.5 / 26.5 / 39.6 / 52.5 s | **14.3 / 27.9 / 41.8 / 55.5 s** | +5.7% |
| per-user prefill | 13.0 / 13.1 / 12.9 s | **13.6 / 13.9 / 13.7 s** | +5.4% |
| total decode stall | 79.4 s | **110.0 s** | **+38.5%** |
| round, median | 256.3 ms | **296.9 ms** | **+15.8%** |
| verify trace | 157.73 ms | 157.84 ms | +0.07%, unchanged |

Worse on every axis that matters, and not marginally.

**The experiment is valid, not null.** v34's log carries
*"setting MAX_PREFILL_CHUNK_SIZE to 4 for compatibility"* twice; v35's carries it zero
times, because `os.getenv` at `model_config.py:2422` short-circuits to the `else` branch
at :2489 and never reaches the `except KeyError` that logs it. The absence of that
warning is the positive control.

## 3. The new stall v35 introduced

The extra 30.6 s of stall is not a longer version of the old one. It is a **new,
synchronised** stall that v34 does not have at all:

| round index | user 0 | user 1 | user 2 | user 3 |
|---|---:|---:|---:|---:|
| #27 | 1.3 s | 1.3 s | 1.3 s | 1.3 s |
| #28 | 1.7 s | 1.7 s | 1.7 s | 1.7 s |
| #29 | 2.2 s | 2.2 s | 2.2 s | 2.2 s |
| #30 | 1.2 s | 1.2 s | 1.2 s | 1.2 s |

All four users freeze on the same four rounds, for the same durations, ~25.6 s in total —
which accounts for the difference almost exactly. This is mid-decode, long after every
prefill has finished, so it is not a prefill artefact; it is a global engine event that
the larger prefill allocation brought into existence.

The verify trace is unchanged at 157.8 ms, so the extra 40 ms per round sits **outside**
the traced region. Both observations point the same way: the 32,768-token chunk's
buffers leave the decode path with less headroom, and the cost lands on the untraced
work and on whatever periodic event rounds 27-30 represent.

## 4. Verdict

**Closed.** Raising the prefill chunk size does not help here and actively hurts. The
vendor hint is sound advice for a model that has the L1 headroom for it; this endpoint
is running four concurrent users, a packed decode trace and a 512 MB trace region, and
does not.

This also **refutes the inference that motivated the experiment**. Lever N section 1
measured the managed endpoint at 4,572.8 tok/s against the 2,423 tok/s measured here,
and the chunk-size fallback looked like the obvious explanation for a ~1.9x shortfall.
It is not: at a chunk size eight times larger the rate went *down*. Whatever separates
those two numbers — different prompt length, different concurrency, a different engine
configuration, or simply not being comparable — it is not this.

**Not worth another point on the ladder.** The direction tested is monotonically wrong
and the mechanism now has a plausible story (allocation pressure) that predicts more of
the same at 8 or 16. A different hypothesis should come before another rig run.

## 5. What this leaves

The 79.4 s stall stands, and the two remaining routes are both more expensive than this
one was:

- **#41, four fabric links on the prefill collectives.** Still un-measured, still
  bandwidth-bound in principle, and now the only cheap prefill-throughput lever left.
- **#31, Lever N on the fast path.** The five-blocker build in
  `docs/lever-n-fastpath-scope-2026-09-22.md`. It attacks the stall directly rather than
  the prefill rate, by letting decode run between prefill chunks, and it is the only
  route that removes the staircase rather than shortening its steps.

Note that prefill *rate* work could only ever have shortened the steps. Even a 2x faster
prefill would leave a 13 s / 26 s / 39 s staircase at four users, halved. Only
interleaving removes it.
