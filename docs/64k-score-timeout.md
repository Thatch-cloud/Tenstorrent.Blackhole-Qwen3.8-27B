# 64K combined score-layout audit: timeout diagnosis

Run `34927938233`, commit `6c831fc`, failed with request exit **124** at the
unchanged **420-second watchdog**. Container evidence reports `OOMKilled=false`.
The signature fix worked: execution reached proposal capture and replay.
This is neither a correctness pass nor a measured throughput result.

Completed log intervals before termination:

| Host interval | Count | Total | Mean |
|---|---:|---:|---:|
| Full target-state snapshot | 11 | 144.71 s | 13.16 s |
| Full history snapshot | 12 | 21.06 s | 1.76 s |
| Blocking proposal replay | 3 | 0.204 s | 67.86 ms |

These intervals can nest; do not sum them into device time. Termination interrupted
another history snapshot. No clean shutdown or complete request acceptance exists.

## Fix the test cost before rerunning

`dspark_request_experiment.py:196` implements target KV hashing with serial
device slices, two-shard readback, host layout conversion, and SHA256. Each slice
contains at most 64 pages of 64 tokens: at a 65,536-token frontier there are
16 slice/readback groups per cache. This is a concrete audit-cost candidate;
the aggregate snapshot also includes GDN and inactive-state hashing, so the log
does not yet isolate how much time belongs to KV alone.

Next experiment: compare a bounded bulk-readback audit against the existing
digest on identical allocations, preserving every page, shard, valid-prefix
boundary and digest ordering. Check partial pages and mutation detection before
admitting it to the combined request. Record peak host memory and readback count.
Do not remove boundary checks, increase the watchdog, or launch an identical
retry. Keep the old audit as the reference until equivalence is demonstrated.

The score-layout candidate remains unqualified. After complete correctness
acceptance, measure clean combined PP / CTX / committed TG separately. The
200 committed tok/s objective remains open; audit replay time is not TG.

Evidence: `runner-evidence.local/34927938233/qwen-splitk-combined-34927938233/`
contains the request log, exit status, partial report and container state.

## V3 candidate

The audit-only callback now reads up to 256 pages at a time, then produces the
same 64-page, cache-major, page-group-major, shard-major SHA256 stream on the host.
CPU tests cover partial pages, full 64K, a changed frontier, both-shard mutations,
invalid input, borrowed allocations and release on readback failure. Fake-device
read counts fall from 32 to 8 per cache at 64K; this is not hardware speed evidence.

Before selecting the new callback, V3 compares every digest with the original
on the loaded hardware at both 65,536 and 65,535 tokens. Any mismatch fails closed.
It keeps the old callback if the candidate is not faster across those checks.
Every subsequent boundary is still checked. Neither timeout nor request coverage
changes. Logged host tensor bytes exclude conversion scratch and are not peak RSS.
No new kernel math is introduced; these local tests run Python/Torch in the
TT-Sim environment, not a simulator kernel validation.

V3 run `34928953853` failed before admission with a Python frontier guard, not a
timeout or digest mismatch. `full_request.py:112` first hashes KV after the gold
decode (65,552 here), before the experimental prefill/capture. V4 admits this
ordering, compares 65,536, 65,535 and the actual first frontier against the oracle,
and returns the digest for the requested frontier. A regression covers the gold
check followed by the experimental request's return to the 65,536 frontier.
