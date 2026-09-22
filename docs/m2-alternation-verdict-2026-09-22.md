# M2 item 1 verdict: the alternation works, and the bridge is the wall

Run **35711818636 (v67)**, against **35707860782 (v65)** as control.

## Result

A decoding user advanced by a full speculative decode step **during another user's
prefill**. That had never happened before in this programme.

| | v65 (control) | v67 (alternation) |
|---|---|---|
| alternation events | 0 | 2 |
| spec decode executes mid-prefill (`execute total=4`) | 0 | 1 |
| decode steps completed while a partial prefill was in flight | 0 | 1 |

Same image (`sha256:06fd3a7f...`), same `M3NATIVE_PREFILL_CHUNK_TOKENS=2048`, same
arm. The only behavioural difference is that in v65 the graft sat on
`TTLaneCoordinator`, which the platform never constructs, so it could not run. The
comparison is structural (did a decode step occur between prefill steps), not a timing
delta, so run-to-run variation does not bear on it.

The `TT_PREFILL_DECODE_INTERLEAVE=0` arm (v68) is therefore **not needed** - v65 is
already the interleave-off control at the same configuration. The tag stays wired for
a future timing comparison; it was not run.

## The yield, in full

```
09:44:39.254 [PHASE] execute total=2048 new=1        <- user 2 prefill chunk
09:44:39.954 Finished batched prefill of 1 user(s), starting decode...
09:44:39.996 [PHASE] propose cmpl-95f0b27... end 41.7 ms
09:44:39     [PINDIAG] m2 alternation: yield_decode=True streak=1 credit=0 r=1
09:44:39.997 [PHASE] execute total=4 new=0 cached=1 spec=1   <- the yielded DECODE
09:44:40.000 [CARRY] op=restore request=cmpl-95f0b27... layers=48
09:44:40.110 [CARRY] op=save    request=cmpl-95f0b27... layers=48
09:44:40.110 [PHASE] step cmpl-95f0b27... end 111.3 ms       <- a real token step
09:44:40     [PINDIAG] m2 alternation: yield_decode=False streak=1 credit=1 r=1
09:44:40.153 [PHASE] execute total=2048 new=0 cached=1 spec=0 <- back to prefill
09:44:40     ERROR EngineCore encountered a fatal error
```

The policy fired, yielded exactly one decode step (`r=1`), the step completed in
111.3 ms with its carry state restored and saved across 48 layers, and the policy
returned to prefill. That is section 3.3 behaving as specified.

## Why the run still failed, and what it proves

```
ValueError: The scheduled requests and the prepared requests must be the same set:
  scheduled_only=['cmpl-a007c5bc...']   <- user 2, the prefill
  prepared_only=['cmpl-95f0b271...']    <- user 1, the decoder
```

Note **which step failed**. The decode yield succeeded, because the request the fast
path had prepared (`cmpl-95f0b271`, which took pool slot 0 at 09:44:37) *was* the
decoder. The failure is the step that returns to prefill: the scheduler scheduled
user 2 while `FastRunnerBridge` was still bound to user 1.

This is the **load-bearing concurrency pin**, unchanged and unmasked: the fast path
holds exactly one prepared request. It is not caused by the alternation - v65 hit the
same error with the same shape and no alternation at all. The alternation simply
reaches it one step sooner.

## Consequence for the programme

M2 item 1 is **done and proven**, but it **cannot deliver its benefit until the
concurrency pin is lifted**. The two are coupled, not independent: every step the
alternation hands back to prefill is a step where the scheduler picks a request the
bridge has not prepared. Alternation converts the stall into an error rather than into
throughput, because there is nowhere for the second user's prefill to be prepared.

The remaining blocker for four concurrent single-streams is therefore entirely T3's
unbuilt work, exactly as the standing STATE describes it:

- singular session state in `serving_lifecycle`
- `FastRunnerBridge` bound to one request  <- **this is what v67 hit**
- trace bucket 1
- `capture_plan` capping verify rows at 32 against the 64 four T16 users need

Nothing measured here changes the reachability assessment recorded in
`memory/goal-200tps-concurrent.md`. What it removes is one of the two unknowns: the
scheduler-side stall is now a solved problem, and the device-side single-occupancy of
the bridge is the whole of what is left.
