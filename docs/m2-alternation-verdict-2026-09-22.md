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

---

# v69: the routing fix lands, and the next pin is the lifecycle's prefill slot

Run **35714211185 (v69)**, image v88 (`sha256:a3432592f5`, built from 8629382f, whose
build ran `test_serving_worker_hook` inside the image).

## What moved

| | v67 | v69 |
|---|---|---|
| `scheduled`/`prepared` set mismatch | fatal | **0 occurrences** |
| alternation events | 2 | **30** (15 yield, 15 no-yield, exactly r=1) |
| resumable prefill calls | 15 (one user) | **30** (two users x 15 chunks) |
| users reaching a first token | 1 | **2** (`ttft=[14.91, 30.11]`) |
| steps with a decoder live beside a partial prefill | 1 | **15** (`partials=1 decodes=1`) |
| max simultaneous decoders | 1 | **2** (`partials=0 decodes=2`) |

The chunked-prefill routing fix did what it was meant to: a second user prefilled
through fifteen chunks while the first decoded, and the alternation held 1:1 across the
whole of it.

## The caveat, stated plainly

**The fast path served only user 1.** All 94 `[PHASE] propose` / `[PHASE] step` records
name `cmpl-91e3c647...`, and only one `pool slot 0 acquired` appears. User 2 prefilled
and produced its first token on the PLAIN path - which is where the routing fix sends a
step the hook does not own. So this is not two fast-path streams; it is one fast-path
stream that no longer freezes while a second user prefills beside it.

The second TTFT is 30.11 s against the first's 14.91 s, i.e. still ~2x. That is the
expected result and not a disappointment: the alternation yields decode steps to the
DECODER, protecting its throughput. It does nothing for the prefilling user's TTFT, and
the staircase is a separate problem (#37 says so explicitly).

## The next pin

```
serving_lifecycle.py:122
ValueError: Fast serving requires one complete fresh prefill:
  prefill_slot='cmpl-a5975da9...'  new=['cmpl-be4fd8c8...']  cached=[]
```

A third request arrived while the lifecycle's singular prefill slot was still held by
the second. The step immediately before it is
`m2 one-in-flight: partials=0 decodes=2 allowed=1 hidden=False`.

That line is the whole diagnosis. The **scheduler** had stopped counting user 2 as a
partial prefill - its chunks were done, so it is no longer `is_prefill_chunk` - and
therefore un-hid the waiting queue and admitted user 3. The **lifecycle** had not yet
run user 2's prefill-to-decode handoff, which is what clears `self.request_id`
(serving_lifecycle.py:268). The two disagree for a window of one step, and a new arrival
inside that window is fatal.

This is the "singular session state in `serving_lifecycle`" item of the standing pin
list, now located precisely: not a missing capability - `decoding_ids` is already a list
and `FastWorkerHook.attach()` already admits further bridges - but a race between two
views of "is anyone still prefilling".

## Honest position

Two things are now true that were not before: the scheduler-side stall is solved and
proven, and a second user can prefill to completion beside a decoding one. Two things
remain: the second user is never adopted onto the fast path, and the lifecycle's
prefill-slot handoff races the scheduler's queue-hiding. Neither is a throughput
question yet, so no tok/s claim can be made from this run.

---

## Correction to the v69 section above (same day)

Two claims in the v69 write-up do not survive a closer read of the artifact. Both are
corrected here rather than edited away, because the reasoning that produced them is the
thing to avoid repeating.

**1. "User 2 prefilled and decoded on the PLAIN path" is withdrawn.**

The log shows `Prefilling 1 user(s) into slots [0]` **16 times** and
`into slots [1]` **16 times** - one one-shot plus fifteen resumable each. Both users
prefilled in the model, through the lifecycle's own continuation path. User 2 was not
on some separate plain path; it simply never reached
`serving_request_factory` (one `nothing to adopt` line, for user 1 only), so it was
never adopted onto the fast path for DECODE.

The inference was drawn from the absence of `[PHASE] propose`/`step` records for user 2.
That absence is real and still means the fast path decoded only user 1 - but it does not
locate where user 2's prefill ran, and I stated that it did.

**2. The "handoff race" mechanism is downgraded to one of two candidates.**

I wrote that the scheduler un-hid the queue after user 2's chunks finished while the
lifecycle had not yet cleared `request_id`. That fits. So does a second reading:

- user 2's seed committed and the handoff ran (consistent with
  `partials=0 decodes=2` and `recorded=2` TTFTs),
- user 3 (`a5975da9`) was then admitted into the lifecycle's prefill slot,
- user 4 (`be4fd8c8`) arrived before user 3's first chunk existed, so
  `partial_prefills` was still empty, the queue stayed un-hidden, and the refusal fired.

The artifact cannot separate them: a step the lifecycle handles alone never reaches the
hook, so it prints no `[PHASE] execute` line, and an admitted-but-not-yet-prefilling
request is invisible in the log. `a5975da9` appears **only** in the error text, which is
equally consistent with "user 2, slot not yet cleared" and "user 3, just admitted".

**What is common to both, and therefore safe to act on.** In each reading the lifecycle
holds `request_id` while the scheduler admits another fresh request, because the
scheduler's test for "a prefill is in flight" is `is_prefill_chunk` in `running` - which
is false both before a prefill's first chunk and after its last. The lifecycle's slot and
the scheduler's partial-set disagree at both ends.

So the fix that is correct under either reading is the one that gives the scheduler the
lifecycle's own answer instead of inferring it. The next run should also carry a marker
naming the request id at admission and at handoff, which settles which reading was true
rather than leaving it to inference a second time.

---

# v73: all four users reach a first token, and the alternation's cost is visible

Run **35718626867 (v73)**, image v89 plus the GDN slot-remap graft (mounted, no rebuild).

| | v71 | v73 |
|---|---|---|
| users reaching a first token | 2 (`incomplete=True`) | **4** (`incomplete=False`) |
| ids the FAST path served | 1 | **2** (486 and 604 propose/step records) |
| resumable prefill calls | 30 (two users) | **60** (four users x 15) |
| alternation events | 30 | **60** |
| `gdn/tp.py` IndexError | fatal | gone |

This is the first run in the programme where every one of the four users produced a
token. It is also the first where the fast path decoded more than one.

## The cost, which must not be buried

```
v73 TTFT:              14.61   29.66   71.61   86.24
known staircase:       13.5    26.5    39.6    52.5
```

Users 3 and 4 are **substantially worse** - 71.6 s against 39.6 s, 86.2 s against
52.5 s. That is the alternation doing exactly what it was designed to do, seen from the
other side: every decode step yielded to a running user is a step not spent on the
queued user's prefill. The policy protects the DECODER's throughput and charges the
later arrivals' TTFT for it.

Two things follow. First, `r=1` (one decode step per prefill chunk) is not obviously the
right operating point at four users, and `TT_DECODE_STEPS_PER_PREFILL_CHUNK` exists
precisely to move it - but that is a tuning question to settle with measurements, not by
argument, and it needs a run that survives to steady state first. Second, no tok/s claim
can be made from v73 either: the run still dies, so there is no sustained decode window
to measure.

## The next failure

```
ValueError: Fast serving requires one complete fresh prefill:
  prefill_slot=None new=[] cached=['cmpl-9629ed6b...'] spec={'cmpl-9629ed6b...': [...]}
```

No new request, one cached request carrying speculative tokens: a decode step, judged by
a clause that exists to vet an admission. It reached that clause because `self.hook` was
None - `_release_request` tore the hook down once every request the lifecycle TRACKS had
finished, while an untracked one was still decoding.

The narrow fix is to delegate a step with no new request to the stock path, as the two
branches above already do for every other shape the fast path does not own, and to log
it so the delegation is visible rather than silent.

**That is not the root cause.** The root cause is that v73 adopted only two of the four
users onto the fast path; the other two decoded untracked, which is why releasing on the
tracked set was premature. Adopting every user is its own task, and until it is done the
fast path is serving two streams, not four.

---

# v75: the engine survives, and four users are measured for the first time

Run **35720619574 (v75)**, image v92. No `ValueError`, no `EngineCore encountered a
fatal error` - zero of each across the artifact. Every earlier run in this sequence died.

## Measured, four concurrent users at 32,768-token prompts

Per-stream inter-token gaps, from the gate's own `gaps_ms`:

| stream | tokens | first quarter | last quarter | best observed |
|---|---|---|---|---|
| 0 | 213 | 371 ms (2.7 tok/s) | 105 ms (9.6 tok/s) | 95 ms (10.5 tok/s) |
| 1 | 256 | 111 ms (9.0 tok/s) | 101 ms (9.9 tok/s) | 90 ms (11.1 tok/s) |
| 2 | 212 | 357 ms (2.8 tok/s) | 103 ms (9.7 tok/s) | 91 ms (11.0 tok/s) |
| 3 | 256 | 100 ms (10.0 tok/s) | 55 ms (18.0 tok/s) | 51 ms (19.8 tok/s) |

So **steady state is roughly 10 tok/s per user**, one stream reaching 18-20. The
first-quarter figures for streams 0 and 2 (2.7-2.8 tok/s) are the prefill-overlap
window: a decode step costing roughly one 2048-token prefill chunk plus its own time.

TTFT: `14.51 / 29.72 / 70.41 / 85.14` s, against the `13.5 / 26.5 / 39.6 / 52.5`
baseline. Users three and four remain much worse, as in v73.

## Correctness

Three of four users are token-identical to their single-user references
(`identical_prefix: true` for users 0, 2 and 3). **User 1 diverges**
(`identical_prefix: false`). That is a real correctness failure on one stream and it is
not explained yet.

## Speculation is working - a claim I nearly got wrong

The final metrics line reads `Mean acceptance length: 1.00, Accepted throughput: 0.00`,
which invites the conclusion that the T16 draft contributes nothing. It does not.
Across the run acceptance reaches **4.00**, with 3.82 and 3.88 also recorded; the 1.00
readings are the tail windows after streams finish. v73 shows the same shape (4.00,
3.88, 1.34, then 1.00s). Reading only the last line would have produced a confident
wrong claim about the biggest lever in the system.

## What this says about the target

Against **200 tok/s per user**, the measured 10 tok/s steady state is **20x short**, and
the best single stream seen anywhere in the run (19.8 tok/s) is still 10x short. Nothing
measured in this programme suggests a 20x lever exists on this path: the round is
already speculative with acceptance near 4, the native decode graft is engaged, and the
per-token cost at four users is dominated by work that scales with the number of live
streams.

Against the reframed target in `memory/goal-200tps-concurrent.md` - four users near the
44 tok/s single-stream rate - the gap is **4.4x** at 10 tok/s, or 2.2x against the best
stream. That is a large gap but not obviously a closed door, and the honest position is
that it is unproven either way until the per-token cost at four users is attributed.

**No claim is made here that the reframed target is reachable.** What is now established
is the starting number: four concurrent users, 32k prompts, ~10 tok/s each, three of
four token-exact.

## Four-user output is not token-exact, and the victim moves

Both four-user runs fail equality on exactly one stream, but not the same one:

| run | diverging user | that stream's text |
|---|---|---|
| v73 (35718626867) | user 0 | coherent (`" would ...veniss ourclassraw ..."`) |
| v75 (35720619574) | user 1 | **degenerate** (`" ...venvenvenven..."`, one token repeated 256 times) |

In v75 stream 1 emits its first token and then repeats it for all 256 - the signature of
decode state that never advances. In v73 the same stream was fine and a different user
diverged with a merely different continuation.

**The victim moving between runs is the diagnostic.** A deterministic fault in the GDN
slot-remap padding added for v73 would be expected to hit the same slot each time. It
does not. That weakens, but does not clear, that change as a suspect: the padding only
acts when a condense occurs, and when a condense occurs is itself timing-dependent, so a
varying victim is not inconsistent with it either. Both runs carried the graft, so there
is no four-user control without it - v71 crashed before two users decoded.

What can be said without guessing: **the four-user configuration is not token-exact**,
in two runs out of two, and that blocks qualification regardless of throughput. The gate
exists to require equality and it is correctly refusing.

This outranks the throughput work. A 10 tok/s four-user configuration that produces
wrong tokens is not a slower correct system; it is an incorrect one.

---

## RETRACTION: v75 is not "the first four-user number", and 10 tok/s is not a baseline

Two claims in the v75 section above are withdrawn.

**1. "The first four-user throughput measurement in this programme" is false.**
`docs/200tps-verdict-2026-09-22.md` records run **35658854824 (m3native v34)**: four
concurrent users at 32768 context, `gate_passed: true`, **token-exact against each
user's own single-user reference**, mean **23.0 tok/s per user**
(20.2 / 26.2 / 21.5 / 24.2). That is better than v75 on both axes and predates all of
this work. I wrote the claim without reading a doc that was already in the repository -
the second time in this session that skipping that check produced a wrong statement.

**2. The 10 tok/s figure is not comparable to 23 tok/s, so it is not a regression.**
The two arms are not the same configuration:

| | v34 | v75 |
|---|---|---|
| K64 kernel graft (`KOPGRAFT64`) | yes | no |
| `M3NATIVE_TRACED_PROPOSAL` | yes | **no - so the arm passes `QWEN_FAST_EAGER_PROPOSAL=1`** |
| `M3NATIVE_PIPELINED_COMMITS` / `_PROPOSALS` | yes | no |
| `M3NATIVE_PACKED_PROPOSAL` | yes | no |
| `M3NATIVE_GDN_USER_BATCH` | yes | no |
| `M3NATIVE_FAST_COMMIT` | yes | no |
| `M3NATIVE_TRACED_PUBLISH` | yes | no |
| `M3NATIVE_GDN_STATE_COPY_BATCH` | yes | no |
| chunked prefill + alternation | no | yes |

v75 ran with **none** of the eight optimisation flags and with eager proposals, which
task #43 exists to retire. The ~10 tok/s says what an unoptimised arm does; it does not
measure the cost of Lever N.

**The token-exactness comparison is confounded for the same reason.** v34 was 4/4;
v73 and v75 are 3/4. Several of the missing flags touch per-user state directly
(packed proposal, GDN state-copy batch, traced publish), so the divergence cannot be
attributed to the M1/M2 work on this evidence either. It remains real and unexplained -
see task #62 - but "Lever N broke token-exactness" is **not** supported.

**What this does not change.** The 200 tok/s verdict stands on its own measurements and
is untouched by any of this: at 23.0 tok/s the stack is 8.7x short; the round budget for
200 tok/s is 29.4 ms against a measured 253-258 ms; the packed verify trace alone is
157.7 ms, 5.4x the whole budget; and deleting 100% of SDPA, its companion and every
matmul still leaves 66.0 ms, 2.2x over - at 32768 context, a fifth of the 163,840 the
target names.

**The comparable experiment has not been run.** v77 should carry v34's full flag set
PLUS chunked prefill and the alternation. Only that isolates what Lever N costs or buys.

---

# THE CONTROLLED RESULT: Lever N makes this configuration slower and incorrect

Three runs, everything held constant except the named variable - same image
(`sha256:67a28229`), same seven optimisation flags, same K64 kernel graft, same four
users at 32,768-token prompts, same 256 completion tokens each.

| | v83 control | v80 Lever N | v34 reference |
|---|---|---|---|
| Lever N model graft, chunked prefill, alternation | **no** | **yes** | no |
| `M3NATIVE_GDN_USER_BATCH` | no | no | yes |
| `gate_passed` | **true** | false | true |
| token-exact streams | **4 / 4** | 3 / 4 | 4 / 4 |
| mean acceptance length | **7.17** | 1.00-4.00 | 5.89 |
| TTFT (s) | 13.6 / 26.5 / 39.6 / 52.5 | 15.5 / 30.0 / **65.4 / 79.5** | 13.5 / 26.5 / 39.6 / 52.5 |
| wall for all four users | **65.1 s** | 98.1 s | - |

**Lever N is 51% slower end to end** (98.1 s against 65.1 s) and fails the equality gate
the control passes.

## Reading the rate correctly

The gate's `tokens` field counts streamed CHUNKS, not tokens, and `gaps_ms` is the gap
between chunks. v83 shows 43 chunks and 256 completion tokens because its acceptance is
7.17; v80 shows 256 chunks for 256 tokens because its acceptance collapses toward 1. An
earlier reading of these gaps as per-token times made the control look four times slower
than the treatment, which is the exact opposite of the truth.

## What the evidence supports

Acceptance falls from **7.17 to between 1.00 and 4.00**, and exactly one stream loses
token-equality - in v75 and v80 that stream is user 1, in v73 it was user 0. Both
symptoms point the same way: the M1 chunked prefill and the M2 alternation disturb
per-user speculative and recurrent state. A draft that is rejected is a round spent for
one token, which is sufficient to explain the wall-clock loss without any other cause.

The alternation does exactly what it was designed to do - it yields real decode steps
mid-prefill, proven in v67 against the v65 control - but the design assumed the decode
steps it wins are worth more than the prefill progress it defers. **At four users this
measurement says they are not.** TTFT for users three and four goes from 39.6/52.5 s to
65.4/79.5 s, and nothing in the decode column pays that back.

## Honest conclusion about this line of work

`docs/200tps-verdict-2026-09-22.md` already established that 200 tok/s per user is
unreachable, on v34's own measurements, by arguments this work does not touch. What
these three runs add is that **Lever N as built does not improve the four-user
configuration - it degrades it on every axis measured**: gate, correctness, acceptance,
TTFT and total wall.

That is a negative result about my own work, and the right response is to say so rather
than to keep tuning `r`. The stall M2 targets is real, but the cure costs more than the
disease at four users, and the correctness regression is disqualifying on its own.
