# TTFT and the decode stall: what the four-user arm actually pays

**Date:** 2026-09-22. **Source:** m3native v34, run `35658854824` - four concurrent users, 32,768
prompt tokens each, `gate_passed: true`, token-exact. Every number below is from that run's
`m3native-gate.json` stream records, not modelled.

## 1. The measurement

| user | TTFT | first inter-token gap | later gaps | total wall |
|---:|---:|---:|---:|---:|
| 0 | 13.5 s | **39.5 s** | 253 ms median | 65.5 s |
| 2 | 26.5 s | **26.5 s** | 253 ms median | 65.1 s |
| 1 | 39.6 s | **13.4 s** | 257 ms median | 62.9 s |
| 3 | 52.5 s | 0.5 s | 258 ms median | 63.9 s |

Two separate costs hide in those columns, and they have the same cause.

## 2. Prefill is strictly serial, at 13.1 s per user

Sorted TTFTs are 13.5, 26.5, 39.6, 52.5 s. The deltas are **13.0, 13.1, 12.9 s** - three
intervals agreeing to within 0.2 s. One prefill of 32,768 tokens takes ~13.1 s (2,423 tok/s), and
the next user's prefill does not begin until the previous one finishes. The last user waits
**3.9x** the first user's TTFT for no reason other than queue position.

## 3. Every user then freezes for exactly the remaining prefills

The first inter-token gap is not noise. It is, for each user, the time the *other* users still
need to prefill:

| user | first gap | remaining prefills at that moment | predicted |
|---|---:|---:|---:|
| 0 | 39.5 s | 3 | 39.3 s |
| 2 | 26.5 s | 2 | 26.2 s |
| 1 | 13.4 s | 1 | 13.1 s |
| 3 | 0.5 s | 0 | 0 s |

Every row matches to within 0.4 s. A user gets its first token, then stops dead until every
outstanding prefill in the engine has finished, and only then decodes normally at ~253 ms per
round. The decode path itself is untouched by this - the later gaps are the same 253-258 ms the
steady state shows.

## 4. The size of it

**79.4 seconds of decode stall across four users, against 257.4 seconds of total user-facing
wall: 31%.** Nearly a third of the time a user spends waiting on this endpoint is spent frozen
behind somebody else's prefill, after that user has already been told the model is responding.

For comparison, the entire decode-side optimisation programme to date - GDN user batching, packed
proposals, publish fusion, pipelined commits - moved the round from 294 ms to 255 ms, about 39 ms
per round, or ~2 s across a 50-round stream. The stall is forty times larger than everything the
decode work has won.

## 5. Why this is the right target, and what it is not

It is **not** the 200 tok/s question. `docs/200tps-verdict-2026-09-22.md` settles that
separately: 200 tok/s per user is unreachable here, and no amount of prefill work changes it,
because the verify trace alone is 5.4x the round budget the target implies.

It is the question of whether this endpoint is usable by more than one person at a time. At four
users the answer today is that three of them watch a frozen stream for between 13 and 40 seconds.
That is a worse user-visible defect than the decode rate, and unlike the decode rate it is not
bounded below by attention bandwidth - it is a scheduling property.

`docs/lever-N-prefill-decode-interleave.md` (branch `lever-n-prefill-decode-interleave`) specifies
the fix: section 3.1 makes prefill resumable so it has chunk boundaries to yield at, and section
3.3 alternates decode steps between prefill chunks. Its own section 4 predicts the stall drops
from the full prefill to ~0.5 s per chunk. Applied here, that would turn 39.5 s into something
near the round time.

Two caveats this document cannot settle, which belong to the scoping work that follows it:

- Lever N was designed against the managed endpoint, where **speculative decoding is off** (its
  section 6 says so explicitly). This arm is the fast path: dflash T16 drafts, four packed users.
  Whether the graft reaches this prefill path at all is an open question.
- Lever N v1 keeps the **one-in-flight prefill** rule, which fixes the stall but not the serial
  ramp of section 2. The last user still waits behind three prefills; only v2 scratch parking, or
  batching prefill across users, addresses that.
