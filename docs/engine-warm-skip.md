# Phase-1 quick wins: ledger off, and the engine warm skip

Two changes for the per-request engine build, each behind its own default-off flag and usable without Stage E. Neither is
measured on hardware yet; the estimates below are derived from figures already in the Stage E design, and each flag has a
byte-comparison plan for the combined window (`quickwin-ledger`, `quickwin-warm`).

## 1. `QWEN_FAST_MEMORY_LEDGER_OFF=1`

The serving image sets `QWEN_FAST_MEMORY_LEDGER=1` in its environment, because the gates read the ledger's lines out of every
arm. So a production profile cannot simply leave the ledger flag unset. `memory_ledger.enabled()` now returns false when the
new flag is `1`, whatever the image flag says. `begin()` then builds no ledger and every hook returns at its first line: no
allocator reading, no host walk of device tensors, no report file. Nothing the ledger reads is returned to a caller that acts
on it (the `before` token only brackets an `after` log line), so no admission, placement or trace depends on it. Gate arms
that want the readings leave the flag unset. The harness promises no ledger marker for an arm that sets it.

Not covered: the lifecycle plans that time a drop on the ledger's prefill lines (`--drops ...:build`) need the ledger and
must not set the flag; the plans in this change do not use drops.

## 2. `QWEN_FAST_ENGINE_WARM_SKIP=1`

`VerifierEngine.__init__` runs one warm-up eager forward per captured bucket (widths 1, 2 and 4 beside the packed block),
then captures the bucket's trace. With the flag, the caller passes `skip_compiled_warm=True`; the engine skips the warm of a
bucket whose key (`warm_key`) an earlier build in the process warmed to completion. The key is what the compiled programs
depend on: the bucket's rows and GDN retention, the page table's width, the engine's options, the model and mesh, and (only
when a replay plan keys buckets by position) the capture position. A bucket no earlier build warmed still warms, and a warm
that fails is not recorded. Off, the engine is called exactly as before and records nothing.

The parked attach (Stage E) builds four engines of one shape, so with the flag on the last three skip their warm forwards
and so does a single rebuild.

### What a warm forward writes, and what overwrites it before any read

The warm forward runs `operation(fixture)` on a throwaway fixture at the bucket's capture position. Its writes:

| Written | Next access | Why nothing reads the warm's bytes |
|---|---|---|
| Native GDN slot 0 (all 48 layers) | `restore_initial()` before the next warm or capture, and once more after the last capture | `restore` copies the whole slot-0 state from `initial`, which was saved before any warm. A skipped build leaves slot 0 equal to `initial` already, and the capture loop restores it anyway. |
| Pooled per-width GDN checkpoints | the eager commit publications of the capture loop write them; the bucket's verify trace writes them in-trace | A checkpoint is read only by `publish`, which needs `phase == 'verified'`, so a verify replay has rewritten it first (`test_every_pooled_checkpoint_and_tap_is_written_by_its_bucket_s_verify_trace_before_anything_reads_it`). |
| Pooled feature taps | the verify trace writes them first | Read by the publication after a verify. The one read that precedes a verify is the publish prewarm's, whose products go to the drafter's spare banks and are discarded (`publish_prewarm`, "why it is exact"); it reads whatever the taps hold, finite either way. |
| The fixture's entry state and inputs | freed with the warm fixture (`warm.close()`) | Dead. |
| The K/V cache at rows `[P, P + r)` of the request's own pages (`P` the position, `r` the bucket's width) | rewritten by the verify replay | See below. |
| Transient intermediates and logits | freed | Dead. |

A capture executes nothing on the device, so it neither reads nor writes any of the above; the trace's replay is the first
device access after the warm, and the census test counts a replay as its trace's first access to each tensor.

### The K/V rows

The warm writes `W` over `[P, P + r)`. A verify of width `w` at frontier `F` writes `[F, F + w)` and reads `[0, F + w)` (a
row attends to every earlier position and to itself, and to nothing later). The frontier then advances by the `a` accepted
tokens, `1 <= a <= w`, over rows that verify wrote. By induction every position below the frontier was written by the
prompt's prefill (below `P`) or by a verify (at or above it), and the verify that reads a position at or past the frontier
wrote it in the same forward. So no verify reads a warm-written row. The simulation `test_a_verify_never_reads_a_row_the_warm_wrote`
runs random `(w, a)` sequences at six prompt lengths, and its negative control (a verify that writes fewer rows than it
reads) does see the warm's rows.

The same fact holds without the warm: a request's pages past `P` are whatever the previous holder of those blocks left, which
vLLM reuses without clearing, so correctness cannot rest on their content. Skipping the warm only changes which finite
values sit there.

### What the skip does not change

`SkipEquivalenceTests` runs the census world (the real `VerifierEngine`, session, request and pool over a fake `ttnn` that
logs every allocation, read, write, free, capture and replay) twice, once with the flag and once without, for a second
request whose shape the first already warmed. From the engine's first verify capture on, through six rounds of both
requests and the closes, the two device event sequences are equal with tensors renamed. The skipped run has fewer events
overall (the warm's own). Flag off, `FlagOffParityTests` in `test_parked_census` still holds the churn trail equal to the
base commit's, event for event.

### Residual risks (for the hardware comparison, not for the CPU tests)

- Program compilation: the warm exists to compile before capture. The key covers what the programs depend on as far as the
  host can see; whether an ordinary bucket's programs are independent of the capture position is the same fact the parked
  design and every replay at another position already lean on. A miss would show as a compile inside a capture, not as a
  different byte; the plan's kernel-cache growth judgement watches it.
- Allocator layout: the warm's intermediates are freed before the capture, so the capture sees the same free space with or
  without it, except for anything the first forward of a shape allocates and keeps; the skip only applies after that shape's
  first warm.

## 3. Estimated saving per arrival

From the Stage E design's inputs (all for the per-request build on phase 1): a warm build costs 1.65-2.23 s including the
ledger (measured); the ledger's readings on one build or rebind cost up to 0.35 s (measured); the three warm forwards cost
about 0.3 s (derived from the constructor's 1.47-1.71 s and the bucket count). Ledger off: up to 0.35 s. Warm skip: about
0.3 s. The design's combined phase-1 figure is 0.4-0.75 s per arrival. Nothing here is measured yet; the plan arms record
the per-arrival build time on both sides, from the engine line and the `[PINDIAG] dram after engine` timing.

## 4. Gate arms

`quickwin-ledger` and `quickwin-warm` (`c2_serving_gate.py --plan ...`, job key `C2_GATE_PLAN`) each run four arms on a sticky
or S2 profile, real text: a solo chain over eight prompt lengths (page-aligned and not, answer budgets 2 to 256 with
`ignore_eos` on every user, so the warm's bucket set varies) and four concurrent users, each with the flag off and on. The
judgement is the strict exactness policy: every user of an on arm identical to the off arm's, solo and concurrent. The
ledger plan also requires the off arms to show ledger lines and the on arms none. The warm plan requires the on arms to
warm the first build and skip a later one, and the off arms to log no warm line.
