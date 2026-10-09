# TP4 engine reuse: parked per-slot engines at eight seats x 262k

Branch `tp4/engine-reuse`, from `tp4/levern-prefix`. Everything here is behind flags that default off; with every flag off the serving path is the parent
commit's, call for call (tests compare against the parent). Nothing has run on a card. Every claim about speed is an estimate until the card gate (section 7)
has run; every claim about exactness is CPU-proved on a fake four-chip world and is to be re-proved on cards by the audited pair (A0 and A1).

## 1. What it does

Today every admitted request builds its own verifier engine, drafter device and proposal captures: measured on the eight-seat image at 2.1 to 2.7 s with the
audits off, and 2.3 to 2.9 s audited. The build is device-exclusive, so every live seat stalls for it. A fresh quad capture is 0.39 to 0.56 s, a pair
0.20 to 0.23 s and a single rebuild 0.20 to 0.28 s, again with every seat stalled.

Engine reuse (`QWEN_FAST_PARKED_ENGINES=1`) builds one engine per pool slot at attach, on a synthetic request (one prompt token, budget 16, every page table
entry on page 0), and PARKS it. A request then REBINDS the parked engine of the slot it takes: rezero the slot's state, a windowed projection, K/V reseed, the
page table rewrite and the carry save, in place of a build. A rebind is estimated at 0.2 to 0.4 s. Closing a request returns the engine to its slot parked, not
destroyed. With engine reuse alone (no 2c) the detach also retires the departing slot's pair, quad and single traces and rebuilds its single when the DRAM
split allows: a 0.2 to 0.28 s stall of every seat that decodes, which is what the E1 arm measures against 2c.

Slot-bound drafter traces (`QWEN_FAST_PARKED_DRAFTS=1`, "2c") go further: the pair and quad drafter traces are bound to their slots for the process, so a
block re-formation at admission or departure no longer recaptures them. Single-user traces are kept per slot.

The deadline governor (`QWEN_FAST_LEVERN_BUILD_MS`) can charge a pending prefill the cost of what its slot will really do: the fixed 2,500 ms stays the
default, a number replaces it, and `learned` charges each pending prefill an EWMA of the rebinds and builds the bridge factory measured.

## 2. Exactness rules

Output must be byte-identical to a cold engine. The rules the code and the tests hold:

- R1: a rebound engine's state equals a cold engine's before its first step (rezeroed state, reseeded K/V, rewritten page table, a fresh carry).
- R2: no foreign trace replay may run between a verify and its publish. The replay ledger records captures and replays; only a foreign replay fails (the
  attach's own prepared proposal replays once). At TP4 the verify override never set the base class's replay mark, so the mark is set in the TP4 engine.
- R3: a rebind that fails after its first device write is not repaired in place: the slot unparks and the request is served by a cold build over the SAME
  prefill capture (one owner of the capture, one ExitStack), on today's terms. A rebind that is refused on the host, before any device write, falls back
  without touching the device.
- R4: a rebound engine asks and serves the widths a cold engine of the same budget would (1, 2 and 4 captured widths are restricted by the request's budget).
- R5: placement equals the pool's rule applied over SERVING slots. A parked slot is always lent out in the pool's accounting, so the pool's own rule would
  see every block full; one shared function computes placement for both the set and the pool.

Six defects were found in the Stage E code by reading it against the TP4 path, and fixed in the port: the TP4 verify override never set the R2 mark; the TP4
close released carry traces only in some phases (a parked engine closed at shutdown leaked them); the rebind imported the two-chip draft history class
unconditionally; the page-table bindings and the audit digest assumed two chips; the coordinator's generation and release keys ignored the eight-seat quad
blocks; and slot choice took the lowest free slot where the pool places by block occupancy.

## 3. Memory

Per card at TP4, measured: a parked engine adds 0.47 to 0.50 GB of allocator growth and 40 MB of trace region; eight take about 3.8 to 3.9 GB. With 2c every
single is kept as well: a bound single trace is 206 MB per chip, a pair 188 MB and a quad 375 MB, about 1.65 GB more. After attach the card has 8.9 GB free
with the audits off and 7.6 GB audited, so the unaudited case keeps about 2 GB beside a parked arrival's need of about 1.3 GB; audited with 2c the free
DRAM after attach is about 2.0 GB and the trace region about 80 MB, which is thin: that is what the ballast ladder (section 7) is for. `QWEN_FAST_PARKED_DRAFTS=0`
frees about 1.5 GB.

The admission terms change with the parked engine. When the slot the next request takes holds a parked engine, nothing is built at admission, so the free term
before the prefill is the prefill transient, the rebind's peak R (100 MB with the windowed projection), a released single's rebuild S, and the reserve; the
post-prefill backstop asks R, S and the reserve. The trace term asks the single's trace only when S applies. The free term also counts a CREDIT: what the
release ladder could free (the book's pairs, other slots' singles, and one other idle parked slot), so that a departing decoder freeing little does not hold an
arrival until the server drains. An arrival that is short at its backstop runs the ladder before it is refused (an arrival on a slot with no parked engine, after a
fault, too: on today's terms, with the same ladder). A held arrival logs how long the hold lasted, counted from the start of that request's own hold.

## 4. Lifecycle and the kill switch

- Faults: `QWEN_FAST_PARKED_FAULT=park` refuses the first park once and `=rebind` refuses the first rebind on the host; both are gate only. The slot unparks,
  a cold build serves, and the slot re-parks at the next idle moment (no decoder, no prefill, a step that schedules nothing), never at a detach: a detach
  happens while other seats decode, and a re-park is a full synthetic build with its captures, a stall of every one of them.
- Kill switch: a file named `parked.off` under the serving image's state directory, polled at the top of every execute (at most once a second), unparks every
  idle slot, and every serving slot at its close, and restores today's per-request path for every later request, mid-traffic, with a log line naming the
  count. The draft book (2c) is unregistered with it, so singles are released and traces closed with their members as today. For the gate there is also
  a trigger of its own, `QWEN_FAST_PARKED_OFF_AFTER=<n>`, which latches the switch after the nth rebind (job K1).
- Instance state: a parked device or engine that gains per-request instance attributes (which would shadow methods and leak a request's state into the next)
  is detected and logged; the audit treats it as a failure.
- Gate-only knobs: `QWEN_FAST_PARKED_AUDIT` (digest every rebind against a cold twin's reference and null tables), `QWEN_FAST_PARKED_NEGATIVE`
  (`carry`, `pages`, `widths`, `drafter`: deliberately break one rule so the exactness gate can be shown to fail), `QWEN_FAST_GATE_DRAM_BALLAST` (hold bytes
  per chip from after attach to close, in whole tiles in replicated buffers of at most 32 MiB), `QWEN_FAST_PARKED_OFF_AFTER` and `QWEN_FAST_PARKED_OFF_PATH`
  (the kill switch's trigger and its file). Under the audit with no parked engines (the control arm) a replay between a verify and its publication is a log line
  (`R2 violation`) the gate counts, not a raise.

## 5. Flags and profiles

`QWEN_FAST_PARKED_ENGINES` and `QWEN_FAST_PARKED_DRAFTS` are refused outside a gate-only profile until the owner decides to ship them (the contract's
`parked_problems`; a traffic profile carries them only under its owner traffic waiver, docs/tp4-ship-ln-w2-er.md), and the gate instruments (the audit, the negative controls, the faults, the ballast, the kill switch's trigger and file) outside a gate's
profile and in the process environment of any profile that does not name them; `QWEN_FAST_LEVERN_BUILD_MS` is the governor's own flag and is not parked-specific.
The production profile is unchanged. The gate profiles are generated twins of the ship-prefix-levern
profiles (`make_parked_profiles.py`), never hand edited: the arms control, `-parked-e1` (engine reuse alone), `-parked` (with 2c and the learned governor
cost), `-parked-audit`, the negative controls, the two fault profiles, `-parked-kill` and a four-level ballast ladder (512, 1024, 1536 and 2048 MB). `-parked`
against `-parked-e1` is 2c and the learned governor together; neither alone is isolated.

## 6. CPU proof

On a fake four-chip world (the census): the parked set, engine, device, wiring, admission, drafts and profile suites (`test_parked_tp4_*`), the gate-side judge,
markers and compare scripts and the smoke tests against a fake server (`test_parked_tp4_judge`, `test_parked_tp4_smoke`), the governor's cost
(`test_levern_admission_cost`), the job pack (`test_tp4_engine_reuse_jobs`) and, on the installed vLLM 0.25.1 scheduler, `test_parked_scheduler_vllm` (the
parked terms switching under a real scheduler, a failing terms reader admitting, a hold not starving the decoder, every block free at the end). The real-vLLM
test runs from the `experiment/fast-vllm-cpu-v*` tag workflow only.

## 7. Card gate (templates in `scripts/ci/references/tp4-engine-reuse-jobs`)

Templates for the combined development window; no tag is pushed from this branch. Order: X0 status, rescan and reset; B0 build; A0 (audited control) and A1
(audited parked arm), compared by `parked_compare.py`; C0 and C1 (churn, abort and reuse); N0 to N4 (the negative controls must FAIL their own judges); F1 and
F2 (injected faults); K1 (the kill switch, latched by the server); M1 to M5 (more than 200 admissions with the ledger on, the 253,920-token worst corner, the ballast ladder); H1 to H3
(hang shapes, arrivals after at least 100 block rounds, three clean runs in a row); R1 to R6 (ER5: closed-loop coding turns on eight seats, control and arms
alternated ABAB, paired per turn by index, never by unpaired medians); Z reset. The cards are never handed back and the node agent is untouched.

Adoption is the owner's decision. Proposed rule: every exactness and lifecycle gate passes, and ER5's committed tokens per elapsed second is higher in every
pair with no seat's worst gap higher, beyond an A/A noise floor taken from the repeated control.

## 8. Review outcome

Taken from the review of the design: the audited capacity numbers above (audited free DRAM about 2.0 GB with 2c, trace region about 80 MB) are what the
ballast ladder and M1's floor of 0.25 GB per card are built around; the single owner for the fallback's prefill capture; the host-only refusal before any
device write; the instance-leak check; keeping `verifier_engine.py` byte-identical (it is in the image's bundle inventory), so the parked phase lives in the
TP4 subclass. Not taken: merging the parent's main branch (21 conflicting files plus unrelated work: the modules and tests were copied and adapted, the shared
hunks ported by hand); parking engines at TP2 on this branch (the flag is refused there); enabling 2c in the ship profile (it stays a gate arm until the
card gate has run).

## 9. Open items

Never run on a card. The attach adds about 20 to 30 s per restart (eight synthetic builds and the drafter warm; estimate). The merge into the W2 branch is
expected to conflict in the profile file, the overlay list and the CI lists: merge, take the union of the lists, regenerate the twins from the merged parents,
re-run the census tests after committing, and build once.
