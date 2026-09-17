# Compact draft score selection

Status: simulator and combined hardware correctness pass. **Not promoted:**
complete-cycle gain is only 1.02%, below the predeclared paired threshold.

## Combined hardware result

Run **35268945497**, commit `8a86be5617dd4bb229b24b0f78cf21dc5d63af5c`,
finishes in **6m29s** on both P150A cards. Fresh control/candidate audits and
four ABBA requests pass at 4,096 context tokens, one stream.

| Arm | PP tok/s | Committed TG tok/s | Accepted / proposed |
| --- | ---: | ---: | ---: |
| Unchanged T16 | 3,322.37 | 124.72 | 224 / 300 |
| Compact selection | 3,314.59 | 125.99 | 224 / 300 |

Both arms commit 242 timed tokens with identical output, acceptance and target
state. Paired TG changes are **+0.56% / +1.48%**; neither exceeds 2%.
Mean T16 draft time changes 24.585 -> 24.154 ms, verification/readback
66.529 -> 66.312 ms, selection/commit 5.047 -> 4.829 ms, and whole cycle
96.972 -> 95.997 ms. These observed differences do not isolate causality;
the unchanged verifier also varies. Logical score-byte savings are not a
proportional end-to-end gain. Do not repeat this unchanged candidate as a
qualified speedup or promote it into serving.

The independent combined report checker passes. All 873 script, 1,520 native
and 14 adapter-source hashes are unchanged; scopes restore and candidate calls
match the prepared feedback route. Process exit is zero and device closure is
clean. Report SHA-256:
`f8d81e5a6ed8f289fd5d8acf8c07948434f4796d9daee63d61e340c185cbf058`.
Held-out coding quality and sustained serving remain unqualified.

## Design and qualification history

The current fused score-layout kernel adds one FP32 base row to the unchanged
FP32 Markov bias, then writes all 248,320 scores to DRAM for native argmax.
One logical row is 993,280 bytes; fifteen sequential proposal steps write
14,899,200 bytes per chip before selection reads. These are logical payloads,
not measured bus traffic. The rank-256 dot-product costs are additional.

Candidate: retain that exact SFPU addition, reduce its outputs to one score/token
pair per worker, then reduce the compact winners. At 110 workers, the winner
payload is 880 bytes per step before alignment. No vocabulary truncation, top-k
approximation or learned-weight precision change is permitted.

`compact_score_device.execute_local_winners` now builds the first stage. It uses
contiguous whole-tile partitions and leaves the SFPU addition body unchanged.
The data-movement processor compares FP32 encodings using integer keys; each
worker writes a 32-byte record containing score bits, original token ID and a
nonfinite flag. Reserved words are zero. At 110 workers this actual record
payload is 3,520 bytes, rather than the idealized 880-byte pair-only payload.
Scratch reuse waits for the last compute consumer before overwriting its input
buffer. No host argmax or production feedback route has been introduced.

`reduce_winners` adds a one-core reducer on each chip. It reads the compact
records, verifies token bounds and validity, and explicitly breaks ties using
original token IDs. The result contains token ID, invalid flag and score bits.
Any nonfinite/invalid worker record produces an invalid-token sentinel and an
error flag, not a usable feedback token. Production integration must never feed
that sentinel into embedding. The current simulator-only guard prevents serving
or hardware use. Eight host tests cover semantic ordering, record validity,
partition coverage, unchanged SFPU source and bounded scratch allocation.

`compact_score_selection.py` specifies finite FP32 comparison via integer keys,
canonicalizes signed zero, and resolves ties to the lowest original token ID.
Contiguous partitions avoid a worker-order tie policy accidentally selecting a
higher token. Three host tests cover full vocabulary, random finite bit patterns,
partition-boundary ties, signed zero and invalid/nonfinite inputs.

Before device admission: compare with native TT argmax including ties, signed
zeros and subnormals; establish a fail-closed policy for nonfinite scores; verify
changed-input trace replay and both replicas. Native tie behavior is not yet
proven by the host tests. Avoid using scalar floating-point arithmetic on the
data-movement processor or silently changing SFPU rounding.

This is not sufficient on its own for 200 TG: the measured T16 cycle would remain
below target even with drafting removed. A useful candidate must reduce complete
combined request time, while verifier work continues separately. If local
reduction overhead exceeds saved writes/argmax cost, reject it rather than
claiming the payload ratio as a speedup.

Simulator run 35265211320 stopped in eager execution after 20.48 seconds:
the reducer read a DRAM record into an L1 address with incompatible alignment
(`src=0x6866c0`, `dst=0x1b3a0`). This was not a timeout or a numerical failure.
The reducer now spaces its 32-byte records at 64-byte L1 boundaries and reserves
8 KiB scratch, including the aligned final result. External records remain
32 bytes; arithmetic and acceptance checks are unchanged.

## Simulator qualification

Run **35265936552**, commit `2fc4252399080611cad5da7816091139cfb03b4c`,
passes in **4m24s**. The downloaded report independently passes
`compact_score_report.py`: 20 exact two-chip eager/replay comparisons and
40 immutable-input checks, complete 248,320-token vocabulary, steps 0 and 14,
random inputs, cross-partition ties, signed zeros and subnormals. Every worker
record matches its native SFPU score partition, and final tokens match native
argmax and CPU argmax of those native scores. Ten source hashes are unchanged
and match the candidate checkout; device teardown succeeds.

Report SHA-256:
`1d26ba993ff5f1e0cf1b5213a522fba82c45202c6f9ef3e3390f5155c2a1ae0b`.

This qualifies finite-score selection, not learned Markov feedback, nonfinite
device handling, combined hardware throughput or coding quality. The next gate
must validate token-only feedback into the existing Markov loop without feeding
an invalid sentinel into embedding. Then measure the combined matched runtime;
retain native selection unless complete-cycle TG improves with correctness intact.

### Feedback extension awaiting qualification

`compact_markov.py` preserves the native embedding, BF16 weights and HiFi4
FP32 bias matmul. It replaces only score materialization/argmax with compact
selection and a one-token slice. The reducer adds an explicit safe-feedback
field: token zero on invalid input, while retaining the invalid flag and
sentinel in the diagnostic fields. Every step's diagnostics must pass host
validation before any proposal is usable; safe zero is not an accepted fallback.
There is no serving or hardware route for this simulator-only implementation.

Returned proposal tokens now preserve the reducer's invalid sentinel, separately
from safe internal feedback IDs. Thus `PreparedDSparkProposal.read_tokens` can
reject an invalid chain at its existing range check, without fifteen additional
diagnostic readbacks per proposal. The probe also injects NaNs at feedback step
four and requires the whole chain to be rejected after safe device execution.

The extended probe keeps full-vocabulary selection checks, adds a two-chip
15-step rank-256 feedback chain at vocabulary 64 (eager and changed-input
replay), and tests device NaN rejection. This does not qualify full-vocabulary
learned feedback. Twelve host tests pass. The extended source/report schema
supersedes the earlier finite-selection-only gate, so run 35265936552 does not
qualify the changed reducer or new feedback loop.

Run **35267717551** at `64540405bbb339fe3cfedcc729f878533cd89105`
now passes the extended gate in **5m5s**. Independently validated: 20 selection
checks, 40 unchanged-input checks, 60 feedback comparisons, device NaN flags,
and whole-chain rejection with safe internal feedback. All 13 source hashes
match; exit and container cleanup succeed. Report SHA-256:
`22243fad7110052d740d9931fa4ff115b09008cddce64c65a95a529e2883b820`.

The hardware adapter preparation changes only execution guards and the Python
import binding; AST tests enforce unchanged function bodies after those guards.
The request-scoped hook validates admitted sources and exact generated adapters,
requires full fifteen-step vocabulary, counts calls, and restores on failure.
Combined staging now passes locally against the frozen winning recipe and the
downloaded simulator evidence. The hardware workflow runs two fresh feature/state
audits, then control/candidate/candidate/control complete requests at 4K. Both
arms retain HiFi4 drafting, norm prefetch, shared QK, fused T16 MLP, native target
attention and incremental publication. Candidate calls must match the prepared
score-layout calls exactly. Output and acceptance must match; TG includes the
entire decode loop, not isolated selection timing. No serving default changes.
Twenty-one focused host tests pass; measured hardware acceptance remains pending.
