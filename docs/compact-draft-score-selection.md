# Compact draft score selection

Status: host specification and simulator-only two-stage reduction kernel source.
No device compilation, simulator qualification, hardware measurement or runtime
integration yet. Neither kernel has been compiled or executed on device.

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

The extended probe keeps full-vocabulary selection checks, adds a two-chip
15-step rank-256 feedback chain at vocabulary 64 (eager and changed-input
replay), and tests device NaN rejection. This does not qualify full-vocabulary
learned feedback. Twelve host tests pass. The extended source/report schema
supersedes the earlier finite-selection-only gate, so run 35265936552 does not
qualify the changed reducer or new feedback loop.
