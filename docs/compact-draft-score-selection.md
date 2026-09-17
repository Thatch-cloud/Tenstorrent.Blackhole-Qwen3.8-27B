# Compact draft score selection

Status: host semantic specification only. No kernel, simulator qualification,
hardware measurement or runtime integration yet.

The current fused score-layout kernel adds one FP32 base row to the unchanged
FP32 Markov bias, then writes all 248,320 scores to DRAM for native argmax.
One logical row is 993,280 bytes; fifteen sequential proposal steps write
14,899,200 bytes per chip before selection reads. These are logical payloads,
not measured bus traffic. The rank-256 dot-product costs are additional.

Candidate: retain that exact SFPU addition, reduce its outputs to one score/token
pair per worker, then reduce the compact winners. At 110 workers, the winner
payload is 880 bytes per step before alignment. No vocabulary truncation, top-k
approximation or learned-weight precision change is permitted.

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
