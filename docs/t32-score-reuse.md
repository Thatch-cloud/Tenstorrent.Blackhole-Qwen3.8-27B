# T32: reuse the T16 score-layout optimisation

The integrated T32 drafter still uses the older native Markov score conversion
path. Comparing it against the fast T16 recipe would therefore change both
proposal width and score-layout policy.

The simulator-only candidate `dspark_t32_score_layout.py` reuses the existing
T16 fused base-row-plus-bias kernel, without changing the native dot product,
argmax, full vocabulary or sequential feedback. It keeps the original T32
7/7/7/7/3 segmentation and per-query observer indices. The T16 kernels and default
T32 implementation remain untouched; no speed improvement is claimed.

`dspark-t32-markov-probe.py --fused-score-layout` selects the candidate explicitly.
Start with the weight-free vocabulary-64 fixture, then qualify learned full
vocabulary and changed-input replay using the existing numerical oracle and
tolerances. The report records which score path executed and fingerprints all
five candidate/score-kernel dependencies. Hardware is rejected by the candidate.

Pending: simulator qualification, then instance-scoped integration into complete
T32 proposal capture and a matched complete-request T16/T32 comparison. Synthetic
component success alone does not establish acceptance, coding quality or TG.
The first bounded run, 35177093623, stopped at the host-I/O gate before Docker:
15.63% full I/O stall measured over 15 seconds, against the unchanged 1% limit.
The simulator step was skipped; no numerical result exists. The whole job
finished in 35 seconds rather than waiting through container setup.
CPU regression run 35177082045 passed all 144 tests. This is not device evidence.
Wait for the host backlog to clear before retrying; do not weaken the admission gate.
