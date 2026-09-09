# Separate experiment: faster approximate drafting

**Proposed, not implemented or qualified.** Keep the current exact live-query
candidate separate. The failed native-SDPA numerical tests remain failed; do not
relax their tolerances or relabel them as passes.

## Different acceptance question

A speculative drafter proposes tokens; the unchanged greedy target decides what
can be committed. Draft arithmetic may therefore differ without changing final
tokens, provided verification, rejection, repair and publication remain correct.
That does not make an inaccurate kernel an exact replacement. It permits a
separate approximate-drafter experiment with explicit end-to-end gates.

The current exact ABBA requires identical proposal and acceptance trajectories.
That is appropriate for the live-query optimization, but would reject a different
drafter before measuring whether it actually improves committed throughput.

## Required gates

1. Simulator-first native BF16 draft-attention checks: valid masks, finite
   outputs, stable changed-input replay and cleanup. Record numerical differences;
   do not claim the old 0.01/0.01 accuracy gate passed.
2. Keep target weights, target kernels, target sampling and rollback unchanged.
   Do not install a global attention patch that could alter target execution.
3. Audit all proposed/accepted/rejected tokens and publication boundaries.
   Include forced rejection, mixed acceptance and EOS cases.
4. Require every complete request's committed tokens, target GDN, valid KV and
   inactive state to match the native greedy reference exactly. Proposal counts
   and acceptance may differ between arms, but must be reported, not hidden.
5. Compare complete uninstrumented PP / CTX / TG and setup-inclusive latency on
   the same coding prompts. Report acceptance and draft/verify/commit costs.
6. Run the held-out executable coding gate before adoption. This proposal does
   not certify stochastic sampling, serving behavior or coding quality.

Native attention has no qualified hardware speed figure here. This route is
worth measuring, not assumed faster. Target-verifier optimization is still
required for 200 committed TG at the current T8 acceptance rate.
