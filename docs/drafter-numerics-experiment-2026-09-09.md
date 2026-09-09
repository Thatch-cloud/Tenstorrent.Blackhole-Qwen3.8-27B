# Separate experiment: faster approximate drafting

**Proposal-only native attention and its simulator gate are implemented.**
No complete-request integration or performance result yet. Keep the exact
live-query candidate separate. The failed native-SDPA numerical tests remain
failed; their tolerances and results are unchanged.

## First implementation

- Calls native BF16 GQA with explicit T8 masks; no SDPA precision graft, target
  hook or serving change. The simulator-only packer compatibility exception is
  explicit and must be restored before any hardware qualification.
- Separate 31-row learned-operand and 2,048-row synthetic-history simulations
  record numerical errors and the original 0.01/0.01 comparison outcome. A
  proposal mask/replay pass does **not** certify attention accuracy.
- Each context requires76 checks: six finite-output diagnostics, eight exact
  eager/replay comparisons, 56 unchanged-input checks, two stale controls and
  four masked-key/value perturbation checks, covering both chips and all32 rows.
- Nine experiment and15 native source hashes are bound to the report. Native
  sources must remain unchanged within the declared simulator packer scope,
  and the outer wrapper must exit successfully.
- 1,008 CI tests and57 simulator-harness tests pass. Simulation is next, then
  request integration with exact native target tokens/state and separate
  proposal-policy acceptance reporting. No hardware dispatch is authorized by
  a finite-output check alone.

The first original-packer attempt, `20260909T102155Z-400`, exits1 at TT-Sim's
`tensix_pacr: Disable_pack_zero_flags` limitation, before producing a numerical
report. Preserve that failure. Native SDPA internally uses packer accumulation
even though the public matmul configuration disables it. The retry uses the
same reviewed, owned packer compatibility patch as the earlier MLP simulations;
neither SDPA compute sources nor runtime binaries are changed. Re-qualification
against the restored original packer is required afterward.

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
