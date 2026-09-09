# Separate experiment: faster approximate drafting

**Both proposal-only native-attention simulator gates pass.**
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
- 1,008 CI tests and57 simulator-harness tests pass. Next is request integration
  with exact native target tokens/state and separate
  proposal-policy acceptance reporting. No hardware dispatch is authorized by
  a finite-output check alone.

The first original-packer attempt, `20260909T102155Z-400`, exits1 at TT-Sim's
`tensix_pacr: Disable_pack_zero_flags` limitation, before producing a numerical
report. Preserve that failure. Native SDPA internally uses packer accumulation
even though the public matmul configuration disables it. The retry uses the
same reviewed, owned packer compatibility patch as the earlier MLP simulations;
neither SDPA compute sources nor runtime binaries are changed. Re-qualification
against the restored original packer also passes. The packer matches its original
backup byte-for-byte, both native binaries retain their original hashes, and
the owned compatibility lock is released.

## Simulator results

| Draft history | Operands | Maximum absolute error | Original 0.01/0.01 test | Mask/replay gate |
| ---: | --- | ---: | --- | --- |
| 31 | Saved learned layer0/rank0 | 0.392848 | Fails, unchanged | Pass:76 checks |
| 2,048 | Synthetic | 0.004196 | Passes this fixture only | Pass:76 checks |

The learned case has mean absolute error0.048489 and RMS0.064525. Both cases
replicate the same operands onto two chips; these are not learned rank-one or
full-model tests. Changed-input and fully masked K/V perturbations also pass.
These history lengths are not model CTX benchmark rows. No hardware latency,
proposal acceptance, coding-quality result or TG gain is established.

| Completed run | Report SHA256 |
| --- | --- |
| `20260909T102545Z-403` | `bbe085e7f873820e9e01c286653f213b20452dcdc6252638d499e6de14e3c2c7` |
| `20260909T102636Z-527` | `fea42395d6cb9203663016a007636d7d873f147a2d97f1a2e7dcd34dcc6dd42a` |

Both outer exits are zero; their shared exit-file SHA256 is
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
Reports and exit files are checked in byte-identically under
`scripts/ci/proposal-native-attention-simulator-{31,2048}.*`.

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
