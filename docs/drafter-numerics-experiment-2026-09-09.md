# Separate experiment: faster approximate drafting

**Both proposal-only native-attention simulator gates pass.**
The complete-request comparison is dispatched; no hardware result yet. Keep the exact
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
- The primitive gate passes1,008 CI tests and57 simulator-harness tests. The
  subsequent request integration passes1,021 CI tests and the same57 harness
  tests, including separate proposal-policy acceptance reporting. A finite-output
  check alone does not authorize hardware promotion.

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
These history lengths are not model CTX benchmark rows. Simulation alone
establishes no hardware latency, acceptance, coding quality or TG gain.

| Completed run | Report SHA256 |
| --- | --- |
| `20260909T102545Z-403` | `bbe085e7f873820e9e01c286653f213b20452dcdc6252638d499e6de14e3c2c7` |
| `20260909T102636Z-527` | `fea42395d6cb9203663016a007636d7d873f147a2d97f1a2e7dcd34dcc6dd42a` |

Both outer exits are zero; their shared exit-file SHA256 is
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
Reports and exit files are checked in byte-identically under
`scripts/ci/proposal-native-attention-simulator-{31,2048}.*`.

## Complete-request comparison

Suite: `full-dflash-native-proposal-request`, opt-in only.

[CI34342721182](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34342721182)
uses immutable tag `ci-qwen-hardware-f1c1b69`. The dedicated hardware step and
post-container validator pass; independent validation of the downloaded artifact
also passes, including clean teardown and both source-pinned simulator gates.

| Setting | Control and candidate |
| --- | --- |
| Workload | One coding request at CTX4,096; captured T8 DFlash2 |
| Changed | Candidate uses native BF16 attention only inside the drafter |
| Unchanged | Target kernels/sampling/rollback, cached draft K/V, fused convolution, commit-only GDN |
| Order | One audit per policy, then uninstrumented control/candidate/candidate/control |
| Exactness | Native target tokens, active GDN, valid KV and inactive state in every request |
| Draft policy | Proposals may differ between policies; each must reproduce its own audited trajectory |
| Reporting | PP / CTX / committed TG, setup-inclusive latency, acceptance and per-block costs |

Every proposal mask is validated before upload/capture. Failed updates revoke
the prior mask authorization. Each audited policy retains eager/trace, cached/
full-history projection, convolution and committed target-feature checks.
Proposal counts, matching prefixes and committed frontiers are independently
reconciled against the final token tape, including EOS truncation.

Both simulator reports and outer exits are checked before build/device work.
Twenty-six integration/source files are fingerprinted in each request; the
original native packer is required. The final artifact must record clean device
teardown and pass the host-side policy validator after the container exits0.
Existing exact-arithmetic ABBAs still reject this different-proposal policy.

## Hardware result: 22.68% higher committed TG at 4K

One coding prompt, B1, two uninstrumented requests per arm, each committing
121 decode tokens through EOS. Audit requests are excluded from rates.

| Arm | PP tok/s | CTX | TG tok/s | Individual TG | Mean total request |
| --- | ---: | ---: | ---: | --- | ---: |
| Composed attention control | 3,338.37 | 4,096 | 61.47 | 59.69 / 63.37 | 7.41 s |
| Native approximate draft attention | 3,322.74 | 4,096 | **75.42** | 77.37 / 73.56 | 6.62 s |

TG includes draft, verification/readback and publication after the prefill seed;
it excludes prefill and fresh setup. Total includes prefill, fresh setup and
decode, excluding model loading. Rates are summed tokens divided by summed time;
all stalls remain included. This is offline request timing, not endpoint streaming.

| Mean per-block cost | Control | Candidate |
| --- | ---: | ---: |
| Draft | 40.32 ms | 20.35 ms |
| Input staging | 2.15 ms | 1.86 ms |
| Target verification/readback | 61.71 ms | 61.72 ms |
| Selection/publication | 11.16 ms | 10.06 ms |
| Complete cycle | 115.68 ms | 94.27 ms |

Both policies accept 105/119 proposals per request (**88.24%**), committing121
tokens in17 blocks. Their trajectories differ: the control has12 fully accepted
and5 mixed blocks; the candidate has13 fully accepted,3 mixed and1 zero-acceptance
block. Each reproduces its own audited trajectory. All six requests match native
target tokens, active GDN, valid KV and inactive state, with feature/cache/
convolution/trace audits retained. Different draft arithmetic is not an exact
attention replacement; the failed original numerical comparison remains failed.

Report SHA256:
`dcf0b1a00fc7d0439412093b073d8b96adec0ab03b54b3a4c49a29cd918f9b26`.
Artifact10100693555 is220,927 bytes, archive SHA256:
`6357e83d308be6c855bdf6003b0e65e4500c2305759d3767bd26072be5ad206e`.

Same-code [repeat34343945544](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34343945544)
uses the same immutable tag and passes the dedicated hardware step, clean
teardown and independent downloaded-artifact validation. All proposal counts
and each policy's trajectories reproduce the first run.

| Repeat arm | PP tok/s | CTX | TG tok/s | Individual TG | Mean total request |
| --- | ---: | ---: | ---: | --- | ---: |
| Control | 3,292.38 | 4,096 | 61.08 | 61.12 / 61.05 | 7.38 s |
| Candidate | 3,326.31 | 4,096 | **73.16** | 76.39 / 70.19 | 6.67 s |

The repeat gain is19.77%. Drafting is40.24/20.87ms per block; verification
61.73/61.80ms; publication11.68/12.15ms. All samples and stalls remain included.
Across both runs, four timed requests per arm pool to:

| Arm | PP tok/s | CTX | Committed TG tok/s | Mean total request |
| --- | ---: | ---: | ---: | ---: |
| Control | 3,315.21 | 4,096 | 61.28 | 7.39 s |
| Candidate | 3,324.52 | 4,096 | **74.27** | 6.65 s |

This is a repeat-confirmed **21.21%** gain, not a held-out coding-quality claim.
All12 complete requests, including four audits, retain exact target tokens and
state. There are still only two independent ABBA runs on one coding prompt.
Serving defaults stay unchanged.

Repeat report SHA256:
`5e08cc943e5b0be222330822124f6c72d08b06f1877c8019d02977df5a7afc28`.
Artifact10101166229 is220,861 bytes, archive SHA256:
`3b3cba1a058e0b14f93b14c77a574e0ca325414c2976f1647c64ff9e2462c893`.

At7.12 committed tokens/block, 200 TG requires a complete cycle of35.59 ms.
The unchanged verifier alone takes61.72 ms, before drafting and publication.
Draft attention is a measured improvement, not a complete route to200 at T8.

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

Target-verifier optimization is still required for 200 committed TG at the
current T8 acceptance rate. The hardware comparison does not replace held-out
executable coding acceptance or authorize serving changes.
