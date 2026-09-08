# DFlash2 native attention: work in progress

**Not a speed result. The last measured single-stream rate remains 58.33 committed TG.**
The proposed replacement removes the composed attention pipeline in favour of
one native SDPA operation. Target verification and serving defaults are unchanged.

## Findings

| Experiment | Result | Decision |
| --- | --- | --- |
| Native BF16 streaming | Fails original synthetic numerical gate | Do not promote |
| Precise main exponential, original buffers | Passes six synthetic chip/context cases and changed-input trace replay | Primitive prerequisite only |
| Same correction, learned layer-zero attention/convolution | 65 of 16384 rank-zero attention values fail | Do not promote |
| Also use FP32 intermediate buffers | 41 values still fail on learned inputs | Not sufficient |
| FP32 buffers, one 64-key chunk, captured learned operands | 44 values fail on each replicated chip | Not just cross-chunk correction |

All numerical tests retain `rtol=0.01, atol=0.01`. There is no tolerance relaxation,
full-drafter claim, hardware timing, coding-quality result or new committed TG.
The captured case contains rank-zero operands replicated on both simulated chips;
it is not a substitute for validating original rank-one weights.

## Implementation

- `native_draft_sdpa.py` corrects the hardcoded approximate exponential only for
  the masked, noncausal TP2 DFlash2 signature: B1, 16 Q/4 KV heads, head width128,
  query chunk32, FP32 accumulation, `exp_approx_mode=False`.
- Other signatures, including the target's width256 attention, retain the old
  exponential path. Pinned source hashes reject drift before installation.
- A parent process owns the temporary patch and restores it after the probe
  child exits, including an abrupt child exit. Concurrent installation is rejected.
- Prepared learned attention has an opt-in native branch. Hardware promotion
  remains disabled in the probes while learned-input correctness fails.
- `native-draft-operands-probe.py` replays the saved learned Q/K/V/mask directly,
  checks the captured SHA before loading, and uses `torch.load(weights_only=True)`.
  This avoids repeating expensive learned projections to debug one attention op.

The separate `draft-sdpa-fp32-intermediates.patch` is an unsuccessful prototype,
not a runtime default. Its factory source changes from
`a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc` to
`53d3a1cd85901f9016ad58d39d6940055f1b327f1aafa74758a6a2c8b9bef73c`.
Simulator execution needs the already-audited packer-zero-flags compatibility
graft; a pass does not qualify the unmodified simulator or physical fabric.

## Evidence

Local reports are preserved under `hardware-evidence.local/native-draft-sdpa-sim/`.

| Simulator report, September 8 UTC | Scope | SHA256 |
| --- | --- | --- |
| `111018Z-338` | Scoped precise exponential, synthetic | `0d2bb14461fe62b465f1193be31c042ae5e0791c96c4876fa2a69ba1aa0cb94f` |
| `111212Z-291` | Four eager, six replay, two stale-input controls | `44175e489784652f1aeac3b5134f977e7374b49db8f848274293ca31bacaf204` |
| `111627Z-380` | Learned attention failure, original intermediate formats | `8e7f60dd0f2c668bf3b47f1ded645d6807ebc523cb0fdb3455c5cb5d10d56de4` |
| `112456Z-292` | Learned attention failure, FP32 intermediate formats | `a33a0beb7a7fc46af24f2bc190a1fbf851bb5ac4c9ec59c7a8ab4ace78d64934` |

Captured operand SHA256:
`0974cf572f0db9291f56b4ee322829d60f61484e2521b4b69c8393035b541e49`.

The local MLP fixture also failed content/conversion integrity checks and was
not used to certify a complete layer. Historical integrity failures remain
unresolved; re-reading successfully is not a fix. No hardware run was dispatched
for the failed native-attention candidate.

Validation: 759 host tests and 60 speculative-harness tests pass; shell syntax
and patch checks pass. Temporary simulator source, packer and shared-library
changes were restored to their original hashes after testing.
