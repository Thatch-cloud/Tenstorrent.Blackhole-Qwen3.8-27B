# Register-resident rounded MLP epilogue

**Combined hardware correctness passes; performance screen fails. Keep the control.**

## Combined hardware outcome

[35240478989](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35240478989),
source `f3f69186e025b80ea64b731e44c7bca3f349e0d9`, completes in **6m26s**.
Independent report validation confirms both fresh native audits, complete ABBA
timing, exact output/state/features, unchanged sources and all-layer packed
weights. Devices close cleanly; the container exits zero without OOM.

| 4K / one stream | Cold PP | Committed TG | Accepted / proposed |
| --- | ---: | ---: | ---: |
| Unchanged winning T16 | 3,371.07 | 121.67 | 224 / 300 |
| Register-resident epilogue | 3,303.60 | 123.87 | 224 / 300 |

The aggregate gain is **1.81%**, but matched pairs are **+3.56% / +0.11%**;
the required repeatable improvement screen fails. Each arm commits 242 tokens.
Report SHA-256: `c6708a01a05c2671ce11989001e2aa1c5234a6d9e505028c1d501137d23fbedf`.

| Timed request order | Draft ms/block | Verify/readback ms/block | Commit ms/block | Whole cycle ms |
| --- | ---: | ---: | ---: | ---: |
| Control A | 27.80 | 66.56 | 4.94 | 100.08 |
| Candidate B | 24.46 | 66.47 | 4.96 | 96.64 |
| Candidate B | 26.39 | 66.53 | 4.98 | 98.63 |
| Control A | 26.61 | 66.46 | 4.86 | 98.74 |

Mean verifier/readback changes by only **0.009 ms**, not a meaningful target
speedup. The larger first-pair gain accompanies faster drafting, which this
target-only kernel does not directly optimize. Do not attribute that entire
gain to epilogue fusion or rerun the unchanged candidate. The rounding fix is
useful correctness knowledge, not an accepted performance optimization.
Retain the original epilogue; no serving change or held-out quality claim.

## First simulator outcome

[35235850613](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35235850613),
source `db35b7315f16a4d7f82c67727fe3362bdedfc794`, finishes in 2m05s.
Compilation and all four packed-weight comparisons pass. The first chip's eager
output has **216 mismatches out of 139,264 values**; the test stops before replay
or chip-1 numerical acceptance. Exit is 1; container cleanup succeeds. This is
a numerical rejection, not a timeout. No hardware run is permitted for this version.
Report SHA-256: `eba85c2867612bbe44efa1bd5245dc7984a5772a345d17a247f2f5a03075e353`.

The next simulator-only diagnostic moves SiLU to MATH but retains the original
BF16 pack/reload and product. This separates activation relocation from explicit
register rounding and product scheduling. It is not a faster candidate or an
acceptance override. Mismatch values and row distribution are now retained.
The v2 workflow explicitly selects `--diagnose-activation`; a pass qualifies
only this diagnostic, not the rejected register-resident epilogue.

## Activation diagnostic passes

[35236664694](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35236664694),
source `133107e83313977558b71e4118359507d06809a7`, passes in **3m53s**.
Independent artifact validation confirms both eager outputs, 12 changed-input
replay comparisons, four stale-input controls and four packed-weight checks.
Exit and all container cleanup statuses are zero. Reconstructed staged source,
executed source fingerprint, helper fingerprint and pinned runtime match.

Report: `196582cf05e57534b98a3c28cda896904c5ec0113625a75b31819bf81f92c2bf`.
Generated compute: `70d5b1e73dca6a1d98d6bdf92312743e2838b9390ae1651bae0470a19f700449`.

This rules out activation relocation alone on this fixture. It does **not**
prove the register-resident product correct: explicit rounding, removal of the
pack/reload conversion and product scheduling remain different in the rejected
version. Next isolate explicit rounding while keeping the passing diagnostic's
pack/reload and product; do not waive the 216 mismatches or promote this
diagnostic as a throughput improvement. Hardware and serving are unchanged.

## Rounding isolation

The v3 workflow selects `--diagnose-rounding`: the exact activation-control
source plus two explicit FP32-to-BF16 casts before its unchanged native packing.
The post-loop unpack and BF16 product are byte-identical to the passing control.
This asks whether the casts themselves change the observed numerical result;
it does not yet test removing native packing. Mode flags are mutually exclusive
and recorded in both the staging manifest and executed kernel metadata.

### Rounding isolation rejects the explicit casts

[35237711938](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35237711938),
source `48e4c459b08d1eaf2279628c7966064304590a27`, finishes in **2m03s**.
It reproduces **216 / 139,264** eager mismatches on chip 0, with finite values
and maximum absolute error 1.0. All four packed-weight checks pass. Exit is 1;
container cleanup succeeds. Replay and chip-1 output acceptance are not reached.
Generated and staged source fingerprints match local reconstruction.

Report: `36777917414802c921fbc1f7ede24c191976d70da511f0bd1c35b28883eee148`.
Generated compute: `f46ba4560d86034768d2707b5cea088edc8b698122d2945950757d3dc952c3cd`.

All 32 retained mismatch examples have smaller candidate magnitude, across
positive and negative outputs; failures occur in every token row. The matching
mismatch count does not establish identical failed coordinates to v1, whose
report did not retain them. The passing activation diagnostic and this failed
diagnostic show that introducing the explicit casts is sufficient to break
exactness with packing/product otherwise retained.

Next investigate the pinned packer's rounding rule and the explicit cast's
tie handling before changing the register candidate. A rounding-policy
hypothesis must pass exact simulation, not a PCC or tolerance waiver. The
current cast is rejected; the register-resident candidate remains barred from
hardware. There is no new committed-TG result.

## Nearest-away hypothesis

The current simulator route selects `--nearest-away` and returns to the complete
register-resident epilogue, not the diagnostic retaining pack/reload. It changes
only the explicit cast's two programmable constants: an unconditional `0x8000`
integer bias replaces the ties-to-even `0x7fff + retained-LSB` bias. Both operands
are still rounded before the same product. Native packing remains the oracle;
nearest-away is a hypothesis, not a claimed packer specification.

Before program construction the adapter checks the runtime's cast instruction
chain and original constants, and records cast/packer header hashes. Changed
implementations fail closed. Host tests enumerate every finite BF16 exponent/
mantissa bin and both signs at five rounding boundaries, confirming that the
two host policies differ only on even midpoints. This is not device validation.
The unused CB, readers, K accumulation and serving defaults stay unchanged.

### Full register-resident candidate passes simulation

[35238822290](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/35238822290),
source `b6764c410e21d064196efa6630a10caef22f326f`, passes in **3m53s**.
Independent validation confirms two exact eager outputs, 12 changed-input
replay comparisons, four stale-input controls and four packed-weight checks.
Exit and all cleanup statuses are zero. The complete register-resident version,
not the packing-retaining diagnostic, was executed. No latency is qualified.

Report: `cda39d8936cbe2056510c0cbcadb41df8e5c478ad277b3641f13fb121986e543`.
Generated compute: `d212a495967dd0db31ee447ed7761b7ce6af3306b35fa42611e1296c4a1b1a82`.
Staged projection: `d26d67ab4c893de327c8a2c6aa9ec949d904332a9b2a91353c5889f30272b6d4`.

`mlp_register_epilogue_gate.py` binds admission to that report, both unchanged
readers, both exact helper sources, the projection, native baseline and runtime.
It checks the entire replay/weight matrix, not just `passed`. Local tests admit
the downloaded artifact and reject altered readers and the wrong physical packer.

The simulator uses the existing packer-zero graft (`8aaf199a...`). Physical
admission explicitly requires the original packer (`87b9c251...`), observed in
the retained native-source inventory of combined run 35233247736. The cast
header must stay `1cfea093...` on both. This known simulator/physical distinction
is explicit in admission and still requires full combined hardware correctness;
the passing simulator is not hardware acceptance. No serving defaults change.

Next: wire this admitted source into the existing loaded 4K T16/DSpark ABBA
comparison, retaining all winning features, fresh native token/state/feature
audits, all 64 target layers and complete-cycle TG. Do not benchmark the
activation-only diagnostic or report simulated time as a speed improvement.

## Combined comparison prepared

`qwen-mlp-register-epilogue-combined.yml` stages the frozen winning runtime and
the admitted candidate separately. One model load covers two fresh correctness
audits followed by control/candidate/candidate/control timing at 4K, one stream.
Only the scoped T16 gate/up projection changes. Shared Q/K, norm prefetch,
incremental history, draft-tail assembly, native down projection and four-link
collectives remain active in both arms. Serving is untouched.

The adapter rechecks physical runtime admission before and after the request
series. Its independent report validator checks all-layer packed weights,
tokens, state, inactive slots, five feature taps and complete-loop TG. Both
paired gains must exceed 2% to pass the preliminary improvement screen; passing
does not qualify held-out coding quality or establish the 200-TG goal.
Local full frozen staging, evidence admission and staged imports pass. The
12-minute job cap and existing host-I/O pressure gate remain in place.

The combined clock diagnostic identifies gate rounding/packing and the rounded
product as measurable work. This candidate removes the intermediate BF16
pack/reload and separate post-loop product, not the already-tested weight reader.

| Kept | Changed |
| --- | --- |
| T16, three gate/up pairs per worker | SiLU runs on MATH rather than PACK |
| Native K loop and FP32 packer accumulation | Explicit FP32-to-BF16 rounding of both operands in DST |
| BF4 weights, readers, CB allocation and output placement | Same BF16 product executes before final output pack |
| Original runtime and serving defaults | Simulator-only staged compute source |

The unused intermediate CB remains allocated for this first comparison, so a
memory-footprint change cannot explain the result. Activation implementation,
rounding and processor scheduling must pass exact simulation; they are not
assumed equivalent merely because the algebra matches. No precision gate is
relaxed. Only two destination tiles are live in the new branch.

1. Run bounded T16 native eager and changed-input replay checks on both simulated
   chips, with packed-weight integrity and stale-input negative controls.
2. If exact, pin the generated source and test the unchanged combined 4K recipe
   with fresh audits and matched ABBA committed TG. No standalone speed promotion.
3. Reject a numerical failure or a non-repeatable whole-request gain. Preserve
   the accepted control. Longer contexts follow combined acceptance.

Workflow: `qwen-mlp-register-epilogue-sim.yml`, explicit immutable experiment tag,
12-minute whole-job cap. It does not load full model weights or expose cards.
The 200 committed-TG objective remains unmet.
