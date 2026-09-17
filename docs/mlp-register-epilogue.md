# Register-resident rounded MLP epilogue

**Candidate only. No new throughput or correctness acceptance.**

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

The current workflow selects `--diagnose-rounding`: the exact activation-control
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
