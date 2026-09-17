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
The current workflow explicitly selects `--diagnose-activation`; a pass qualifies
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
