# Register-resident rounded MLP epilogue

**Candidate only. No new throughput or correctness acceptance.**

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
