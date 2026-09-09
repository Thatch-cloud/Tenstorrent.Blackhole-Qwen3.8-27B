# DSpark: real target-bound proposals

**Prepared, not hardware-qualified. No new PP or committed-TG result.**

This joins previously tested components instead of repeating a whole-model
simulator run. One hardware session loads the target and drafter once.

| Boundary | This test |
| --- | --- |
| Target | Actual pinned Qwen3.8-27B; two P150A cards |
| Target context / draft history | 32 / 32 tokens; complete history, no hidden truncation |
| Features | Actual post-layer outputs at 5, 19, 33, 47, 61 |
| Proposal | Actual target embedding, learned FC, five layers, target LM head, Markov feedback |
| Vocabulary | All 248,320 IDs; all seven query rows, including row zero |
| Fabric | Four explicit links for proposal collectives |
| Cases | First 32 tokens of two coding fixtures; replay order A/B/A |
| Verification | Eight native serial target predictions, before and after proposal execution |
| State | All recurrent slots, valid KV prefix and borrowed buffer identities |
| Learned parameters | All 60 used tensors checked before and after; all 62 checkpoint hashes retained |
| Performance | Not a batched verifier, publication loop, coding-quality score or TG benchmark |

## Faster execution

- Reuse the content-addressed native build from the successful backbone run.
- Check target imports, snapshot, tokenizer and component provenance before building.
- Prepare real features and eager references before capture. No target forwards
  or new device-input allocations while the proposal trace is live.
- Compare complete logits, Markov scores and IDs, rather than re-auditing every
  internal backbone stage already covered by the 2,398-check hardware run.
- Preserve numerical failures. Functional replay does not clear their gate.

The new lookup-layout prerequisite passes 44 simulator checks with clean exit 0.
The full model is **not** run in TTsim.

## After this gate

Connect full 4K history, qualified wider proposals and the batched target verifier
with committed-feature publication. Then screen matched single-stream PP / CTX /
committed TG and advance winners through coding/context tests. A CTX32 integration
pass cannot establish the 200 TG target or replace the 4K benchmark.

Serving defaults remain unchanged. Select the explicit `dspark-target` CI suite
only while the pair is allocated to this experiment.
