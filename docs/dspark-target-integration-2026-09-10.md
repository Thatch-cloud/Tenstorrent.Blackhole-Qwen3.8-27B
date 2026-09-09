# DSpark: real target-bound proposals

**Target-bound proposal integration passes. No new PP or committed-TG result.**

[Hardware run 34417423120](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34417423120),
code `d426fbaa534e332dbf53b8b852263357b81815ec`, completes cleanly with exit 0.

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

## Observed result

| Check | Result |
| --- | --- |
| Actual proposal eager versus changing-input trace | Exact on A/B/A, both chips |
| Target tokens, all recurrent slots and valid KV | Unchanged; both serial oracles reproduce exactly |
| Learned device parameters | All 60 tensors exact before/after on both chips |
| Contiguous accepted proposals | Python prefix **3/7**; Rust prefix **0/7** |
| Repeat control | Python prefix repeats 3/7; not an independent sample |
| Native build/cache setup | **2 seconds**, verified cache hit; previous first build 262 seconds |
| Complete integration probe | **112.55 seconds**, including about 75.49 seconds loading the target |

The short fixtures are truncated prompt prefixes, not complete held-out coding
requests. Their low acceptance does not establish useful drafting speed or model
quality. All earlier numerical failures remain open. The artifact contains 71
source hashes, 1,517 native fingerprints, all 62 checkpoint hashes and complete
target-state digests; the independent audit reconciles the full recorded matrix.

Report `scripts/ci/dspark-target-hardware.json`, SHA-256
`399971973e34fa043577bc7c194ca54baee94934948820d97c16c0fb9998cdcb`.

## Faster execution

- Reuse the content-addressed native build from the successful backbone run.
- Check target imports, snapshot, tokenizer and component provenance before building.
- Prepare real features and eager references before capture. No target forwards
  or new device-input allocations while the proposal trace is live.
- Compare complete logits, Markov scores and IDs, rather than re-auditing every
  internal backbone stage already covered by the 2,398-check hardware run.
- Preserve numerical failures. Functional replay does not clear their gate.

The lookup-layout prerequisite passes 44 simulator checks with clean exit 0.
The full model is **not** run in TTsim. Build reuse is now measured, not assumed.

## After this gate

Connect full 4K history, qualified wider proposals and the batched target verifier
with committed-feature publication. Then screen matched single-stream PP / CTX /
committed TG and advance winners through coding/context tests. A CTX32 integration
pass cannot establish the 200 TG target or replace the 4K benchmark.

Serving defaults remain unchanged. Select the explicit `dspark-target` CI suite
only while the pair is allocated to this experiment.

The next targeted simulator experiment covers **4,096 historical rows and 15
proposal rows**. It combines 2,048 / 2,048 / 64-key chunks with a single global
softmax, preserving the existing bounded SFPU kernels rather than increasing
their on-core buffers. This new composition is not yet simulator-qualified,
connected to a full-history learned proposer, or a hardware speed result.
