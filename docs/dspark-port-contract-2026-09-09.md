# DSpark v2: port contract, not a speed result

**Status: native selector prototype passes its small simulator gate; learned full-vocabulary test running.**
No integrated DSpark backbone, acceptance or hardware throughput result exists yet.
The native-attention DFlash2 candidate now reaches74.27 committed TG at CTX4,096
across two matched runs. Its roughly62ms T8 verifier exceeds the entire35.59ms
cycle budget needed for200TG at the measured acceptance. Wider useful verification
is therefore worth investigating alongside target kernels; it is not assumed faster.

## Pinned checkpoint

[RadixArk DSpark v2](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/tree/b9a5dbdf03bc999c6c73c426b19c2d9041cea393)
is inspected at revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
The initial metadata intake fetches only36,763 bytes: config, two source files retained as inert text,
the8-byte safetensors prefix and6,640-byte header. No remote code is imported;
that step alone does not download or hash-verify any weight payload.

| Boundary | DSpark v2 requirement |
| --- | --- |
| Payload inventory | 62 BF16 tensors;1,857,358,337 parameters; exact contiguous bounds |
| Backbone | Five5120-hidden layers;32 Q /8 KV heads, dimension128 |
| Context | Full attention; no automatic DFlash2 2K rolling-window substitution |
| Rotary | YaRN, factor32, original context8192; not default DFlash2 tables |
| Target features | Taps5/19/33/47/61; concatenate, learned projection and normalization |
| Draft selector | Rank256 vanilla Markov correction over the full vocabulary |
| Confidence | Trained5376-input confidence projection; future policy needs separate qualification |
| Published serving | Seven noise/proposal rows; anchor plus seven proposals gives eight target rows |
| Training | Sixteen future positions; **not proof of qualified15-proposal serving** |

These boundaries come from the pinned
[configuration](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/config.json)
and [model source](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/dspark.py).
Matching dimensions do not make this a DFlash2 checkpoint swap. DFlash2's current
eight physical noise rows and top16 selector have different semantics.

## Selector contract

For each proposal position, add a vocabulary-wide bias from the previous token's
rank256 embedding and the successor matrix, then select the greedy token. That
new proposal becomes the previous token for the next position. Reusing the anchor
at every position or truncating base logits to top16 first is not equivalent.
The [SGLang implementation](https://github.com/sgl-project/sglang/blob/708f51e44bc64f546a60fa9631f0e7d99493d0a0/python/sglang/srt/models/dspark.py)
provides this explicit serving contract. Its source SHA256 is
`d70ebbbbb81b93c0ad9e5baf1cec73d8fea5ab90851d14ca9128afa23fb1d75a`.

The checkpoint's inherited `spec_generate` helper instead samples base logits
without calling the Markov head. Do not use that helper as the DSpark serving
oracle. This finding is specific to the inspected
[pinned helper](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/blob/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/dflash.py).

`dspark_markov.py` is a CPU semantic reference only. The initial14 tests cover
sequential feedback, batch isolation, full-vocabulary selection, ties, empty
blocks, nonfinite/overflow rejection, input immutability and metadata mismatch.
The width15 synthetic case tests indexing, not trained acceptance or TT support.
It does not certify BF16 GPU/TT rounding or stochastic sampling.
The expanded DSpark suite now has33 CPU tests; the complete CI host suite passes
1,054 tests and the simulator harness passes57. No qualified DFlash2/target
runtime source is changed by this separate port work.

## Native Markov selector prototype

`dspark_markov_device.py` chains embedding lookup, native HiFi4 matmul, FP32 bias
addition and full-vocabulary argmax. Each selected token feeds the next lookup
on device, without a CPU round trip. This composes existing kernels; it is not
yet a fused Markov kernel. Each chip owns a replicated head; fabric/sharded-head
integration is not covered by this gate.

| Simulator gate | Scope | Result |
| --- | --- | --- |
| `20260909T113815Z-406` | Synthetic64-token vocabulary;3 sequential proposals; both chips | Pass:80 checks, clean close, outer exit0 |
| `20260909T113949Z-365` | Learned full248,320-token vocabulary;7 sequential proposals; both chips | Running; no result yet |

The small gate covers18 eager token/score checks,24 changed-input trace checks,
28 input-immutability checks,8 weight checks and2 missing-update controls. Every
toy score is exact against the CPU reference, including all-equal ties. The
independent gate verifies all coordinates and unchanged runtime/source hashes.
Original native binaries and packer remain unchanged; no simulator graft is used.
Its report SHA256 is
`f3246494e332724490f5e85398c74c2674d7bc67a1e7d57e3add57ef767253fa`.
The report and outer exit are preserved in `scripts/ci/dspark-markov-simulator-64.*`.

The learned test must additionally pass100 checks with both complete learned
matrices, synthetic base logits, exact greedy IDs and stable device feedback.
It compares FP32 prototype scores at1e-4 relative/absolute tolerance, not
bitwise SGLang BF16 arithmetic. No backbone, target features, full model,
accepted-token rate or hardware timing is established by either selector test.

## Complete learned weights staged

The3,714,723,322-byte object is downloaded to the D: worktree and verified
against its pinned Hugging Face LFS SHA256:
`2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`.
Header geometry and all62 tensor extents pass. The previously range-staged
Markov matrices also match their subranges in this complete object:

| Matrix | Shape | SHA256 |
| --- | --- | --- |
| Predecessor | 248320 x256 BF16 | `80cdfb6122576b2cf02a3df40e83209fd5514aed82ef381d6c551d6c925d3c57` |
| Successor | 248320 x256 BF16 | `39ba4adba9d613e61ba2ff4ef4151fdfda7ac898dab12e4603d3cff6a5e4d2e2` |

Downloads are bounded, refuse existing/partial paths and publish only after
size/hash validation. No checkpoint Python is executed. The whole-object report
is `scripts/ci/dspark-checkpoint-v2.json`, SHA256
`da56ad1b2f262e18fedd4753f364cc24994a6556f9aa8f3f4348f8fe061a0250`.
This is weight integrity, not learned backbone correctness.

## Evidence and next gates

`scripts/ci/dspark-intake-v2.json` preserves the successful metadata report.
Report SHA256:
`a3b8c987b36b3129622fe3ef006a1014955f78c6d6be188538b409cc5f710673`.
The header SHA256 is
`992cdd260cf8761176cb7d6e94a64339189819e74609e7f0c2989ec33ee182f1`.
All simulator/hardware/serving eligibility flags remain false.

1. Complete the learned Markov gate; validate target-feature, YaRN and full-attention
   primitives without reusing incompatible DFlash2 assumptions. Weights are staged.
2. Port learned primitives to TTsim, including changed inputs, masks, trace
   ownership and exact target-verifier isolation. CPU tests are not that gate.
3. Run a complete seven-proposal hardware correctness/acceptance baseline through
   CI, then separately qualify wider proposals. Keep all rejections and overhead.
4. Compare committed PP / CTX / TG on held-out executable coding tasks against
   the retained DFlash2 candidate before any serving change.

Reproduce intake without cards, into a new directory:

```sh
python scripts/ci/dspark_intake.py --output hardware-evidence.local/dspark-v2-new-intake
```

There is no DSpark hardware rate, model-quality score or200-TG result here.
