# DSpark v2: port contract, not a speed result

**Status: bounded metadata intake and CPU selector semantics pass. No TT port yet.**
The native-attention DFlash2 candidate now reaches74.27 committed TG at CTX4,096
across two matched runs. Its roughly62ms T8 verifier exceeds the entire35.59ms
cycle budget needed for200TG at the measured acceptance. Wider useful verification
is therefore worth investigating alongside target kernels; it is not assumed faster.

## Pinned checkpoint

[RadixArk DSpark v2](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/tree/b9a5dbdf03bc999c6c73c426b19c2d9041cea393)
is inspected at revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`.
Only36,763 bytes are fetched: config, two source files retained as inert text,
the8-byte safetensors prefix and6,640-byte header. No remote code is imported;
no weight payload is downloaded or hash-verified.

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

`dspark_markov.py` is a CPU semantic reference only. Fourteen tests cover
sequential feedback, batch isolation, full-vocabulary selection, ties, empty
blocks, nonfinite/overflow rejection, input immutability and metadata mismatch.
The width15 synthetic case tests indexing, not trained acceptance or TT support.
It does not certify BF16 GPU/TT rounding or stochastic sampling.
The complete existing CI host suite also passes:1,035 tests. No qualified
DFlash2/target runtime source is changed by this separate intake/reference work.

## Evidence and next gates

`scripts/ci/dspark-intake-v2.json` preserves the successful metadata report.
Report SHA256:
`a3b8c987b36b3129622fe3ef006a1014955f78c6d6be188538b409cc5f710673`.
The header SHA256 is
`992cdd260cf8761176cb7d6e94a64339189819e74609e7f0c2989ec33ee182f1`.
All simulator/hardware/serving eligibility flags remain false.

1. Stage hash-verified learned slices; validate target-feature, YaRN, full-attention
   and Markov semantics without reusing incompatible DFlash2 assumptions.
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
