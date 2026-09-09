# DSpark v2: port contract, not a speed result

**Status: small simulator gate passes; learned FP32-score gate fails, with native matmul rounding isolated.**
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
The expanded DSpark suite has36 CPU tests; the complete CI host suite passes
1,057 tests and the simulator harness passes57. No qualified DFlash2/target
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
| `20260909T113949Z-365` | Learned full248,320-token vocabulary;7 sequential proposals; both chips | Fails original FP32 score gate; clean close, outer exit1 |

The small gate covers18 eager token/score checks,24 changed-input trace checks,
28 input-immutability checks,8 weight checks and2 missing-update controls. Every
toy score is exact against the CPU reference, including all-equal ties. The
independent gate verifies all coordinates and unchanged runtime/source hashes.
Original native binaries and packer remain unchanged; no simulator graft is used.
Its report SHA256 is
`f3246494e332724490f5e85398c74c2674d7bc67a1e7d57e3add57ef767253fa`.
The report and outer exit are preserved in `scripts/ci/dspark-markov-simulator-64.*`.

The learned test was required to pass100 checks with both complete learned
matrices, synthetic base logits, exact greedy IDs and stable device feedback.
It compares FP32 prototype scores at1e-4 relative/absolute tolerance, not
bitwise SGLang BF16 arithmetic. No backbone, target features, full model,
accepted-token rate or hardware timing is established by either selector test.

### Learned failure and arithmetic diagnostic

The first learned score comparison fails:53,427/248,320 values (21.5%) exceed
the original1e-4 relative/absolute threshold. Maximum absolute difference is
0.0025434494 at vocabulary ID1340. Only the four initial matrix-immutability
checks complete; no eager numerical, replay or device-feedback qualification
is recorded. Native/source fingerprints remain unchanged and cleanup succeeds.
The failed report remains rejected by the original gate. SHA256:
`adefd2e09ab7f835fba635129db85ad82bad1ba19378c2b2a0eda468d757dcc9`.

Diagnostic `20260909T120503Z-397` isolates the same learned anchor1596 and
vocabulary columns1312:1376, keeping the full256-term reduction. On both chips:

| Diagnostic comparison | Maximum absolute difference |
| --- | ---: |
| Native matmul versus ordinary CPU FP32 | 0.0025445223 |
| Native matmul versus existing Blackhole grouped-product rounding reference | **0, bitwise exact** |
| Default addition versus FP32 addition of the actual native bias | **0** |
| Explicit FP32 output flag versus default addition | **0** |

Independent recomputation from saved operands confirms these results. The
isolated issue is native BF16 grouped-product arithmetic, not the bias-add
operation. The narrow diagnostic is not a full-vocabulary pass and does not
certify every proposal. No tolerance is widened or failed report relabeled.

Diagnostic report SHA256:
`9559e951228f2caaa2708df8471737744d39b2ee8c7d34327d29a11d9648966f`.
Saved operand SHA256:
`d050d3be174cf85dc3b87bbd0aa0323f6fbd308fa57b3ffd68eadbbd2d049d5c`.
Both simulator processes terminate cleanly; no hardware job is dispatched.

### Separate native-arithmetic experiment - September 10

`--native-reference` is opt-in. It requires bitwise-exact native scores against
the grouped-product oracle, while recording both same-input FP32 differences
and an independent FP32 proposal trajectory. Each trajectory uses its own
previous selected token; disagreement cannot be hidden by reusing the native ID.
The original FP32 gate rejects this policy. No tolerance is widened.

| Gate | Current result |
| --- | --- |
| Small native-policy simulator, 64 IDs / 3 proposals | **80 checks pass**, both chips, exact scores, changed-input replay and clean exit |
| Learned native-policy simulator, 248,320 IDs / 7 proposals | First eager trajectory passes on both chips; `20260909T123008Z-401` continues, not qualified yet |
| Unchanged target tokens/GDN/KV with DSpark | Not integrated or measured |

The small run is `20260909T122916Z-419`; report SHA256:
`a92ea83d79602e14603757ad1ab42f6b1472f2a643dc99302e5261490d36346c`.
Its toy weights are exactly representable and do not establish learned accuracy.
The learned run retains all100 required checks and a bounded three-hour timeout.
Progress distinguishes CPU reference generation, enqueued steps, synchronization
and completed audits. An enqueued step is not reported as completed computation.

Reconcile the separate policy with `dspark_markov_gate.py --native-reference`;
without that flag, the CLI continues to require the original FP32 policy.
Full-vocabulary deterministic feedback and eventual unchanged-target token/state
audits remain mandatory. Neither selector gate certifies a complete model.

### YaRN CPU tables - September 10

DSpark now has separate positional tables; DFlash2's unscaled tables are not reused.
The checkpoint's factor32 YaRN scales **both** cosine and sine by1.3465735903,
including position zero. Its correction band is14..29 of the64 frequencies.

| Check | Result |
| --- | --- |
| All64 inverse frequencies and attention scale | Exact against pinned Transformers5.8.1 functions |
| FP32 and BF16 cosine/sine, all128 columns | Bitwise exact at21 selected absolute positions, including8191/8192 and262143 |
| Chunked tables and seven-row query suffix | CPU tests pass; absolute positions preserved |
| TT rotary operation, learned attention, full model | Still pending |

The reference checker verifies source hashes before extracting only the reviewed
YaRN parameter and rotary-forward functions. No Transformers package is installed,
no checkpoint model code is imported, and dynamic decorators are disabled for this
static YaRN comparison. This is CPU table evidence, not262K model execution.
Report: `scripts/ci/dspark-yarn-cpu-reference.json`, SHA256
`97424ecd6a1355d24c7f4c73bfd07c9ca3c8035e452b032cf91d20d6412043f9`.
Validation:1,077 CI host tests and57 simulator-harness tests pass, including
56 DSpark CPU tests. These counts are separate from actual simulator execution.

Primary sources at Transformers commit
[`cc832f9`](https://github.com/huggingface/transformers/tree/cc832f9055ba11c8c55f918ab4bda9472b910d48):
[YaRN parameters](https://github.com/huggingface/transformers/blob/cc832f9055ba11c8c55f918ab4bda9472b910d48/src/transformers/modeling_rope_utils.py),
[Qwen3 rotary forward](https://github.com/huggingface/transformers/blob/cc832f9055ba11c8c55f918ab4bda9472b910d48/src/transformers/models/qwen3/modeling_qwen3.py).

### Native rotary gate prepared

The simulator-only rotary composition is queued behind the complete Markov gate;
it must not open a second simulator while the first is active. No rotary device
result is claimed yet. The native operation receives FP32-widened BF16 heads and
YaRN tables, then returns BF16 proposal heads. This is not a new fused kernel.

| Coverage | Required check |
| --- | --- |
| TP2 query heads | 16 heads/chip, seven live rows, padded to32 |
| TP2 key heads | Four heads/chip,39 and4,103 live rows, padded to64 and4,128 |
| Absolute positions | Changed tables alone change live output; includes the8192 boundary and last position262143 |
| Padding | Poisoning only unused rows must leave every live output bit unchanged |
| Trace | Five changed-input replays per shape; all padded outputs and addresses match their own eager reference |
| Ownership | Every borrowed head/table input remains bitwise unchanged; missing table updates are detected |

The planned matrix has234 checks. CPU comparisons retain the existing widened
draft-rotary threshold of0.01 relative/absolute; raw full/valid-row errors and
bitwise equality are reported separately. Exact CPU YaRN tables do not imply
bitwise-exact device multiplication. Replay and padding isolation require exact
bits regardless of that numerical threshold. Final target-token/state and coding
quality gates are unchanged and still outstanding.

Validation:1,085 host tests and58 simulator-harness tests pass, including64 DSpark
CPU tests. `dspark_rotary_gate.py` independently rejects missing coordinates,
failed controls, stale sources, unclean teardown and unsupported arithmetic policy.

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

1. Qualify the separate native-arithmetic Markov policy without erasing the FP32
   failure; validate target-feature, YaRN and full-attention primitives. Weights are staged.
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
