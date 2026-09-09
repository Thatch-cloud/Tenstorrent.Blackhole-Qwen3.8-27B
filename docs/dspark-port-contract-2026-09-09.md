# DSpark v2: port contract, not a speed result

**Status: learned native-selector and composed-rotary gates pass; full-attention accuracy remains open.**
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
| Published serving | Seven query rows: anchor plus six masks; sample all seven, then verify anchor plus seven proposals |
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
| Learned native-policy simulator, 248,320 IDs / 7 proposals | **100 checks pass**, both chips, exact native-policy scores, changed-input replay, immutability, stale controls and clean exit0 |
| Unchanged target tokens/GDN/KV with DSpark | Not integrated or measured |

The small run is `20260909T122916Z-419`; report SHA256:
`a92ea83d79602e14603757ad1ab42f6b1472f2a643dc99302e5261490d36346c`.
Its toy weights are exactly representable and do not establish learned accuracy.
The learned run `20260909T123008Z-401` is complete and independently reconciled:
28 eager,42 replay,20 input,8 weight and2 stale checks. Its report is
`scripts/ci/dspark-markov-native-simulator-learned.json`, SHA256
`cd53b2cf20f4be1a2cae1b4f52b3162f6e5527fa194795dd53068cae88cd3a5a`.
All28 same-input FP32 score comparisons still fail the original tolerance;
worst difference is0.0034258366. Greedy tokens agree on these28 checks against
independent FP32 trajectories, not a universal token-equality or quality claim.
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

### Native rotary failed; composed alternative passes

The native rotary run `20260909T132948Z-2600` fails its unchanged0.01 relative/
absolute gate and exits cleanly. Six query-head checks pass before the poisoned
padding case fails one padded element. Key shapes and trace controls are not
qualified by this run. Its report is `scripts/ci/dspark-rotary-simulator-failed.json`,
SHA256 `8ef56e1c600bb996c9ee82d426018ebeff3e38cae4761fc376d465af2d27276d`.

Diagnostic rerun `20260909T133332Z-387` reproduces the failure. All three BF16-to-FP32
input casts and the final BF16 cast are exact; the discrepancy is inside native
rotary math. At the failing coordinate, CPU FP32 gives0.3603515625 and native
FP32 gives0.375, rounding to BF16 values0.359375 and0.375 respectively. This does
not establish a particular mantissa-rounding mechanism. The failing row is
padded, but the whole-padded gate remains required rather than being removed.

An explicit `--composed` candidate uses separate slice/negate/concatenate,
FP32 multiply and add operations, followed by BF16 output. It is **not fused**
and makes no speed claim. Its CPU reference is unchanged, with a stronger
bitwise gate on every padded output. The original native policy remains rejected.

Composed run`20260909T134544Z-456` now passes **all234 checks**, independently
reconciled against unchanged sources and native runtime, with clean exit0.
All24 padded CPU comparisons are bitwise exact (maximum error0), all30 replays
are exact, and input/padding/position/stale controls pass. Report:
`scripts/ci/dspark-rotary-composed-simulator.json`, SHA256
`769c301f85ce5816495f0fdc763bbc7a4b4bc6868b32332cd094098d61fee5c7`.
Qualify with`dspark_rotary_gate.py --composed`; the default native-policy gate
does not accept this candidate's report.

This reference widens products and addition to FP32 before BF16 output. It is
**not** the upstream eager backbone's BF16-product-rounding policy. Matching
this primitive reference does not establish bitwise upstream backbone equality;
learned proposal, acceptance and unchanged target-token/state gates remain open.

| Coverage | Required check |
| --- | --- |
| TP2 query heads | 16 heads/chip, seven live rows, padded to32 |
| TP2 key heads | Four heads/chip,39 and4,103 live rows, padded to64 and4,128 |
| Absolute positions | Changed tables alone change live output; includes the8192 boundary and last position262143 |
| Padding | Poisoning only unused rows must leave every live output bit unchanged |
| Trace | Five changed-input replays per shape; all padded outputs and addresses match their own eager reference |
| Ownership | Every borrowed head/table input remains bitwise unchanged; missing table updates are detected |

The matrix has234 checks. Native-policy CPU comparisons retain the existing widened
draft-rotary threshold of0.01 relative/absolute; raw full/valid-row errors and
bitwise equality are reported separately. Exact CPU YaRN tables do not imply
bitwise-exact device multiplication. Replay and padding isolation require exact
bits regardless of that numerical threshold. Final target-token/state and coding
quality gates are unchanged and still outstanding.

Validation after the composed candidate:1,102 host tests and58 simulator-harness
tests pass. `dspark_rotary_gate.py` independently rejects missing coordinates,
failed controls, stale sources, unclean teardown and unsupported arithmetic policy.

### Full-context attention: separate mask and numerical gates

`dspark_attention.py` does not reuse DFlash2's sliding mask. Each of the seven
live queries sees every historical key and all seven proposal keys. Padded query
rows see only the anchor, avoiding all-masked softmax rows. Host tests cover mask
geometry through the positional bound; that is not a262K device-attention run.

The simulator matrix uses31 and4,096 historical rows, two distinct chip shards,
five input patterns and six trace replays per shape. Controls poison masked K/V,
change the oldest historical value (outside a2K window in the4K case), and change
the final proposal value while checking the first query. Missing input updates
must be distinguishable. All236 coordinates are required by the independent gate.

| Attempt | Result |
| --- | --- |
| Original packer, `20260909T135328Z-399` | TTsim aborts on `Disable_pack_zero_flags`; no numerical report, no pass |
| Compatibility packer, native exponential, `20260909T135618Z-418` | First short-context comparison fails:441/65,536 values, worst failing-value error0.0348141; clean exit1 |
| Compatibility packer, precise exponential, `20260909T135936Z-400` | Six initial comparisons pass; oldest-value stress case fails one live value, error0.0756164; clean exit1 |

The packer compatibility hunk is the existing reviewed
[upstream PR53805](https://github.com/tenstorrent/tt-metal/pull/53805), not removal
of packer accumulation. The precise exponential uses the existing owned,
compile-signature-limited proposal graft. Both variants are explicit report/CLI
scopes, not changes to serving or target attention. Original header/SDPA sources
and runtime binaries are restored and verified between attempts.

Failure reports in`scripts/ci/`:

| Report | SHA256 |
| --- | --- |
| `dspark-attention-simulator-failed.json` | `b5a2492d964b7a9a1140363004cf4c26960d55ba005d9c799b78c9ed756877e0` |
| `dspark-attention-precise-simulator-failed.json` | `ffca4f5f5c8653daa18f1265b80ef6d9cb2aa80acb9863c6ff32d465106806e8` |
| `dspark-attention-precise-matrix-failed.json` | `3fce141c65e10bb4a4032875733ba3fb8850fdb727c8e41773dbeb3f1c6f5bc1` |

Full precise matrix`20260909T140234Z-417` completes all236 coordinates. Independent
source/runtime/coordinate reconciliation confirms216 passing structural checks,
19 passing numerical comparisons and one numerical failure. All ten4K numerical
comparisons pass; the single failure is context31, oldest-value pattern, chip0.
All24 replays match their own padded eager outputs exactly. Both full-history and
future-proposal dependency controls pass. This is still a **failed accuracy gate**.

| Historical rows | Numerical comparisons | Largest full/live absolute error |
| ---: | --- | --- |
| 31 | 9/10 pass | 0.139591 /0.139591 |
| 4,096 | 10/10 pass | 0.015625 /0.011263 |

These raw maxima include passing values; the gate combines relative and absolute
tolerances, so an absolute error above0.01 does not alone imply failure.
Original packer, both SDPA compute files and both native binaries are verified
restored after the final run; the temporary ownership locks are released.

The probe continues after finite numerical mismatches to collect the remaining
structural evidence. A mismatch still forces final exit1 and
`passed=false`; successful mask/replay checks cannot override it. Tolerances and
the stress inputs are unchanged. Nonfinite values still abort immediately.
No learned attention, full backbone, coding-quality or hardware-speed pass follows.

CPU diagnostic on the saved operands also compares live rows with the verified
upstream BF16 eager policy, rather than FP32 SDPA: native exponential has523
failing values (maximum0.0390625); the precise stress case has41 (maximum0.125).
Changing the reference to upstream BF16 does not make these fixtures exact or
pass the same0.01/0.01 tolerance. It is diagnostic only; no reference is replaced.
After implementing this matrix,1,112 host tests and58 simulator-harness tests pass.

### Next attention test: 64-key chunks

The remaining short-context failure spans two 32-key native chunks. The next
explicit candidate uses `--precise-native --key-chunk-size 64` to test whether
removing that intermediate merge helps. This is a hypothesis, not a diagnosed
root cause or a numerical pass. The default remains 32 keys.

All original queries, keys, values and stress inputs are preserved bitwise.
At 4K, only masked padding grows from 4,128 to 4,160 keys; the extra rows are
poisoned in the padding-control fixture. Full history, seven proposal keys,
HiFi4/FP32 accumulation and the 0.01/0.01 CPU threshold remain unchanged.
The qualifier requires the matching explicit chunk policy and all 236 checks;
earlier 32-key evidence cannot qualify it. Host tests verify operand preservation,
mask semantics, actual dispatch arguments and rejection of numerical failures.

Preparation passes 1,123 host tests and 58 simulator-harness tests. Simulator
run `20260909T154029Z-441` is active after the weight-reader MLP passed and its
runtime was restored. The saved historical failing operands and FP32 reference
are independently verified bitwise-identical in the 64-key candidate. Partial
checks are not qualification; there is no new hardware, quality or TG result.

### Learned CPU backbone matches upstream

All five learned layers now execute in the CPU reference. An independent control
uses the checkpoint's reviewed, hash-pinned forward methods plus the pinned
Transformers eager attention, RMSNorm, MLP and YaRN methods.

| Coverage | Verified result |
| --- | --- |
| Learned feature projection and normalization | Full5120 outputs; taps5/19/33/47/61 in checkpoint order |
| Five complete layers | Q/K/V projections, head norms, YaRN, full noncausal GQA, output projection, SwiGLU and residuals |
| Four CPU cases | Two feature/noise patterns, absolute starts0/8190 and a repeated first case |
| Stage comparison | **48/48 bitwise-exact BF16 matches** against the reviewed upstream control |
| Inputs and weights | Borrowed inputs unchanged; checkpoint hashes checked before/after, every loaded tensor rehashed |

Each case uses32 synthetic feature rows and seven synthetic noise embeddings.
These are not tokenized coding prompts or PP/CTX/TG measurements. The reference
does not include the target embedding/LM head, Markov selection, confidence policy,
target verifier or serving path. It does not establish GPU/TT arithmetic equality.

The control executes only reviewed function bodies, with training, dynamic
decorators and alternative attention backends disabled; it does not import
checkpoint modules or install Transformers. Earlier intake/loader checks executed
no checkpoint code. This separate upstream comparison explicitly does.

| Evidence | SHA256 |
| --- | --- |
| `dspark-backbone-cpu-reference.json` | `2dcaf7f4e5d1052da206dcead5ccc54b15b0a5d9d81dc3f8605e2a5fd349b912` |
| `dspark-backbone-upstream-reference.json` | `eb79532aeee5121b184f24121b9d2012aa9905f0a202100c98093b3657f9ed16` |
| Saved CPU stage tensors,4,769,305 bytes | `ede89f463be2df136298444e092c9bffaa272bb1968e2cc8336cf0ae0aef02fc` |

The reports are in `scripts/ci/`; tensor data stays under `/opt/ttsim/results/`.
Independent report/tensor reconciliation also passes. All1,096 host tests and58
harness tests pass, including75 DSpark tests. Next port the learned operations
and full-context mask to TT without substituting DFlash2's2K window or eight-row
proposal convention. CPU correctness is not permission to skip the target audits.

### Serving row convention: anchor plus six masks

The pinned SGLang DSpark adapter defaults `sample_from_anchor=True`. For the
published seven-proposal block, the boundary is:

| Stage | Rows |
| --- | --- |
| Draft query IDs | Anchor/bonus token followed by six mask IDs248070 |
| Sampled draft outputs | All seven rows, including row zero |
| Target verification input | Anchor followed by all seven proposals: eight rows |

`dspark_inputs.py` encodes this convention with ID, shape and absolute-position
checks. It rejects a full block without room inside262,144 positions; a shorter
end-of-context fallback is not implemented. Do not copy DFlash2's eight-query-row,
drop-row-zero convention. This helper is not integrated into target execution.

Primary sources at SGLang commit`708f51e44bc64f546a60fa9631f0e7d99493d0a0`:
[configuration](https://github.com/sgl-project/sglang/blob/708f51e44bc64f546a60fa9631f0e7d99493d0a0/python/sglang/srt/speculative/dspark_components/dspark_config.py)
and [draft input/output construction](https://github.com/sgl-project/sglang/blob/708f51e44bc64f546a60fa9631f0e7d99493d0a0/python/sglang/srt/speculative/dspark_components/dspark_draft.py).
The inert source copies hash to`73c3dc2986fa57eebcf61fdf47a817920a918f35e1414a724a0a0a3e44a7e577`
and`ee063303d98d0f592a61b7168dbbf4fd1713d4cde8b6b0166149b1258d909c36`
respectively. Neither module is executed.

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

1. Retain the qualified native-arithmetic Markov and composed-rotary policies,
   preserve their original failures, and resolve the full-attention numerical gate.
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
