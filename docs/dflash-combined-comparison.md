# DFlash2 versus the promoted combined runtime

Status: T32 combined-runtime preflight passed in **35311905218**. Full matched
T16/T32 hardware comparison **35312183859** is underway; no T32 throughput is
validated yet. DFlash2 is not promoted; serving defaults are unchanged.

## Current combined experiment

Both arms use cached native DFlash proposals and the qualified target recipe:
register MLP, block-stream weights, shared-Q/K GDN with scatter normalization,
direct convolution windows and the wider MLP-down grid.

| Setting | T16 control | T32 candidate |
|---|---|---|
| Context / streams | 4,096 / 1 | 4,096 / 1 |
| Maximum verification rows | 16 | 32 |
| Output budget | 256 tokens | 256 tokens |
| Model and weight pool | Shared, loaded once | Same pool |
| Measurement | Audit, then two timed requests | Audit, then two timed requests |

Timed order is T16, T32, T32, T16. Exact target tokens, state and features must
pass; proposals and acceptance may differ by width. PP and committed TG include
their documented request costs, not isolated kernel rates. Setup is reported
separately. Passing this pilot alone does not establish held-out coding quality.

Preflight checked all five source-bound component admissions against the
restored runtime, without weights or device execution. It caught and fixed a
missing ZIP tool, an overwritten pinned geometry file and staging-only imports.
Report and kernel pins were not relaxed.

## Latest matched DFlash2 result

Follow-up **35298946581** retains native drafting and changes only target MLP
weight transport: **113.89 -> 116.97 TG** at CTX4096, one stream (+2.71%).
See [combined transport results](mlp-block-stream.md) for setup/memory costs
and exact proposal/output/state checks. It is not promoted.

The next width experiment must preserve cached native drafting and the target
recipe rather than repeat the old uncached T32 run. At the latest 93.92-ms
cycle, even 16 committed tokens would yield only about 170 TG; 200 requires
lower latency and/or more useful committed tokens per cycle. T32 is not an
assured improvement: historical eager DFlash T32 accepted only 8.82 committed
tokens/block. The full-32-query native-attention helper now has simulator replay
evidence and an explicitly scoped hardware route; learned acceptance and combined
target correctness remain pending the full request test.

T32 native proposal attention run **35299919048** passed its mask-isolation and
changed-input replay gate in **2m13s** on the exact combined runtime binary.
Downloaded reports were independently revalidated against generated probe/gate
sources and native fingerprints. Each context covers six eager outputs, eight
replays, 56 input-preservation checks, two stale-input controls and four
masked-key invariance checks across both simulated chips.

| Draft history | Replay/mask gate | Legacy `.01/.01` numerical diagnostic | Maximum absolute error |
|---|---|---|---:|
| 31 | Pass | All six comparisons fail | 0.0268841 |
| 2048 | Pass | All six comparisons pass | 0.00525287 |

This is **approximate proposal attention**, not an exact target replacement.
The short-history diagnostic is retained, not relaxed or relabelled. There is
no learned acceptance, target-state or throughput result for this new T32 path.

Cached T32 source preparation now has a simulator-only adapter, leaving the
admitted T8/T16 source files unchanged. Seven local host tests pass, including
cache publication and rollback across every accepted prefix length from 1 to
32. These are host regression checks, not a device cache-replay qualification.
Run **35301426428, attempt 2** passed the combined synthetic cache/native-attention
replay check in **3m39s**. Both simulated chips passed initial replay, discard
and commit for prefixes 1, 16 and 32; unpublished cache updates correctly
blocked replay. All 17 checks passed and the simulator closed cleanly.
The first attempt stopped at the storage gate (23.8% full I/O stall), before
any simulation. The successful rerun passed that same unchanged gate.

The probe uses synthetic K/V projection, not learned drafting. It validates
cache publication together with actual captured T32 native attention, but
does not establish learned acceptance or target-state correctness. The adapter
rejects bare hardware execution; the combined experiment supplies source-bound,
request-scoped admission. Matching T32 target validation remains pending.
The MLP probe retains the register arithmetic and block-stream reader,
changing only token width and simulator routing. Its T32 simulator evidence
is recorded below; hardware execution remains unqualified.
Serving selection remains unchanged; only the explicit experiment enables T32.

Cache replay report SHA256:
`2ad1b72e02ab978b82cfc252d8226a98a2c1d9a3296a15c13b85da418164e7a6`.

### T32 combined-recipe parity checklist

Source audit, 18 September: changing only the draft width would **not** retain
the measured T16 recipe. Keep the T16 control intact while closing these gaps.

| Component | Current T32 coverage | Required before combined acceptance |
|---|---|---|
| Native draft attention and cache replay | Synthetic simulator checks pass | Learned proposal/output and acceptance checks |
| Register MLP and block-stream reader | Run 35302273096 attempt 3 passed T32 eager/replay; runtime preflight passed | Combined target checks and all 64 layer hits |
| Shared-Q/K GDN and scatter normalization | Run 35304765890 passed exact replay; T32 scope and preflight ready | Accepted-prefix continuation and all 48 layer hits |
| Direct convolution windows | Run 35305449655 passed native output/checkpoint replay; T32 scope ready | Combined per-layer execution and exact state checks |
| Wider MLP-down grid | Run 35305677689 passed exact replay; T32 scope ready | Combined execution in all 64 layers |
| Full request integration | Dedicated T32 wrapper, cached/native routing and matched driver pass CPU checks | Hardware exact tokens/state/features, then matched timed requests |

Source pointers: `gdn_shared_qk_scope.py`, `gdn_direct_window_scope.py`,
`mlp_down_grid_scope.py`, `dflash_combined_request.py`, `full_dflash_request.py`
(all under `scripts/ci`). Existing T16 reports must not be relabelled T32.

Execution order: bind the passed MLP evidence, close the GDN/window/down
width gaps, then integrate and run one matched full-request comparison. Report
actual optimized-layer hits and fallbacks, not merely enabled flags.
At 200 TG, an average of 11 committed tokens permits **55 ms per whole cycle**;
20 permits **100 ms**. Wider drafts only help if measured acceptance and total
cycle latency satisfy that budget; a simulator pass cannot establish either.

A separate simulator-only shared-Q/K T32 builder adapter is now prepared.
It changes host tensor shapes and runtime token counts, not recurrence,
normalization or scatter kernel arithmetic. In particular, 16-row tile-face
offsets remain unchanged: they are physical tile indexing, not draft width.
Run **35304765890** passed this adapter on both simulated chips: 24 exact
output/state/bridge checks, 48 immutable-input checks and six changed-input
stale-state controls. All 32 recurrent prefixes were compared. Source hashes
matched the staging manifest and remained unchanged; cleanup succeeded.
The probe took **387.42 seconds**, inside its 420-second cap. Earlier attempts
failed at staging and a staging-only import, not numerical comparisons.
This is not accepted-prefix continuation, combined target or hardware admission.
Report SHA256:
`c88ceb06a8983b156c90161354f7f9f7da918b86e4329e03ad54ebf909bb66b3`.

The next direct-window adapter fixes a separate width dependency: T16's
linear row offsets do not address the bottom tile faces correctly at T32.
The generated reader uses explicit 16-row face offsets for all 32 rows,
while retaining native convolution arithmetic and the untouched T16 sources.
CPU run **35305118623** passed. Simulator run **35305449655** subsequently
passed in **43.02 seconds of probe execution**: 56 exact native output/checkpoint
comparisons and 88 immutable-input checks across two chips, with L1 projected
inputs. Changed-input trace replay overwrote NaN-poisoned outputs. Source
hashes matched the staging manifest and were unchanged; container cleanup
succeeded. This does not yet qualify the full T32 request or hardware timing.
Report SHA256:
`bdc8b31ec1ed80a401ecec1b581baf3006e3d72e3699c3e73417ef494a6962e1`.

T32 MLP-down run **35305677689** passed in **147.11 seconds of probe execution**:
12 exact eager/replay comparisons, 14 activation/weight integrity checks and
three poisoned-output replay controls. Both chips used unchanged BF8 weights,
L1 outputs and native arithmetic; only the worker grid differed (11x3 versus
11x8). Staged probe/helper hashes matched the report, source hashes remained
unchanged and cleanup succeeded. Combined runtime layer coverage and TG remain
unmeasured for T32. Report SHA256:
`5768dc71a94c74424beae53919316700a1bc0b7b12e5bcfcc8d24487254cadac`.

T32 block-stream MLP run **35302273096, attempt 3** passed in **5m17s**:
two exact eager comparisons, 12 exact changing-input replay comparisons and
four stale-input controls across both simulated chips. Stream bytes remained
unchanged and container cleanup succeeded. Attempts 1 and 2 stopped before
simulation at the unchanged storage gate (31.7% and 1.33% full I/O stall).
This is MLP correctness evidence, not combined T32 acceptance or throughput.
Report SHA256:
`0fbdcbdd9c688a5a2e497b682ab3cd4fadcb40279fda952c8e24323a92856766`.
Executed projection SHA256:
`4e1127829c6aa567f5af3ea403d067923d3f545e983a0107a2a61f7ea81485f3`.

Report SHA256 (31):
`aeb9e55a2426c31ea695bd191124abb233ca652dc6252bccc8c4e1cf858b8408`.
Report SHA256 (2048):
`3e4e153046d2ebcebb3f9d822b081a04b1e8453a2890cee8499c6644d1bc6857`.

| Attention | CTX | Streams | PP tok/s | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| Composed control | 4096 | 1 | 3271.03 | 93.53 |
| Native candidate | 4096 | 1 | 3332.93 | **112.09** |

Two audited arms followed by composed/native/native/composed timed requests.
Native attention improves complete-request TG **19.84%**. Both arms commit
242 decode tokens over two timed requests, accept 222/330 proposed tokens,
and preserve exact target output/state. This remains a coding-fixture pilot,
not held-out coding quality or serving certification.

Evidence: run 35293989426, `cumulative-validation.json` and independently
revalidated `dspark-captured-publication-request-hardware.json`, SHA256
`f6bb5cca6cfba301a8a726de3f6bb1f425369028d01aace89b520985338b7b03`.

| Mean block cost | Composed ms | Native ms |
| --- | ---: | ---: |
| Draft | 40.48 | 21.11 |
| Verify/readback | 62.64 | 62.99 |
| Select/commit | 11.99 | 11.61 |
| Complete cycle | 117.48 | 98.01 |

**Next priority: target verification, alongside acceptance.** At the measured
11 committed tokens/block, 200 TG requires a 55-ms whole cycle. Verification
alone is approximately 63 ms: even eliminating all drafting/publication costs
would only reach a conditional 174.63 TG. Another selector-only tweak cannot
meet the goal. The native candidate still needs roughly 44% less total decode
time, or substantially better acceptance combined with lower verification cost.

## Earlier matched DSpark control

Run 35287866254 compared the user-promoted T16/DSpark stack with composed
DFlash2. It was a different run, not a paired control for the new candidate.

| Drafter | CTX | Streams | PP tok/s | Committed TG tok/s |
| --- | ---: | ---: | ---: | ---: |
| DSpark | 4096 | 1 | 3333.91 | 131.05 |
| DFlash2 | 4096 | 1 | 3360.78 | 97.58 |

Source: run 35287866254 artifact `cumulative-validation.json`, two timed
requests per drafter after correctness audits. This is a coding-fixture pilot,
not held-out quality certification or serving acceptance. DFlash2 is 25.54%
slower here; target optimizations are shared but drafter optimizations are not.

## DFlash2 optimization sequence

The user released hardware after the CPU/simulator preparation phase.

1. Qualify a separate T16 native proposal-attention candidate. Preserve the
   existing T8 gates and pinned shared sources. Check all 16 live queries,
   finite padding, masked-key isolation, changed-input replay and stable bindings.
2. Integrate only after simulator evidence, then compare against composed
   attention in the same combined runtime when hardware becomes available.
   Native proposal arithmetic is not assumed identical: exact target output
   and state, acceptance, and complete-request TG remain mandatory gates.
3. Reduce DFlash2 history-publication and selection overhead, measuring each
   change in combination rather than substituting kernel timing for TG.

`dflash_t16_native_attention.py` remains an opt-in experimental adapter.
CPU tests cover mask rejection, second-half live-row
diagnostics, dispatch configuration and reference masked-key isolation. They
do not certify device numerics, trace replay, performance or coding quality.
Its simulator probes use actual T16 operands, not relabelled T8 learned fixtures.

Simulator run 35289299758 failed during checkout, before any probe execution:
the shared workspace root was not a Git repository during checkout cleanup.
The simulator workflow now uses a run-specific child checkout, as the other
staged experiments do, rather than cleaning the shared root. This failure
provides no evidence about native attention.

### T16 simulator result

Run 35289482417 passed in 2m51s on CPU TTsim, using no model weights or cards.
Both 31-row and 2048-row histories passed finite-output, masked-key isolation,
borrowed-input preservation, stable-binding and changed-input replay checks on
both simulated chips. The downloaded reports also passed independent local
validation against current source hashes.

| Draft history | Maximum absolute difference from FP32 reference | All values within legacy tolerance? |
| --- | ---: | --- |
| 31 | 0.030413 | No |
| 2048 | 0.004695 | Yes |

This qualifies mask/replay behavior only. Native proposal arithmetic is **not
numerically equivalent** to the reference at short history; tolerances were not
relaxed. Do not replace target attention or declare coding-quality acceptance.
The subsequent exact-runtime simulator qualification and combined hardware
audits are recorded below. Native arithmetic remains proposal-only; target
attention and the promoted DSpark runtime are unchanged.

The combined request API now has an opt-in `native_attention_evidence` argument.
It requires both pinned simulator reports and matching source/native hashes,
then enables the T16-specific mask validator and native proposal operation.
The admission is rechecked on exit and recorded with each request; summary
validation rejects missing admission. All existing target optimization scopes
remain active. The default comparison driver does not select this candidate;
the explicit native-comparison driver produced the latest measured result.

The separate `dflash_native_comparison_experiment.py` driver prepares two fresh
target audits followed by composed/native/native/composed complete requests in
one loaded session. It stops before timings if either audit fails. Its report
validator checks target components, exact committed output, source/cache policy,
per-arm proposal repeatability, acceptance accounting and stage costs. Different
draft proposals between arms are allowed; different committed output is not.
This driver is CPU-tested. The existing combined workflow selects it only for
an explicit `experiment/cumulative-t16-full-v*-dflash-native` tag. That route
downloads the pinned simulator artifact, validates it before and after staging,
and selects the independent native-comparison report validator. The successful
hardware tag is `experiment/cumulative-t16-full-v8-dflash-native`.

### CPU selector preparation candidate

`dflash_compact_selector.py` gathers the same active codebook entries, validates
all query rows together, and converts them to FP64 once rather than once per
eight-query chunk. The sequential greedy arithmetic and first-candidate tie
rule are unchanged. The existing selector remains the runtime default.

CPU tests match every score and selected token bitwise for 1/7/8/15/16/31
queries, batches one/two and two seeds. They also check ties, input immutability,
late-row NaNs, duplicate/out-of-range IDs and arithmetic overflow.

Local Windows synthetic T16 measurement, PyTorch 2.14.0+cpu, two threads,
five warmups and 40 alternating samples per arm:

| Selector | Median ms | 90th-percentile sample ms |
| --- | ---: | ---: |
| Existing | 12.549 | 22.335 |
| Single preparation | 9.868 | 33.849 |

The median improved but the tail worsened on the shared desktop. This is not
an AMD-runner result or combined TG gain, and is insufficient for promotion.
Keep it separate from the pending native-attention comparison so gains or
regressions can be attributed before combining candidates.

### Budget for 200 committed TG

Recomputed from the two timed DFlash2 requests in run 35287866254: 242 committed
tokens over 22 blocks, or 11 tokens per block. At that acceptance, 200 TG needs
an average complete cycle of **55 ms**. Measured decode time needs a **51.21%**
reduction. Verification/readback alone averages about **62.65 ms** per block.

With that acceptance and serial verification cost unchanged, even eliminating
every other cost gives a conditional ceiling of **175.57 TG**. This is a bound,
not a performance prediction. Native draft attention alone therefore cannot
reach 200 unless acceptance improves enough; verifier latency must also be
addressed otherwise. The combined comparison now reports this budget per arm
alongside actual TG and acceptance, without claiming unmeasured gains.

## Comparison contract

- Start at CTX4096, one stream, the same coding fixture and output budget.
- Keep target weights, precision, four-link policy and T16 verification matched.
- Keep direct windows, wider MLP down, scatter normalization and register
  epilogue active in both arms, with per-request execution evidence.
- DSpark retains compact Markov selection. DFlash2 uses its own selector;
  absence of DSpark selection is intentional, not a missing target optimization.
- Preserve each drafter's checkpoint and native history policy. DFlash2's 2K
  draft history is not truncation of the target's 4K KV context. Report this
  difference rather than calling the comparison history-matched.
- Two fresh correctness audits followed by ABBA complete requests. Compare
  target outputs and state across arms, not draft proposals or acceptance paths.
- Report PP/CTX/committed TG, setup, draft, verification, publication and
  accepted tokens per block. Keep timing stalls, source hashes and raw reports.

## Implementation sequence

1. Separate drafter-specific selection from shared target scopes. Implemented
   with host tests for identity, route engagement and exception cleanup.
2. Prepare T16 masks, absolute rotary inputs and global vocabulary candidates.
   Host utilities now support 16 rows; old eight/32-row behavior is retained.
3. Device/cache execution now supports explicit T16 composed attention, with
   cached history and captured proposals. Capture is deferred until verifier
   persistent allocations exist. Native proposal attention and live-query QK
   remain T8-only; their gates are not widened. The DFlash2 request owns the
   same four target scopes directly, without claiming DSpark history/selector
   hooks executed. Full target-output, state, feature, cached-versus-uncached
   proposal and changed-input replay checks remain mandatory.
4. Qualify changed-input proposal replay and full-request target state on the
   combined hardware path before timed ABBA. Use tiny simulator cases only if
   kernel changes require them; do not load model weights in the simulator.
5. Compare DFlash2 T8 separately after matched T16; retain width/history labels.

The comparison driver runs both drafters in the same loaded target session.
The `experiment/cumulative-t16-full-v*-dflash` tag family selects the adapter
in `qwen-cumulative-t16.yml`; other cumulative tags retain the existing driver.
It uses cached pinned DFlash2 fixtures only (revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`), fails if missing, and verifies tensors
with the existing fixture loaders. It does not start a multi-hour download.
The independent report validator checks both target component identities,
within-drafter proposal repeatability and exact shared target outputs, while
allowing different proposals between drafters.

Host tests are preparation evidence, not device numerical or performance
acceptance. Serving defaults and the immutable promoted DSpark recipe remain
unchanged. Source-qualified artifacts must be regenerated/reconciled where
the new host-source hashes differ; old admissions are not silently reused.

## First hardware launch: preflight failure

### Native T16 comparison: runtime admission mismatch

Hardware run **35291967062** passed staging and the disk gate, then stopped
after weight loading at native-attention source admission. No request audit
or timed comparison completed; this is not a kernel failure or a TG result.
The simulator used the original SDPA program factory, while the combined
runtime contains the promoted DSpark statistics-factory patch. Its source
hash differs (`a263559f...` versus `fd8c0676...`).

Native admission now reports the exact differing file and runs after runtime
construction but before model loading. The strict source gate remains intact.
The candidate still needs reconciliation with the actual combined factory;
the successful original-factory simulator report does not qualify that change.

The replacement simulator lane restores the exact cached combined binary
(`4b7299c1...`) read-only and reconstructs its factory byte-for-byte. Both
library paths are hashed before and after replay. It uses synthetic operands,
no target weights, no device access and no native rebuild. Fresh passing
reports must be pinned before the next combined hardware comparison.

**Exact-runtime replay passed:** run **35293698929**, 1m53s. Both contexts
(31 and 2048 draft-history rows) passed mask isolation, changed-input replay,
stable bindings and input preservation on both simulated chips. The factory
and both runtime binaries match the combined hardware recipe. CPU integration
run **35293655254** also passed. This admits the hardware comparison, not TG,
held-out coding quality or serving promotion.

Run 35285659419 attempt 3 passed the disk gate but stopped before device
execution: the shared simulator-source gate detected modified `draft_attention.py`.
The integration had widened a host mask argument check in a file shared with
DSpark. This was a staging regression, not a kernel crash or numerical failure.
The follow-on Apport `FileNotFoundError` was only error-reporting noise.

The fix restores that file exactly to its simulator-pinned bytes and puts the
T16 host mask specialization in `dflash_attention_mask.py`, used only by DFlash2.
The comparison stage now runs the real shared-source preflight before and after
its overlay. Independent CPU visibility-formula tests cover T16 masks; the
original source-pin regression and existing widths remain checked. No source
hash, numerical threshold or simulator admission was bypassed.

Run 35287122216 subsequently reached both audit generation paths, but failed
at the DFlash2 summary before timed requests. The new driver omitted
`ended_with_eos`, `sampler_num_links` and `fabric_sources`, previously populated
by `full-prefix.py`. The retained DSpark audit also lacked these fields. This
is an integration-reporting failure, not an accepted benchmark or kernel-speed
result. DFlash2's request was not retained because validation preceded append.

The driver now derives completion from emitted/EOS IDs, reads the active sampler
link count, and checks fabric-source hashes against the runtime audit before
attaching them. It records completed requests before summary validation so a
future rejection retains its evidence. Source identities are rechecked after
the comparison; validation thresholds and exact-state gates are unchanged.
