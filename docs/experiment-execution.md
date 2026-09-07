# Two-P150A execution ledger

Updated 2026-09-06. Target: 200 committed tokens/s for one coding stream, not
aggregate throughput. No adoption or serving restart is authorized by a test pass.

## Current frontier (2026-09-07)

- Retained full-model attention replay34070163839 passed24 static batch cases,
  16 rollback checks,4 negative-control pairs and36 changed-metadata replay cases.
  The dynamic reader engaged in24 T16/T32 cases; twelve T2 cases retain native
  attention. Every replay also checks two corrected native continuations.
- Matched actual-request benchmark `full-attention-engine` is now prepared.
  Both arms use norm batching and the same bounded family proposal routing;
  only the candidate enables dynamic parallel attention. Each arm owns a fresh
  prefill and trace lifetime. ABBA validation requires identical emitted tokens,
  proposal/acceptance accounting and routing, and reports setup cost separately.
  This remains the existing synthetic lookup pilot, not coding-quality evidence.
- Full-model parallel attention34067681095 passed24 batch fixtures,16 rollback
  checks,4 negative-control pairs and12 paired timing fixtures. Matched T32 costs
  99.331691->95.884002ms at4K and105.968306->102.058394ms at16K; T16 costs
  76.966126->74.991160ms and80.284747->78.267385ms. Both arms use identical DMA
  layouts and GDN. The4K control shifted across runs; do not add cross-run gains.
- Hardware mask-replay34068177963 passed48 exact checks across24 captured
  position refreshes. Reusable reader TTsim20260907T001643Z-681 passed T16/4K,
  both chips, forward/rollback positions, mask refresh engagement and borrowed
  query lifetime. Its hardware attention-composition gate is next.
- Actual matched request decode remains22.46/18.67 committed tokens/s at4K/16K;
  the200 target is not achieved. New attention changes are not in that request path.
- Grouped attention full-model34062230310 passed: T32 verification103.31/109.92ms,
  versus110.02/127.56ms with the same norm batching and serial B1 attention.
- Direct attention DMA passed micro34063065156, real-layer34064053339 and
  full-model34064443450. Matched T32 costs106.98->102.99ms at4K and
  109.94->105.94ms at16K; compare within runs because the4K control shifted.
- Allocation-only native tree scratch34064917232 timed out during compilation,
  at976/1091 build steps after600s, before hardware testing. Retry allows1800s
  for the disposable-container build; this is not evidence of a device hang.
  Retry34066556771 completed compilation but failed importing ttnn: rebuilt
  `_ttnncpp.so` lacked the existing `ttnn::transformer::attn_decode_prep` symbol.
  No device was opened. Audit native graft registrations and ABI before retrying;
  do not replace or modify the serving runtime.
- Parallel group scheduling passed hardware34065899606:36 exact fixtures,
  4320 timed replays and72 changed-query checks. T32 attention microbenchmarks
  improve0.853514->0.640465ms at4K and1.268948->1.026474ms at16K, including DMA
  layouts. T16 improves0.463693->0.355681ms and0.690229->0.568410ms.
  Three groups preserve16 workers per KV head and allow96 SDPA workers on the
  110-core grid. These are component timings, not committed request throughput.

Prepared parallel reader simulator `20260906T232103Z-427` passes T16/start16383/
seed2 on both chips, including boundary groups, DMA outputs and borrowed-query
lifetime. The `attention-parallel-groups` suite now runs its matched real-weight
layer gate after the microbenchmark; both layer arms use identical DMA layouts,
ordered K/V writes and projections. Only group scheduling differs. Native scratch
cold-build retry `34066556771` runs separately with no serving-image changes.

Real-weight parallel layer `34067099712` passed48 exact checks,24 negative-control
pairs and72 timing blocks. Both arms use DMA, ordered cache and the same batched
projections. Mean paired layer times: T16/T32 at4K0.827588/1.350228ms control
versus0.720095/1.135140ms candidate; at16K1.054529/1.763848ms versus
0.932434/1.519554ms. T1/2/4 native fallback and T8 boundary plan are unchanged.
The next `full-attention-parallel` gate retains the same six widths, native full
logits, GDN prefix/rollback and K/V checks, with DMA in both timing arms. Dynamic
request replay remains disabled. A read-only native registration/symbol audit
also accompanies this run to diagnose the separate scratch-library build issue.

Dynamic-mask prerequisite: simulator `20260906T234825Z-409` and final harness
`20260906T235115Z-512` pass48 exact same-address checks each. Six geometries cover
4K/16K capacities, one/three/four query rows, one/two/three parallel groups and
forward/backward position refresh. The dataflow kernel writes only the final256
columns of a zero-initialized BF16 mask; preceding columns stay zero. A host ticket
guard rejects crossing the captured256-token capacity family. No request engine
uses this yet. The hardware `attention-mask-replay` gate will capture the same
preallocated program and refresh its position input across24 trace replays; it
does not claim attention correctness, cache-write correctness or request speed.

The queued mask run34068081774 was cancelled before hardware execution after a
container-layout source-audit path bug was found. Corrected gate34068177963 is
queued behind full-model34067681095; source hashes now resolve beside the imported
helper, independent of the simulator harness mount path.

Dynamic mask plus parallel attention composition passes TTsim
`20260906T235604Z-310`: T16/start4096/seed1, both chips, changed start4103 then
rollback4096 with the same query/mask buffers. All four changed-position output
checks match native causal B1 exactly; prepared-reader lifetime also passes.
The probe is read-only K/V and slow dispatch, not a retained full-model verifier
or hardware trace certification. The16K composition counterpart is next.

The16K counterpart `20260907T000136Z-300` also passes: T16/start16384/seed2,
forward16391 then rollback16384, four exact chip/position checks and prepared
reader lifetime. The reusable `ReplayAttentionReader` is now a local prototype:
fixed T8/T16/T32 family metadata, one same-address position input, per-call mask
refresh, protected borrowed buffers and poisoning after a failed update/dispatch.
Six host tests cover these contracts; the complete host suite passes380 tests.
It is not connected to `ModelBatch` or the request engine. Its own TTsim lifetime
and output check `20260907T001643Z-681` is running before any hardware promotion.

That reusable-reader simulator run subsequently passed. The native scratch
diagnostic from34067681095 shows the on-disk transformer CMake registrations
exclude existing attn_prep/GDN graft sources, although their source directories
and Python bindings are present. A cold rebuild therefore loses runtime symbols.
Re-register the exact graft sources and verify the loaded-library ABI before
another scratch experiment; this issue is separate from the passing unmodified
runtime attention tests.

Next hardware gate `attention-replay`: twelve fixtures across T8/T16/T32,4K/16K
and two seeds. Each captures one prepared reader, refreshes both query and position
inputs at the same addresses, checks forward/end-of-family/rollback tickets against
independently computed native B1 outputs on both chips, and verifies full K/V
buffers remain unchanged. Expected coverage is96 chip/ticket checks and48 trace
replays. No cache writes or GDN histories are involved yet; real-layer and retained
full-model replay certification remain required before request-path integration.

Hardware reader composition34069798251 passed all12 fixtures,96 exact chip/ticket
checks and48 trace replays; complete K/V buffers stayed unchanged. The next
`full-attention-replay` gate integrates one shared reader across all16 attention
layers and stages its position input in the same transaction as tokens/RoPE/native
positions. It uses aligned4K/16K starts4096/16384 so both verification blocks stay
within their prepared256-token family; crossing a family is rejected before any
metadata copy. T1/2/4 retain native attention. It retains the existing full-logit,
GDN prefix, corrected-continuation and valid-K/V replay checks. Request-pilot and
sampling modes remain rejected until this retained-state gate passes.

Native scratch build repair: source registration hash
`a2fb5b6ea5769c57f79353f2a8c3859f85d161f3a94ced173862d10cd5adca10`
is pinned from34067681095. A separate patch adds only15 existing host implementation
files for attn_prep, gdn_conv_gates, gdn_norm_gate, gdn_decay and
decode_gated_delta_rule. No native kernel math or Python binding is edited. The
disposable build records every source hash, checks existing transformer imports
before opening devices and records dynamic-library resolution. It still applies
the simulator-certified allocation-only scratch patch and uses the same paired
T4/T8 group microbenchmark; it does not alter the serving image or host runtime.

Host-only request capture plan now specifies all mask-family buckets before any
trace is captured. At request start4078 with128 tokens remaining, it prepares
native T1/2/4 once, T8/T16 for capacity4096, and T8/T16/T32 for capacity4352.
Large tickets cannot cross their captured256-token family; native small-width
tickets may cross using the already certified native dynamic positions. Four
tests enumerate every possible ticket position across representative4K/16K
budgets, reject unprepared widths/ranges and refuse unsupported long-context mask
families. This plan is not yet wired into `VerifierEngine`; retained replay and
request-owner/commit gates must pass first. No allocations under active traces or
new request throughput claims are authorized by these host tests.

`VerifierEngine` now has an unexposed opt-in `attention_replay` path using that
plan: all native/family fixtures are allocated before capture, warmup uses each
family's actual capture position, proposal limits expose safe widths, and
publication uses the bucket key sealed during verification rather than rerouting
after the decision. Default native bucket keys and behavior are unchanged. Three
additional engine tests cover future-family forwarding, opt-in guards and exact
bucket ownership during publication. The complete host suite passes393 tests;
the speculative harness passes60. Request CLI/benchmark wiring remains disabled
pending full-model retained replay34070163839.

Parallel simulator evidence: `20260906T224631Z-303` T16/start4096/seed2 uses host
packing; `20260906T225115Z-307` T16/start4096/seed0 includes DMA packing and
output layout; `20260906T225312Z-461` T16/start16383/seed1 covers the boundary,
device layouts and reusable adapter. All match native B1 exactly on both chips.
The bundles are groups of queries from one speculative verification stream,
not separate user requests or aggregate serving throughput. The hardware
`attention-parallel-groups` screen compares serial and parallel groups with
identical DMA layouts, native SDPA math and read-only KV.

## Verified control

Grouped attention hardware microbenchmark `34061073573` (`6f75984`) passed:
36 fixtures, 4320 blocking timed replays, 72 changed-query checks, exact native
outputs on both chips and unchanged KV. Including device packing/unpacking,
T32 attention fell from about 1.48 to 1.11 ms at 4K and 2.59 to 1.52 ms at 16K.
Small widths regress; the real-weight layer gate keeps T1/2/4 on native SDPA.
This is synthetic read-only attention, not full-model throughput. Real-weight
integration now has a separate `attention-group-layer` gate: identical ordered
cache writes in both arms, whole physical cache checks, native output oracle,
static long-context positions, and valid wrong-page negative controls.
Dynamic-position request trace reuse remains uncertified. The latest actual
request rates remain 22.46/18.67 committed decode tokens/s, far below 200.
Prepared-reader simulator pass `20260906T214204Z-304`: T8/start4095/seed2,
groups 1/4/3, exact on both chips after scratch cleanup, borrowed query intact.
Real-weight attention run `34061952468` (`a84002e`) passed all48 eager/trace
checks,24 pairs of negative controls and72 paired timing blocks. Both arms use
ordered cache writes and batched projections. Mean paired T32 layer latency:
4K1.974771->1.599829ms;16K3.073475->2.013567ms. T16:1.110906->0.967897ms
and1.660966->1.194989ms. Whole physical KV and outputs match native B1 exactly.
The next `full-attention-groups` gate keeps row-parallel GDN norm in both arms,
changes only attention grouping, and rejects request/replay modes until static
full-model correctness and position-dependent mask ownership are certified.
Full-model static gate `34062230310` (`add2738`) is running. Separately, the
direct tile-permutation DMA prototype passed TTsim `20260906T215257Z-312`:
T8/start4095/seed0, forward packing versus Torch and grouped SDPA/unpacking
versus native B1 are exact on both chips. It replaces the seven-operation
stock layout chain with one dataflow launch per direction, without changing
SDPA math. No DMA hardware speed result yet.
Long-context DMA TTsim `20260906T215411Z-605` also passed: T32/start16383/seed2,
all token outputs exact on both chips with groups1/4/4/4/4/4/4/4/3. Native
query packing and inverse layouts preserve bits; no simulator speed claim.
The two-row group also passed `20260906T220602Z-302` (T2/start4096/seed1).

### SDPA tree scratch allocation audit (not yet hardware certified)

Pinned factory `05708e6d...` allocates CB19 for `num_cores_per_head - 1`
packets. The paired writer `734c90c0...` sends and reads at tree-round offsets,
not worker offsets; the factory already computes and bounds
`num_tree_reduction_rounds`. The default `SDPAProgramConfig` caps workers at16
per head (`sdpa_config.hpp:20`), even when the supplied grid is11x10. These TP2
fixtures therefore use32 SDPA workers, not110:15 packet slots are allocated
per worker but only4 round slots are addressed. With an explicit55-worker cap,
54 slots versus6 would apply, but that is not the tested configuration.
This over-allocation contributes to the large folded
query L1 failure, not evidence that the extra math intrinsically cannot fit.
The prototype changes only this allocation under
`QWEN_SDPA_TREE_SCRATCH_ROUNDS=1`; reduction order, math and writer are unchanged.
Set the flag before process startup and never toggle it under a program cache.
The default allocation remains unchanged. Native patch is tracked in
`optimisation/sim/sdpa-tree-scratch.patch`, with baseline source hashes in
`scripts/ci/sdpa_tree_scratch.py`. The local incremental build succeeded;
original shared libraries are retained in
`/opt/ttsim/runtime-baselines/sdpa-tree-scratch/`. Candidate library SHA256:
`d2652fc01a6836b4d567a788a9c11d8f6cb238bb480bbf68d0e32ee4037c3e24`.
T32 group test `20260906T221642Z-406` still fails L1 allocation at3419328bytes
(limit1572864); shrinking this one buffer does not make every width fit.
T8 group test `20260906T221933Z-382` passes exactly on both chips atstart4096,
seed1. These are host-packed primitive checks, not device-layout or speed
certification for larger groups. No hardware runtime or serving image changed.

Full-model static grouped run `34062230310` (`add2738`) passed24 mode/width
fixtures,16 rollback checks,4 negative-control pairs and12 matched timing
comparisons. Median paired sample means for T32:4K110.015156->103.311151ms;
16K127.564602->109.918755ms. Both arms retain norm batching and ordered cache.
This remains full-logit verification, not committed tokens/s or dynamic replay.
Direct layout DMA micro run `34063065156` (`4e13492`) passed36 fixtures,
4320 timed replays and72 changed-query checks with exact native outputs and
unchanged KV on both chips. Matched stock-grouped versus DMA T32 component:
4K1.108354->0.854201ms;16K1.521339->1.267841ms. Real-layer DMA gate is next;
small widths remain on native attention regardless of these component timings.
Prepared DMA reader lifetime check `20260906T222439Z-296` passed in TTsim
(T8/start4095/seed1, native scratch allocation mode). The borrowed query remains
intact after temporary-buffer cleanup. The next real-layer run also collects a
read-only native source/compiler audit before considering a hardware runtime build.
DMA real-layer run `34064053339` (`cf3b11a`) passed48 mode checks,24 negative
control pairs and72 matched timing blocks. T32 stock grouped->DMA layer means:
4K1.599494->1.349729ms;16K2.012371->1.763004ms. The read-only source audit
matches all three native SDPA pins and confirms the16-worker-per-head cap.
Both build directories contain Ninja files with `clang++-20` recorded as compiler;
compiler availability under that exact name still needs checking before a build.
Long-context scratch-round TTsim `20260906T222729Z-298` also passed exactly:
T16/start16383/seed0, groups1/8/7, host layout only.
Larger-group stock device packing/unpacking now passes TTsim too:
`20260906T223545Z-304` (T8/start4096/seed2) and `20260906T223905Z-308`
(T16/start16383/seed1, groups1/8/7). The native-tree hardware experiment builds
the audited allocation-only patch inside a disposable, network-isolated test
container, then compares four-row versus eight-row groups with the same compact
allocation in both arms. It does not overwrite the serving image or host runtime.
Full-model DMA gate `34064443450` (`21aa767`) is running separately on the
unchanged hardware runtime.

- Image: `sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465`.
- TT-Metal: `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` with recorded graft changes.
- Plugin: `bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c`; vLLM `0.25.1+empty`.
- Weights/tokenizer: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, cache discovered
  and mounted directly read-only; full-model startup and K/M engagement verified.
- PCIe x16/x4 host attachment; inter-chip collectives use QSFP-DD/P300 fabric.
- Operator-confirmed exclusive card allocation; no host process-scan claim.
- Runner repository access and tested-ref exceptions remain enabled at operator request.

## Execution queue

### Grouped attention hardware screen prepared

`attention-groups` compares independent native B1 attention against serial
query slicing/SDPA/concatenation and the simulator-validated grouped path.
The matrix is T1/2/4/8/16/32 at4095/16383 tokens with seeds0/1/2. Each fixture
checks eager and captured output on both chips, runs three ABBA blocks with ten
blocking replays per sample, then changes the query at the same buffer address
and checks both traces against a separately computed native reference. Read-only
KV tensors must remain unchanged. Matrix totals are36 fixtures,4320 timed
replays and72 changed-input checks.

Timing includes device query packing, SDPA, unpacking and output concatenation.
Masks, page-table views, allocation/warmup/capture, validation readback and query
refresh are outside timing. There are no real model weights, KV writes, dynamic
position changes or committed-token claims. T1 is deliberately measured rather
than hidden: layout overhead can make this approach slower at small widths.
Scratch cleanup protects caller input buffers against no-op slice/reshape aliases.
No serving or verifier default uses the candidate.

### Device attention packing/unpacking passes first simulator cases

`device_layout` now executes query slicing, untilize, reshape, permutation,
reshape and tilize on the device, with the inverse operation after SDPA. It
retains intermediate handles for caller-owned lifetime management; the simulator
deduplicates buffer addresses before final release so reshape aliases are not
freed twice. Output remains BF16 and no arithmetic is added to the layout path.

Both-chip packing is checked against the independent Torch layout oracle before
SDPA, and unpacked SDPA results against native causal B1 outputs:
- `20260906T211544Z-314`: T4/start4096/seed0, one four-query group, exact.
- `20260906T211739Z-300`: T8/start4095/seed1, groups1/4/3, exact including offsets
  and the non-power-of-two tail. Both use negative-infinity masks and close cleanly.

Masks/page views are still host-prepared fixture metadata. This is a correctness
prototype using stock TTNN layout operations, not a fused copy kernel or proven
latency optimization. Hardware timing must include all packing/unpacking and
check capture memory headroom; simulator wall time is not a performance metric.

### Full T32 grouped attention passes boundary simulation

The simulator now executes every planned group, not just the CPU layout oracle:
- `20260906T210330Z-288`: T32/start4095/seed1, finite mask, exact both chips.
- `20260906T210610Z-284`: T2/start4095/seed2, native negative-infinity mask,
  exact both chips after splitting at the boundary.
- `20260906T210645Z-473`: T32/start16383/seed2, native negative-infinity mask,
  exact both chips for every token, with clean close after356.6 simulated seconds.

Both T32 cases use groups1/4/4/4/4/4/4/4/3. The first group has the old rounded
cache extent and the remainder have the next extent; no full T32 circular-buffer
allocation is attempted. The negative-infinity pass shows the finite-mask
workaround is unnecessary for this correctly partitioned fixture. It does not
prove all geometries or pathological inputs.

These tests upload host-folded queries and masks, so no device-side layout cost,
captured trace reuse or performance claim follows. Next build and simulator-test
device packing/unpacking, then measure the complete grouped operation against
native serial attention on hardware. Dynamic position changes must also select
the correct group/cache-view plan; a static plan must not be reused blindly
after crossing a chunk boundary. Serving and current verifier defaults remain
unchanged.

### Attention grouping: bounded groups pass simulator exactness

The head-fold hypothesis is not categorically rejected: the first masked variant
changed the number of key chunks assigned to cores. The native runtime reverses
chunk-to-core assignment as chunk count changes, so even a fully masked extra
chunk can alter reduction ordering. No numerical tolerance was changed.

With finite masks and bounded cache views/chunks, seed0 exact comparisons on
both simulated chips now pass:
- `20260906T205313Z-305`: T1/start63,64-token view and chunk.
- `20260906T205333Z-521`: T2/start63,128-token view and chunk.
- `20260906T205404Z-382`: T4/start127,256-token view and chunk.
- `20260906T205713Z-300`: T2/start4096,4352-token view,256-token chunks.
- `20260906T210011Z-298`: T4/start16384,16640-token view,256-token chunks.

Boundary diagnostic `20260906T205608Z-299` at T2/start4095 differs in2285
elements only for the first token; the next token is exact. This supports
splitting groups at native chunk-count boundaries. A host planner now groups
by dynamic chunk size and rounded valid extent, with a default four-query cap.
It handles non-power-of-two tails without changing the target's verification
block width. This is not yet an integrated device-layout adapter.

T32/start4096 probe `20260906T205751Z-456` was rejected before candidate kernel
execution: static circular buffers need6122688B versus1572864B L1. No hardware
test is authorized by these narrow simulator passes. Remaining gates include
grouped boundary execution, changed inputs/seeds, device-side packing and real
hardware latency including layout and mask work. No attention speedup is claimed.

### Attention head-fold first pass rejected in TTsim

The prior trace attributes about75% of summed B1 kernel time to matmuls; earlier
grid/block/packing sweeps did not yield a large repeatable gain. To investigate
wide verification instead, a new host-layout oracle folds token queries into
KV-grouped query heads with a separate causal mask per virtual head. Three host
tests cover roundtrip, GQA mapping and future-token masking through T32.

The stock non-causal masked SDPA probe is not exact against native causal B1:
- `20260906T204731Z-306`: T2/start63, all6144 elements differ on each chip.
- `20260906T204913Z-393`: T1/start63 diagnostic, all3072 elements differ;
  candidate output is zero, native maximum magnitude0.046875. Both are finite.
- `20260906T205021Z-304`: T1/start63 with finite negative mask instead of
  negative infinity,2554 differences per chip, maximum error0.000732421875.

All simulations closed cleanly with an intentional failing exactness assertion.
The finite-mask result isolates one masked-softmax issue but does not establish
native numerical equivalence. The probe changes causal mode and K chunking;
it does not isolate the head-layout transformation as the cause of residual
drift. No tolerance was relaxed, no hardware run was dispatched, and no serving
adapter uses this path. A future attention optimization must preserve native
per-query reduction/masking behavior rather than assume the stock masked API
is a drop-in exact replacement.

### Matched actual norm-batch requests passed, modest decode gain only

Run34055684657 (`c98f401`) passed eight actual requests in5m1s, one ABBA block
per context. Every arm matched native emitted tokens, active GDN, valid KV and
inactive slots; paired arms also matched proposal inputs, routing and acceptance.
Each arm aggregates two requests and256 post-seed committed tokens:

| Context | Control decode tok/s | Candidate decode tok/s | Control setup-inclusive tok/s | Candidate setup-inclusive tok/s |
| --- | ---: | ---: | ---: | ---: |
|4078|21.485812|22.461923|13.506870|13.237129|
|16363|17.781404|18.668689|12.005900|11.713965|

Decode gains are1.045430x/1.049900x. Candidate capture/setup totals were
7942.475/8141.457ms across two requests, versus7038.479/6925.786ms control.
Consequently the setup-inclusive post-seed result regressed; do not adopt this
as an unconditional short-request latency win. Prefill is excluded from both
rates in the table and recorded separately. The200tok/s target remains unmet.

The new native-tape report passed the48-policy offline scorer for all eight
request records. In the first candidate request per context, one-token matches
accounted for30/38 blocks and32/30 accepted proposals. The4K case had four
two-token matches and one six-token match; the16K case had two two-token matches.

Using observed per-width mean cycle costs only as an in-sample cost estimate,
width8/minimum-match2 predicted5073.490/5649.485ms for128 committed tokens.
The best16K estimate among these policies was width32/minimum-match4 at
5598.317ms. These are offline estimates, not measured speedups or held-out
selection, and remain far from200tok/s. The scorer preserves the native tape
but does not certify runtime rollback. Evidence is in the run artifact and
`hardware-evidence.local/34055684657/lookup-acceptance.json`.

### Norm-batch force-argmax passed; matched request gate enabled

Run34054076401 (`1882a31`) passed all12 width/context fixtures and1440 paired
timing replays. Both-chip full logits, greedy IDs and native GDN/KV state were
checked outside timing. Timed readback uses the first chip in both arms. Median
of six per-arm sample means, including verifier and selection/readback:

| Context | T | Host argmax ms | Device force-argmax ms |
| --- | ---: | ---: | ---: |
|4095|1|48.841846|47.470447|
|4095|8|71.583713|69.883303|
|4095|16|86.592667|83.972513|
|4095|32|115.775775|112.415496|
|16383|1|48.937715|48.023741|
|16383|8|76.086096|74.306094|
|16383|16|95.366206|92.816976|
|16383|32|133.145671|129.999195|

These costs exclude drafting and dynamic commit. With static, retained replay
and selection prerequisites inspected, `full-norm-engine` now enables eight
actual requests: one ABBA control/candidate block at each context. New telemetry
will support offline lookup policy scoring after the hardware request report
passes. No serving defaults, precision, reset or access changes are part of it.

### Norm-batch retained replay passed

Run34053795475 (`729c408`) passed in22m12s:24 eager/trace width fixtures,
16 rollback checks,36 two-block changed-input replay cases and4 negative
controls. Every replay case reports exact logits, refreshed prefix histories,
valid KV, inactive slots and two corrected continuation steps. Cases cover
T2/16/32 at4095/16383 tokens, first committed prefixes0/1/T and second
prefixes1/T. This certifies the row-parallel norm path with retained histories
and captured DMA publication for this matrix, not a real drafter or throughput.

Run34054076401 (`1882a31`) is now executing the separate force-argmax selection
cost gate. The next actual request run must include matched control/candidate
requests with fresh prefills and all custom traces closed between requests,
report setup separately, and retain the new proposal/native-tape telemetry.

The prepared request harness runs one control/candidate/candidate/control block
per context when norm batching is enabled. Each arm completes its entire request
and closes custom traces before the next prefill. The comparison rejects unequal
prompt/output tokens, budgets, proposal inputs/routing, acceptance, inactive-state
checks or zero-token measurements. It reports aggregate committed decode rate
and setup-inclusive post-seed rate separately for each arm; this is one ABBA
block, not a statistically broad benchmark. Per-arm evidence is written as each
request finishes. The CLI now enables norm-batch requests following the inspected
selection pass above; harness preparation is not a hardware request result.

### Lookup acceptance bottleneck and next-request telemetry

Re-inspection of actual request run34045603132 separates accepted proposal
tokens from committed correction/bonus tokens. At4078 tokens, ten T32 blocks
accepted29 proposals and committed39 tokens in1345.410ms. At16363 tokens,
twelve T32 blocks accepted25 and committed37 in1831.929ms. Thus those wide
blocks averaged only3.9/3.083 committed tokens, not32. T8 accepted zero
proposals in all13 blocks across the two contexts. T1 accounted for54/58
committed tokens and2615.003/2862.787ms respectively.

Even the improved static T32 verifier costs require about22.0/25.5 committed
tokens per cycle to reach200tok/s before drafting, selection and publication
overhead. This is an economic lower bound, not a predicted request result.
Kernel acceleration alone has not resolved the lookup acceptance gap.

The next synthetic request report records prompt tokens/hash, proposal input
tokens, position and lookup match length. Match metadata is derived only from
committed history and propagates with the immutable proposal/ticket; target-only
steps report zero, with no stale previous-match state. This changes telemetry,
not routing, verification or serving defaults. Host coverage passes337 CI and53
speculative-session tests. These traces will allow deterministic offline policy
comparisons without presenting oracle-scored acceptance as measured throughput.

### Full-model norm-batch exactness and static timing passed

Run34052200159 (`397fa9d`) passed24 eager/trace width fixtures,16 rollback
checks and4 negative controls. All12 matched timing comparisons were exact.
All48 GDN layers use the candidate only at T>=8; the ordered cache and B1 SDPA
remain identical between arms. Median arm costs from six sample means:

| Context | T | Control ms | Candidate ms | Median paired ratio |
| --- | ---: | ---: | ---: | ---: |
|4095|8|70.024971|67.462975|1.037808x|
|4095|16|89.578100|81.551613|1.098420x|
|4095|32|128.820462|109.972825|1.171447x|
|16383|8|74.434865|71.865865|1.035723x|
|16383|16|98.385381|90.345532|1.089042x|
|16383|32|146.353723|127.515929|1.147720x|

T1/2/4 controls were unchanged within timing noise. These are static full-logit
verification costs, not committed request throughput. The latest actual request
pilot remains21.415/17.741 committed decode tokens/s. The next hardware gate is
`full-norm-replay`: changed token/position inputs, retained histories, captured
DMA publication and corrected continuations.

The selection, retained replay and request helpers now accept and forward an
explicit `norm_batch` option, including warm and captured verifier buckets.
The CLI allows retained replay and the independent static `full-norm-selection`
gate, whose prerequisite is the full-logit pass above. Selection compares host
argmax/full-logit readback with device force-argmax/ID readback using the same
norm-batch verifier, not old versus new GDN kernels. Its device logits and IDs
must match the native reference on both chips. Hardware concurrency serializes
it behind the replay gate. Both gates have now passed; request combinations are
enabled for the next experiment, not certified by wiring alone. Host validation passes
337 CI tests,55 simulator-support tests and51 speculative-session tests.
Serving defaults remain unchanged.

Attention token-to-head folding is a source-level hypothesis only. The pinned
SDPA factory allocates reduction cores from batch and KV-head counts, suggesting
that keeping batch one might preserve the native reduction topology. However,
explicit per-query masks require non-causal mode, head growth increases circular
buffer storage, and causal half-tile selection can change. These are unresolved
exactness and memory gates, not evidence that ordinary batched SDPA is safe.

### Row-parallel norm: real-weight layer gain passed

Run34051597269 (`ccd51ff`) passed36 eager/trace fixtures,414 restored-prefix
two-token continuations,18 stale controls and720 paired timing replays. It used
real TP2 GDN weights, batched input/output projections, fabric reduction, all
packed recurrent/convolution prefixes and native final-state commit. The arms
differed only in recurrence/norm for T>=8; T1/2/4 retained the control path.

| T | Control ms | Candidate ms | Median paired ratio |
| --- | ---: | ---: | ---: |
|1|0.332933|0.332789|1.000x|
|2|0.435132|0.434923|1.001x|
|4|0.500165|0.500006|1.001x|
|8|0.631487|0.568535|1.112x|
|16|0.893762|0.721702|1.240x|
|32|1.417266|1.015059|1.396x|

Timing used seed0; exactness used all three seeds. These are one-layer results,
not full-model gains. The next opt-in `--norm-batch` full-model gate preserves
all attention, precision, checkpoint and commit behavior. All48 GDN layers must
report engagement for T>=8, with zero engagements below8. Actual request-pilot
integration is rejected until its separate lifecycle gate is explicitly wired.
The first `full-norm-batch` gate is static full-logit coding-context verification;
selection and deferred/replayed/request combinations explicitly reject the flag
rather than silently using old helpers. Its paired control keeps ordered cache,
B1 SDPA, every packed history and selected checkpoint identical, disabling only
the new norm path. All333 CI,55 simulator-support and51 speculative host tests
passed before dispatch, along with shell syntax, workflow YAML and diff checks.

### Row-parallel norm/gate: first useful split-kernel gain

Run34050458771 (`7db0198`) passed18 independent native fixtures,2160 paired
composition replays,2160 isolated-stage replays and36 same-address input-refresh
checks. All output rows and recurrent prefixes remained exact on both chips.

| T |24-worker fused ms |96-worker plus row-parallel norm ms | Median paired speedup |
| --- | ---: | ---: | ---: |
|1|0.071138|0.143933|0.493x|
|2|0.100197|0.162794|0.615x|
|4|0.160400|0.194820|0.825x|
|8|0.282678|0.260968|1.084x|
|16|0.521735|0.386469|1.350x|
|32|1.000683|0.640081|1.565x|

Values are medians across three seeds. At T32, isolated norm/gate time fell
from about0.511 to0.160 ms; recurrence remained about0.505 ms. This is a
synthetic component improvement, not measured model or committed-token speed.
Keep native/control paths for T1/2/4. The next opt-in real-layer gate keeps
batched projections, DMA convolution windows, packed histories and commit
identical between arms, changing only recurrence/norm at T>=8. It must pass
all-prefix native continuation and captured timing before full-model integration.

Integrated synthetic convolution/recurrence/norm simulator gates passed with
the no-inner-fence allocating factory and history pruning: T8/seed1
`20260906T180913Z-325` checked every output/state/conv prefix plus9 restored
two-token continuations and1 stale control; T32/seed2 `20260906T181653Z-410`
checked every output/state/conv prefix. Both retained exactly5 checkpoint-history
buffers and closed cleanly. T32 continuation was not requested in that simulator
run; the real-weight hardware layer gate covers every prefix in eager and trace.
Host validation passed330 CI,55 simulator-support and51 speculative tests.

### Row-parallel norm/gate: simulator gates passed

Stage-attribution run34049504379 (`9a2633e`) passed18 exact fixtures,2160
paired composition replays,2160 separate-stage replays and36 input-refresh
checks. T32 median-of-seed isolated costs were0.504833 ms recurrence and
0.510840 ms norm/gate; combined captured cost was0.991795 ms. Separate blocking
stage traces include extra host/dispatch boundaries, so do not sum them as
full-model latency. They expose repeated whole-tile norm/gate work as a concrete
optimization candidate, not just a memory-bandwidth hypothesis.

`gdn_vsplit_norm_batch` keeps recurrence unchanged but packs token outputs into
the32 rows of each whole128-wide head's four FP32 norm tiles. The existing
FNG arithmetic and BF16 round trips run once on those independent rows. Norm
weights repeat across rows, z arrives as full tiles, and the writer exports
four full BF16 tiles per head after zeroing only unused token rows. No norm
reduction spans different tokens or partitions a128-wide head. The only compute
source change is the loop bound; this still requires numeric device validation.
T2/seed1 simulator run `20260906T174736Z-317` passed three prepared executions,
including two changed-input checks on fixed addresses. T32/seed2 run
`20260906T174951Z-316` passed the same three executions and two refreshed-input
checks across every output and recurrent prefix, including token rows15/16/31.
Both closed cleanly. Independent hardware correctness/performance remains a
separate gate. Host validation passed328 CI,55 simulator-support and51
speculative harness tests, with shell syntax, workflow YAML and diff checks.

### Recurrence input prefetch: correct, no latency improvement

Run34049169694 (`b4054bf`) passed18 independent native fixtures,2160 paired
replays and36 changed-input checks on fixed addresses. Median-of-seed candidate
times were0.133481/0.162156/0.217013/0.329596/0.551384/0.997197 ms for
T1/2/4/8/16/32. Paired T32 improvement versus24-worker control was only about
0.4-0.5%, below the plain split's already marginal result. Do not adopt it.
Reducing issued input-read payload did not improve this composition's critical
path. Separate captured recurrence and norm/gate timing follows before further
arithmetic changes. Those isolated blocking timings are diagnostic, not additive
full-model latency or committed-token throughput.

### Value-split captured timing: correct, no useful gain

Run34048090329 (`1b70865`) passed all18 native-oracle fixtures and2160 paired
ABBA timed replays. Median latencies below use the median of the three seed
medians; both arms export every recurrent prefix and use BF16 TILE L1 output.

| T |24-worker fused ms |96-worker plus norm ms |
| --- | ---: | ---: |
|1|0.070349|0.120509|
|2|0.100335|0.152801|
|4|0.160545|0.208492|
|8|0.282692|0.322577|
|16|0.522745|0.546246|
|32|1.001499|0.991198|

The candidate is slower through T16 and only about1% faster at T32. Do not
integrate this variant into the model on those results. More active cores alone
did not lower the critical path enough; these are synthetic component costs,
not a model throughput improvement. The next isolated variant caches the11
Q/K/V/beta/g input pages per recurrence worker once in L1, retaining existing
compute, state feedback, norm stage and output layout. Its full-ring CB reuse
and changed row padding require new simulator and independent hardware gates.

The recurrence-prefetch reader (`70cc1e7f74672a26e047854a684258afe05b863dd506d780ce49b37f8410af53`)
passed three prepared L1-output executions at T2/seed1 (`20260906T172336Z-323`)
and T32/seed2 (`20260906T172513Z-313`), checking every output/prefix and immutable
inputs against the existing generated24-worker serial-T1 FNG control. These are
simulator functional results only. The hardware `gdn-value-split-prefetch` gate
requires18 independent native fixtures,2160 paired replays and36 changed-input
trace checks using the same tensor addresses. All refresh sources and native
reference results are prepared before trace capture; no new device allocation
is introduced while traces are live. Serving and model integration stay gated.
T2/seed2 simulator refresh run `20260906T173517Z-318` also passed: after the
first prepared execution, all four packed input streams changed in place;
two further executions matched the existing24-worker full-block control at
every output and state prefix. The fixture explicitly detected changed state.

### 96-worker recurrence: native hardware correctness passed

The opt-in `gdn_vsplit` prototype splits each128-wide value head into four32-wide
partitions (96 recurrence workers/chip), preserving native state tile order.
It exports an FP32 pre-norm bridge to a24-worker stage using the unchanged native
full128-wide RMS/norm-weight/SiLU arithmetic and BF16 rounding points. It does
not normalize partitions independently or round the bridge to BF16. Static CB
storage is282624 bytes/recurrence worker and204800/norm worker, versus630784
for the existing fused recurrence/norm kernel.

TTSim `20260906T163210Z-322` (T1/seed0), `20260906T163314Z-319` (T2/seed1)
and `20260906T163345Z-499` (T32/seed2) passed every output and recurrent prefix,
finite FP32 bridge and immutable inputs, with clean close. Oracle: serial T1
of the already-certified24-worker generated FNG kernel, not independent native
hardware. No value-split continuation or timing result is claimed yet.

`gdn-value-split` next checks three seeds andT1/2/4/8/16/32 against the independent
native packed FNG operation:18 eager fixtures,207 restored-prefix continuations
and18 stale controls. It is intentionally fenced/eager only. Trace-safe prepared
programs, paired timing, real-layer integration and model performance follow
only if this correctness gate passes. No default or serving integration exists.

Run34046231437 (`fa3e1b9`) passed all18 eager fixtures,207 restored-prefix
continuations and18 stale controls against the independent native packed FNG
operation. No timing was collected. The96-worker recurrence and24-worker
FP32-preserving norm/gate composition is now hardware-correct for these
synthetic cases; prepared/captured execution, real-layer and full-model
integration remain separate gates.

The prepared variant allocates output, prefix history and FP32 bridge once,
then enqueues the same two CQ0 programs without an intervening host fence.
T2/seed1 simulator run `20260906T165618Z-318` passed three repeated executions
with L1 output, exact outputs/prefixes and immutable initial state. T32/seed2
run `20260906T170429Z-430` passed the same three-execution checks with clean
close. The new `gdn-value-split-timing` hardware gate pairs
the existing24-worker FNG control with the96-worker composition at identical
L1 output placement, checking an independent native oracle outside timing.
Three seeds across T1/2/4/8/16/32 require18 fixtures and2160 ABBA timed replays.
This measures synthetic captured kernel cost, not committed model throughput.

### Actual request engine: correctness passed, speed target missed

After ownership replay recertification34044312880 (`1798fc8`:24 width/mode,
16 rollback,36 two-block replay and4 negative-control cases), actual request
pilot34045603132 (`800d13c`) passed exact native autoregressive tokens, final
active GDN, valid KV and inactive-slot checks for128 post-seed tokens per case.

| Actual context | Committed decode tok/s | Decode ms | Per-request setup ms | Accepted/proposed drafts |
| --- | ---: | ---: | ---: | ---: |
| 4078 | 21.415362 | 5977.017801 | 3700.997702 | 39/537 |
| 16363 | 17.741310 | 7214.799911 | 3233.698956 | 30/658 |

These complete-chat repeated-code fixtures are not a representative coding
benchmark. Rates include actual lookup, input updates, verification/readback,
selection and synchronized publication, but exclude the separately reported
request-specific setup. T32 lookup blocks often accept zero drafts, wasting
verification work; blindly widening lookup is not an adoption candidate.
The 200 committed tok/s target remains unmet. No serving changes are approved.

### Actual request engine design and retry history

The opt-in `full-verifier-engine` pilot connects real request-local lookup
proposals to T1/2/4/8/16/32 captured verifiers, force-argmax and synchronized
prefix publication. Capture is per request after prefill; all traces are closed
before the next prefill. Report setup and the post-seed setup gap separately,
without amortization, and time the complete proposal-to-commit loop. Compare
actual emitted tokens, final active GDN, valid KV and inactive slots against
native autoregressive generation. No oracle tokens enter the drafter. These
repeated-code prompts are pipeline fixtures, not representative coding quality.

Host review also corrected accepted-EOS publication: consume the seed and
preceding accepted tokens, but not the terminal EOS, matching native generation
and correction-EOS semantics. All31 possible accepted terminal positions have
an explicit state-row regression check; nonterminal decisions are unchanged.

First pilot run34043059904 (`eb6e16b`) failed before request decoding because
the top-level multimodal Qwen3_5Config has no eos_token_id. The retry reads and
hashes the frozen generation_config.json, validates all terminal IDs, and fails
closed on missing/invalid metadata. No performance result or reset is inferred.

Retry34043284318 (`c9775cc`) returned green but both truncated static prompts
produced terminal prefill seeds: zero decode tokens and no constructed engine.
This is only a terminal-request/accounting check, NOT an engine correctness or
throughput pass. The next pilot preserves the complete chat template using the
existing bounded prompt builder rather than truncating its generation marker;
actual context lengths are reported. Zero-decode fixtures now fail the pilot.

Complete-chat retry34043482490 (`a09c0c6`) reached engine capture, then failed
with a static circular-buffer/L1 allocation clash (buffer start736512 versus
CB end742272). Retained records unnecessarily held projection scratch across
layers and buckets. The fix retains only recurrent plus four convolution
histories, transfers layer output ownership to its caller, and releases other
temporaries during capture just like the non-retained path. Both-chip alias
guards precede any free. Kernel arithmetic is unchanged; simulator ownership/
all-prefix continuation and hardware replay recertification are required.

Simulator retention first pass `20260906T155420Z-326` passed T2/seed1 outputs,
all histories,3 restored continuations and one stale-state control. The stricter
both-chip independence guard then passed T32/seed2 run `20260906T155706Z-301`:
all32 outputs/recurrent/convolution prefixes and32 prefix copies, retaining
exactly5 history buffers per layer after scratch release. Both runs closed
cleanly. The oracle is serial T1 of the same recurrence kernel with native
convolution, not an independent full-model reference or timing result.

The final ownership guard also passed T2/seed2 continuation run
`20260906T161556Z-322` (all3 prefixes, one stale-state control; source6cf8aa03).
Independent engine review found preparation lifecycle gaps: preparation now
excludes proposals, failed construction poisons its request, and all capture
widths are bounded by the remaining generation budget before device work.
The complete budget must fit the page capacity; a last-page T1 request no
longer warms T32 beyond its allocation. Constructor-failure/geometry tests pass.
Capture-finalization or post-decode registration failures still require runtime
teardown rather than in-place recovery; no serving-lifecycle certification is
claimed for those paths.

### Full-verifier device selection (hardware gate passed)

Run `34040918809` at `613a591` passed all12 width/context fixtures and1440
paired timed replays. T32 verifier plus first-chip token readback measured
131.190365ms at4K and148.751581ms at16K, versus host-selection
134.513634ms and151.960231ms. All widths improved; complete pre-gather
logits, selected IDs and native state checks passed. These are component
latencies, not committed throughput, and exclude drafting/dynamic publication.

`full-verifier-selection` compares the same packed-history/ordered-cache verifier
with full-vocabulary gather plus host argmax versus pre-gather logits plus native
TTSampling force-argmax. The sampler's own trace is disabled inside the outer
verifier trace. Warm both arms before capture; compare complete native logits,
token IDs and active GDN/valid KV on both chips. Timed ABBA replays read only the
first chip in both arms, avoiding a diagnostic extra-read bias. Setup/restore
are excluded and explicitly reported; drafting and dynamic publication remain
outside this component measurement. The original sampler prerequisite is rerun
before the full model. No serving sampling default changes.

### Changed-metadata verifier replay (hardware gate passed)

The next width extension is opt-in T32: simulator publication passed all33
prefixes and input staging `20260906T150111Z-317` passed token, packed/singleton
position, native-RoPE and unchanged-page checks at31/4095/16383. The replay suite
now requests T2/T16/T32 (36 two-block fixtures), alongside24 width/mode cases
and16 captured post-verification decisions. Run `34041390068` at `eb4f744`
passed all36 two-block fixtures,24 width/mode checks,16 rollback/commit checks
and4 negative-control pairs. This certifies T32 changed-input replay and
publication, not actual drafting throughput or a multi-request lifecycle.
Host proposal/selection capacity31 requires explicit opt-in; defaults remain15.

`full-verifier-replay` reuses one captured verifier across two blocks at4K/16K,
T2/T16. First decisions0/1/T and second decisions1/T cover abort, partial and
complete publication. Tokens, packed/singleton positions and RoPE are updated
in place. Compare full logits, refreshed prefix histories/end checkpoints,
native active state, valid KV, every inactive slot and two corrected tokens
against fresh native serial references. These24 fixtures are not actual drafting
or a committed-throughput benchmark.

Retained records permit replay only after an explicitly synchronized successful
commit; failed publication, synchronization, callback or rebound native state
poisons reuse. Exactly one decision follows each successful replay epoch.
Input staging already passed simulator T2/T16 at31/4095/16383; no new kernel math
is introduced by this gate. Trace lifetime remains owned by the fixture and
traces are released before their retained buffers.

Run [34039074595](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34039074595),
code `4ed8814`, passed all24 changed-metadata replay cases, in addition to20
width/mode checks,16 dynamic commits/corrected rollbacks and four negative-control
pairs. All refreshed prefix histories, full logits, active GDN, valid KV, inactive
slots and corrected continuations matched. `full_replay.py` SHA256:
`6cec770ae90c37801410166e2a9443c6963fc7f5847513a681ba4140a60fb8a8`.
This establishes the tested two-block trace lifecycle, not an actual drafter,
multi-request serving lifecycle, or measured committed-token rate.

### Captured prefix publication (hardware gate passed)

Prebind and warm all prefix-specific DMA programs, then capture each publication
outside the decision interval. Report this setup separately from readback,
selection and synchronized selected-trace execution. This is a component gate,
not a reusable request executor or committed-throughput result. All existing
full-logit, rollback, valid-KV and inactive-slot checks remain required.

The binding refactor passed simulator run `20260906T133129Z-324`, T2/seed2,
all three prefixes, exact native/checkpoint values and untouched inactive slots.
Kernel C++ remains `ba513253101a62a921ac402ac0fe9e6c14bca73f4077efd4e88c5a2b9af92019`;
Python binding is `257f7b13899737b4d507024c7a77281b6925528e9ecdf94f7fe1600667f62ecb`.
Host suites pass: 219 CI, 55 simulator-support and 40 speculative tests.

Run [34036778172](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34036778172),
code `ac9608b`, passed 20 width/mode cases, 16 selected-prefix rollback cases,
four negative-control pairs and all inactive-slot checks on both chips. Selected
commit execution including binding guards and synchronization measured
1.852-4.571 ms; complete readback/selection/commit measured 9.852-14.092 ms.
These are unpaired component observations, not a committed-throughput claim.
Per-fixture program preparation, warmup and capture cost 80.581-472.999 ms,
reported separately and not yet amortized by a reusable request executor.
The native serial cache writer remains the independent control in this gate.
Health preflight and mesh close passed; no reset or serving change occurred.

### T32 verifier prerequisites (simulator work in progress)

The T16 cost ceiling remains below 200 tokens/s even at perfect acceptance and
zero overhead. T32 tests whether filling the existing 32-row matmul tile can
amortize weight reads further; it does not assume a drafter will accept 31 tokens.
Recurrence plus fused norm/gate passed simulator run `20260906T134145Z-324`,
T32/seed0, every token output and recurrent prefix exact against serial T1 of
the same generated kernel on both chips. This is not an independent native
oracle certification. It also exercises the second vertical tile face (rows 16-31).
Convolution/packed histories passed `20260906T134609Z-603` (seed1): all 32
prefix copies, 33 restored two-token continuations and a stale-state control,
with immutable entry/projected inputs and clean close. Ordered-cache T32 passed
`20260906T140728Z-1111`, seed2/start16383 with 1024 page-table columns,
complete native BF8 cache equality, immutable input and negative/replay checks.
Hardware attempt34038257973 (`94a86f3`) completed36 exact native full-layer
eager/trace cases,414 restored continuations and18 stale-state controls, then
failed an obsolete hard-coded216/15 final counter assertion. Full-model testing
did not start; mesh close succeeded. The retry derives every expected count from
the selected width matrix and includes regression coverage for missing cases.
This is a harness rejection, not a numerical or kernel-liveness failure.
Serving and host
proposal bucket limits remain at T16 until the required integration gates.

Retry [34038451865](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34038451865),
code `59e8940`, passed the independent native GDN layer prerequisite (36 exact
eager/trace cases,414 corrected continuations,18 stale-state controls). Full-model
artifacts contain24 exact width/mode cases,16 corrected rollback cases, four
negative-control pairs and all12 exact timing fixtures (2880 paired replays).
The workflow nevertheless failed its final obsolete `len(lengths)*5` timing-count
assertion: T32 correctly produced12 fixtures rather than10. This last bookkeeping
check is corrected in `4ed8814`; do not describe the original workflow as green.

Paired full-model means, identical packed GDN/B1 SDPA in both arms:

| Context | T | Serial cache writer ms | Ordered cache writer ms |
| ---: | ---: | ---: | ---: |
| 4095 | 16 | 91.584565 | 89.559871 |
| 4095 | 32 | 132.931733 | 128.738358 |
| 16383 | 16 | 100.371342 | 98.328946 |
| 16383 | 32 | 150.487581 | 146.287526 |

T32 ideal ceilings are248.57/218.75 tokens/s at4K/16K, assuming perfect acceptance
and zero drafting, selection, input-staging or commit overhead. These are not
committed throughput. The32-row path now has theoretical room above the target,
but high actual acceptance and low cycle overhead are still necessary.
T32 dynamic prefix publication passed simulator `20260906T143759Z-322`, seed2,
two synthetic layers, all33 prefixes, exact native/checkpoint state and every
inactive slot/source unchanged. Python binding SHA256:
`2674e657b10125be456b1a1eb948ca8b06a613978fd36c92bcba09dfb020830e`;
C++ remains `ba513253101a62a921ac402ac0fe9e6c14bca73f4077efd4e88c5a2b9af92019`.
The stricter `full_matrix.py` checker independently validated complete exact
coverage in the original T32 artifact; its original CI status remains failure.
No serving defaults or precision settings changed.

### Request-scoped speculative accounting (host tests passed)

`GreedySession` now coordinates actual lookup proposals, bucket selection, target
argmax decisions and caller-synchronized publication. Output/history advances
only after device publication succeeds. The prefill seed is excluded from decode
token counts; stale tickets, cross-request calls, reentrancy and failed commits
are guarded. Generation budgets and EOS cannot over-emit. Fourteen coordinator
tests bring the speculative host suite to40 tests. Device trace reuse and a real
coding-throughput executor are still required; this is not a speed claim.

### Next recurrence experiment: value-axis partitioning (not implemented)

The current device-loop recurrence still assigns one worker to each of24 local
heads. A candidate can split each128-wide value axis into four32-wide partitions,
using96 workers while duplicating the128-wide Q/K norm inputs. State pages would
remain in native `[head,K,V]` order: each partition owns every fourth value tile,
not a contiguous quarter of the recurrent tensor. Beta/g indices must continue
to use the original head, rather than the96-way worker index.

Important precision boundary from the audited native compute source: fused
norm/gate consumes the FP32 `qn @ new_h` result before RMS reduction. Routing
partition output through the ordinary BF16 recurrence output and then a separate
norm op would introduce an extra rounding point, so it is not an exact fusion.
A valid prototype must preserve FP32 partial outputs and reassemble all four
value tiles in the native RMS reduction order, then preserve the existing BF16
rounding after norm-weight multiplication, SiLU and the final multiplication.
The extra synchronization/traffic must be timed, not assumed free. Simulator
native-oracle output/state/continuation checks precede any hardware trial.

### Fused retained-state publication (hardware passed 34034469074)

Run34034469074 (`a6f64f1`) passed20 full-model width/mode cases,16 post-output
decisions and corrected continuations,4 negative-control pairs, and exact
before/after checks of all inactive slots in all48 native GDN layers. The gate
used96 workers per chip and the previously certified serial cache writer.
The preflight also passed12 two-chip transfer checks; no reset was needed.

The first commit included cold compilation:344.17ms commit,352.37ms combined
readback/selection/commit. Subsequent commits cost7.35-9.09ms, with total component
times15.65-19.47ms. Full-logit readback still costs7.64-8.64ms and host selection
0.35-1.95ms. These are unpaired component observations, not a speedup ratio or
committed TPS. Verifier execution, drafting and input staging are excluded.
Prebinding/capturing publication and certified device force-argmax are the next
ways to remove descriptor construction and full-logit host transfer overhead.

`gdn_commit_dma` publishes an arbitrary selected prefix for all48 GDN layers in
one arithmetic-free launch:96 workers per chip, two disjoint page partitions per
layer. It reads immutable entry or packed-history buffers and writes native slot0
plus the external checkpoint. Native inactive slots1-7 are untouched. All layers,
buffer aliases and accessor layouts are checked before submitting any write.
The multi-operation commit remains the helper default; the next explicit hardware
gate opts into fused publication. It deliberately retains the certified serial
cache writer, independent of the ordered-writer full-model experiment. A bounded
two-chip transfer probe must pass before opening the model. In addition to the
existing exact logits/active state/valid KV/rollback checks, every inactive slot
of every native GDN tensor is hashed before and after each commit, outside timing.

The simulator fixture checks every prefix against exact BF16 host slices, all
native inactive-slot canaries, external checkpoints and immutable sources. T2,
two synthetic layers, seed0 passed `20260906T120611Z-321`; T16 seed1 passed all17
prefixes in `20260906T120814Z-316`, with clean close and exit0. T4 seed2 passed
all5 prefixes in `20260906T121953Z-315`.
This does not validate96-worker hardware scheduling or commit performance.
The future full-model gate records readback, selection and commit separately,
so host logit transfer cannot be mistaken for kernel execution time.
The new commit remains opt-in. Its source hashes are
`2d1bc9e4267d2576dc4c699a41db1e8afeb2a4bec04ce867d831ffd92d0661b7`
(Python) and`ba513253101a62a921ac402ac0fe9e6c14bca73f4077efd4e88c5a2b9af92019`
(kernel), matching the simulator first pass. No serving defaults changed.

### Post-verification GDN commit (hardware passed 34029984214)

`RetainedGDNBlock` keeps all48 packed histories alive through logit readback and
permits one commit decision. Every owner binding is checked before writes; a
partial device failure poisons the decision rather than permitting a fresh retry.
Record buffers are released only when the owning fixture closes, after trace
release. Downstream-owned layer outputs are excluded from record ownership.

The full-model gate uses the existing `greedy_verify.select_prefix` helper
on actual target argmax rows. The already-emitted seed is not emitted twice:
accepted draft proposals plus the correction/bonus determine `state_rows`.
An explicit abort exercises prefix0. Inputs are forced-rejection fixtures, not
an actual drafter, and acceptance is selected after the full verifier output.
The candidate is no longer given the selected checkpoint in advance for these
rollback cases. It must publish the selected native state itself; an external
restore is deliberately omitted so it cannot hide publication errors.

All earlier full-logit/state/KV/negative-control checks remain. The experiment
passed16 post-verification decisions and readback/selection/commit component times,
not end-to-end throughput. It skips redundant static verifier timing, already
measured in34028729821. The tested kernels are unchanged.200 CI,55 simulator
and26 speculative host tests pass. Hardware run34029984214 at08bd728 passed
all20 width/mode cases,16 corrected rollbacks and4 negative-control pairs,
including4 explicit aborts and12 greedy decisions across4K/16K eager/trace.
All48 layer records survive until the post-verification decision. Readback,
host selection and eager commit cost25.45-41.57ms in these fixtures; those
component timings exclude verification and drafting and are not committed TPS.
No serving settings change.

### Ordered shared-page cache write (hardware passed 34033619168)

Full-model retry34034319922 (`2391339`) passed all20 width/mode cases,16 corrected
rollbacks,4 negative-control pairs and10 paired timing fixtures (2400 replays).
Full logits, active GDN state, valid KV and end checkpoints are exact on both
chips. The comparison keeps all packed-history GDN flags and B1 SDPA identical.

| Context | T | Serial writer control ms | Ordered writer ms | Paired ratio |
| --- | ---: | ---: | ---: | ---: |
| 4095 | 1 | 45.027539 | 45.018834 | 0.999948 |
| 4095 | 2 | 55.566669 | 55.396328 | 1.003080 |
| 4095 | 4 | 60.751795 | 60.247484 | 1.008347 |
| 4095 | 8 | 70.979604 | 70.029573 | 1.013586 |
| 4095 | 16 | 91.592307 | 89.551743 | 1.022750 |
| 16383 | 1 | 45.594663 | 45.616157 | 0.999953 |
| 16383 | 2 | 56.687872 | 56.514909 | 1.003138 |
| 16383 | 4 | 62.953030 | 62.455886 | 1.007989 |
| 16383 | 8 | 75.380580 | 74.444874 | 1.012667 |
| 16383 | 16 | 100.367964 | 98.324729 | 1.020734 |

T1 remains native. The actual full-model improvement is about2%, not the21%
layer ratio. Even perfect T16 acceptance with zero drafting/commit overhead
would yield only178.67/162.73 tokens/s at4K/16K. Target200 is still unmet.
The independent fused-commit run34034469074 (`a6f64f1`) uses the certified
serial cache writer, so its state-publication test does not depend on this change.

The first full-model attempt34033778987 (`ee90045`) cleanly rejected its1024-column
page table at the adapter's512-column guard, before ordered kernel dispatch.
The cap is now1024, also bounded by physical cache block count to match the native
validator. The extended simulator fixture initially used only8 physical blocks
and was correctly rejected by the native oracle (`20260906T124638Z-325`); this
fixture was fixed rather than weakening the native capacity invariant.

With1024 physical blocks and1024-column metadata, T16 at16383/seed1 passed
`20260906T124725Z-316` and T2 at4095/seed2 passed `20260906T124834Z-318`.
Both chips' complete physical caches match native ordered B1 writes, with repeat,
immutable-input and omitted-write checks, clean close and exit0. The generated
kernel math is unchanged; only metadata capacity/CB size changes. The full-model
retry needs no reset because the rejected attempt closed cleanly.

The corrected layer gate at51aebda passed60 exact eager/trace cases,30 complete
KV negative-control pairs and90 ABBA timing blocks (10800 timed replays).
Paired controls use identical batched projections and B1 SDPA; only the writer
changes. Means below combine both seeds and starts31/63/65; ratios are medians
of the18 paired blocks per width.

| T | Serial writer control ms | Ordered writer ms | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 | 0.282626 | 0.291087 | 0.971070 |
| 2 | 0.320980 | 0.310411 | 1.034293 |
| 4 | 0.377824 | 0.346379 | 1.090584 |
| 8 | 0.495695 | 0.429324 | 1.153374 |
| 16 | 0.721910 | 0.592336 | 1.214686 |

T1 regressed and remains native in the full-model candidate. Multirow widths
advance to full-model long-context correctness and paired timing; these are
layer timings, not committed TPS. All native cache kernel hashes match the
simulator-certified sources. No serving defaults change.

Run34031827886 passed T1 eager/trace output and complete KV equality, then
failed before timing on an unwarmed `InterleavedToShardedDeviceOperation` in
the previous serial-writer control. The candidate was warmed, but the newly
isolated control composition was not. An active capture then prevented clean
teardown until the shell timeout. This is not evidence of a cache-kernel mismatch
or a multirow kernel pass. No timing was collected.

The retry warms and validates both arms before either timing trace is captured,
and closes/releases failed captures before propagating operation errors. Stage
markers distinguish native reference, candidate warmup and each timing arm.
Transfer health check34033532881 passed12 exact two-chip readbacks, clean close
and0.226s inside the probe. No reset was needed or dispatched.212 CI,55 simulator
and26 speculative host tests pass. The same failed-capture cleanup and all-arm
warmup are also applied to the opt-in full-model timing path before its next run.

The remaining attention adapter dispatches per-token slice, reshard and native
paged-cache writes. A bounded candidate stages each prepared KV row on one worker,
then chains the native reader/writer semaphore so each BF8 read-modify-write
finishes before the next token reads the shared page. Native untilize/tilize math
and packing remain unchanged. This is one launch with ordered device work, not
unsafe parallel updates to a shared BF8 tile. B1 SDPA remains unchanged.

`ordered_cache.py` hash-pins all three native kernel sources before generating
the reader's interleaved-input staging change. Simulator fixtures compare the
entire physical cache against native serial B1 writes, cross tile/page boundaries,
verify immutable inputs, detect omitted writes and repeat after restoring the
initial cache. No hardware dispatch or performance adoption precedes simulator
success. T1/2/4/8/16 passed with starts15/31/63/65 across three seeds; simulator
fixture IDs are in `optimisation/sim/README.md`.206 CI,55 simulator and26
speculative host tests pass. Hardware `attention-timing` compares the exact same
batched projections and B1 SDPA with serial versus ordered cache writers, keeping
native complete-cache/output correctness and negative controls.

### Packed convolution checkpoints (hardware layer passed 34028407207)

The next isolated experiment retains the four post-convolution `[1,T,5120]`
tensors as immutable packed histories, rather than materializing four separate
compact tensors for every prefix. `gdn_conv_prefix_copy` restores any nonzero
prefix after verification with one aligned, arithmetic-free DMA launch. Prefix0
continues to use the entry snapshot. Guards reject out-of-range selections,
shape mismatches and aliases on either chip before submitting writes.

AtT16 this changes convolution-prefix storage from20MiB to1.25MiB per layer per
chip (BF16 tile padding included); recurrent-prefix exports are unchanged.
It does not implement an engine-level draft/accept/commit lifecycle. In particular,
the current full-model adapter still selects a checkpoint in advance and releases
its other records at the layer boundary; no dynamic serving claim is made.

Standalone prefix-copy simulator fixtures `20260906T103225Z-321` (T4/seed2) and
`20260906T103340Z-466` (T16/seed1) passed all4/all16 selections with clean close.
The development fixture `20260906T103050Z-328` passed kernel comparisons but failed
because its new cleanup block referenced model-adapter-only variables. Cleanup
scope was corrected before these passing retries; the failed run is not a pass.

Packed integration fixtures `20260906T103845Z-544` (T2/seed0) and
`20260906T104119Z-721` (T16/seed1) passed every output/recurrent/history prefix.
T2 additionally passed all3 corrected-prefix continuations and a stale-state
control; T16 passed synthetic local output projection. Adapter SHA256:
`84d8af2bd14ffd0d133c06b3cf777635671cb40d7497abcb16d37ebe3c4baa03`.
Prefix-copy Python/kernel hashes:
`06bfde0ee1d8503128b44818a98ce4cc0048b86f37c3f2d11cf926426a7cab38` /
`31fc2a0e886eb0b85239fa196ca2c571d4eb4124b84c4eb840b6be2f2a71fa7f`.
195 CI and55 simulator unit tests pass. The hardware layer gate compares packed
storage against the previous DMA-window/batched-convolution candidate with
separately materialized prefixes, preserving all logical checkpoints in both arms.
Packed model-adapter fixture `20260906T104508Z-326` also passed all3 prefix
selections atT2/seed2, exact final active-slot publication and inactive-slot
preservation, with clean close and exit0. This gate does not enable packed
checkpoints in the full-model suite yet.

Run34028407207 (`7a20340`) passed30 native-oracle eager/trace checks,
216 two-step corrected-prefix cases,15 stale controls and600 paired layer replays.
The19213-byte artifact confirms matching adapter, restore and prefix-copy hashes.

| T | Separate-prefix control ms | Packed-prefix candidate ms | Median paired ratio |
| --- | --- | --- | --- |
| 1 | 0.334818 | 0.333533 | 1.002951 |
| 2 | 0.663980 | 0.435348 | 1.525094 |
| 4 | 0.994694 | 0.500429 | 1.987312 |
| 8 | 1.658873 | 0.631632 | 2.626395 |
| 16 | 3.005572 | 0.911663 | 3.360829 |

The next full-model gate probes packed histories at every T>1, versus the previous
DMA-window full-model candidate atT8/T16 and its retained compact path atT2/T4.
T1 remains native. These layer ratios are not extrapolated to full-model speed.

Full-model run34028729821 (`46ec864`) passed20 width/mode cases,16 corrected
rollbacks,4 negative-control pairs and10 timing fixtures (2400 model replays).
The84522-byte artifact confirms simulator/hardware kernel hashes. Results:

| Context | T | Previous best control ms | Packed-history ms | Median paired ratio |
| --- | --- | --- | --- | --- |
| 4095 | 1 | 45.013186 | 45.012427 | 1.000035 |
| 4095 | 2 | 56.134482 | 55.574778 | 1.010103 |
| 4095 | 4 | 72.867452 | 60.740132 | 1.199637 |
| 4095 | 8 | 74.180274 | 70.989847 | 1.044967 |
| 4095 | 16 | 94.920985 | 91.546888 | 1.036916 |
| 16383 | 1 | 45.593229 | 45.583486 | 0.999956 |
| 16383 | 2 | 57.248886 | 56.679426 | 1.010053 |
| 16383 | 4 | 75.076121 | 62.958131 | 1.192472 |
| 16383 | 8 | 78.589817 | 75.390329 | 1.042499 |
| 16383 | 16 | 103.710163 | 100.335609 | 1.033609 |

Every multirow width improved, though T2's gain is small. T16 ideal acceptance
without any drafting or commit cost still permits only about174.77/159.46 tokens/s,
so200 committed tokens/s remains unachieved. The full-model gate still preselected
an external checkpoint; the next gate removes that requirement for rollback.

### Parallel causal convolution windows (hardware passed 34026768263)

`gdn_batched_conv.py` builds each token's four-tap window from entry history and
projected input, then calls the unchanged native convolution/gates kernel once
with `batch=T`. This is input-window parallelism, not parallel GDN recurrence.
Native BF16 arithmetic and the device-loop recurrence/norm kernel are unchanged.
Every convolution prefix is retained; acceptance need not be known in advance.
The working final history is copied back to stable addresses. T1 remains native.

Simulator fixtures passed on both simulated chips, with clean close and exit 0:

| Fixture | Width/seed | Additional checks |
| --- | --- | --- |
| `20260906T100224Z-326` | T2/0 | All3 corrected-prefix continuations; stale-state control |
| `20260906T100449Z-630` | T16/1 | All16 output/state/history prefixes; local output projection |
| `20260906T100806Z-838` | T4/2 | All5 corrected-prefix continuations; stale-state control; immutable projected input and final working history |

Adapter SHA256: `a34fc390f39263f791c1bf6c31e03ee1f35ed36168e4acf7e13900d6fe88d8e3`.
187 CI and55 simulator unit tests pass. Simulator timing is not a card performance
result. Next `gdn-multitoken-conv` CI uses the new candidate against the previous
serial-convolution/device-loop full-layer adapter, rather than a naive serial
projection control. Native-oracle all-prefix eager/trace and corrected continuation
checks still gate the paired measurements. No full-model routing or serving change.

Run34026768263 (`2288bab`) passed all30 real-weight eager/trace cases,
216 two-step restored-prefix cases and15 stale-state controls. All five paired
timing fixtures passed exactness (600 replays); the17624-byte artifact confirms
the simulator adapter hash. Results versus the prior serial-conv/device-loop layer:

| T | Control mean ms | Batched-conv mean ms | Median paired ratio |
| --- | --- | --- | --- |
| 1 | 0.332765 | 0.332644 | 1.000074 |
| 2 | 0.574181 | 0.968120 | 0.592697 |
| 4 | 0.972758 | 1.313739 | 0.745849 |
| 8 | 1.770593 | 1.967011 | 0.900074 |
| 16 | 3.371502 | 3.302826 | 1.020746 |

T16 improved only about2%; T2/T4/T8 regressed. Do not promote this composition
as a general performance win. Next remove the input-window layout/concat/slice
chain with one arithmetic-free DMA kernel, then remeasure before full-model use.

The first DMA prototype failed safely in simulator fixture `20260906T101509Z-321`
(exit1): a32-byte read from history row0 into an odd output row violated source
versus destination NOC alignment. It was never run on hardware. The correction
stages five full tiles (projected input plus four entry states) in aligned L1,
rearranges rows locally and writes four complete output tiles. Corrected T2/seed0
fixture `20260906T101726Z-332` passed every prefix, all3 corrected continuations,
the stale-state control and clean close. T16/seed1 fixture `20260906T101946Z-516`
also passed every output/state/history prefix and synthetic local output projection.
Model-adapter coverage follows; no full-model routing is changed by this kernel gate.

Aligned-DMA run34027345128 (`5759def`) passed30 native-oracle eager/trace cases,
216 two-step prefix continuations,15 stale controls and600 paired layer replays.
The19241-byte artifact's adapter/DMA hashes match simulator evidence:

| T | Serial-conv control ms | DMA-window batched-conv ms | Median paired ratio |
| --- | --- | --- | --- |
| 1 | 0.333619 | 0.332588 | 1.000039 |
| 2 | 0.573486 | 0.663122 | 0.864568 |
| 4 | 0.973454 | 0.996604 | 0.977645 |
| 8 | 1.770995 | 1.659346 | 1.067371 |
| 16 | 3.370352 | 3.004254 | 1.121892 |

The full-model experiment therefore changes only T8/T16; T1/T2/T4 retain their
previous paths. Its paired control retains compact-prologue device-loop recurrence
with serial convolution, not the older row-layout-only candidate. Simulator
fixture `20260906T102349Z-322` passed the actual DeviceLoopState adapter atT2/seed2:
all3 prefix selections, final native-slot publication and inactive-slot preservation.
This validates adapter plumbing, not full-model logits or timing. Full-model
eager/trace/rollback/valid-KV and paired4K/16K costs are the next hardware gate.

Full-model run34027510486 (`de6fd22`) passed20 width/mode cases,16 corrected
rollbacks,4 negative-control pairs and10 timing fixtures (2400 model replays).
All full logits, active GDN state and logically valid KV matched on both cards.
The84002-byte artifact matches simulator adapter and window-DMA kernel hashes.

| Context | T | Compact-prologue serial-conv ms | DMA batched-conv ms | Median paired ratio |
| --- | --- | --- | --- | --- |
| 4095 | 8 | 103.766911 | 74.177437 | 1.399093 |
| 4095 | 16 | 163.262062 | 94.881889 | 1.720690 |
| 16383 | 8 | 108.173005 | 78.591578 | 1.376383 |
| 16383 | 16 | 172.040179 | 103.655336 | 1.659846 |

T1/T2/T4 retained their previous paths and unchanged paired timings. These remain
fixed verifier blocks: ideal T16 acceptance without draft/commit overhead would
give about168.63/154.36 tokens/s. Actual committed throughput is not established,
and the200 goal is not met. No serving defaults changed.

### Projected-input convolution integration (passed 34019928513)

The shared `gdn_multitoken_conv.run_projected` adapter composes native serial
convolution/gates with one multi-token recurrence/norm call. It retains all four
convolution state snapshots per token and the recurrent prefixes; it does not fuse
the convolution token loop or include output projection/CCL.

Seven dual-chip synthetic fixtures passed exact outputs, every recurrent prefix,
all convolution prefixes, unchanged initial recurrence and clean close with exit0:

| Seed | Tokens | Simulator run |
| --- | --- | --- |
| 0 | 1 | `20260906T073404Z-412` |
| 0 | 2 | `20260906T073143Z-327` |
| 0 | 4 | `20260906T073427Z-458` |
| 0 | 8 | `20260906T073533Z-568` |
| 0 | 16 | `20260906T073730Z-684` |
| 1 | 2 | `20260906T074120Z-868` |
| 2 | 2 | `20260906T074203Z-941` |

Reports/logs are in `/opt/ttsim/results` on D-drive `TT-Sim`. This first pass uses
synthetic weights/projected rows and serial T1 calls of the same helper, not an
independent native recurrence oracle. Generated norm kernels remain identical to
the hardware-validated `36b9eed` versions. Local validation:59 GDN unit tests pass,
Python AST, shell syntax and workflow suite routing pass.

Run [34019928513](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34019928513)
(`8e75a95`) passed all30 eager/trace cases across three seeds and T1/2/4/8/16,
using93 captured real-weight native projections. Native GDN gated outputs and every
recurrent/convolution prefix matched exactly on both cards; entry snapshots and
stable working convolution addresses also passed. The report reached clean close
and `passed=true`, with the expected checkpoint, generated kernel and runtime header
hashes. No reset, serving change, rollback-continuation certification or timing claim
is part of this completed gate.

Next: restore every accepted prefix (including zero/all) into independent stable
native recurrent and convolution buffers, then compare two corrected native steps
against independently saved native-oracle prefixes. The extended suite requires216
prefix/mode cases (432 steps) and15 stale-final-state negative controls. The shared
restore helper rejects snapshot aliasing on either chip before writes.

Simulator continuation passed: development T2/seed0 `20260906T080711Z-332` checked
all3 accepted prefixes and detected its stale-state control. After adding stricter
shape/alias guards, final T1/seed1 `20260906T081239Z-328` and T4/seed1
`20260906T081508Z-676` checked all2/all5 prefixes respectively, both with detected
stale-state controls, clean close and exit0. The final two reports pin adapter SHA256
`f5e91983b050f443129e2d00af8e5373a02093c871cb5b0c0c287a0767fd062f`.
Generated recurrence kernels are unchanged. These are synthetic two-row continuation
comparisons, not native hardware or full-model token acceptance results.

Local validation:67 focused GDN tests pass on Windows; all174 CI and55 simulator
unit tests pass under WSL. The broader suites require Linux paths/Torch and failed
in Windows before being rerun successfully in WSL. AST/shell/workflow syntax passes.
An initial WSL VM startup timed out before the simulator ran; restarting only the
dedicated `TT-Sim` distribution restored operation. No card reset occurred.
The extended native hardware continuation gate passed in
[34021612139](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34021612139)
(`241be95`):30 eager/trace cases,216 accepted-prefix/mode two-step native
continuations (432 compared steps),15 detected stale-state controls and clean close.
The hardware adapter hash matched the simulator's final guarded version.

### Full GDN layer projection and paired timing (next)

The composed gate now has optional `--full-layer --paired-timing`: real batched
input projection, the validated convolution/device-loop recurrence path, batched
output projection/native fabric reduction, and final state committed into stable
native buffers. Native serial full-layer output is the independent oracle. All30
eager/trace and216 two-step prefix-continuation cases remain mandatory.

Simulator `20260906T083558Z-338` passed T2/seed0 composition plus exact synthetic
3072x128 local output projection against serial T1, clean close/exit0. It does not
simulate the fabric reduction or real projection weights. Native input/output
projection and CCL are reused; no recurrence kernel math changed. Local validation:
68 focused GDN tests,175 CI and55 simulator unit tests pass.

Timing uses seed0 T1/2/4/8/16, three ABBA blocks and120 replays per width. Both arms
return every DRAM state prefix and commit their final native state. Initial restore
is outside timing. The control is native serial input/output projections, not the
previous optimized full-model row-layout control; candidate batches projections.
Native recurrence uses L1 state while the candidate loads immutable DRAM state and
loops locally. Report these differences and raw paired samples; no full-model or
committed-token speedup follows from this isolated layer experiment.

First full-layer run34022404895 (`e6b134f`) stopped during eager T1 result cleanup
after exactness and both continuation prefixes passed: `ValueError: Both chips
required` in the owned-buffer ledger. Device close completed normally; this was
not a device timeout. The output helper now follows the native projection tail's
ownership contract: consume/deallocate gated input after `_row_proj` and remove it
from the retained ledger instead of inspecting it again at fixture cleanup.
Regression coverage includes a projection that already consumes its input.
Simulator T2 `20260906T084141Z-327` passed with the corrected helper (adapter hash
`829e24de72c04759e8950e766c7e8d8aebaa2badc5d36cc7f3266642cb187f1a`),
clean close/exit0;69 focused GDN,176 CI and55 simulator unit tests pass. Hardware
retry is required before any timing or full-layer adoption claim. No reset needed.

Retry [34022668338](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34022668338)
(`6b7d2b2`) passed all30 full-layer eager/trace cases,216 two-step continuations,
15 stale-state controls and five timing fixtures (600 timed replays), with clean
close. The ownership correction is hardware-validated. Seed0 mean block times:

| T | Native serial ms | Candidate ms | Median paired ratio |
| --- | --- | --- | --- |
| 1 | 0.31330 | 0.33316 | 0.9398 |
| 2 | 0.67204 | 0.57339 | 1.1718 |
| 4 | 1.31505 | 0.97451 | 1.3498 |
| 8 | 2.60140 | 1.77148 | 1.4681 |
| 16 | 5.17571 | 3.37186 | 1.5355 |

Native T1 must remain the deployed/control path. These gains are against serial
full-layer projections, not the faster previous full-model row-layout control.

### Full-model device-loop integration (next)

`full-gdn-device-loop` wires the validated path into all48 GDN layers for T>1,
retaining native T1 and the16 existing B1 attention/KV adapters. Independent compact
entry/working buffers protect the stable native B8 allocation. It publishes the
selected prefix separately from final active-slot state, leaving inactive slots
untouched. Every internal prefix is still materialized, which may offset kernel
gains; do not infer a full-model win from the layer timing.

Simulator `20260906T084858Z-363` passed T2/seed0 at accepted prefixes0/1/2:
exact outputs/recurrent prefixes, selected recurrent/convolution checkpoints,
final active state and all seven inactive slots on both chips, clean close/exit0.
It ran the actual active/compact DMA kernels and the new state adapter; projection
inputs were synthetic, not full-model weights/attention. Adapter hash:
`fd5ca6746a84184d105dd0698a4d4716705c678fa5296baea8ad1cbd86cc85bf`.

Hardware gate:4K/16K coding contexts, five widths, eager/trace full logits and state,
corrected rollback, valid-KV checks, stale-state/wrong-page controls, and paired
full-model timing against the previous compact/input-reuse/selective-clone/row-layout
candidate from run34011273093. No serving change or200 committed-tok/s claim.

Run [34023117059](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34023117059)
(`6a2ca23`) passed20 width/mode checks,16 corrected rollback cases, four negative
control pairs and all10 timing fixtures. Full logits/GDN state/valid KV were exact.
**Performance rejected:** all T>1 widths were slower than the previous optimized
row-layout control. T1 remained native and unchanged within measurement noise.

| Context | T | Control mean ms | Device-loop mean ms | Median paired ratio |
| --- | --- | --- | --- | --- |
| 4095 | 2 | 56.138 | 62.037 | 0.9051 |
| 4095 | 4 | 72.858 | 83.246 | 0.8752 |
| 4095 | 8 | 106.285 | 125.435 | 0.8473 |
| 4095 | 16 | 173.072 | 210.138 | 0.8236 |
| 16383 | 2 | 57.238 | 63.133 | 0.9067 |
| 16383 | 4 | 75.109 | 85.456 | 0.8785 |
| 16383 | 8 | 110.697 | 129.859 | 0.8524 |
| 16383 | 16 | 181.859 | 218.918 | 0.8307 |

Next bounded intervention: `--compact-prologue` restores the earlier hoisted
projected-row layout strategy and removes unneeded convolution checkpoint clones.
Only the selected/final convolution prefixes are materialized (four rather than64
convolution clones at T16/end); all recurrent prefixes remain. Missing convolution
prefix restores fail before any writes. Native T1 and kernel math are unchanged.
This bundles two known overhead sources; a gain must not be attributed separately
to either without ablation. The full-model comparison still uses the stronger
previous row-layout baseline, not the slower device-loop candidate. No adoption.

Compact-prologue simulator `20260906T092329Z-324` passed T2/seed0 at prefixes0/1/2,
including selected checkpoints, final active state, all inactive slots, clean close
and exit0. The standard all-prefix path is also exercised as its control. Local
validation:75 GDN,184 CI and55 simulator unit tests pass; Python/shell syntax and
diff checks pass. Full-model compact-prologue hardware results are pending.

Run [34024642720](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34024642720)
(`8cb31c8`) passed20 width/mode checks,16 corrected rollbacks, four stale-state/
wrong-page negative-control pairs and10 timing fixtures (2400 timed model replays).
Both adapter hashes matched simulation. All logits, GDN state and valid KV checks
were exact; device close and CI completed successfully.

| Context | T | Previous optimized mean ms | Compact-prologue mean ms | Median paired ratio |
| --- | --- | --- | --- | --- |
| 4095 | 2 | 56.134 | 58.244 | 0.9638 |
| 4095 | 4 | 72.860 | 72.552 | 1.0044 |
| 4095 | 8 | 106.304 | 103.774 | 1.0245 |
| 4095 | 16 | 173.067 | 163.227 | 1.0604 |
| 16383 | 2 | 57.244 | 59.348 | 0.9644 |
| 16383 | 4 | 75.063 | 74.780 | 1.0036 |
| 16383 | 8 | 110.692 | 108.155 | 1.0234 |
| 16383 | 16 | 181.856 | 172.023 | 1.0572 |

All three paired blocks favored T8/T16 at both contexts. T2 regressed; T4's gain
was marginal. The follow-up experimental routing policy therefore uses compact
device-loop GDN only at T8/T16, retaining native T1 and the previous optimized
compact/row-layout path at T2/T4. This routing-only follow-up has185 CI +55 simulator
unit tests passing; the timings above belong to `8cb31c8`, before that selector.
No serving defaults changed and the200 committed-tok/s target is not demonstrated.

These are fixed verifier blocks with a preselected checkpoint, not a dynamic
speculative acceptance/commit pipeline. Next: attribute remaining convolution,
packing and recurrent-prefix export costs; test a true convolution/gates token loop
that writes packed outputs directly. Dynamic acceptance still needs arbitrary-prefix
state recovery (for example, validated reconstruction of convolution history from
retained projected rows), plus drafter/commit and executable coding-quality gates.

### Multi-token norm/gate hardware exactness (passed 34019033933)

Run [34019033933](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34019033933)
(`36b9eed`) passed after the full15-fixture simulator matrix, without another reset.
All30 three-seed T1/2/4/8/16 eager/trace cases matched native output and every prefix
state exactly on both cards. All216 restored-prefix native continuations matched;
all15 stale-state controls were detected. No errors or failed exactness checks.
The 18065-byte artifact records the same generated kernel hashes as simulation,
and both runtime-header hash guards passed on the hardware image.

This validates the CB5 producer-counter fix and synthetic recurrence plus fused
norm/gate against the native implementation. No timing was collected. Native oracle
state is L1, candidate prefix exports DRAM; do not infer a matched speedup. Real-weight
convolution, full-model integration, device rollback plumbing and committed-token
throughput remain subsequent gates. Next validate real-weight GDN integration in
simulation before returning to hardware native-oracle and matched timing checks.

### CB5 producer-counter handoff fix (15 simulator fixtures passed)

The local Blackhole LLK pack implementation caches `tiles_received` in the packer
thread and publishes that cached value on push. The dataflow reader increments
the shared received counter instead. Our one-time full-ring CB5 reader push left
the packer's private count at zero. After the initial16 tiles were acknowledged,
the first feedback push published16 rather than32; the next token therefore saw
zero available tiles and waited. Full-ring pointer alignment alone was insufficient.

The candidate seeds CB5's packer-local `tiles_received` to `kv` once before the
token loop, using `PACK(...)` so only the producer thread updates its private
counter. It does not publish shared availability early. The reader remains the
only producer until its initial full-ring push; initial-state consumption and the
normal math dependencies precede compute's first feedback push. All subsequent
CB5 production is compute-only. Math, BF16 rounding, buffers and writer are unchanged.

Because this touches runtime internals, norm/gate execution now checks the SHA256
of `llk_io_pack.h` and `dataflow_api.h` (recorded in `HANDOFF_HASHES`) and fails
closed on drift. The fixed local T2 run `20260906T070713Z-307` passed exact output
and every prefix state on both simulated chips, unchanged initial state and clean
close. All15 three-seed T1/2/4/8/16 fixtures subsequently passed with exit0 and
`last_stage=complete`, exact outputs/every prefix state on both simulated chips,
unchanged initial states and clean close. Generated compute SHA256 is
`9512188a20fd63f2853f1ac427f1b7dfaf96bab18f9bfb32e73c2d647283a9e0`;
reader/writer hashes and the recurrence-only kernels remain unchanged.
This is simulator evidence against
serial T1 of the same kernel, not hardware/native-oracle certification or speedup.
No physical-card kernel job or reset was dispatched for this fix.

Reports/logs/zero exit-status files are in `/opt/ttsim/results`, with these run IDs:

| Seed | T1 | T2 | T4 | T8 | T16 |
| --- | --- | --- | --- | --- | --- |
| 0 | 20260906T070857Z-312 | 20260906T070911Z-451 | 20260906T070929Z-500 | 20260906T071004Z-591 | 20260906T071105Z-682 |
| 1 | 20260906T071300Z-784 | 20260906T071312Z-830 | 20260906T071330Z-876 | 20260906T071403Z-932 | 20260906T071458Z-978 |
| 2 | 20260906T071651Z-1138 | 20260906T071703Z-1188 | 20260906T071721Z-1234 | 20260906T071751Z-1280 | 20260906T071847Z-1326 |

### Simulator-first kernel debugging and second authorized recovery

Recovered-device run 34016749007 (`ae5b71b`) timed out at seed0/T2
`candidate-warm-submitted`; all three stacks point to synchronization after the
custom kernel. Both native T2 oracle tokens and native continuation references had
completed. T1 passed eager/trace, four continuations and its negative control.
This isolates the original stall to candidate warm-up, before T2 trace capture;
it does not yet identify the internal deadlock mechanism.

At the operator's renewed request, recovery run 34017283126 reused the guarded
`9c4eaed` recovery workflow. Reset/reinitialization of devices0/2 completed at
06:45:20 UTC; all12 transfer checks and clean mesh close passed (0.391 s).
No further physical-card kernel run is authorized by this checkpoint: first use
the [simulator-only harness](../optimisation/sim/README.md).

The harness uses the same hash-checked source transformation and generic-op builder
with two simulated Blackhole chips, slow dispatch and disabled SFPLOADMACRO. It
compares all output/prefix states against serial T1 executions of the same kernel;
this is not an independent native-oracle certification. The existing local runtime
can compile the generated source without installing the production GDN op.
Initial local run `20260906T064618Z-338` completed both serial T1 calls and reached
T2 readback, where repeated stack dumps show it waiting until the bounded process
failed at the timeout. This is not a simulator pass. Recurrence-only T2 control
`20260906T064957Z-301` then passed exact output/prefix-state comparisons on both
simulated chips, unchanged initial state and clean close. The local source-built
extension hash is `9fa83a78556aaf99e3cf59f59e44199d8ffa1839b09505e399c26ad2018de5aa`.
This provides a local failing norm/gate case and passing recurrence control; it
does not prove the internal cause. Hardware remains idle while the circular-buffer
protocol is investigated locally.

### Authorized recovery (passed 34016364842)

Run [34016364842](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34016364842)
(`9c4eaed`) verified the two expected boards and PCI functions, then reset host
devices 0 and 2 once with installed `/home/thatch/.local/bin/tt-smi`. Reset and
reinitialization completed at 06:25:32 UTC. The subsequent health test passed all
12 exact readbacks on both chips, synchronization and clean mesh close (0.450 s).
The 9326-byte artifact confirms transfer health recovery, not norm/gate correctness.
Next rerun the unchanged instrumented norm/gate suite without reset authorization
to locate the original post-T1 stall on recovered devices. No automatic reset loop.

The operator explicitly approved controlled card reset through CI after health
run 34015497253 (`3db69df`) failed during mesh opening with `Device 0 init: failed
to initialize FW! Try resetting the board.` Zero transfer checks executed; the
7981-byte artifact confirms firmware initialization failure, not the original
post-T1 kernel stall's cause. Runner was online and idle.

`device-recovery` requires a separate default-false reset authorization input and
exclusive card allocation, verifies the exact known board IDs, exactly two TT PCI
functions and no running containers with accelerator-capable mappings. It uses
installed host TT-SMI once, without installing software or changing services.
Noninteractive privilege/tool failures stop before reset. A successful command
must be followed by all 12 transfer checks and clean close before recovery is
declared. There are no automatic reset retries, reboots or firmware updates.

### Transfer health isolation (failed 34015497253)

Diagnostic run 34014926676 (`5cdd443`) timed out before any native oracle or
custom kernel: all three stack dumps identify `host(initial)` / `ttnn.to_torch`
at seed 0 / T1. The 10114-byte artifact's last stage is `fixture-upload`.
This does not reproduce or explain the original post-T1 stall. Residual unhealthy
device/dispatch state after forced termination is a hypothesis, not a confirmed cause.

`device-readback` isolates transfers in the same pinned image and 1x2 fabric mesh.
It uploads BF16 TILE DRAM tensors of shapes `[1,1,32,32]` and `[1,24,128,128]`,
using three distinct exact integer patterns and reading each chip separately.
Twelve exact checks, synchronization and successful mesh close are required.
Durable stages identify the shape/pattern/chip; stack dumps repeat every 30 seconds.
The process is bounded to 180 seconds plus 15-second kill grace. No model imports,
weight loading, trace capture, custom kernels, card resets or serving changes.
Do not retry the GDN experiment until this gate passes or recovery is authorized.

### E4 multi-token norm/gate prerequisite (timed out 34013517498)

Run 34013517498 (`3e0846f`) timed out after seed 0 / T1 passed eager, trace,
all four restored continuations and the stale-state control. The 11311-byte
artifact does not locate the next blocking call; T2 and larger are not certified.
The unused-reader-helper compiler warning is nonfatal. A diagnostic retry preserves
the same kernels, writes durable stage markers before blocking phases, and dumps
Python stacks every 120 seconds. Its test timeout is reduced to 420 seconds plus
a 30-second kill grace. Do not infer a CB5 deadlock or blame trace allocation from
the last printed line alone; native oracle execution also occurs before each candidate.

`gdn-multitoken-norm` extends the token loop with the native fused output math,
including BF16 rounding of norm-times-weight, SiLU and final multiply. CB30/31
retain their native norm-weight/scratch roles. Recurrent feedback moves to CB5:
the reader pushes exactly one complete 16-tile initial-state ring, compute consumes
it, then compute alone produces/consumes full-ring BF16 feedback blocks. No reader
state pushes occur for subsequent tokens. CB storage is 630784 bytes/core.

Each of 24 head-owning cores assembles its T rows into four local output tiles,
zeroing padding and writing full pages to `[1,T,3072]` TILE L1 after the loop.
No cross-core assembly semaphore is needed for this fixed T<=16 geometry. Every
prefix state is still exported to DRAM. The native oracle uses L1 state, so this
first gate measures correctness only, not comparative speed. Required coverage:
30 eager/trace cases, 216 exact restored-prefix continuations, 15 stale-state controls.
Synthetic QKV/beta/g/z and norm weights only; convolution, real model weights,
device rollback integration and full-model performance remain subsequent gates.

### E4 paired recurrence latency (passed 34013199242)

Run [34013199242](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34013199242)
(`2272b91`) passed 30 exact cases, 216 continuations, 15 negative controls and all
1800 timed replays. Across three seeds, T16 serial medians were 0.531716-0.532623 ms
versus 0.339825-0.340785 ms for the device loop (about 36% lower median latency).
All nine T16 paired blocks favored the candidate, but spikes affected both arms:
per-seed median paired ratios ranged 1.169-1.620. Do not hide those spikes or treat
the ratio of marginal medians as the paired estimator. The 14710-byte artifact is
recurrence-only evidence, not a full-model or committed-token speedup.

The `gdn-multitoken` suite now retains the passing correctness matrix and adds
native serial B1 versus device-token-loop captured traces. Both arms export every
token output and every prefix state, with an immutable initial state. Three ABBA
blocks, ten blocking replays per sample, across three seeds and T1/2/4/8/16 yield
1800 timed replays. Each arm is warmed and validated before timing; every timing
sample is followed by exact output/all-prefix/initial-state checks on both chips.
Capture, compilation, input packing/upload, host reads and continuation checks are
outside timing. Report raw samples and paired block ratios; these are fenced host
trace latencies, not device-only kernel time or committed-token/model throughput.
The verified result above is bounded to this synthetic recurrence fixture.

### E4 multi-token recurrent kernel prerequisite (passed 34012883902)

Run [34012883902](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34012883902)
(`855f8de`) compiled and passed on both cards: 30 exact seed/width/eager-trace
cases, 216 exact restored-prefix continuations and 15 detected stale-state negative
controls. The artifact is 12602 bytes. This certifies the bounded recurrence-only
prototype below, not full GDN integration or a performance improvement.

`gdn-multitoken` is the first true device-side token-loop prototype, not another
Python unroll. It derives three kernels from hash-pinned native source without
patching the installed runtime. Each of 24 head-owning cores per chip processes
T1/2/4/8/16 sequential packed rows. The reader loads the initial recurrent state
only for token zero. Compute retains subsequent state in a dedicated 16-tile BF16
L1 feedback ring (32 KiB/core), preserving the native inter-token state rounding;
FP32 math scratch is not silently carried as persistent higher-precision state.
The writer exports every token output and every prefix state to disjoint buffers.
Total configured CB storage is 618496 bytes/core, including feedback. Compilation
and resource compatibility passed for this configuration on the paired cards.

This initial operator gate uses synthetic packed QKV/beta/g inputs and the native
packed recurrence without fused output norm/gate as oracle. It does not include
real weights, convolution, output normalization/gating, attention or full-model
integration. Native arithmetic statements are preserved; strict replacement anchors
change only iteration/state supply/feedback. Independent initial state must remain
unchanged, and every emitted state/output must equal the corresponding serial B1
result on both chips. Reloaded prefix states must produce exact native continuation
outputs and states. Inputs are replicated synthetic fixtures, not coding evaluation.

Required coverage: 30 seed/width/eager-trace cases, 216 restored-prefix continuation
checks and 15 stale-state negative controls. No performance timing or throughput
claim is attached to this prerequisite. Subsequent gates must add native norm/gate,
real-weight conv integration, device checkpoint/rollback integration and paired
full-model timing against run 34011273093 before adoption. Keeping all-prefix DRAM
exports intentionally preserves observability; this prototype removes repeated
state reads but does not yet eliminate checkpoint writes.

### E3 remaining-work attribution (passed 34004475100)

Run 34004475100 (`f4443cf`) passed all four T1/T16 context fixtures, including
three fenced passes, unfenced eager controls and three captured replays each.
All checked full logits, active GDN state and complete valid KV prefixes matched
native serial B1 exactly. The artifact is 167635 bytes, without raw profiler dumps.

| Context | T | Trace median ms | Unfenced eager median ms | Fenced root median ms | GDN native-row exclusive median ms |
| --- | --- | --- | --- | --- | --- |
| 4095 | 1 | 44.997 | 95.125 | 127.052 | 18.613 |
| 4095 | 16 | 235.188 | 670.426 | 783.768 | 421.565 |
| 16383 | 1 | 45.589 | 145.552 | 264.602 | 33.916 |
| 16383 | 16 | 243.955 | 828.717 | 1006.306 | 563.365 |

Each stage median is computed independently across three passes. Fences and host
dispatch substantially inflate and vary these intervals: do not divide the GDN
numbers by trace time or interpret them as device critical-path percentages.
The repeated trace controls corroborate the previous 235/244 ms block costs;
this run does not demonstrate a speed improvement or committed-token throughput.

The pinned native GDN source shows that B1 input with Bmax8 slices recurrent state
and invokes `_write_recurrent_state_prefix` for every token. Consequently
`QWEN_GDN_FUSED_INPLACE=1` does not request in-place recurrence on this path:
the native guard requires B == Bmax. The prefix writer already avoids idle rows
using a sharded active-prefix write, so this is **not** a whole-B8 copy finding.
Conv/gates and recurrence/norm/gate math are already fused; the remaining question
is how much redundant movement surrounds them.

The second bounded attribution revision (passed 34004870686, `28b8dd4`) adds separate coverage for
projected-row slice/clone, fused conv/gates, packed recurrence/norm/gate, active
state slicing and active-prefix writeback. Each must engage exactly 48*T times.
It uses only cloned function namespaces and reversible instance wrappers, preserving
native state geometry, precision, kernels and execution order. An active-B1
working-state/in-place candidate remains a hypothesis, not an implemented speedup;
it will need exact rollback, continuation, idle-slot and trace-address gates.

All four fixtures and twelve fenced passes passed exact logits/state/valid-KV
checks. Each of the five new categories engaged exactly 48*T times in every pass.
The downloaded artifact is 228636 bytes. T16 trace medians remain 235.112 ms at
4095 tokens and 243.885 ms at 16383 tokens; this is instrumentation, not a speedup.

| T16 fenced exclusive category | 4095-token median ms | 16383-token median ms |
| --- | --- | --- |
| Projected-row slice/clone | 87.957 | 90.081 |
| Fused conv/gates | 137.818 | 138.192 |
| Packed recurrence/norm/gate | 97.843 | 97.943 |
| Active recurrent-state slice | 31.583 | 31.730 |
| Active recurrent-state writeback | 62.328 | 62.268 |

These independent medians include fences/dispatch and must not be subtracted from
trace time to predict savings. Fenced root medians were 732.041/734.234 ms versus
unfenced eager medians of 439.437/425.633 ms. The native-row exclusive category now
excludes its five measured children and is not directly comparable with revision 1.

The next implementation candidate keeps an isolated active recurrent working state
through the T-token GDN block, enabling native B1 in-place updates instead of 768
active-state slice/write pairs across 48 layers. Copy into/out of that working
state at bounded block/checkpoint boundaries, preserving native live-buffer addresses
and idle slots. First certify a single-layer candidate with every rollback prefix
and corrected continuation in eager/trace; only then integrate and run paired
full-model timing. Keep projected-row clone removal separate: ownership/aliasing
must be established before deleting it. No arithmetic or precision changes are
justified by this attribution pass, and no serving promotion has been made.

### E3 isolated in-place working state (passed 34005306635)

Run 34005306635 (`7625f31`) passed 216 exact prefix/continuation checks, all 30
stale-state negative controls, all 15 isolation checks, and 279 in-place request
and both-chip buffer-alias checks. This certifies the tested single-layer state
geometry change, not full-model integration. T16 single-sample trace times were
3.750/3.749/3.746 ms across three seeds, including all-prefix staging. These are
diagnostics without a contemporaneous paired control; do not infer a speedup.

`gdn-inplace-timing` (completed 34005485199, `94c5726`) reruns the full correctness gate and
captures a control with the same batched input/output projections and all-prefix
checkpoint semantics, but native B1-in-B8 state handling. Candidate traces include
working-state entry/publication transfers. Each seed/width uses three ABBA blocks,
ten replays per arm sample, restoring identical initial live state outside every
timed replay. Full outputs, final state including idle slots, and every prefix
checkpoint must match the serial oracle after each sample. Fifteen fixtures and
1800 timed replays are required; report paired block-cost ratios, not tok/s.

All 15 timing fixtures and 1800 timed replays completed with exact validation.
The gate again passed 216 prefix/continuation checks, 30 negative controls,
15 isolation checks and 279 in-place request/alias checks. However, the performance
result rejects promotion of this candidate: all 15 fixture median paired ratios
were below 1.0. T16 was consistently slower across every ABBA block and seed.

| T16 seed | Control mean ms | Candidate mean ms | Median paired control/candidate | Paired slowdown |
| --- | --- | --- | --- | --- |
| 0 | 3.607 | 3.764 | 0.956 | 4.61% |
| 1 | 3.591 | 3.753 | 0.955 | 4.69% |
| 2 | 3.564 | 3.758 | 0.944 | 5.89% |

Means summarize six samples per arm; slowdown uses the median of the three paired
ratios, not the ratio of the displayed means. Shorter widths are noisier but none
has a winning median ratio. A green CI run certifies completion/exactness, not a
performance improvement. Keep native B1-in-B8 state handling as the control; do
not integrate this all-prefix working-state variant into the full model.

The precise cause is not isolated. This comparison includes compact entry/exit
transfers and different checkpoint implementations: five generic compact copies
per prefix versus one direct active-slot DMA in the control. Next distinguish
checkpoint overhead from recurrence savings using paired no-checkpoint and
end-prefix-only diagnostic arms, with identical checkpoint semantics per pair.
The end-prefix arm would match the current full-model timing fixture, but neither
arm alone certifies dynamic rollback or a complete speculative commit pipeline.
Retain the full all-prefix correctness gate; do not hide the failed all-prefix
performance result or predict savings from fenced eager attribution.

`gdn-checkpoint-cost` implements that diagnostic (passed 34005748778, `5fac974`). It retains
the all-prefix correctness/rollback/continuation/isolation matrix and reruns the
all-prefix paired control alongside `none` and `end` checkpoint policies. Both
arms use the same policy; compact entry/publication transfers remain timed even
when checkpoints are disabled. Native fused arithmetic and precision are unchanged.
Each warmup/capture asserts the exact checkpoint callback sequence. End-only
staging is overwritten with the initial active state outside every timed replay,
and that poison is required to differ from the expected end state, preventing a
missing checkpoint write from passing on stale results. Every timing sample checks
all outputs and full final live state, plus the selected checkpoints.

The expanded matrix requires 45 seed/width/policy fixtures, 5400 timed replays and
651 eager/capture in-place request/alias checks. Report policy-specific paired
ratios rather than mixing samples or treating reduced-checkpoint diagnostics as
a complete speculative verifier. This can separate checkpoint-related costs from
the combined recurrence/working-state-transfer cost; it does not isolate recurrence
kernel latency alone. No full-model integration or serving promotion is enabled.

Run 34005748778 passed all 45 fixtures / 5400 timed replays, 216 exact prefix and
continuation checks, 30 negative controls, 15 isolation checks and 651 in-place
request/alias checks. T16 paired control/candidate ratios across seeds were:

| Policy | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| All-prefix | 0.942 | 0.942 | 0.942 |
| None | 1.060 | 1.062 | 1.057 |
| End-only | 1.050 | 1.085 | 1.052 |

End-only candidate means were 3.302/3.301/3.299 ms. Every T16 none/end paired block
favored the candidate, while all-prefix remained slower. This supports checkpoint
cost as the factor erasing the benefit in this fixture. Policies run sequentially,
so do not interpret cross-policy differences as a precise isolated kernel cost.
These are layer block-cost ratios, not full-model or committed-token throughput.

### E3 compact checkpoint DMA (passed 34005970668)

Run 34005970668 (`07f2b59`) passed 45 fixtures / 5400 timed replays, 216 exact
prefix/continuation checks, 30 negative controls, 15 isolation checks, 651 in-place
alias checks and 354 compact checkpoint DMA calls. T16 median paired ratios across
seeds were 1.042-1.054x with all-prefix checkpoints, 1.054-1.060x end-only and
1.058-1.062x without checkpoints. The all-prefix median regression is removed,
although one of nine T16 all-prefix paired blocks was slightly below 1.0 (0.996x).
T1 remained slower across all policies and seeds. These are single-layer gains,
not a measured full-model improvement or committed-token rate.

`gdn-checkpoint-dma` changes only compact checkpoint saving from five generic
copies to one mesh generic-op invocation, reusing the existing unchanged 48-core
data-movement kernel. A new strict compact-to-compact mode validates both ends as
one recurrent state plus four conv taps, BF16 interleaved DRAM tiles, with ten
non-aliased buffers per chip. Existing active-slot transfers remain a separate
full-to-compact/compact-to-full mode. Recurrence copies complete tiles; conv copies
only the two logical-row face segments, not padded rows. No arithmetic or native
fused kernel changes are made; padding is not promoted to logical model state.

The hardware gate repeats all three checkpoint policies, the full exactness matrix
and 5400 paired replays, with 354 compact checkpoint DMA calls required during
warmup/capture/eager (trace replay does not increment Python counters). It also
records the unchanged C++ kernel hash and new Python adapter hash. The old generic
copy path remains selectable; no speedup or full-model integration is assumed.

### E3 full-model compact GDN integration (passed 34006233354)

Run 34006233354 (`1be2572`) passed 20 width/mode full-model checks, 16 rollback
cases with corrected continuations, four stale-GDN/wrong-page negative-control
pairs, and ten exact timing fixtures with 2400 timed decode replays. Full active
logits, active GDN state, selected end checkpoints and complete valid KV prefixes
matched the native serial oracle. Both 4K/16K contexts passed eager and trace gates.

| Context | T | Batched native-state control ms | Compact candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 1 | 45.025 | 45.023 | 1.000x |
| 4095 | 2 | 58.764 | 57.930 | 1.014x |
| 4095 | 4 | 84.246 | 82.031 | 1.027x |
| 4095 | 8 | 134.541 | 130.705 | 1.029x |
| 4095 | 16 | 235.197 | 227.075 | 1.036x |
| 16383 | 1 | 45.629 | 45.594 | 1.001x |
| 16383 | 2 | 59.900 | 59.065 | 1.015x |
| 16383 | 4 | 86.480 | 84.227 | 1.026x |
| 16383 | 8 | 138.942 | 135.106 | 1.028x |
| 16383 | 16 | 243.981 | 235.797 | 1.035x |

Times are means over six samples per arm; ratios are medians of three paired
ABBA blocks. Every T>1 paired block favored compact state. T1 explicitly did not
enable compact state; tiny timing differences there are not optimization gains.
T16 saves about 8.1-8.2 ms per full-logit block against the existing batched control.
This is a measured full-model verification gain, not committed-token throughput.
With perfect acceptance and zero drafting/selection/commit overhead, 16 divided
by these block times gives only 70.46/67.85 tok/s for this T16 implementation.
Reaching 200 requires blocks below 80 ms even before that overhead; this improvement
does not close the remaining gap or establish a hardware-wide speed ceiling.

Retain the opt-in exact candidate for subsequent experiments; do not change serving
defaults. Next investigate projected-row preparation/copy overhead and remaining
per-token work using the new candidate as a contemporaneous control. Keep attention
numerics and GDN precision unchanged, and require full-model paired gains rather
than extrapolating layer timing or fenced attribution.

### E3 redundant GDN input preparation (passed 34007693367)

Run 34007693367 (`8b6f429`) passed 20 width/mode exact checks, 16 rollback cases,
four negative-control pairs and ten timing fixtures with 2400 timed decode replays.
The contemporaneous control already uses compact GDN state and checkpoint DMA.

| Context | T | Compact control ms | Reused-input candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 4 | 82.033 | 77.671 | 1.056x |
| 4095 | 8 | 130.708 | 118.293 | 1.105x |
| 4095 | 16 | 227.118 | 201.741 | 1.126x |
| 16383 | 4 | 84.396 | 80.090 | 1.053x |
| 16383 | 8 | 135.122 | 122.758 | 1.101x |
| 16383 | 16 | 235.884 | 210.584 | 1.120x |

Every T16 paired block favored input reuse. T2 showed no meaningful median gain;
T1 remained unchanged. T16 saves about 25.3 ms versus the compact-state control.
These full-model verification gains preserve exact logits, state, valid KV and
corrected rollback but do not include drafting or a complete commit pipeline.
Idealized perfect-acceptance/zero-overhead T16 bounds are now 79.31/75.98 tok/s,
still well short of 200. No serving defaults are changed.

Next, inspect pinned slice/copy and tensor ownership code before attempting to
remove the projected-row clone. The `gdn-source` exporter now includes bounded
source-only ownership roots (maximum 200 files / 4 MiB additional source), recording
missing optional roots rather than assuming them. This audit uses no cards and
does not execute model code. The existing clone path remains unchanged until its
aliasing and deallocation behavior can be established from the actual image.

### E3 projected-row layout hoisting (passed 34011273093)

Run 34011273093 (`6ded69e`) passed 20 width/mode checks, 16 corrected rollback
cases, four negative-control pairs and ten exact timing fixtures / 2400 timed
decode replays. Full active logits, GDN state, selected end checkpoints and valid
KV matched the native serial oracle. T1 did not enable the optimization.

| Context | T | Selective-clone control ms | Hoisted-layout candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 2 | 57.554 | 56.142 | 1.026x |
| 4095 | 4 | 76.834 | 72.868 | 1.054x |
| 4095 | 8 | 116.988 | 106.337 | 1.100x |
| 4095 | 16 | 197.494 | 173.065 | 1.141x |
| 16383 | 2 | 58.635 | 57.281 | 1.024x |
| 16383 | 4 | 79.062 | 75.086 | 1.053x |
| 16383 | 8 | 121.388 | 110.716 | 1.096x |
| 16383 | 16 | 206.296 | 181.846 | 1.135x |

Every T>1 paired block favored layout hoisting. T16 saves about 24.4 ms against
the contemporaneous selective-clone control. This is a full-model verification
gain, not committed-token throughput. Perfect acceptance with zero drafting/commit
overhead gives only 92.45/87.99 tok/s at T16; the 200 goal remains unmet.
The bounded layout arm is complete. Retain the opt-in candidate as the next control
and prioritize E4's true multi-token state-retaining recurrence. No serving defaults
are changed and no new hardware experiment is dispatched by recording this result.

`full-gdn-row-layout` converts the packed projection to row-major once per T>1
block rather than allowing each nonzero TILE row slice to repeat that conversion.
It slices B1 row-major tensors and tiles each output separately. Both-chip ownership
guards require independent converted/source/sliced/output buffers before releasing
intermediates. The first-row clone remains; T1 and default behavior are unchanged.
Conversion setup failures release the original projection without changing the hook.

The direct paired control retains compact state, checkpoint DMA, input reuse and
selective clone removal from 34009858516. Layout conversion adds live scratch and
changes padding preparation, so savings and correctness must not be assumed.
The full 4K/16K exactness/rollback gate precedes 2400 timed decode replays. Once this
bounded arm is resolved, prioritize E4's multi-token state-retaining recurrent kernel.

### E3 selective projected-row clone removal (passed 34009858516)

Run 34009858516 (`215b933`) passed 20 width/mode checks, 16 corrected rollback
cases, four negative-control pairs and ten exact timing fixtures / 2400 timed
decode replays. Slice ownership guards and per-layer clone-skip engagement passed.
The paired control already uses compact state, checkpoint DMA and input reuse.

| Context | T | Reused-input control ms | Selective-clone candidate ms | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 2 | 57.944 | 57.508 | 1.008x |
| 4095 | 4 | 77.665 | 76.868 | 1.010x |
| 4095 | 8 | 118.283 | 116.995 | 1.011x |
| 4095 | 16 | 201.794 | 197.479 | 1.022x |
| 16383 | 2 | 59.048 | 58.648 | 1.007x |
| 16383 | 4 | 79.885 | 79.052 | 1.010x |
| 16383 | 8 | 122.857 | 121.524 | 1.011x |
| 16383 | 16 | 210.808 | 206.547 | 1.020x |

Every T>1 paired block favored clone removal. T1 explicitly kept cloning and
native state handling; its small timing variation is not a gain. T16 saves another
4.3 ms per full-model verification block. Full active logits/GDN state/end
checkpoints/valid KV and corrected continuations remain exact. The opt-in candidate
is suitable as the next experiment control, not an automatic serving promotion.
Idealized perfect-acceptance/zero-overhead T16 bounds are 81.02/77.46 tok/s, still
far from 200 committed tok/s. Current evidence does not justify extrapolating to
a pure kernel speedup or removing the first-row alias-protection clone.

Ownership audit 34009341359 passed and exported the pinned slice implementation.
The no-op path can return input storage; nonzero row starts within these T<=16
tiles take the row-major slicing path, whose device op allocates a fresh output
when no preallocated output is supplied. Native GDN consumes/deallocates the
projected row, so the first-row clone is retained conservatively, including T1.

`full-gdn-row-clones` skips only clones after row zero. The exact slice frontend
and device-op source hashes are required before model startup; every skipped clone
also requires distinct source/destination buffer addresses on both chips. A
failure aborts rather than silently falling back or reporting a fake optimization.
Each active compact layer must skip exactly T-1 clones per eager/capture invocation,
removing 720 clone calls at T16. Native slice, fused math and precision are unchanged.

The paired control is the current compact/DMA/reused-input candidate, not the
older native-state baseline. Full-logit/GDN/valid-KV/rollback exactness at 4K/16K
must pass before 2400 timed replays. T1/default paths stay unchanged. Ownership
source evidence and local tests do not substitute for hardware exactness, trace
replay or a measured speedup; no serving promotion is enabled.

`full-gdn-input-reuse` isolates input-row preparation against the exact full-model
compact/DMA control from 34006233354. Batched projection already consumes the full
input and the projected-row hook supplies each token's distinct projected data.
Pinned native decode otherwise reads its input only for shape/reshape preparation.
The candidate therefore prepares one B1 input row per layer and reuses that
shape-only argument across T native row calls, instead of preparing T input slices.
At T16 this removes 720 input-slice calls across 48 layers; it does not remove
projected-row slicing/cloning or change their ownership. A source AST guard rejects
new input consumers outside shape/projection preparation, in addition to pinned
source hashes and existing exact projection-hook engagement checks. One owned
input tensor is released once, not once per repeated reference. T1 stays unchanged.

The gate repeats full-logit/GDN/valid-KV exactness and corrected rollback at 4K/16K
before 2400 timed replays. Its direct paired control now also enables compact state
and checkpoint DMA: only distinct versus reused input rows differ. Reports identify
that control explicitly; do not mislabel this as another native-state comparison.
Both simultaneously captured arms allocate compact working sets (192 MiB/chip
combined at T>1, separate from shared snapshots). No speedup or serving-default
change is assumed before hardware evidence.

`full-compact-gdn` opts the static verifier into compact state plus direct compact
checkpoint DMA for T2/4/8/16 across all 48 GDN layers. T1 and the default path keep
native B1-in-B8 state handling. B1 SDPA reads, projection batching, model precision
and live native buffer ownership are unchanged. Every eager/capture invocation
requires all 48 compact layers to perform exactly T in-place updates and one
selected-prefix checkpoint. Working sets are fixture-owned, preallocated before
capture and released only after their traces; they add 96 MiB per chip at T>1.

The gate repeats 4095/16383-token full-logit/state/valid-KV exactness at all widths
in eager/trace and T16 rollback prefixes 0/1/8/16 with corrected continuations and
negative controls. Only after that matrix passes, timing captures native serial,
the existing batched native-state control, and the compact candidate. Both batched
arms include one end-prefix checkpoint, checked against native serial state after
each sample. Restore remains outside each timed replay; no prefill occurs while
candidate trace buffers are live. Three ABBA blocks compare compact versus the
existing batched control in addition to the serial-versus-batch comparison.
Ten fixtures require 2400 timed decode replays across the two comparisons, plus
separate restore measurements. Do not extrapolate layer gains or call verifier
block throughput committed tok/s. No serving promotion or default change is made.

`gdn-inplace` runs the existing real-weight single-layer matrix with an opt-in
compact B1 working set (recurrence and all four conv taps), copied from native B8
slot zero once per block. Instance-local B/state bindings enable the native
in-place guard during the sequential fused rows. A cloned function namespace
requires the in-place request and verifies returned recurrence buffer addresses on
both chips at every eager/capture call. No installed source or math is changed.
All-prefix checkpoints copy the compact state directly; final publication uses
the already-certified active-slot DMA and restores native B8 Python bindings even
on failure. Working tensors are preallocated and remain stable across trace replay.

The gate requires 216 exact prefix/continuation checks over three seeds and
T1/2/4/8/16 in eager/trace, 30 stale-state negative controls, full-state equality
immediately after each candidate block (including idle slots), stable live
addresses, and 279 successful in-place wrapper calls across warmup/capture/eager.
Trace replays execute captured device work, not Python counters. The existing
15 active-restore isolation cases remain enabled. This changes working-state
geometry, so local tests alone do not certify native-kernel compatibility or
numerical equivalence. Reported single-sample layer times are diagnostic only;
paired measurements above establish a performance regression. Full-model
integration remains withheld pending a genuinely faster exact candidate.

`full-batch-attribution` targets the remaining 235-244 ms T16 path with a bounded,
JSON-only in-situ profiler, avoiding the historic multi-GB raw profiler exports.
It compares T1/T16 at 4095/16383 tokens, three eager profiling passes each, with
separate unfenced eager and captured trace controls. Exact full logits, active GDN
states and the complete valid KV prefix are checked against native serial B1.

Instance-local wrappers distinguish GDN input/output projections, each native
fused row, selected-prefix checkpoint, attention projections/KV/SDPA with row
packing, decoder direct-call components, final norm and LM head. Nested exclusive
times reconcile to the full-model root; call-count gates require all 64 layers,
48*T GDN rows and all attention paths. The profiler synchronizes the mesh at each
boundary, so it changes overlap and includes dispatch/CPU overhead. Report fence
calibration and unfenced/trace controls; do not treat category sums as a decomposition
of the earlier traced 235-244 ms critical path or as a throughput improvement.

### E3 coding-context verification cost (passed 34002876975)

`full-coding-cost` extends exactness to 4095/16383-token deterministic repeated-code
fixtures, crossing the 4K/16K boundaries during verification. It checks all T widths
in eager/trace and rollback prefixes 0/1/8/16 with corrected continuations. Valid KV
hashing now covers the whole request prefix in bounded 64-page chunks across both
chips and all attention layers; no raw KV/logit arrays are exported. This is a
synthetic context-length probe, not a coding-quality benchmark.

Only after the full correctness matrix passes, compare captured native serial B1
and batched full-logit blocks using three ABBA blocks, ten replays per sample. Every
replay restores the initial GDN state outside timing; captured candidate execution
includes one preselected end-prefix checkpoint. Full logits/state/KV are rechecked
before timing and after each sample. Separately measure captured active-state
restore. These measurements exclude drafter generation, dynamic acceptance,
all-prefix checkpoint staging/selection and a complete speculative commit pipeline.
Report block milliseconds and paired ratios, not committed-token throughput.

Attempt 34002210390 (`c31021a`) found a real long-context divergence: at length
4095, eager T=1 and T=2 passed exact logits/state/KV, but T=4 differed in 921936
logit values per chip (maximum absolute difference 0.59765625). Timing was not
entered. The previous short-context pass therefore cannot justify long-context use.

The next controlled arm changes only the SDPA read in the candidate: keep batched
QKV preparation/output projection and serial cache writes, but run native B1 SDPA
for each query before concatenating results. B1 query geometry preserves the native
per-query grid/reduction path, unlike a larger batch that shares cores among rows.
This is an isolation hypothesis until hardware passes; the exactness gate is not
relaxed. `full-coding-cost` now explicitly selects `--serial-sdpa`; other suites
retain their original batching policy.

Retry 34002555664 (`c74b2be`) passed all five 4K eager widths, with exact logits,
active GDN states and complete valid KV prefixes, plus the four eager rollback
cases. This controlled result isolates the earlier eager drift to the batched SDPA
read path. The first T1 traced candidate then produced non-finite logits after the
harness replayed long prefill between candidate capture and execution. This is a
separate trace-lifetime problem, not a reason to relax arithmetic comparison.

The next retry saves candidate-entry GDN state in a fourth preallocated compact set
and restores it in-place after capture, without replaying prefill while candidate
trace buffers are live. KV writes touch only candidate/future positions, which the
position mask excludes until rewritten; exact KV checks remain mandatory. Timing
also prefills before allocating candidate fixtures. Snapshot allocation is now
384 MiB/chip. Structural regression tests guard both trace/prefill boundaries.

Run 34002876975 (`6937529`) passed the complete selected matrix: 20 width/mode
checks, 16 rollback cases, four native baseline mode checks and four negative-control
pairs. All full active-token logits, active GDN states and entire valid KV prefixes
matched exactly on both chips. Ten timing fixtures passed all pre/post checks,
covering 30 ABBA blocks and 1200 individually state-reset timed replays.

| Context | T | Serial block mean (ms) | Candidate block mean (ms) | Median paired speedup |
| --- | --- | --- | --- | --- |
| 4095 | 1 | 43.799 | 44.985 | 0.973x |
| 4095 | 2 | 87.650 | 58.716 | 1.493x |
| 4095 | 4 | 175.233 | 84.217 | 2.081x |
| 4095 | 8 | 350.618 | 134.474 | 2.607x |
| 4095 | 16 | 701.383 | 235.098 | 2.983x |
| 16383 | 1 | 44.349 | 45.555 | 0.974x |
| 16383 | 2 | 88.756 | 59.833 | 1.483x |
| 16383 | 4 | 177.423 | 86.422 | 2.053x |
| 16383 | 8 | 354.974 | 138.879 | 2.556x |
| 16383 | 16 | 710.095 | 243.876 | 2.912x |

Separate active-state restore averaged 0.866-0.867 ms. Candidate timing includes
batched projections, sequential native fused GDN recurrence, B1 SDPA reads, ordered
KV writes, native MLP/norms/LM head and one preselected end checkpoint. T1 is slower
than native serial, so this wrapper is not a replacement for the normal B1 path.

**200 tok/s assessment:** the optimistic `16 / block_seconds` bound is only
68.06 tokens/s at 4K and 65.61 at 16K for this implementation, assuming perfect
acceptance and zero drafter/selection/commit overhead. These are derived ceilings,
not measured committed throughput or hardware-wide limits. A 200-token/s T16 path
would need at most 80 ms per block before overhead: another 2.94-3.05x reduction in
the measured candidate cost. The previous 8.45x result was one short-context
attention layer, not this full-model/long-context execution policy.

Next prioritize attribution of remaining per-row GDN and B1 attention work and
evaluate a time-fused GDN block or B1-equivalent parallel attention kernel against
this exact oracle. A neural drafter alone cannot overcome this target-path bound.
All-prefix checkpoint staging, dynamic acceptance and true commit costs still need
implementation; no serving promotion is enabled. Local c31021a passed 113 CI/helper
and 26 drafting tests; subsequent affected helper/trace-boundary tests passed on
Windows Python 3.11 after WSL startup timed out (no VM/service reset attempted).

### E3 full-model batch integration (passed 34001403540)

The `full-batch` suite reuses the serial oracle without changing its default path.
The candidate invokes the native 64-layer `_forward_decode` with instance-local
GDN and attention adapters. It retains native normalization, residuals, MLP and LM
head. Each GDN layer batches input/output projection, runs the pinned fused native
recurrence per token, and saves its own active state at the requested prefix;
attention batches arithmetic with ordered B1 shared-page writes. A third reusable
all-layer compact snapshot set holds candidate checkpoints separately from reference
and scratch, so the reference snapshot cannot accidentally repair the candidate.

Gate: 30 full-logit/state/KV comparisons (T=1/2/4/8/16, eager/trace, prompt
lengths 63/64/65), then all 102 T=16 prefix rollback cases and existing negative
controls. All candidate shapes are compiled before parking native prefill/decode
traces. No installed-source patch, serving eligibility, drafter or speed claim is introduced.

First attempt 34001190409 (`7423438`) compiled all candidate widths and passed the
first native eager baseline comparison, but stopped before batched comparison:
the serial Generator API returned Bmax8-padded host logits, not a single row.
The comparator now explicitly accepts native one/eight-row API geometry and selects
the active row, with padding-canary and invalid-geometry regression tests. No model
arithmetic was changed and this harness failure is not evidence of numerical drift.

Retry 34001403540 (`028c23f`) passed in 8m11s:

- All 30 batched-width cases: full active-token vocabulary logits on both chips,
  active states across all 48 GDN layers, and valid KV prefixes across all 16 full
  attention layers exactly match native sequential B1.
- All 102 T=16 rollback cases: every prefix 0..16, prompt lengths 63/64/65,
  eager/trace; candidate-owned per-layer checkpoints restore the reference state,
  and two corrected continuation steps preserve exact full logits/state/KV.
- Six native eager/trace baseline comparisons and six pairs of stale-GDN/wrong-page
  negative controls passed. Rejected future rows do not change accepted-prefix
  logits in the tested cases.
- Three reusable compact snapshot sets consume 301989888 bytes per chip (288 MiB).
  Candidate checkpoints are separate from reference and scratch buffers.

Local validation: 108 CI/helper tests, 26 drafting tests, shell syntax and diff
checks passed. This closes the first integrated **short-context correctness** gate;
it does not measure full-model block latency, prove long-context equivalence, or
enable a serving/drafter integration. Next measure paired T=1/2/4/8/16 verification
and commit cost at coding-length contexts, repeating exactness gates there first.

### E3 attention shared-page gate (passed 34000512864)

Source-only run 34000023393 exported the actual K-image fused attention source
(`attention/tp.py`, SHA256 `e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a`).
Its decode batch means independent users, with one KV writer core per user.
The paged update validator explicitly rejects `share_cache`. This is a shared-page
write hazard to avoid, not evidence that production B1 cache writes are broken.

The new `attention-batch` suite batches native fused QKV preparation, attention and
output projection while replacing only the two paged writes with ordered B1 calls
through an instance-local tail function. Installed source and global TTNN functions
are untouched. Two seeds, T=1/2/4/8/16 and start positions 31/63/65 cover tile/page
boundaries with nonzero BF8 caches and a nonidentity shared page map. Eager and trace
outputs plus the entire physical K/V pool must match native sequential B1 exactly
on both chips; omitted writes and wrong-page writes must be detected. Static host
position/page fixtures are not a device-dynamic verifier or serving integration.
This gate deliberately makes no throughput claim; numerical drift blocks promotion.

Hardware run 34000512864 (`fbfefa4`) passed all 60 checks, with zero unequal
output/KV values on either chip, and 30 pairs of omitted-write/wrong-page controls.
This validates one real-weight attention layer (layer 3), not all 16 attention
layers or a full-model batched verifier. `attention-timing` adds paired captured
serial/batched layer timing: three ABBA blocks, 30 replays per sample, with exact
output and complete KV revalidation before and after measurement. No device-dynamic
page/position integration has been implemented.

Timing run 34000694699 (`9d9c6ec`) passed the same 60 correctness checks and 30
negative-control pairs, plus all pre/post timing checks and 90 paired ABBA blocks.
For each T, 18 blocks cover two seeds and three start positions, with three repeats
of the ABBA sequence per fixture and 30 trace replays per sample:

| T | Serial block mean (ms) | Batched block mean (ms) | Median speedup | Speedup range |
| --- | --- | --- | --- | --- |
| 1 | 0.2875 | 0.2826 | 1.017x | 1.015-1.018x |
| 2 | 0.5610 | 0.3071 | 1.826x | 1.822-1.830x |
| 4 | 1.1106 | 0.3388 | 3.278x | 3.272-3.285x |
| 8 | 2.2090 | 0.3985 | 5.544x | 5.530-5.552x |
| 16 | 4.4063 | 0.5218 | 8.451x | 8.425-8.460x |

These are warmed host-wall timings of captured **one-layer** token blocks, not
per-token full-model latency. Candidate timing includes the serialized shared-page
KV writes; reset and host validation are excluded from both arms. Positions are
31-80, not coding-length contexts. Neither this gain nor T=1's small difference
establishes serving throughput. Next combine this attention path with the exact
GDN batching/rollback path in a full-model verifier, then measure verification and
commit cost at realistic contexts before introducing a drafter.

| Track | Current status | Required next evidence |
| --- | --- | --- |
| Hardware prerequisite | Correctness-passed, run 33941853075 | Repeat for changed native kernels |
| E0 baseline | Benchmarked: run 33943034757 | 19.45 to 18.39 client-estimated tok/s at B=1; no-logprobs and engine timing still required |
| E1 cache | Occupancy observed, no active-zero recurrence | 0-12.4893% occupancy; full-model cache lifecycle gate remains required |
| E2 interleaving | Boundary and three-request mixed-traffic gates passed at all ratios | Repeated workload/long-context/load/cancellation sweeps; full device KV lifecycle remains unverified |
| E3 verifier | Exact full-model layout-hoisted candidate passed 34011273093; T16 costs 173.065/181.846 ms at 4K/16K | Prioritize multi-token GDN; reach <80 ms before drafting overhead and implement dynamic selection/commit |
| E4 fusion/pipeline | State/copy changes now win in full-model paired tests; earlier 91-core MLP fusion missed repeatability | True multi-token state-retaining recurrent kernel and memory-pipeline experiments; no serving promotion |
| E5 drafting | Lookup policy host-tested; MTP historical integration; DFlash2/DSpark checkpoint/config review only, not card-tested | Shared E3 verifier/rollback gate, then matched MTP/lookup/DFlash2/DSpark runs; no non-greedy semantic substitution |
| E6 coding quality | Corpus not frozen | 200 independent executable fixtures, isolated code execution and paired outcomes |
| E7 prefix reuse | Dependency-gated | E1/E2 lifecycle gate; hybrid KV plus recurrent/conv state identity and isolation |
| E8 precomputation | Planned audit | Bound removable cost before table prototypes; no unapproved arithmetic changes |
| E9 spare-core work | Layer/full-model attribution, direct DMA and guarded force-argmax measured | Persistent L1 state, dedicated staging and broader mappings; fenced attribution is not traced critical-path timing |
| E10 disaggregation | Eight-slot TP1 pool fails analytical capacity bound | Smaller-pool TP1 and hybrid-state handoff remain unverified; E2 mixed-traffic control first |

The queue is not a claim that all tracks already have runnable implementations.
Run dependent arms only after their entry gates pass. Preserve failed and negative
results; do not convert skipped or blocked experiments into successful test counts.

## Baseline protocol

Manual workflow suite `baseline`, `cards_allocated=true`. Network-disabled disposable
container, loopback-only endpoint, read-only existing weight cache, separate labeled
experimental weight/kernel cache volume, unchanged K-image flags and host sampling.
No production volumes are written. Only the container created by the job is removed.

Warm every workload, then three alternating-order repetitions at concurrency 1 with
target prompt lengths 128/4096/32768/65536, capped below the 65536 context limit to
reserve 1024 output tokens. Actual templated lengths are recorded. Concurrency 2 and
8 at 4096 tokens are separate aggregate controls. Ignore-EOS is verified by output
counts, exact streamed token IDs cross-checked against usage, metrics scraped every 250 ms.
Client timing is labeled an estimate; no engine-token timestamps means this cannot
certify 200 committed tokens/s. Historical run 33943034757 used logprobs and token
strings; new runs request exact token IDs without logprobs by default. The old
results remain valid only for that original host-sampling workload.
Coding quality and greedy token-ID equivalence require their separate gates.

Initial full-model run 33942471680 reached readiness and logged K/M engagement,
but no throughput requests ran: Transformers returned a BatchEncoding rather than
a token list. The client now normalizes the actual input_ids and has regression
tests for both return shapes. Run 33943034757 retries the baseline with the frozen
snapshot and unchanged model flags. Do not count the failed warm-up as a benchmark.

The [payload feasibility estimate](decode-payload-bound.md) gives approximately
19.92 GB of dense projection value payload per ordinary decode step under current
mixed bfloat4/bfloat8 layouts. It is not measured traffic, but establishes the
approximately 3.98 TB/s payload requirement of a non-amortised 200-step/s path.

## Layer profile gate

Run 33944471561 passed GDN and MLP numerical checks with two-chip device timing
and no dropped markers. Some operations use all 110 available workers; this is
not a uniform 36-core workload. These eager fixture timings are not full-model
traced decode timings. Attention failed before completing its numerical check:
the stock fixture allocated BF16 KV while the shipped BF8 flag casts fill inputs
to BF8. The retry stages only the fixture cache allocation dtype to match that
flag, recording original/staged hashes. Model precision and PCC thresholds are
unchanged. The failed attention profile is not accepted as performance evidence.

Retry [33945221856](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33945221856)
passed all three fixtures on both chips with no dropped markers. Attention
paged-versus-concat PCC: prefill 0.999696, decode 0.999757. GDN position-zero
PCC 0.99985; MLP PCC 0.999230. No thresholds were relaxed.

Device-0 eager attribution: the GDN fixture's two steps used 333.358 us in four
matmuls (32/43 cores), 93.131 us in two recurrent kernels (24 cores), and 70.693 us
in two fused conv/gate kernels (81 cores). The MLP fixture's single step used
303.913 us in three matmuls (32/39 cores). Some copy/elementwise operations use
110 cores. Attention includes both prefill and decode, reference and paged paths;
its aggregate operation totals must not be called a single decode latency.

Local validation at commit 35a0d3f: 84 host tests passed (31 CI, 38 continuation/
interleaving, four stream rig, 11 lookup policy). These are not 84 silicon tests.
Interleaving run 33945305689 passed the control's eight boundary prompts and
post-cancellation output-ID check without requesting logprobs. The ratio-1 arm
failed before model execution because vLLM offline mode rewrote the canonical
model ID to the pinned local snapshot path. The guard now accepts only that exact
reviewed snapshot in addition to the canonical model ID; arbitrary model paths
and other revisions remain rejected. Ratio 2/4 did not execute. Historical
zero-cache reports remain observation-only unless current measurements reproduce them.

Retry 33945632554 again passed the control, then failed during ratio-1 API startup:
`Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger than
max_num_batched_tokens (2048)`. This is multimodal admission/encoder budgeting,
not a KV-usage or kernel failure. Interleaving v1 is deliberately text-only but
the launcher had left image/video limits at their defaults. All interleaving
arms, including the control, now set image/video limits to zero and disable MM
embedding inputs. The 2048-token chunk budget and numerical gates are unchanged.
The prior baseline/profile suites retain their original settings.

Run 33945913888 confirmed the encoder-budget fix: the ratio-1 endpoint loaded and
became ready. Its first text request then hit `Continuation v1 is text-only`.
The pinned plugin represents an absent per-request image as `pixel_values=[None]`,
not only `None`. The continuation guard now accepts exactly one absent visual
row, while rejecting tensors, populated/nested rows, malformed row counts and
vision tokens. Regression cases cover both acceptance and rejection. This is
still a failing integration result, not a passed interleaving correctness gate.

Run 33946277920 progressed past the visual-row guard and failed on position
validation. The pinned runner builds prefill positions and prompt lengths as
NumPy integer arrays; our validator accepted only Python integers and torch
integer scalars. It now accepts `numbers.Integral`, normalizes to Python int,
and still rejects floating-point, Boolean and array-valued positions. The
integration regression now feeds the actual plugin visual-metadata gatherer
and NumPy position/prompt-length representations through `submit_prefill`.

Run [33946645240](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33946645240)
cleared all three runtime/configuration errors. Control and ratio-1 each completed
eight boundary requests and the post-cancellation check. Cross-arm comparison
correctly failed: the 2049-token prompt differs at generated token index 1
(the second output token). First four control IDs: `[74830,8,198,727]`;
ratio-1 IDs: `[74830,1590,198,262]`. All 32 IDs match for the other seven lengths
(63/64/65/2047/2048/4096/8193) and after cancellation. This is now a genuine
full-model equivalence failure, not a launcher error. Its cause is not yet
established; the matching first output token does not prove equal logits or state.
Do not relax the exact-output gate or adopt interleaving. Ratios 2/4 did not run.
Next diagnostic: repeat the isolated 2049 case and compare recurrent/conv/KV state
and logits around positions 2048/2049 and the first decode step. The previous
raw startup errors remain preserved in their original artifacts.

Diagnostic run 33947341479 localized the failure: for 2049-token prompts, ratio-1
records one 2048-token intermediate prefill, then enters decode at position 2048.
There is **no final GDN slot write**. The 8193-token case likewise has no final
slot write, despite previously matching output tokens by coincidence. The pinned
runner's `_is_still_prefilling` assumes one remaining token means decode. That
assumption is invalid for this prototype's separate B=1 prefill scratch: even a
one-token final prefill must execute and commit the request's GDN state to its
decode slot. The fix uses the existing request/epoch completion ledger only when
continuation is enabled; ordinary decode and the flag-off path keep their original
classification. A regression executes the transformed nested runner classifier
for a one-token tail, completed request, no remaining tokens, and flag-off mode.
Diagnostic reruns also gate three isolated 2049-token repetitions against control,
in addition to the original nine checks. Host fingerprint instrumentation performs
no additional device operations; its timings must not be used as speed evidence.

Fix verification [33947778844](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33947778844)
passed: ratios 1/2/4 each match the control on all 12 exact-token checks (eight
boundary prompts, three isolated 2049 repetitions, and post-cancellation output).
Each arm also has 13 paired diagnostic request records, including the cancelled
request. All host recurrent/conv snapshots supplied to the decode-slot writer
match the control byte-for-byte, as do the recorded final-prefill and first-three
decode logits. No missing final slot writes remain. This is not a complete direct
device KV snapshot or multi-request isolation certification.

Mixed-traffic run 33948332548 separately tests one 512-token decoder, an injected
8193-token prefill and a short coding request. It requires exact token-ID equality
against isolated requests and verifies actual request overlap. Boundary checks
remain enabled in every arm. Diagnostic hashing is disabled for this run, and
KV metrics continue to be collected. No production serving changes or automatic
adoption follow from these tests.

Mixed-traffic [33948332548](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33948332548)
passed all four arms. The live decoder, long request and short request each match
their isolated output IDs exactly; post-run comparison also matches all three
against the ratio-0 isolated control. Actual overlap and short-request arrival
before the long request's first token were verified. One smoke trial per ratio:

| Decode steps/chunk | Live decoder maximum gap (s) | p99 gap (s) | Client decode tok/s | Long TTFT (s) | Short TTFT (s) |
| --- | --- | --- | --- | --- | --- |
| Control, no interleaving | 3.116 | 0.055 | 17.671 | 2.590 | 2.813 |
| 1 | 0.654 | 0.556 | 17.470 | 2.836 | 3.308 |
| 2 | 0.646 | 0.552 | 17.249 | 3.078 | 3.605 |
| 4 | 0.647 | 0.546 | 17.148 | 3.517 | 4.132 |

The maximum interruption falls approximately 79%, but one long stall becomes
multiple chunk-sized interruptions: p99 worsens, and new-request TTFT increases.
This is a responsiveness tradeoff, **not** a decode-throughput improvement or
evidence of 200 committed tok/s. Ratio 1 is the next candidate for repeated
paired measurements, not an adopted default. Cache occupancy was nonzero
(peak 1.60-1.61%), with zero active-zero-cache samples across 1230 active samples
and zero scrape errors. Production serving and default-off experiment flags
remain unchanged.

## E9f: guarded TP2 greedy device sampling

Native sampler prerequisite [33949480633](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33949480633)
passed all 30 checks: real 248320-token vocabulary, random/boundary/tie/near-tie
cases, eager and two trace replays, on both chips. Median synchronized sampler
latency was 2.607 ms over 20 warm iterations. This is a 32-row sampler-only
measurement, not B=1 full-model throughput.

The pinned model disables generic device sampling on TP2 because each vocabulary
shard exceeds its TopK limit. The opt-in experiment enables only force-argmax,
with a runtime guard that fails before any generic TopK execution. Original
sampling eligibility checks remain intact; non-greedy requests, penalties,
multi-request batches and unsupported requests retain host sampling.

The full-model test uses three ABBA blocks per prompt length (approximately 128
and 4096 tokens), B=1 and 512 output tokens. Both arms omit explicit seeds to
permit internal greedy sampler tracing, and omit logprobs in timed requests.
Every output must match exact token IDs and text; device-path engagement is
recorded per request. A device-labelled logprobs request must fall back to host.
This is separate from the historical logprobs-enabled baseline. KV metrics are
collected throughout; experiment flags and production defaults remain unchanged.

Full-model attempt 33950052748 stopped safely during startup: the shared warmup
sweep requested non-greedy TopK. The experiment now passes the existing
`greedy_only=True` warmup option only for its TP2 model. Host/no-sampler warmup
remains included, and the runtime TopK guard is retained.

Full-model [33950377324](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33950377324)
passed at `dd42e2e`. All 24 measured requests completed 512 tokens and matched
their prompt-length control exactly (token IDs and text). All 14 device requests
(12 measured, two warmup) recorded force-argmax engagement with model trace mode
`all`; both logprobs negative controls correctly selected host sampling.

| Actual prompt tokens | Host mean client tok/s | Device mean client tok/s | Mean paired gain | Three-block gain range |
| --- | --- | --- | --- | --- |
| 109 | 19.850 | 21.317 | +7.389% | +7.334% to +7.455% |
| 4078 | 19.800 | 21.113 | +6.629% | +6.278% to +6.975% |

Means weight the three ABBA blocks equally. These are B=1 client estimates,
not engine-committed timing or evidence of 200 tok/s. Exact equality on these two
coding prompts is not a general coding-quality benchmark. KV occupancy peaked
at 0.8782%; no active-zero occupancy occurred across 2364 active samples out of
2454 scrapes, with zero scrape errors and zero preemptions.

Decision: retain guarded device force-argmax as a promising experimental candidate,
not a production default. Next gates are longer-context paired runs, explicit-seed
and non-greedy/penalty fallback checks on hardware, and full-model traced operation
attribution before changing projection grids or low-level GDN kernels. Multi-request
device sampling and composition with prefill interleaving are not certified here.

The `sampling-extended` suite next runs three ABBA blocks at approximately 32K
and 64K input tokens, still B=1 with 512 output tokens and exact ID/text gates.
Before timing it compares host/device-labelled requests with explicit seeds 123
and 456, seeded non-greedy sampling, each penalty type separately, and a final
greedy recovery request. Non-greedy and penalty cases must actually select host;
seeded greedy and post-fallback greedy must engage device force-argmax. These
checks change request parameters only, not eligibility or sampling mathematics.

Extended [33952227890](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33952227890)
passed at `9ca28d0`, following 47 passing local CI-helper tests. All seven
fallback/seed pairs matched exact IDs and text, with the expected actual sampler
selection; device greedy resumed correctly after penalty requests. Both long-context
logprobs negative controls also selected host and matched greedy control tokens.
All 24 long-context measured requests completed 512 tokens and matched their
prompt-length control exactly across three ABBA blocks:

| Actual prompt tokens | Host mean client tok/s | Device mean client tok/s | Mean paired gain | Three-block gain range |
| --- | --- | --- | --- | --- |
| 32752 | 19.231 | 20.522 | +6.709% | +6.553% to +6.828% |
| 64504 | 18.724 | 19.968 | +6.642% | +6.464% to +6.831% |

KV occupancy peaked at 12.3918%; zero active-zero observations across 2563 active
samples out of 4581 scrapes, zero scrape errors and zero preemptions. This is
exporter evidence, not direct certification of every attention/GDN state tensor.

The sampling improvement now reproduces at all four tested context lengths.
Explicit-seed checks establish correctness and path selection, not seeded throughput
or internal sampler-trace reuse. Timings still exclude logprobs and use client
stream events, not engine commit timestamps. Production defaults remain unchanged.
The next performance investigation is full-model traced operation attribution:
separate projection/GDN math, collectives, sampling and host dispatch before tuning
core grids. Existing eager module profiles cannot establish that critical path.

The `model-profile` probe loads every model layer through the served Qwen Generator,
retaining max-batch 8, 64K context and the observed 8200-block KV pool. Only the B=1
host-sampling decode bucket is warmed/captured; HTTP, scheduling and other resident
batch traces are excluded. It checks the first 16 tokens against endpoint run
33950377324, flushes device profiler buffers between decode steps, and requires
15 complete trace replay sessions on both chips. Attribution filters by the actual
decode trace ID, excluding eager compilation and prefill. Summed kernel durations
are attribution evidence, not non-overlapping critical-path time or tok/s.

Attempt 33954595293 completed all 16 exact-token checks on the full 64-layer
Generator, but is not a passing profile. Warmup overflowed profiler buffers and
the report fell into device-only Python analysis of an 11.5 GB raw log; it was
cancelled after exceeding its inner timeout. No kernel bottleneck conclusion is
drawn from that attempt. The retry explicitly enables host operation metadata
before Tracy imports, uses partial Python tracing and runtime C++ analysis without
the large raw CSV dump, and adds a hard kill-after timeout. Warmup overflow warnings
are reported separately: any overflow during the explicitly bounded decode region
still fails, as do missing or inconsistent replay rows on either chip.

Retry 33957235279 again passes exact generation and now captures host operation
metadata, but the all-operations report correctly rejects missing warmup data
(operation 11738113, device 1). The next revision scopes the pinned report join to
the recorded decode trace before joining device measurements. Its existing missing
operation assertions remain active for every selected operation; the final checker
also retains both-chip, replay-count, per-replay coverage and measured-overflow gates.
Profiler CSV sidecars are preserved explicitly even on failure, rather than relying
on hidden-directory artifact upload. No model/kernel arithmetic is modified.

Full-model trace [33957724963](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33957724963)
passes at `f62c298`: all 64 layers, exact first-16 endpoint tokens, 15 measured
replays on each chip, and 1433 operations per replay. Native C++ operation counts
also match the joined report by chip, replay and operation name. No marker drops
occur in measured decode; 1100 excluded warmup warnings remain explicitly recorded.
This certifies the selected decode evidence, not the discarded warmup profile.

| Operation group | Chip 0 summed kernel ms/step | Chip 1 summed kernel ms/step |
| --- | --- | --- |
| Matrix multiplications | 32.069 | 32.051 |
| GDN recurrent kernel | 2.256 | 2.255 |
| GDN convolution/gates | 1.699 | 1.699 |
| All-gather | 1.999 | 1.770 |
| Reduce-scatter | 1.413 | 1.606 |
| All operation groups | 42.670 | 42.657 |

Each entry is the median of per-replay summed durations; the last row sums those
group medians. Matmul accounts for approximately 75% of this attribution budget,
GDN recurrence plus convolution/gates for 9.3%, and the two collectives for 8%.
Do not sum chips or equate these sums to non-overlapping critical-path latency.

Largest matrix shapes, chip 0 (chip 1 agrees closely):

| Weight dimensions (padded) | Calls/step | Weight dtype | Cores | Summed kernel ms/step |
| --- | --- | --- | --- | --- |
| 5120 x 8704 | 128 | BFLOAT4_B | 39 | 11.803 |
| 8704 x 5120 | 64 | BFLOAT8_B | 32 | 7.819 |
| 5120 x 8256 (8240 logical) | 48 | BFLOAT8_B | 43 | 5.754 |
| 3072 x 5120 | 64 | BFLOAT8_B | 32 | 2.984 |
| 5120 x 7168 | 16 | BFLOAT8_B | 56 | 1.939 |
| 5120 x 124160 | 1 | BFLOAT8_B | 108 | 1.769 |

Decision: prioritize MLP gate/up and down projection grid/blocking experiments,
then the GDN input projection. Preserve the existing BF4/BF8 weight dtypes and
accumulation fidelity; extra cores are not automatically a bandwidth improvement.
The vocabulary projection already uses 108 cores, disproving a universal 36-core
limit. Next gates are real-shape numerical checks and synchronized traced kernel
A/B timing before any full-model adoption. No speedup beyond the earlier sampling
result is claimed from profiling. The host scheduler and device-sampling path
remain outside this probe. The enormous raw host-times CSV is not preserved in
future runs; final operation reports, native C++ data and host operation metadata
remain available. Local CI-helper validation now includes shape/coverage checks.

The `mlp-sweep` gate loads real layer-0 weights and compares the frozen production
44/33 gate/down grid budgets against separate larger/smaller grids and K-block
sizes 4/16 (control 8). It preserves BF4 gate/up, BF8 down, LoFi, FP32 destination
and packer L1 accumulation. Fifteen configurations run three seeded B=1 vectors
against the existing Torch PCC threshold (0.97) and exact baseline outputs, eager
and traced. All compilation precedes trace capture. Three ABBA blocks per candidate
time 20 replay submissions followed by synchronization per sample. Promotion to
a full-model gate requires exact control equality and over 2% lower latency in
every block; this is only a single-layer screen, with replicated DRAM input and
both-card reduce-scatter, not a serving-throughput or coding-quality benchmark.

Grid/block sweep [33958859849](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33958859849)
passes at `50d63d7`: all 90 eager/traced seeded outputs match control exactly;
Torch PCC ranges 0.999203-0.999417. No candidate clears the predeclared 2% latency
gate in every block. The best, gate grid budget 110, reduces whole-layer latency
only 1.691% (about 0.338 to 0.332 ms). Larger down grids are 0.48-2.43% slower;
K blocks 4 and 16 are 3.96% and 5.72% slower. The frozen control remains unchanged.

Follow-up `mlp-packing` keeps the control grid budgets but separately sets gate/up
or down output tiles per core to 8, 12 or 16, permitting a four-tile FP32 subblock
instead of the control's one-tile subblock. It retains the same precision, seeded
exact-output and paired timing gates. This tests tile packing, not a promise that
more active cores improve bandwidth.

Packing [33959155664](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33959155664)
passes at `86b9d83`, with all 42 seeded eager/traced outputs exactly equal to control.
No candidate meets the promotion gate:

| Change from frozen control | Mean whole-layer latency change |
| --- | --- |
| Gate/up 8 output tiles/core | -0.062% (one block slower) |
| Gate/up 12 output tiles/core | +2.300% |
| Gate/up 16 output tiles/core | +4.204% |
| Down 8 output tiles/core | +1.686% |
| Down 12 output tiles/core | +4.354% |
| Down 16 output tiles/core | +8.817% |

Across the two runs, 20 non-control configurations were screened without changing
weight or accumulation precision. None qualifies for full-model adoption under
the existing rule; the 44/33 control remains the default. These single-layer
results do not prove DRAM saturation, but they do rule out a large gain from the
tested grid/K-block/output-packing changes. They are not a full-model speedup.
Next investigation: combine gate/up work or reduce surrounding memory transfers,
with weight-layout/extra-allocation accounting and exact-output gates before
serving tests. Do not extrapolate a one-layer gain to the 200 tok/s objective.
Local CI-helper suite: 56 tests pass; hardware access and serving remain unchanged.

## GDN upstream parity audit

Read-only source audit [33960203974](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33960203974)
passed. The image includes ctxbot's fused recurrent kernel hardening plus packed
QKV, native norm/gate folding and broadcast rank-one updates. No missing listed
core fusion fix was identified; do not overwrite these extensions with the PR.
See [source comparison and next experiment](gdn-source-audit-2026-09-05.md) for
exact upstream heads, evidence limits, two failed audit attempts and the proposed
active-prefix state-write experiment. B1/Bmax8 currently cannot use the existing
in-place flag. No new throughput result or production change is claimed.

Offline attribution of the existing 33957724963 hardware trace now isolates the
three recurrent-state copy stages across all 48 GDN layers, 15 replays and both
chips. Median summed copy time is 0.548383/0.548504 ms per step on chip 0/1,
about 1.285%/1.286% of summed kernel time (not critical-path time). Native call
coverage, chain adjacency and BF16 state shapes are checked. Active-prefix state
writeback remains a secondary candidate; combined gate/up projection work takes
priority. The analyzer is added to future full-model profiles; eight regression
tests pass. This is new analysis of old measurements, not a new speed benchmark.

## Fused gate/up decode probe

The `mlp-fusion` suite compares the frozen real-weight layer-0 MLP with native
`minimal_matmul(fuse_swiglu=True)` using tile-pair-interleaved gate/up weights.
It retains BF4 gate/up, BF8 down, LoFi FP32 destination and packer L1 accumulation;
fusion can still alter rounding/order, so exact control output remains mandatory
for promotion. Three candidates use M/K blocks 1/8, N blocks 8/16/32 and grid
11x2. The existing minimal matmul partitions M across grid rows, so adding rows
for B1 is not automatically useful N-parallel work. No custom kernel is added.

The timed path includes the original input-to-L1 transfer, unchanged down
projection and TP reduction. All candidates compile before traces; three seeds
are checked eager and traced against Torch PCC >=0.97 and exact control output.
Three ABBA blocks with 20 replays/sample require >2% lower latency in every block
and exact outputs before any full-model gate. A non-exact result is not promoted.
An additional packed layer weight is allocated only inside the experiment;
host packing is checked reversible and no weight cache is written. Full-model
duplicate-weight memory capacity remains unverified. This is not serving adoption.

Attempt 33962274337 passed the three eager control seeds but rejected the first
fused candidate before execution: the operator requires grid dimensions >=2x2.
Retry uses 11x2 rather than 11x1, without relaxing the native validation. No fused
correctness or timing result is claimed from the rejected attempt.

Retry [33962376720](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33962376720)
completed at `460959c`. All three seeds passed Torch PCC >=0.97 eagerly and under
trace for control and every candidate (24 checks total). All 18 candidate
seed/mode comparisons failed bit-exact equality against control. Candidate PCC
was 0.999201477-0.999417543, close to control's 0.999202549-0.999416769;
this is not evidence of coding-quality equivalence or degradation.

| Fused N block | Mean layer latency increase | Three-block range | Approximate candidate ms |
| --- | --- | --- | --- |
| 8 | +70.565% | +70.481% to +70.674% | 0.576 |
| 16 | +68.031% | +67.801% to +68.285% | 0.568 |
| 32 | +57.923% | +57.889% to +57.942% | 0.535 |

Control was approximately 0.338 ms. No candidate qualifies on either exactness
or speed; no full-model arm is warranted for these configurations. Execution
success is not an optimization pass. This does not rule out a decode-specialized
fused kernel: the existing implementation uses a 2D M/N partition instead of the
control's 1D N-parallel map. Its SwiGLU stage multiplies gate/up in destination
registers before the final pack, unlike control's separately rounded BF16 gate
and up intermediates. These source differences are hypotheses for the result,
not experimentally isolated causes.

Next native candidate must preserve the control's 1D N-parallel distribution,
keep complete gate/up tile pairs together, reproduce BF16 intermediate rounding
and account for packed-weight allocation before repeating the same gates. Do not
extend the negative prefill-oriented grid sweep or relax precision/exactness to
claim a win. The 65 local CI-helper tests pass; production remains unchanged.

## Decode-specialized 1D gate/up prototype

`projection-1d` uses a separate `generic_op` program, not the prefill-oriented
minimal matmul. The native 1D compute source is copied into a source-code kernel
descriptor; its final-output branch and a post-loop epilogue change. The original K-block loop and
FP32 packer-L1 partial accumulation remain. No installed source or binary is edited.
Native and generated compute-source SHA256 values are recorded in the result.

The program maps 272 output-tile pairs to the same 39 N-workers (seven pairs/core,
six on the last), within an 11x4 rectangle. An activation-multicast reader includes
the five non-compute receiver cores. Separate weight-reader/output-writer code
streams pair-interleaved BF4 weights, including zero padding on the last worker,
and writes compact BF16 output directly. These readers are new code, not a claim
of byte-identical native reader scheduling. K-block size stays eight; paired
subblocks have two tiles instead of control's single tile.

The epilogue runs native pack-side SiLU on the gate tile only, packs gate and up
to a private fourteen-tile BF16 circular buffer (seven pairs per worker), then multiplies those rounded values
and packs BF16 output. Packed device weights must dequantize bit-identically to
the original gate/up tensors on each chip. This establishes the intended rounding
contract, not a correctness claim before silicon testing.

First gate is projection-only (no down projection or CCL reduction): three real
layer-0 input seeds, both chips, eager and traced exact comparisons, finite/PCC
diagnostics, then three ABBA blocks with 40 trace replays/sample. Inputs already
reside in L1 for both arms. Only exact output and >2% lower latency in every block
permit a whole-MLP experiment. Preserve non-exact/negative results and do not
replace the serving path on the strength of this prerequisite test.

### Initial 1D silicon attempts

All runs below are isolated projection probes, not full-MLP or serving results.
Execution/PCC success is separate from the exact-output promotion gate.

| Run | Outcome |
| --- | --- |
| 33963133460 | Descriptor setup failed: unpack-mode vector constructor not exported; use the exposed property instead. |
| 33963238683 | Descriptor validation required 64 unpack-mode entries rather than 32. |
| 33963346556 | Inline per-pair product interfered with subsequent native matmul state; first comparison PCC 0.95914, stopped before timing. |
| 33963532517 | Deferring product until after all matmul work restored PCC above 0.99988, but LoFi FPU multiply left 8,040–8,121 mismatches per 8,704-element shard. Projection latency increased 83.89–85.39%. |
| 33963794203 | SFPU product JIT failed because its binary API header was missing. |
| 33963923741 | Explicit SFPU API plus waiting for all rounded tiles reduced discrepancies to 52–83 elements per shard, maximum absolute error 0.015625. Still not exact. Control 0.19695–0.19733 ms versus candidate 0.36276–0.36490 ms, 83.83–85.29% slower. Not eligible for whole-MLP testing. |

The next probe isolates rounded gate and up outputs independently before checking
their product. It also corrects the new readers to the native 1D NoC assignments:
input multicast on NoC 1, weight reads on NoC 0. NoC 1 multicast endpoints reverse,
but receiver acknowledgements still target the original sender. Compare only the
ordinary product arm in timing; diagnostic intermediates compile before traces.

Run 33964284985 confirmed all twelve rounded gate/up comparisons exact across
three seeds and both chips. Product mismatches were unchanged. With corrected NoC
routing, candidate projection latency was 0.20523–0.20580 ms versus control
0.19672–0.19693 ms: only 4.33–4.50% slower, rather than approximately 85% slower.
This remains a rejected candidate, but identifies a substantial reader defect.

The native Blackhole `calculate_sfpu_binary_mul` explicitly applies BF16
round-to-nearest-even and zero-input handling when its arithmetic template flag
`is_fp32_dest_acc_en` is false. Calling the ordinary SFPU API inside our FP32
matmul kernel passes true and skips that narrowing. The next candidate keeps
FP32 destination addressing/accumulation for the matmul but calls the same native
multiply calculation with its BF16 narrowing branch enabled. No accumulation
precision is lowered, and the existing exact-output gate remains mandatory.

Run 33964555688 confirmed the correction: every rounded intermediate, eager
product, and traced product comparison was exact on both chips for all three
seeds. Projection latency remained 5.00–5.10% above control (0.20688–0.20733 ms
versus 0.19695–0.19732 ms). Correctness is established only for this bounded gate;
there is still no performance promotion or whole-model quality claim.

The subsequent isolated grid sweep retains this arithmetic, K-block eight,
two-tile subblocks and NoC routing. It compares 39/55/68/91 N-workers using
7/5/4/3 pairs per worker respectively. Only the N partition, corresponding buffers
and multicast receiver rectangle vary. All candidates compile before trace
capture and each receives its own three ABBA timing blocks and exact-output gate.

Run 33964688595 retained exact outputs for every grid but promoted none:
39 cores +13.22–13.30%, 55 cores +96.00–96.54%, 68 cores +11.38–11.70%,
91 cores +14.29–14.56% projection latency versus control. The 39-core arm itself
regressed relative to the preceding single-candidate probe. Generalizing the
reader had inadvertently changed its inner tile-loop bound from compile-time
to runtime; the retry restores compile-time specialization and compact core
ranges before drawing conclusions about the additional cores.

Run 33964808840 restored the 39-core control candidate to +5.07–5.16%. All
four candidates remained exact. The 55-core variant was +85.50–85.57%; the
68-core variant improved 1.94–2.06% but failed the >2% requirement in one block.
The 91-core variant passed the prerequisite: 0.18833–0.18882 ms and 4.07–4.32%
lower projection latency in every block. This is not end-to-end tok/s.

`projection-1d` now conditionally follows its projection probe with the existing
whole-MLP harness. It independently verifies each selected candidate's twelve
eager/traced shard checks and all three timing blocks, then requires an identical
compute manifest. Rejected candidates are not forwarded. Whole-MLP testing keeps
the native BF8 down projection, DRAM input/output and TP2 reduction; three seeds,
Torch PCC checks, exact native-control outputs and paired traces still apply.

Run [33964993645](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33964993645)
repeated the projection gate and completed the conditional whole-MLP test:

| Stage | Result |
| --- | --- |
| 91-core projection | Exact on both chips, eager and traced, all three seeds; 4.24–4.47% lower latency across all three paired blocks. |
| Complete layer-0 MLP, including down projection and TP2 reduction | Exact native-control output for all three seeds, eager and traced; Torch PCC 0.99920–0.99942, unchanged from control. |
| Complete MLP latency | Control 0.33735–0.33777 ms versus fused 0.32892–0.32962 ms; 2.37–2.50% lower latency, mean 2.447%. |
| Promotion | Eligible for the full-model gate only. No serving default changed and no end-to-end tok/s result claimed. |

The 68-core arm again missed the >2% projection requirement in one block and was
not forwarded; 39 and 55 cores remained slower. Local helper validation passed
75 tests, with shell syntax and whitespace checks passing. These results correct
the prior inference from the prefill-oriented 2D kernel: properly routed,
rounding-preserving 1D fusion can improve this decode workload, although the
measured gain is small and does not establish the 200 tok/s target.

Next gate: opt-in full-model integration with pair-packed weights replacing
(not retaining alongside) gate/up allocations, source/precision checks on every
layer, trace-safe B1-only dispatch and native fallback elsewhere. Verify output
tokens and memory headroom at short and long contexts before paired serving
timings. Do not extrapolate the layer microbenchmark to end-to-end speed or
claim coding-quality coverage from three layer-input seeds.

## Full-model fusion gate (2026-09-06)

Source inspection found an important allocation detail: the native TP model already
loads a pair-packed `w_gate_up` tensor for its fused prefill path. The full-model
experiment reuses that existing allocation instead of building another copy.
Native `w1`/`w3` remain available for unmodified fallback, with no weight-memory
increase relative to the current baseline. This replaces the preceding proposed
weight-replacement approach; hardware must verify all 64 layers and both shards.

`full-model-fusion` is an isolated Generator test, not a serving deployment.
It checks pinned compute/reader hashes, BF4 gate/up and BF8 down precision,
dequantized packed-weight equality for every layer/shard, and unchanged weight
addresses. It records per-chip DRAM/L1/trace allocator snapshots. Both arms share
the model and its KV pool but use separate decode trace stores. All decode kernels
compile before trace capture; prefill always uses the native path. Only explicit
B1 decode inputs select the fused path, with capture engagement checked on all
64 layers. No installed source or serving default is patched.

Predeclared work: short (~109 tokens) and long (~64,504 tokens) coding prompts,
16-token eager/traced token and full-logit equality checks, then three ABBA blocks
of 128-token generations per context. Timing includes host argmax and readback,
excludes prefill and HTTP/scheduler overhead, and counts 127 decode steps rather
than counting the prefill-produced first token as decode. Every request resets
state through native prefill. Require exact outputs and >2% lower latency in every
block before marking eligibility for a subsequent serving gate. Preserve failures
and smaller/negative gains; none establishes 200 committed tok/s.

### Completed run 33996306217

[CI evidence](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33996306217),
code `84db0ab`: execution/correctness passed, `eligible_for_serving_gate=false`.
All six eager/traced logit/token checks were exact, as were paired generation
tokens and packed weights on all 64 layers/both chips. Weight addresses were
unchanged and additional weight allocation was zero.

| Context | Control host tok/s, three blocks | Fused host tok/s, three blocks | Decode latency change |
| --- | --- | --- | --- |
| Short (~109 input) | 19.576 / 19.652 / 19.713 | 19.789 / 20.245 / 20.067 | -1.076% / -2.931% / -1.766% |
| Long (~64,504 input) | 18.410 / 18.626 / 18.201 | 18.406 / 18.705 / 18.746 | +0.018% / -0.421% / -2.907% |

Only two of six blocks clear the predeclared >2% latency threshold. The exact
kernel is viable but the full-path improvement is not repeatable enough for
promotion. These are Generator timings with host sampling, not the separate
device-argmax serving arm or engine-committed throughput. No serving change made.

### Multi-drafter groundwork

Added lookup-first routing to one explicitly selected neural callback, with native
target fallback when mode/verifier readiness is unsupported or no proposal exists.
Nine new host tests cover selection, skipped neural work, invalid proposals,
explicit commit, failures and isolation, alongside the existing eleven lookup tests.
Callbacks are test doubles: no DFlash2, DSpark or EAGLE3 device adapter is claimed.
The E5 plan now separates routing, adaptive selection, cascades and tree ensembles.
The shared exact E3 verifier and device state lifecycle remain the next dependency.

## E3a native GDN prefix gate (2026-09-06)

Added the `gdn-prefix` hardware suite. The historical multi-token implementation
used older composed conv/gate/norm operations, so it is not reused as the current
control. The new candidate calls the audited native packed input projection once
for T rows, then supplies independent row copies to the unchanged fused native
decode remainder. Output projections and collectives remain per token in this
first isolation test; it is not a complete batched verifier or optimal kernel.

The gate loads one real GDN layer, retains the pinned K-image BF16 recurrence and
decode settings, and uses B1 within an eight-slot state allocation on both chips.
Nonzero priming precedes three seeds at T=1/2/4/8/16. Both eager and trace replay
must match every native sequential output and recurrent/conv snapshot exactly.
Each prefix 0..T is restored in place, followed by two correction-input steps,
with exact outputs and final state required. Deliberately restoring stale
recurrence or convolution separately must be detected in state and continuation.

Device snapshots are preallocated outside trace capture and all live state
addresses must stay fixed. Timings include snapshot copies and are diagnostic
only. Attention KV, full-model logits, EOS/cancellation/slot epochs and all-layer
verification remain later gates; no host test or layer pass certifies those.

Run [33997659654](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33997659654),
code `b4da5c4`, passed all 216 prefix/continuation cases (both chips, three seeds,
five row widths, eager/trace) and all 30 stale recurrence/convolution negative
controls. This establishes input-projection batching with native fused GDN state
evolution for this layer fixture, not a complete verifier.

Next `gdn-block` arm batches the output projection and TP reduction as well. It
extracts the pre-projection native decode body from the SHA-pinned audited source,
with a structural tail check, preserving the internal fused GDN arithmetic. This
is an instance-local experimental function, not an installed source patch.

The host `greedy_verify.py` contract selects the longest exact proposal prefix
against K+1 target argmax rows. Retain state after seed plus accepted proposals;
the target correction/bonus is emitted but remains the next unconsumed input.
Accepted EOS suppresses later rows; correction EOS terminates without consuming
it. Six tests cover all rejection positions, bonus/off-by-one cases and EOS.
This selector neither calculates target predictions nor commits device state.

Run [33997961604](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33997961604),
code `82b9e3b`, passed the output-batched arm: all 216 exact prefix/continuation
cases and all 30 stale-state negative controls again passed. The layer now reads
input and output projection weights once per T-row block and performs one TP
output reduction; convolution/recurrence/norm-gate still use the native fused
per-token path, not a new time-fused recurrence kernel.

Single synchronized traced diagnostic observations, three seeds:

| T | Input-batched only, snapshot-inclusive ms | Input + output batched, snapshot-inclusive ms |
| --- | --- | --- |
| 1 | 0.411-0.468 | 0.409-0.414 |
| 2 | 0.666-0.675 | 0.703-0.720 |
| 4 | 1.195-1.222 | 1.195-1.199 |
| 8 | 2.247-2.251 | 2.191-2.194 |
| 16 | 4.354-4.389 | 4.176-4.193 |

These are separate correctness runs, not paired ABBA performance trials. Each
row snapshots the entire eight-slot recurrent/conv allocation even though only
one slot is active. Do not use these costs as an optimized verifier curve,
extrapolate them over 48 layers, or claim a serving speedup. Next isolate active-slot
snapshot/restore cost, retain full-state comparisons for correctness, and test
attention KV masks/positions plus all-layer target logits before drafter integration.
Current local validation: 87 CI/helper tests and 26 drafting/selector tests passed;
shell syntax and whitespace checks passed. No serving default or runner access changed.

## E3 active-slot storage (2026-09-06)

The `gdn-active` arm snapshots only logical slot zero: recurrent axis 0,
convolution axis 1. It retains BF16 dtype and padded tile storage, so convolution
snapshots do not necessarily shrink physically. Restore clones the saved tensors
before the native consuming writers and changes only slot zero, preserving live
buffer addresses. No installed source or arithmetic changes.

Repeat the 216 prefix/continuation and 30 stale-state gates, now comparing complete
live state against the full-state reference after each active restore. Additionally
change all eight slots, restore only zero and assert slots 1..7 stay unchanged on
both chips. Compare full versus active save and restore separately with three ABBA
blocks, 30 trace replays per sample, synchronized outside the measured loop.
This is a snapshot-cost experiment, not full-model speculative throughput.

Run [33998325637](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33998325637),
code `da39648`: all 216 prefix cases, 30 stale-state controls and 15 changed-idle-slot
checks passed. Padded snapshot allocation fell from 7,602,176 to 2,097,152 bytes per
chip (72.4% smaller). However active save was 11.1-11.4% slower (~0.0593 vs 0.0533 ms),
and restore was 13.61-13.66 times the full-copy cost (~0.732 vs 0.0537 ms).
Do not promote this native-writer restore as a latency optimization.

The native convolution slot writer reconstructs the whole buffer with slice/concat
and copy. Next `gdn-direct` tests a 48-worker generic-op DMA kernel: full tiles for
slot-zero recurrent state, but only the two 32-byte BF16 row-zero face segments per
convolution tile. It uses no compute kernel or arithmetic, does not overwrite other
rows, and avoids native restore's reconstruction. Both directions use preallocated
snapshot/live addresses and the identical correctness/isolation and ABBA gates.
Shapes, interleaved DRAM placement, dtype, both chips and non-aliasing are guarded.

Direct-copy runs 33998799270 and 33998989061 failed exactness; preserve their timings
as invalid-for-promotion. Diagnostics isolated 2,560 wrong values per convolution
tap/shard, starting at channel 16, before trace capture; recurrence matched. The
second face-row used a scratch offset of 32 against a DRAM face offset of 512.
The retry preserves the same NoC address alignment with scratch offset 512;
do not confuse this with changing the BF16 face layout or arithmetic.

Run [33999114362](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33999114362),
code `2b38da1`, passed 216 exact prefix/continuation cases, 30 stale-state controls
and 15 changed-idle-slot checks. The alignment fix retained exact recurrence and
all convolution channels on both chips. Copy kernel SHA256:
`4f921500b63817a5f26f288725a9da1fee9223841a68d058d96fac0f67a23428`.
In three paired ABBA blocks, save took 0.02266-0.02282 ms versus full-copy
0.05319-0.05346 ms (57.16-57.48% less); restore took 0.02265-0.02272 ms versus
0.05320-0.05336 ms (57.35-57.55% less). Padded allocation remains 72.4% smaller.
This is a validated per-layer snapshot microbenchmark, not a decode-speed claim.
The full-model oracle pins this kernel hash as its prerequisite.

## Full-model rollback oracle

Added `full-prefix`, gated behind the direct-copy layer test. It retains native
sequential target decode, not the experimental multi-row GDN forward. Compare
eager and trace logits, then freshly prefill each branch at lengths 63/64/65.
For retained prefixes 0..16, advance through a deliberately wrong rejected tail,
restore all 48 GDN slot-zero snapshots, rewind the explicit decode position and
compare two corrected steps against an uninterrupted native reference.

Attention KV rollback is logical: future rejected entries remain physically present,
but only positions below the current valid length may affect attention. Compare
dequantized valid KV tensor values on both chips for all 16 attention layers,
including the 64-token page transition, and all active GDN states/full logits.
Wrong logical-to-physical page mapping and omitted GDN restore are negative controls.
Fresh prefill per branch prevents one branch's correction writes contaminating the
next branch's valid prefix. Snapshot buffers are preallocated before trace capture.

This oracle is a prerequisite for the batched target verifier. It does not validate
tree branches, stochastic acceptance, request cancellation/epoch reuse, long contexts,
or a serving speedup. It must not turn the policy's `verifier_ready` on in serving.

Initial full-model run 33999230778 loaded and warmed native decode, then failed in
the harness's KV reader: native `ttnn.Shape` supports integer indexing, not Python
slice indexing. No full-model correctness case ran. Convert the shape to a tuple
before extracting dimensions, add a native-like shape regression fixture and retry.

Run [33999532634](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/33999532634),
code `d203018`, completed successfully. Artifact evidence confirms:

- All six eager/trace baseline comparisons matched full logits exactly.
- All 102 rollback cases passed: lengths 63/64/65, eager and trace, every retained
  prefix 0..16. Both correction steps matched full logits, all 48 GDN active states
  and logically valid KV values in all 16 attention layers, on both chips.
- All six negative-control configurations detected both stale GDN state and wrong
  page mapping (12 checks), rather than merely exiting successfully.
- Two reusable all-layer compact snapshot sets occupied 201,326,592 bytes per chip.

This establishes full-model serial rollback correctness for the tested page-boundary
fixtures. Rejected KV entries need not be physically erased in this path: rewinding
the decode position and restoring GDN state kept the rejected suffix from affecting
the checked continuation. It does not establish that concurrent multi-row KV writes
to shared pages are safe. The next gate is batched attention/target verification
against this oracle, including shared-page write hazards, then its measured cost
curve. No speculative serving flag, precision setting or deployment changed.
