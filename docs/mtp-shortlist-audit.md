# MTP short-block follow-up

## Source audit

NInfer branch `feat/qwen3.8-nvfp4full`, pinned at
`53d4efcfb4a4eb84e0a3938fdae910e7f7557e44`:

- [Draft head conversion](https://github.com/cometkim/ninfer/blob/53d4efcfb4a4eb84e0a3938fdae910e7f7557e44/tools/convert/qwen3_6/common/draft_head.py)
  selects head rows using corpus token frequencies and carries an explicit global-ID map.
- [MTP execution](https://github.com/cometkim/ninfer/blob/53d4efcfb4a4eb84e0a3938fdae910e7f7557e44/src/targets/qwen3_6/impl/runtime/mtp_impl.h)
  retains draft/target hidden buffers and performs autoregressive proposal steps.

Our `speculative-decoding` branch already contains `Qwen36MTP`, paged draft KV,
hidden retention, and traced-draft groundwork. Its generation loop uses the full
target LM head and gathers the full vocabulary for each draft. Reuse that work;
do not start a second MTP implementation or confuse it with lookup drafting.

## Bounded next experiment

| Item | Requirement |
| --- | --- |
| Draft lengths | K1 and K3 against corresponding exact target verification widths |
| Draft head | Full vocabulary control versus a fixed precomputed shortlist |
| Ranking | Separate training/calibration data; never future target answers from the evaluated request |
| Special tokens | Required IDs cannot be silently discarded to meet the shortlist size |
| Target | Full vocabulary, unchanged weights and exact greedy verification |
| State | Correct MTP catch-up and rollback at every rejection position |
| Metrics | Draft/head, verify, commit, acceptance and total committed TG; include zero-acceptance rounds |

`draft_vocabulary.py` implements host-only shortlist selection, weight-row packing
and global-ID mapping. Rows are stored in ascending token-ID order for deterministic
ties. It is not yet a device draft head, integrated MTP runtime, or speed result.
Reducing draft vocabulary can reduce acceptance; measure the complete trade-off.

`draft_shortlist_device.py` now provides experimental device preparation and
single-row selection: replicated BF16 32K/64K head, row-major argmax, then a
UINT32 global-ID gather. There is no full-vocabulary collective or host logits
read in this path. Replication trades additional head memory for no draft-head
collective; its cost must be measured rather than assumed beneficial.

Host tests check operation ordering, ownership, ID mapping, and rejection of
unsupported shapes. Device correctness, tie handling, A/B/A trace replay, and
latency remain unqualified. Run those simulator gates before integrating this
path with the existing MTP runtime or promoting it to hardware.

The existing CI workflow selects the shortlist simulator with
`suite=learned-attention`, `simulator_only=true`, `learned_stack=false`.
It runs both shortlist widths with synthetic, analytically known winners,
including a tie and a high global special-token ID, on both simulated chips.
It checks eager execution, A/B/A captured replay, input preservation, buffer
addresses and a stale-input control. This is not learned-weight acceptance or
hardware latency evidence. `learned_stack=true` retains the five-layer gate.

## Target hidden-state integration

The old MTP branch intercepts `_lm_head` input. The current verifier requests
sharded logits, so integration must not depend on that head method executing.
`mtp_hidden_capture.py` instead wraps `_final_norm_decode` within an explicit
instance-local scope and copies its result to caller-owned fixed storage.
It performs no allocation or global environment switching inside the forward.

`VerifierEngine(retain_mtp_hidden=True)` now allocates each bucket's hidden buffer
before warming or capturing any verifier trace. Warm and captured forwards use
the same fixed-buffer scope. `verified_mtp_hidden(ticket)` exposes borrowed rows
only for the current verified ticket; retention defaults off.

Before live MTP requests, establish the initial anchor hidden from prefill and
publish only the last consumed target row after prefix selection. Prefix-zero
abort must preserve the previous anchor. Bucket retention is host-tested but
still needs device qualification and is not enabled in the candidate runtime.
MTP KV catch-up and rollback still require independent exact tests; merely
capturing every speculative row does not authorize using rejected rows.

## Request bridge implementation

- Reused `Qwen36MTP` from branch commit `1fcbd0cdf8341ddbb7ab9dc93da917f42f2709c0`
  (source blob `e00758ecbd0b06a24a225a29538e3000b90023ed`), rather than a second model implementation.
- `MTPDeviceStep` stages persistent embedding/hidden/position/RoPE inputs and
  prepares separate proposal and head-free catch-up traces. It supports the
  native full-vocabulary force-argmax control or the experimental shortlisted head.
- `MTPRequestRuntime` supplies proposals to the real request coordinator, then
  refreshes consumed MTP KV rows using **verified target hidden states**, not the
  recursively drafted hidden states. The next logical position excludes rejected rows.
- `measure_request(mtp_runtime=...)` enables hidden retention and neural-only
  routing. Its total decode clock includes proposals, verification, catch-up and
  commit. K1/K3/K7/K15/K31 are supported; short K is not assumed sufficient for 200 TG.

Host tests cover all selected prefixes, abort/reproposal, catch-up failure and
complete coordinator execution. The native device step is **not hardware qualified**.
`full-mtp-request` now connects these pieces for one real coding request (K7/T8):
full prompt features, aligned independent MTP KV, fixed-buffer accepted rows,
native full-vocabulary force argmax and exact native target/state comparison.
Preparation and MTP prompt initialization are separately reported and included
in setup-inclusive totals. Hardware qualification and measured TG are pending.

### Alignment and fixed-row preparation

`initialize_prompt` now pairs token `prompt[position + 1]` with target hidden
row `position`, fills exactly `prompt_length - 1` draft KV rows, and retains the
last valid target hidden as the seed anchor. Padding is excluded. The paired
`AlignedMTPStep` maps target token positions to draft cache/RoPE positions by
subtracting one. This follows the input-token shift with unchanged hidden/position
rows in [vLLM's base proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py).

`VerifierEngine` also prepares native slice traces for every valid hidden row in
each retained bucket, writing into one fixed destination. MTP catch-up uses these
through `engine.mtp_row`; the hot path does not create new row-extraction buffers
or dispatch uncaptured Python slice operations. These additions are host-tested,
not device-certified. The launcher captures the last decoder layer during fresh
eager prefill, applies the native distributed final norm, checks both replicas,
and initializes MTP before binding the request runtime. No future target outputs
are supplied to the drafter. The 705-test host suite passes.

The hardware suite rebuilds the audited lazy-link fix, then proceeds directly to
the coding request rather than exiting after another collective microbenchmark.
It uses the physical `p150_x2` descriptor and four-link sampler, not a global
four-link override. No serving defaults or device reset are part of this run.

First hardware launch `34191983233` failed before drafting: native and fresh
candidate prefill seeds differed. It produced no MTP throughput result. The
launcher had skipped prefill warmup before parking the native decode trace,
contrary to the runtime's prefill/JIT ordering requirement. The retry restores
bounded warmup at the actual coding context and needed verifier widths, warms
the feature capture and norm before trace capture, and checks seed equality
there as well. Hardware confirmation is still required; the guard is not relaxed.

Retry `34193704110` confirms the prefill fix: native and captured warmup seeds
match, and both post-trace request prefills return token `71093`. It then fails
before MTP construction because the new embedding path guard rejects ordinary
Hugging Face snapshot symlinks into the sibling `blobs` directory. The corrected
guard permits that store only for the pinned model/revision, while rejecting
index traversal and external symlinks. Read-only host header inspection confirms
the exact embedding and all 15 `mtp.*` tensors; index SHA256:
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.

The next launch validates checkpoint paths and headers before rebuilding or
opening cards. Allocated hardware builds use eight compile jobs within the
existing 24-CPU/96-GiB container limits; CPU simulator builds retain two jobs.
Neither change is a model throughput improvement. MTP TG remains unmeasured.

Run `34195514722` loads MTP, captures its proposal/catch-up traces, and initializes
all 169 shifted prompt rows. It then fails while preparing verifier hidden-row
extraction: unaligned native slice internally switches to row-major, where the
preallocated tiled destination has incompatible padding (32 physical rows versus
one required row). The caller now lets native slice finish its layout conversion,
then copies into the fixed destination; single-row buckets copy directly.
Temporary ownership stays within warmup/capture, not hot-path replay.

Real-TTNN simulator gate `20260908T065049Z-415` passes T1/T2/T4/T8 A/B/A replay:
90 exact row checks on both chips, 24 source-preservation checks, and 30 stale-row
negative controls. No hardware throughput is inferred. The hardware request suite
now runs this exact row test before loading the full model, then continues into
the complete coding request. The 709-test host suite also passes.

## Complete hardware MTP request - 2026-09-08

[Run 34196777661](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34196777661)
on `c0810fd` passes the hidden-row gate and the full coding request. This supersedes
the pending native-MTP device qualification above; the shortlisted head is still
unqualified and unused. No future target answers or shortlist calibration are used.

| Metric | Measured result |
| --- | --- |
| CTX / streams / draft K / verify T | 170 / 1 / 7 / 8 |
| Output | 150 committed decode tokens; EOS reached |
| Native / MTP committed TG | 19.7357 / 48.8346 tok/s |
| Acceptance | 125 of 182 proposals; 26 blocks |
| Equality | All emitted tokens, final active GDN, valid KV and inactive slots |
| MTP setup / verifier setup | 3058.08 / 4345.42 ms |
| MTP prefill / decode | 298.06 / 3071.59 ms |
| Prefill + setup + decode | 10773.15 ms MTP; 7900.36 ms native |

The request is slower including unamortized setup; target model loading is outside
both totals. This one task is not a
held-out coding-quality, context/concurrency or sustained-serving qualification.
Target 200 TG remains unmet. Mean block costs are 66.29 ms verifier/readback,
38.82 ms drafting and 12.04 ms repair/commit. Sampling currently pads each draft
to 32 rows; test native logical rows without changing full-vocabulary greedy
semantics, and then measure the complete request rather than extrapolating TG.

Artifact: `full-mtp-request.json`, SHA256
`86a92b560ced248ce91e20a1ec19ea2007ac7a73be64748b5c517b724bfc305c`.

### Next experiment: native logical sampling rows

The opt-in `full-mtp-request` suite now compares padded/native/native/padded
requests with K7/T8 fixed. Only logical-row padding in the native full-vocabulary
sampler changes, in both MTP and target verification. Target quantization, four
fabric links, prompt, sampling semantics and cache repair remain unchanged.
Exact proposal routes, acceptance, outputs and target-state checks are mandatory.
MTP setup is included alongside verifier setup in the paired inclusive totals.
The 716-test host suite and 60 harness tests pass; no candidate hardware TG yet.

Simulator reduction gate `20260908T071725Z-311` exceeded its 600-second budget
after exact T1 checks and partial T2 checks. It is not a pass. The retry removes
redundant padded-control simulation and tests the changed native untilize/argmax
against Torch at the real 248320-token vocabulary. It does not simulate fabric.
The hardware gate retains both complete samplers, changing inputs, cross-shard
ties, boundary IDs and input-preservation checks before the request comparison.

Retry `20260908T072747Z-398` completes with exit 0 and clean mesh closure: 48 exact
native-row output checks, 48 source-preservation checks, and 8 stale controls.
The real vocabulary and all T1/T2/T4/T8 widths are covered on both simulated chips.
Report SHA256: `c109f779e3454be96144be2536ddc9f61ed82b8d83cc8f52e382b2aaf8473b8e`.
Sampler helper SHA256: `86f86a286a860529b95232e0251227fb9dca4df261d783932570750a4d50f744`.
[Hardware comparison 34200129693](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/34200129693)
is running on `fb17bbb`; no native-row TG claim until its complete requests pass.

### Native-row hardware result

Run `34200129693` completes successfully on `fb17bbb`. The full four-link sampler
passes 96 exact output checks, 48 source checks and 16 stale controls. All four
coding requests then pass exact tokens, final active GDN, valid KV and inactive
slots, with identical proposal routes and 125/182 accepted drafts per request.

| Metric | Padded control | Native-row candidate |
| --- | ---: | ---: |
| Complete requests / committed decode tokens | 2 / 300 | 2 / 300 |
| Committed TG | 49.7851 | 53.6001 |
| Mean draft block ms | 37.2168 | 29.0068 |
| Mean verifier/readback ms | 66.1132 | 65.3123 |
| Mean repair/commit ms | 11.6290 | 12.1784 |
| Mean complete cycle ms | 115.8518 | 107.5969 |
| Setup-inclusive post-seed TG | 19.3967 | 20.3595 |

Decode improves 7.66% in this ABBA block. Candidate repetitions individually
measure 52.8842/54.3356 TG; the reported 53.6001 uses both, not the faster sample.
Their matched native reference is 19.8221 TG. Mean candidate prefill + setup +
decode is 7690.81 ms versus 7890.12 ms native, with target weights already loaded.
No serving default changes, 200 TG claim, held-out quality or context sweep.
The remaining verifier cost, not another sampling sweep, is the next bottleneck.

Artifact `full-mtp-request.json` SHA256:
`cda6127029826c86cc59b9c2e3ea775d0850840a0ef580cce0afc113d5398b9d`.

### Next verifier comparison: short-context parallel attention

Simulator `20260908T075510Z-308` passes T8 replay at capacities 256/512/768,
including the actual position 170: 24 exact native-B1 output checks, 24 exact
causal-mask checks, 12 unchanged-KV checks and 6 stale-query/position controls.
The BF8 cache format matches the completed hardware request. Report SHA256:
`ce0412905dced6ba5cd93cf5b5a318c3fbff764478da99041e9d5d40faaaceff`.

An explicit short-context option is restricted to T8/four-row groups, with the
first 256-token family starting at position 128 so native chunk size stays 256.
The default long-context guard remains unchanged. Native T1/T2/T4 fallback handles
family boundaries; no padding changes the actual coding prompt. The same bounded
routing plan is applied to both arms, so neither acceptance nor proposal work may
change between serial and parallel attention. It can differ from the earlier
unrestricted T8 request, making the new paired control essential.

The next `full-mtp-request` compares serial/parallel/parallel/serial attention,
keeping native-row sampling, K7, target precision and four sampler links fixed.
The candidate shares each refreshed causal mask across all 16 attention layers.
All three short families warm before the native decode trace is parked. A same-
source hardware component gate precedes model execution, then the complete
requests must retain exact tokens, active GDN, valid KV and inactive slots.
The 722-test host suite and 60 harness tests pass; no short-attention hardware
or end-to-end speed claim yet.
