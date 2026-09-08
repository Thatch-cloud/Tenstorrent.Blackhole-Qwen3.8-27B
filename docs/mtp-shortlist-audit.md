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
