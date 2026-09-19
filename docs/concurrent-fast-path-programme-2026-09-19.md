# Programme: concurrent users on the fast path

**Target.** 200 tok/s per user at 161k context and 30k prefill, 4 concurrent users
first, then 8.

**Blocker found 2026-09-19.** The fast T16 path refuses the configuration outright.
`scripts/ci/serving_fast_policy.py:21`:

```python
if (scheduler.max_num_seqs != 1 or scheduler.async_scheduling is not False
        or parallel.tensor_parallel_size != 1 or ...):
    raise ValueError('Synchronous single-request single-worker serving required; TT mesh supplies TP2')
if (cache.block_size != 64 or cache.enable_prefix_caching is not False
        or config.lora_config is not None or config.model_config.max_model_len != 4352):
    raise ValueError('Initial fast profile requires 64-token pages, 4352 positions, ...')
```

So today you may have **either** speculation **or** concurrency, never both:

| Path | Concurrency | Context | Speculation | 200 tok/s/user? |
| --- | --- | --- | --- | --- |
| Plain | 4 users proven (run 35410432516) | 161k proven | no | No — 19.5 ms weight-pass floor against a 5 ms budget, 3.9x short |
| Fast T16 | pinned to 1 | pinned to 4352 | yes, ~12 committed/pass | Only for one user at 4352 |

The arithmetic needs both: speculation to get ~12 tokens per weight pass, batching to
amortise that pass across users.

**Why this looks tractable.** `internal_batch_capacity()` already returns
`native_gdn_slots=8`, and `serving_plugin_patch.py` rewrites the plugin's
`get_tt_max_batch_size` to use it. The device side is already provisioned for 8. The
pin is in the scheduler contract, and the message calls it the *initial* fast profile.
That is a conservative starting constraint, not a discovered hardware limit — which is
a hypothesis this programme tests first, not an assumption it builds on.

## Tasks

Each task states its gate. A task is not done until its gate produces evidence.

### T1 — Map every pin to the assumption behind it

| Pin | Verdict | Evidence |
| --- | --- | --- |
| `max_num_seqs == 1` | **Load-bearing** | serving_lifecycle.py:20 holds one request_id and one capture; line 57 rejects a second scheduled request; FastRunnerBridge binds one request's runner state; serving_runtime.py:39 sets trace bucket 1 |
| `max_model_len == 4352` | **Conservative** | capture_plan is parameterised by position with a general 256-capacity loop; coding_context_request.py already contemplates 8192 and 65536 |
| `async_scheduling is False` | Load-bearing | FastRunnerBridge requires non_dp_async_scheduling false |
| `tensor_parallel_size == 1` | Conservative label | TP2 comes from the TT mesh, not vLLM's parallel config |

A further ceiling sits behind the concurrency pin: capture_plan caps max_verify_rows
at 32 while four T16 users need 64, and capture buckets are prepared fixtures rather
than runtime-computed.
Read the fast-path runtime and find what actually depends on `scheduler_requests=1`,
`verifier_rows=16`, `context_tokens=4096` and `max_model_len=4352`. Classify each pin:
**conservative** (nothing downstream assumes it) or **load-bearing** (something breaks).
*Gate:* a written table, pin by pin, citing the code that does or does not depend on it.
*Risk:* if most are load-bearing this is a much larger job and the programme re-scopes.

### T2 — Lift the context pin

**Code done; hardware gate outstanding.** The equality became a floor (any whole
number of 64-token pages at or above 4352) and context_tokens is derived as
max_model_len - 256. Eight tests pass: 163840 accepted, 4353 and 8100 rejected as
partial pages, concurrency pin asserted to still hold, canary's 4352 unchanged.

serving_plugin_patch.stage() copies the policy into src/vllm_tt_plugin at image build
time and the plugin imports from there, so mounting scripts/ci does not override it.
The gate needs an image graft - the delivery path Lever N uses.
Allow `max_model_len` above 4352 where nothing depends on it. The plain path already
serves 163,840, so the model handles the positions; the question is whether the fast
path's capture planning does.
*Gate:* fast path starts at a context above 4352 and serves one correct request.

### T3 — Lift the concurrency pin

**Re-scoped by T1: this is not a pin lift.** Session state is singular from
serving_lifecycle down, so it means per-session capture and bridge state, a trace
bucket sized for N, and a verify-row budget above 32 with capture buckets prepared for
the wider rows. Shares its core with **Lever N milestone M2** (alternation policy in
the plugin scheduler) - extend that rather than designing a second one, and sequence
after Lever N M1 since resumable prefill is what lets a second session start.
Allow `max_num_seqs` up to `native_gdn_slots` (8). The verifier processes 16 rows at
T16 for one request; establish what it does for N.
*Gate:* fast path starts with `max_num_seqs=4` and serves 4 concurrent correct requests.

### T4 — Unit tests before hardware
Extend `test_serving_fast_policy.py` for the new contract, and add the module to
`qwen-integration-cpu.yml` — untested policy changes are how a silent regression
reaches the rig. See the 63% dead-test finding.
*Gate:* tests fail against the old policy and pass against the new one.

### T5 — Bring-up at 4 users, 161k, fast path
Combine T2 and T3 on hardware. Reuse `qwen-cycle-bench.yml`, which already stages the
draft config, the dflash fixtures and the serving image.
*Gate:* server ready, 4 concurrent streams, coherent output.

### T6 — Measure inter-token latency
The number that decides the programme. 200 tok/s/user is **5.0 ms ITL**. Report the
fraction of target.
*Gate:* ITL median at 4 streams, first token dropped, host load recorded.

### T7 — Decide on 8 users
8 users at bf8 needs 63.1 GB against 64 GB and does not fit; it forces bf4 KV, which
risks the acceptance rate the whole speculative budget depends on.
*Gate:* only attempted if T6 shows 4 users within reach of target.

## The 4K/256 profile is pinned in four places, not one

Found by lifting them one at a time; each lift revealed the next. This is
defence in depth, not an oversight — the fast path is qualified for exactly one
request shape and refuses to run outside it at every layer.

| # | Location | Constraint | State |
| --- | --- | --- | --- |
| 1 | serving_fast_policy.validate_fast_config | max_model_len == 4352 | **lifted** (T2) |
| 2 | serving_fast_policy.validate_request_sampling | prompt_tokens == 4096, max_tokens == 256 | **lifted** |
| 3 | dflash_device.__init__ line 40 | bounded prefill, five DFlash2 layers, pinned TP2 | not lifted |
| 4 | dflash_combined_request line 80 | len(prompt) == 4096 and max_new_tokens == 256 | not lifted |

Proven on hardware along the way: the engine **starts and allocates cleanly at
65,536 positions with the fast path enabled**, ready in 100 s, twice
(runs 35412244363, 35412592950). The context ceiling is not a hardware limit.
That result stands whatever happens to the remaining pins.

**Why lifting the rest does not reach the target.** All four are context/output
pins. The concurrency pin is separate and load-bearing, so defeating all four
still yields one user at 65k rather than four at 161k. Each lift also moves the
configuration further from anything qualified — every validator here returns
serving_qualified=False by design.

The path that does reach the target is Lever N M1 (resumable prefill) then M2
(scheduler alternation), scoped at 2.5 days in its own doc, and it does not
require defeating qualification boundaries.

## Standing constraints

- Do not change serving defaults without authorisation. This programme changes a
  **validation contract**, not a default; nothing ships to the endpoint from here.
- One device-opening step per CI job. Repeated open/close wedged a card on 2026-09-19;
  recovery is `tt-smi -r /dev/tenstorrent/<id>`.
- Hardware work goes through CI tags allowlisted in org runner group 3. A PATCH to that
  group wipes its five selected repositories — restore them every time.
- Verify edits by grepping for the **new** behaviour. Two runs were lost to patches
  that silently did not match but still parsed.

## Not in scope

- DRAM-core prefetch — measured 24.7% slower, see
  [dram-prefetch-verdict-2026-09-19.md](dram-prefetch-verdict-2026-09-19.md).
- Card B — physically absent, `PresDet-` at switch port `f2:01.0`, needs on-site hands.
- Firmware — 19.12.0.0 is fine; fabric initialises (`FABRIC_1D`, both devices).

## Relationship to Lever N

Lever N (`docs/lever-N-prefill-decode-interleave.md`) addresses prefill blocking decode
on the **managed endpoint**, which already runs 8 scheduler slots without the fast path.
This programme addresses the **fast path** refusing more than one slot. They meet at the
same place — a scheduler that interleaves several streams against one weight pass — and
T3 should not invent an alternation policy that Lever N's M2 already designs.
