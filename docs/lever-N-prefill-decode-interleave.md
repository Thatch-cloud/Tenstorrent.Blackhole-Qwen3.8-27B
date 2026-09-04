# Lever N — prefill/decode interleaving on the Qwen3.8-27B endpoint

**Status:** design (2026-09-04). Nothing implemented. Companion to `optimisation-plan.md` §K/N.
**Owner surfaces:** the Tenstorrent Qwen model in the serving image (`models/demos/blackhole/qwen36/tt/`),
Tenstorrent's vLLM plugin in the same image (`/opt/vllm-tt-plugin/src/vllm_tt_plugin/`), and one
per-model kwargs entry in Thatch.Server (`py/serving/engine.py`).

## 1. The problem, measured

On the managed endpoint (`thatch-inference-Qwen-Qwen3.8-27B`, 2026-09-04 04:23Z) while it served
real traffic:

| engine log line | meaning |
| --- | --- |
| `Running: 2 reqs, Waiting: 3 reqs` | five requests in flight for 8 slots |
| `Avg prompt throughput: 4572.8 tokens/s` | a ~45k-token prompt being prefilled |
| `Avg generation throughput: 0.0 tokens/s` (same window) | every decode stream stalled |
| 60-token probe at the container: 17.5 s to first token, then 38.5 ms/token | queued behind the prefill |
| same probe via `api.thatch.cloud`: 41.6 s to first token, then 46.8 ms/token | plus gateway transit |

A 45k-token prompt is ~10 s of prefill at that rate; a full 65,536-token prompt ~14 s. For that
whole time no other stream produces a token. Coding-agent clients (`pi`) run near-full-context
turns routinely, so this is the steady state of a multi-user endpoint, not a tail event.

## 2. Why, in the code

**No mixed steps.** `vllm_tt_plugin/scheduler.py`: "No mixed prefill+decode batches: each batch is
either all-prefill or all-decode." `lane_scheduler.py::TTLaneCoordinator._negotiate_forced_mode`:
"if *any* lane wants to prefill, the whole step is prefill-only"; `_local_prefill_intent` says a lane
wants to prefill when it has a partial prefill, or a waiting request and either nothing running or
spare capacity. With traffic arriving, decode-only steps happen only when nothing is waiting.

**No chunk boundary to yield at.** The managed container runs `enable_chunked_prefill=False`,
`max_num_batched_tokens=2048` (its own engine log). The prompt is therefore one vLLM step. Inside
that step the model chunks it itself: `qwen36_vllm.py::prefill_forward` → `_prefill_forward_tp_batched`
→ `model.prefill_paged_slots` → `prefill_traced_chunked`, which resets the GDN state, replays the
2048-token chunk trace `num_full` times (`_prefill_traced_chunked_tp`), runs the tail through the
masked bucket, and returns the last-token logits. The scheduler sees one opaque step.

**What already exists for chunked continuation.** The plugin's runner handles it generically:
`model_runner.py::_prepare_model_inputs` sets `input_positions = num_computed_tokens`,
`prompt_lens = input_positions + chunk_lens`, and `intermediate_prefill_mask = prompt_lens < num_tokens`;
`submit_prefill` passes `start_pos=model_input.input_positions` and `prompt_lens`;
`_finish_lane_sync` routes intermediate rows through `_build_chunked_prefill_output`, which samples
nothing for them. The scheduler keeps a partial prefill in `running` with `is_prefill_chunk=True`
and schedules its next chunk on a later prefill step. **The model ignores `start_pos`**: every call
prefills `[0, prompt_lens)` from a reset state. Enabling chunked prefill today would be *correct*
(each chunk recomputes the prefix and rewrites the same KV) and quadratic: a 45k prompt in 2048-token
chunks costs ~22 steps of growing length, ~110 s instead of ~10 s. Do not flip the flag without §3.1.

**State the model carries across chunks.** One persistent B=1 GDN prefill scratch
(`_bind_gdn_prefill_scratch` rebinds every GDN layer's `rec_state` / `conv_states` / `conv_carry` to it;
`_unbind_gdn_prefill_scratch` restores the batched `[B, …]` decode buffers without freeing the scratch).
The chunk trace is captured against the scratch's addresses (`warmup_model_prefill`), so the scratch is
the only place a chunk sequence can run. `prefill_paged_slots` snapshots the scratch to host after the
prompt and uploads it into the request's decode slot with `_write_gdn_slot`. The per-request RoPE
table is host-side (`_build_request_rope`, staged once for the whole prompt at chunk 0; per-chunk
`cos/sin` sliced from it). Attention KV is paged and written per chunk through the page-table slice
`page_table[:, blk0 : blk0 + blocks_per_chunk]`.

**Async decode drain.** The runner waits for all pending async decode steps before a prefill
(`must_drain_pending_async_steps` / `wait_for_all_pending_async_steps`); the scheduler docstring
states the invariant ("every async op is forced to complete before the next prefill"). Any
interleaving pays one drain per transition into prefill.

## 3. Design

Three pieces, in dependency order. Each is independently testable.

### 3.1 Resumable prefill in the model (the enabling change)

Contract, at the vLLM entry (`qwen36_vllm.py`):

```
prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos=…, empty_slots=…)
  for each row u:   process prompt tokens [start_pos[u], prompt_lens[u]) of request u
                    start_pos[u] == 0            → first chunk: reset + stage the request's RoPE
                    prompt_lens[u] == num_tokens → last chunk: tail + logits + slot write
                    otherwise                    → intermediate: no logits (return zeros), no slot write
```

`prompt_lens` is the chunk END (the runner's definition), not the prompt length; the true prompt length
is `tokens.shape[1]` for that row (the runner sends `token_ids_cpu[req, :max(prompt_lens)]`, so the
row holds the prompt up to this chunk's end — the model needs the total length only to know whether
this is the last chunk, which the runner's `intermediate_prefill_mask` already encodes; pass it as a
kwarg `is_last` rather than re-deriving).

Model side (`model.py`), a resumable form of `prefill_paged_slots` / `_prefill_traced_chunked_tp`:

```
prefill_paged_slots_range(token_ids_list, page_table, empty_slots, starts, ends, is_last, valid_lens)
  bind scratch                                   (Python attribute swap; no device work)
  for u:
    if starts[u] == 0: _reset_gdn_state_for_new_sequence(); _build_request_rope(full prompt row)
    stage the full page table                    (already padded/clipped to the captured width)
    for each FULL chunk c with c*2048 in [starts[u], ends[u]):   replay the chunk trace (as today)
    if is_last[u]:
       tail = prompt_len - num_full*2048  →  prefill_masked_bucket(chunk_start = num_full*2048)  (as today)
       logits; snapshot rec/conv → host; _write_gdn_slot(empty_slots[u], …)
    else:
       synchronize; return zero logits for the row
  unbind scratch                                 (every step; the scratch's contents persist)
```

Invariants the design relies on:

- **Chunk alignment.** Continuation starts are multiples of 2048: vLLM's per-step budget must equal
  the model's chunk size. Assert `start % chunk_size == 0` and `max_num_batched_tokens == chunk_size`
  at model load; a mismatch is a configuration error, not something to paper over.
- **One in-flight prefill per lane (v1).** The scratch and the host RoPE table are single-occupancy.
  Between two chunk steps of request A, a prefill of request B (even a 200-token masked-bucket one,
  which resets the GDN state) would corrupt A. The scheduler enforces this (§3.3). Decode steps in
  between are safe: the decode trace runs against the batched buffers, and the scratch is unbound.
- **Slot stability.** `_alloc_prefill_state_slots` re-derives the slot on every prefill step, preferring
  the request's row; the slot write happens only on the last chunk, so an intermediate slot move is
  harmless. `_release_dead_state_slots` drops a preempted request's claim; a preempted partial prefill
  resumes from `start_pos=0` (the plugin replays the prompt plus generated tokens), which the contract
  handles as a fresh first chunk.
- **Numerics.** Per-step replay is the same trace with the same staged inputs as the in-call loop;
  the only difference is a `synchronize_device` at each step end instead of every 8 chunks. Expected
  byte-identical; gated in §5.

**v2 — scratch parking (follow-up, not v1).** Park the in-flight request's scratch (device-to-device
`ttnn.copy` of each GDN layer's rec/conv into a per-request parking buffer, ~38 MB/device per request)
before another prefill runs and restore it after. That lifts the one-in-flight rule: short prompts can
be admitted mid-prefill, and several long prefills can round-robin. Costs a few ms of copies per
switch. Do it once v1 is measured; it is the piece that keeps a 200-token chat snappy while a 45k
prompt is in flight.

### 3.2 vLLM-level chunking on

Thatch.Server `_PER_MODEL_VLLM_KWARGS["Qwen/Qwen3.8-27B"]` gains
`"enable_chunked_prefill": True, "max_num_batched_tokens": 2048`. Nothing else changes in that repo.
With §3.1 absent this setting is the quadratic path above, so the two land together (the image with
§3.1 first, then the kwargs).

### 3.3 Alternation policy in the plugin scheduler

Two small changes, both behind `TT_PREFILL_DECODE_INTERLEAVE=1` (default on when chunked prefill is
enabled, so a stock image is unchanged):

1. `TTScheduler._schedule_prefill_only`: when `running` holds a partial prefill, also hide `waiting`
   and `skipped_waiting` (the same trick `_schedule_decode_only` uses) so the step schedules only the
   continuation. This is the one-in-flight rule; it costs new arrivals a wait behind the in-flight
   prompt (v2 removes that).
2. `TTLaneCoordinator._negotiate_forced_mode`: keep a counter of consecutive prefill steps. If any lane
   has a partial prefill *and* any lane has running decodes, alternate: after each prefill-chunk step
   force `DECODE_ONLY` for `R = TT_DECODE_STEPS_PER_PREFILL_CHUNK` steps (default 1), then
   `PREFILL_ONLY`. Unchanged otherwise: no running decodes → prefill; no prefill work → decode; the
   existing zero-token fallback to decode stays.

Pseudo-code for (2):

```
intent = max(lane_intent)
if intent == PREFILL and interleave and any_running_decode() and any_partial_prefill():
    if self._prefill_streak >= 1 and self._decode_credit < R:
        self._decode_credit += 1;  return DECODE_ONLY
    self._decode_credit = 0; self._prefill_streak += 1;  return PREFILL_ONLY
self._prefill_streak = 1 if intent == PREFILL else 0;  self._decode_credit = 0
return from_prefill_intent(intent)
```

## 4. Expected effect

| quantity | now | after (R=1) |
| --- | ---: | ---: |
| decode stall while a 45k prompt is in flight | ~10 s (65k: ~14 s) | ~0.5 s per chunk (2048 @ 4.6k tok/s + drain) |
| the long request's prefill time | ~10 s | ~11–12 s (+1 decode step + drain per chunk) |
| decode ITL between stalls | 38–47 ms | unchanged (decode path untouched) |
| new arrivals during a long prefill (v1) | wait for it | still wait (v2 fixes) |

The transition cost per cycle is one decode step (~46 ms at B=8) plus the drain of the async decode
pipeline; with 22 chunks that is ~1–1.5 s on a 10 s prefill.

## 5. Measurement and gates

**Harness** (standalone container from the patched image, both cards; `optimisation/rig/`):
eight short-prompt chat streams decoding continuously; at t=5 s inject one 45k-token request.
Record per-token gaps on the eight streams and the long request's time to first token. Report the
streams' p50/p99/max inter-token gap and the long request's prefill wall time, before and after,
interleaved runs.

**Gates, in order:**
1. Chunked-vs-whole equality: greedy output of a 4,095-, 12,288- and 45,000-token prompt identical
   between one-call prefill (today's path) and per-step chunks (§3.1). Byte-identical is the bar.
2. The stock 60-item GSM8K gate at 57/60 (decode untouched; run anyway) and the 262k continuation
   check, both with chunked prefill on.
3. The harness numbers above, plus 8-stream ITL unchanged when no long prompt is in flight.

## 6. Risks and open points

- **`_build_request_rope` and the masked-bucket reset are single-occupancy** → the one-in-flight rule is
  load-bearing in v1; a bug there corrupts a long prompt silently. Gate 1 at 45k catches it only if the
  harness also admits a short prompt mid-prefill — add that case to gate 1.
- **Async-decode bookkeeping under frequent transitions.** The plugin was built for occasional prefills;
  alternating every step exercises `_pending_async_steps` / drain paths constantly. Expect a perf cliff
  before a correctness one; measure R=1 vs R=2.
- **Structured output.** `skipped_waiting` (grammar compilation) is hidden during continuation steps;
  those requests wait like any other new arrival. Same as today's decode-only hiding.
- **Multimodal** stays on the B=1 single-sequence path (`max_num_seqs=1`), untouched.
- **Chunk trace page-table width** (`_chunk_full_page_table_buf`) vs vLLM's per-step page table: already
  padded/clipped per call; unchanged.
- **Speculative decoding** is off on this endpoint; the design does not consider it.

## 7. Plan and effort

| milestone | content | effort |
| --- | --- | --- |
| M1 | §3.1 in the model + gate 1 (equality at three lengths, plus the mid-prefill short prompt) | 1.5 d |
| M2 | §3.3 in the plugin + the plugin's scheduler unit tests extended for alternation and one-in-flight | 1 d |
| M3 | harness, measurements, gates 2–3 | 1 d |
| M4 | graft both into `Dockerfile.k`, kwargs PR in Thatch.Server, image workflow, agent restart | 0.5 d |
| v2 | scratch parking; re-measure new-arrival latency during a long prefill | 1–2 d, after M4 |

Delivery path is the one used for lever M: patches carried in the graft image, upstream afterwards.
