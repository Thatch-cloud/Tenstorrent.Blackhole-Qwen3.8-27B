# What the plugin actually does, and where the Lever N design was wrong

Read from the image's own sources, extracted by the M1 gate's graft job
(`/opt/qwen-fast-plugin/src/vllm_tt_plugin`, `models/demos/blackhole/qwen36/tt/rope.py`).
Everything below is quoted behaviour, not inference. It corrects
[lever-N-prefill-decode-interleave.md](lever-N-prefill-decode-interleave.md) in three
places and re-scopes M2.

## The prefill_forward contract

`model_runner.submit_prefill` builds the kwargs:

```python
kwargs = {
    "tokens": model_input.input_tokens,
    "page_table": model_input.block_tables,
    "kv_cache": self.kv_caches,
    "enable_trace": self.trace_mode in ["all"],
    "prompt_lens": model_input.prompt_lens,
    "start_pos": model_input.input_positions,
}
```

and `_prepare_inputs` defines them:

```python
input_positions = input_batch.num_computed_tokens_cpu[req_indices]
prompt_lens = input_positions + chunk_lens
intermediate_prefill_mask = prompt_lens < input_batch.num_tokens[req_indices]
input_tokens = input_batch.token_ids_cpu_tensor[req_indices, :max_prefill_tokens]
```

| Field | Meaning |
| --- | --- |
| `start_pos` | `num_computed_tokens` per request. **Always present**, 0 on a full prefill |
| `prompt_lens` | the **chunk end**, `start_pos + chunk_len` — not the sequence length |
| `tokens` | rows sliced to `max(prompt_lens)`, so only up to this chunk's end |
| total length | `input_batch.num_tokens` — **never passed to the model** |
| `is_last` | **does not exist**; `intermediate_prefill_mask` is computed and never forwarded |

### Correction 1 — dispatch cannot key off `start_pos` being absent

Section 3.1 implies `start_pos=…` marks a chunked call. It is sent on every prefill, so
`if start_pos is None` never selects the one-shot path. Run 35415521079 is the evidence:
both arms went through `prefill_paged_slots_range`, all three lengths compared
byte-identical, and the gate correctly refused to pass because the baseline arm had not
taken the one-shot path. Equality between two runs of the same new code proves nothing.

The discriminator is a **nonzero start**. A prompt's first chunk keeps the one-shot path,
where the two are equivalent anyway: its end is chunk-aligned, so `tail_real == 0` and no
tail runs either way.

### Correction 2 — `is_last` is neither available nor needed

Section 3.1 says to "pass it as a kwarg `is_last` rather than re-deriving". There is no
such kwarg and the total length needed to derive one is not sent either. It is also
unnecessary:

- **The tail** already runs iff `tail_real > 0`, which is only true on a final chunk that
  misses a chunk boundary.
- **The logits** of a mid-prompt row are discarded by the runner itself:

```python
if (not is_decode and model_input.intermediate_prefill_mask is not None
        and bool(model_input.intermediate_prefill_mask[rows].all())):
    # Every row is mid-prompt; there is nothing to sample and the
    # placeholders are dropped when the output is built.
    next_token_ids = torch.zeros(sz, dtype=torch.int32)
```

So every step may return its logits and write its slot; the final step's values are the
ones that survive. The redundant intermediate slot writes are a v1 cost, not a defect.

### Correction 3 — the RoPE staging concern was unfounded

`rope.py`:

```python
if image_grid_thw is None and video_grid_thw is None:
    self._req_cos = None; self._req_sin = None; self.rope_delta = 0
    return 0                       # text-only: clears, never reads input_ids
...
t = torch.arange(start, start + length, dtype=torch.float32)   # 1D, on demand
```

Text-only clears the table without reading `input_ids`, and `prefill_cos_sin_torch`
computes 1D RoPE on demand for any absolute range. The M-RoPE table self-extends
(`_extend_req_table`). Staging from a chunk-end length changes nothing, and the batched
path is text-only by assertion. No fix required.

## M2 is in a different place than the design says, and is smaller

The machinery already exists. `TTScheduler` has `TTSchedulingMode`
(PREFILL_ONLY / DECODE_ONLY / DEFAULT), `set_forced_mode`, `_schedule_prefill_only`,
`_schedule_decode_only`, and `is_prefill_chunk` is a first-class concept throughout:
`_schedule_decode_only` already hides partial prefills "so their next chunk is not
scheduled into it either".

**Where the stall actually lives.** `LaneScheduler._negotiate_forced_mode`:

```python
intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
```

with, in `_local_prefill_intent`:

```python
has_partial_prefill = any(r.is_prefill_chunk for r in sched.running)
return int(has_partial_prefill or (has_waiting and ((not has_running) or has_capacity)))
```

A partial prefill always votes to prefill, and **any** lane voting prefill makes the whole
step prefill-only. So a chunked prefill in flight forces every step to be prefill-only
until the prompt finishes. Chunked prefill on its own therefore does **not** fix the decode
stall — it splits the prompt into chunks that still monopolise every step.

That makes M2 two small, precisely located changes:

1. **Alternation** in `LaneScheduler._negotiate_forced_mode` (not in
   `TTScheduler._schedule_prefill_only`, where section 3.3 puts it): when a partial prefill
   is in flight *and* some lane has running decodes, yield decode steps between prefill
   chunks instead of always voting prefill.
2. **One in flight** in `TTScheduler._schedule_prefill_only`: hide `waiting` and
   `skipped_waiting` when `partial_prefills` is non-empty, which is what section 3.3
   describes and is the one part it puts in the right place.

The risk is not size but interaction: the PREFILL_ONLY-to-DECODE_ONLY fallback in
`LaneScheduler.schedule` carries `finished_req_ids`, `free_encoder_mm_hashes` and
`preempted_req_ids` across a discarded pass, and an alternation policy must not lose that
bookkeeping.

## Correction 4 — chunked prefill is refused for this model at config time

`platform._apply_chunked_prefill_policy`:

```python
_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}
...
if scheduler_config.enable_chunked_prefill:
    logger.info("Chunked prefill is not supported for `model_type=%s`; disabling it.", model_type)
    scheduler_config.enable_chunked_prefill = False
    if max_num_batched_tokens < max_model_len:
        scheduler_config.max_num_batched_tokens = max_model_len
scheduler_config.long_prefill_token_threshold = 0
```

This model's HF `model_type` is **`qwen3_5`**, which is not in the set, so
`--enable-chunked-prefill --max-num-batched-tokens 2048` was overridden at config time
and the budget bumped back to `max_model_len`. Runs 35415521079 and 35415811328 each
produced exactly one `prefill_forward` call for prompts of 460, 3,524 and 5,918 tokens.

The set is, by its own comment, the model types "whose tt-metal generator accepts a
`chunk_start_idx`" — precisely the capability M1 adds. **M1 is therefore a three-file
graft** (`model.py`, `qwen36_vllm.py`, `platform.py`), not the two the design describes,
and section 3.2's plan to enable chunking through Thatch.Server
`_PER_MODEL_VLLM_KWARGS` would have been silently overridden by this same policy.

Once M1 ships the opt-in should become a plain `"qwen3_5"` entry in the set.

### The multimodal budget follows from it

Taking the allowlisted branch also sets `disable_chunked_mm_input = True`, because a
chunk boundary inside a multimodal item would split its embeddings from their positions.
vLLM then refuses a small batch budget outright:

```
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (16384) is larger
than max_num_batched_tokens (2048). Please increase max_num_batched_tokens.
```

The batched path is text-only by assertion regardless, so the gate declares
`--limit-mm-per-prompt image=0 video=0` on both arms. A production rollout has to make
the same decision explicitly: chunked prefill and multimodal items do not coexist at a
2048 budget.

## The scratch really does persist between steps

Design section 3.1 assumes it; `model.py` confirms it. `_bind_gdn_prefill_scratch` is a
pure Python attribute swap onto each GDN layer, `_ensure_gdn_prefill_scratch` allocates
only on first use, and `_unbind_gdn_prefill_scratch` restores the batched decode bindings
"WITHOUT freeing the persistent scratch". Nothing in bind or unbind zeroes it; only
`_reset_gdn_state_for_new_sequence` does, and the graft gates that on `start == 0`.

Decode steps between two chunks of one prefill are safe for the same reason: they run
against the batched decode buffers while the scratch is unbound.

## What M1 has actually shown on hardware

Run 35415521079, batched path, `max_num_seqs=4`, chunk budget 2048:

| Prompt | Tokens | Completed |
| --- | ---: | --- |
| approx_400 | 460 | yes |
| approx_3000 | 3,524 | yes |
| approx_5000 | 5,918 | yes |

`prefill_paged_slots_range` runs, produces coherent output, and chunked and unchunked
agree byte for byte through it. The gate still reports NOT PASSED because that comparison
was range against range. Equality against the original `prefill_paged_slots` is what the
corrected dispatch tests.

Also settled: the 5,000-token engine crash that wedged a card twice was confined to
`_prefill_forward_tp`, the single-sequence path. `prefill_paged_slots` and
`prefill_paged_slots_range` are only reachable when `model.args.max_batch_size > 1`, so
every run before this one at `max_num_seqs=1` was exercising neither.
