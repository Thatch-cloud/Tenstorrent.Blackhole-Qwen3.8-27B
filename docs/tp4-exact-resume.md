# Exact resume for returning agent turns (TP4, scoping)

**Date:** 2026-10-11. **Base:** `tp4/fusion-2` (a3d6583c). **Status:** sections 0-5 are the scoping (CPU only). Section 6
records what was then built on this branch: P0 (gate-only, default off), the Phase-0 telemetry (counting only) and the
split-invariance probe. Nothing has run on a card yet. Phase-0 numbers come from existing card artifacts (runs 38084230415
tier-timed, 38083229148 tier-returning, 38080393657 levern-hit beside six decoders). Every number is labelled **measured**
or **estimated**.

> **2026-10-19 update (`docs/tp4-exact-end.md`).** P0 is merged onto the ship-4 + Swift stack on `tp4/exact-end` and works beside the tile-aligned drafter window (the merge had switched that
> window off under a carry). The split-invariance probe (SP0, v839) found every 128-aligned resume DIFFERENT; `tp4-exact-end.md` explains why nothing finer than P0 is worth building without a
> kernel or numerics change, and adds the attribution arms (SP1) that say which. Section 5.4's "P1" and 5.5's "P2" stand as the E1 designs they were; their incremental value over P0 is
> re-modelled there (about 12% of today's prefill time, and only with kernel work).

## 0. Summary

- **The problem (measured).** A returning sticky-session turn resumes at `Q = floor2048(P_prev) - 2048`. It therefore
  re-prefills `(P_prev mod 2048) + 2048` tokens it has already processed, plus the new content. On the 13 typical
  continuation hits in the three runs, the tail was 3,485 to 5,762 tokens (mean 4,619). On the three resident hits,
  2.1k to 3.5k of the tail was repeat work. Prefill accounts for 54-63% of a resident hit's TTFT, and the engine rebind
  accounts for 27-38%.
- **The proposed decode-start / end-of-turn checkpoints do not work as stated.**
  - No prefill route can start at an arbitrary position. The KV fill is block-aligned (64), and the flexible SDPA
    q-chunk is 128 (section 1.2).
  - The nearest usable points are `floor128(P - 1)` and `floor128(E - 1)`.
  - No split off a 2048 boundary is proven byte-identical to a cold prefill (the Lever N design says so explicitly).
  - So they are a new exactness class: "equal to a split twin", not "equal to the cold twin".
- **Recommendation: P0 first.** P0 is the *undrop* plus a *drafter-window carry*: resume at `floor2048(P_prev)`, one
  chunk higher than today.
  - It stays byte-exact (E0) under the existing gates.
  - Estimated saving: 0.5 / 0.8 / 1.3 s per typical tool turn at 32k / 128k / 254k.
  - Cost: about 1 week plus one card window.
- **The 128-aligned variants come later, if at all.**
  - P1 (decode-start) adds only about 0.2-0.55 s per turn over P0, and P2 (end-of-turn) about 0-0.4 s more over P1.
  - Both need an owner decision on exactness and kernel or verify-path work.
  - Gate them on a 30-minute split-invariance card probe and on production client telemetry.
- **After P0, the next TTFT lever is the fixed floor, not the tail.** The floor is the engine rebind (0.53-1.35 s,
  bimodal), the per-step overhead (about 0.2 s) and the API side (0.05-0.38 s, growing with P).

## 1. Why `floor2048(P - 2048)` and `drop_last`

### 1.1 The constraints, with evidence

| Constraint | Where | Why |
|---|---|---|
| Checkpoints exist only at 2048 boundaries. | `qwen_prefix_registry.py:27-49` (model contract), `:71-72` (`CHUNK = 2048`, `BLOCK = 64`) | A capture is the GDN state after exactly `pos` tokens. It is taken between two chunk programs, so `pos` is a multiple of 2048 (`capture()` refuses `loop_pos != pos`). |
| A resume starts only on a 2048 boundary. | `lever_n_model_patch.py:151,179` (`assert start % chunk_size == 0`, `chunk_from = start // chunk_size`); `qwen_prefix_model_patch.py:1018`; `serving_lifecycle.py:322` (`R is not a 2048-token boundary`); `dflash_prefill_window.py:111-121` | The resumable chunk loop indexes whole chunks. A non-aligned start would silently re-run a chunk over a state that already contains part of it, so every layer refuses it. |
| Ceiling `Q <= P - 2048`. | `qwen_prefix_scheduler_patch.py:601-603`; `serving_lifecycle.py:324`; `dflash_prefill_window.resume_start`; `levern_policy.valid_step` (`:415-419`) | The DFlash drafter reads target taps for the window `[P - 2048, P)` (geometry `window: 2048`, `drafter_checkpoints.json`). Those taps exist only for positions this prefill computes. The parked-engine rebind projects them into the drafter history (`serving_parked_engines.rebind_device`, `:549`). |
| `drop_last` | `qwen_prefix_scheduler_patch.py:53-69, 297-304, 561-566`; vLLM 0.25.1 `v1/core/kv_cache_coordinator.py:480-496` (`drop_eagle_block = 0 in eagle_group_ids`) | `SpeculativeConfig.use_eagle()` includes dflash. The unitary coordinator therefore drops the last matched block of every hit, so the next turn's hit stops at least 64 tokens short of `floor2048(P_prev)`. The trim lands a chunk lower, and the plan captures `floor2048(P) - 2048` instead of `floor2048(P)`. The drop exists so that an EAGLE drafter can recompute the last block's hidden states. TT's DFlash takes its taps from the prefill capture, so the drop has no function here. |
| The cap publishes only `[0, floor2048(P))`. | `qwen_prefix_scheduler_patch.py:686-700` (`cap`), docstring `:28-34` | vLLM serves the **first** block cached under a hash. Blocks written by the tail program or by decode are not what a full chunk writes, so publishing them would make every later hit over them non-exact for everyone. |
| Only chunk-program states are byte-exact. | `docs/tp4-packed-prefix.md:15-21, 54-62`; `docs/lever-n-tp4-design-2026-10-04.md:316-330` | Splitting at 2048 boundaries is op-sequence identity: the same programs run on the same bytes, and the GDN 128-row sub-chunks align with 2048. A step off 2048 "would run the chunk programs at another M (other matmul program configs and K-blocking, so other rounding) and a padded masked bucket. That is mathematically equal and not proven bit-equal." |
| Prefill program shapes | `fixtures/qwen36_model.py` (byte-identical to the image's `model.py`, sha `c977f380`): `:1943` buckets `(128, 256, 512, 1024, 2048)`; `:2181-2260` every bucket is warmed masked and full, plus every KV fill width 1..32 blocks; `:2292-2293` full chunks plus a tail | The fast path prefills eagerly (`trace_mode decode_only`, row marker `path=eager`), and nothing compiles after the decode traces are parked (#48536: the second request hangs). Full chunks and tails both go through `_forward_prefill_chunk_masked_tp`, with `valid_len` set. They differ only in `M` (bucket) and masking. |

### 1.2 What breaks at other start positions

| Start position | What happens |
|---|---|
| A 2048 multiple | Today's path. Byte-exact (E0). |
| A 128 multiple, off 2048 | Refused by every guard above. **Mechanically possible:** a head segment `[R, ceil2048(R))` through `prefill_masked_bucket(chunk_start=R)`. That uses the warmed bucket programs, the flexible SDPA's device `chunk_start` (one program per bucket for any position), and warmed fill widths, so it needs **no new compile**. **Not proven byte-equal:** the bucket `M` differs from 2048 (matmul K-blocking), and a bf16 rounding of the GDN state at R (`QWEN35_GDN_STATE_BF16=1` in serving, `gdn/tp.py:304`) may not happen inside the fused op. That makes it E1 unless the probe in 5.3 passes. |
| A 64 multiple, off 128 | **Wrong SDPA.** The flexible chunked SDPA runs `q_chunk = k_chunk = 128` and assumes `chunk_start % q_chunk == 0` (`graft/attention/tp.py:854-857`: "valid (chunk_start always divisible by 2048)"). The device-tensor `chunk_start` cannot be checked on the host. The GDN scan's sub-chunk is also 128 (`graft/model_config.py:156`). |
| Arbitrary | **Silent KV corruption.** `blk0 = chunk_start // 64` (`model.py:2049-2053`), and `paged_fill_cache` writes the segment from slot 0 of that block (`graft/attention/tp.py:827-839`). So rows `[floor64(S), S)` are overwritten and every new row is shifted by `S mod 64`. vLLM's prefix cache also shares whole 64-token blocks only. |

### 1.3 Is a decode-start snapshot free?

**Yes, as host bytes. It is unusable where it sits.**

- The final step of every prefill already reads the B=1 scratch to the host and writes the decode slot from that copy:
  - merged route: `levern_route.py:939-940` (`rec_snap, conv_snap = _read_scratch(...)`) and `:959-960` (`_write_gdn_slot(slot, rec_snap, conv_snap)`);
  - stock: `model.py` `prefill_paged_slots` `:1771-1773`;
  - G1 route: `qwen_prefix_model_patch.py:543-545`.
- That copy is the exact state decode starts from, after `P` tokens. `conv_states[1..3]` are the same bytes as the
  `conv_carry` a checkpoint stores (`gdn/tp.py:660-690`).
- Keeping it costs host RAM only: 78.4 MB raw bf16, or 106.95 MB preconverted (59-61 ms to convert, off the critical
  path; **measured**, T1/T3 logs).
- **But:**
  - it sits at `P`, and section 1.2 rules out a resume at `P` unless `P % 128 == 0` (1 in 128 prompts);
  - section 2 shows that a client that does not echo reasoning diverges at `P - 1` anyway.
- **The usable decode-start point is `R = floor128(P - 1)`. It is not free.** There are two ways to get it:
  - **Split turn N's tail at R.** This costs one extra eager forward on turn N: about 0.25-0.6 s (**estimated** from
    the measured T=128 final-forward medians of 0.36 / 0.49 / 0.71 s). That roughly cancels the saving on turn N+1.
  - **Have the fused GDN chunk op emit the state after sub-chunk `(R - floor2048(P)) / 128`.** This is a kernel twin,
    plus the conv carry of rows `[R - 3, R)`.

## 2. The chat template and the round trip

**The served template** is the Qwen3.8-27B snapshot `1d4bf0f2` `chat_template.jinja`, the one the prefix design's P0b
names.

- Line 116 renders `<think>\n{reasoning|trim}\n</think>\n\n{content}` for an assistant message when
  `preserve_thinking is undefined or true`, or when the message comes after the last real user query.
  **By default it strips nothing.**
- `preserve_thinking=false` strips reasoning from assistant turns before the last real user query. The contract warns
  on a server default that sets it (`serving_c2_contract.py:899-909`).
- The generation prompt ends `<|im_start|>assistant\n<think>\n` (`:163-168`).

**vLLM 0.25.1** returns the reasoning to the template only if the client echoes it:

- `protocol.py:484-486` renames `reasoning_content` to `reasoning`;
- `chat_utils.py:1752, 1776-1781` set both names for the template;
- `chat_utils.py:1858` turns a JSON `arguments` string into a dict.

**The Qwen3 parser keeps every tool parameter as `value.strip()`** (`vllm/parser/qwen3.py:59-77`), so values stay
strings, minus all edge whitespace.

**Render study (measured on real model output, CPU).** The study took 25 finished answers from the three runs, each a
complete tool call or stop with its output token ids, between 117 and 712 tokens. For each answer it re-rendered turn
N+1 with the template and the tokenizer of the same snapshot, then compared the result with
`tokens(prompt N) + generated ids`:

| Client behaviour | Tool-result turn | New user message | Useful exact point |
|---|---|---|---|
| Echoes reasoning (the replay harness does: `prefix_agent_corpus.py:438-458`) | 25/25 extend through every generated token, `<\|im_end\|>` included | 25/25 the same | **End of turn**: the state after `E - 1` tokens (the next prompt re-feeds `<\|im_end\|>`) |
| Drops reasoning (plain OpenAI clients) | 25/25 diverge at **`P - 1`**: prompt N ends `<think>` `\n`, the re-render has `<think>` `\n\n` `</think>` | the same | **`P - 1`**, so decode-start minus one, not the decode start |
| `preserve_thinking=false` | 25/25 extend fully (reasoning after the last query is kept) | 25/25 diverge **before P**, at the first earlier assistant's `<think>` | neither: only older 2048 checkpoints help |

**Tool calls (synthetic, same template, tokenizer and parser semantics).**

- These round-trip byte for byte: plain string arguments, content before a call, two calls, integer arguments, JSON
  object arguments (kept as raw strings) and multi-line bash.
- These diverge mid-call:
  - a `write` whose content ends in `\n` (the parser strips it);
  - an `edit` whose `old_string` starts with indentation;
  - reasoning that ends in a blank line (the template trims it).
- Answers cut by `max_tokens` are sent back without their tool call, so they also diverge mid-turn.

**Consequences for the design:**

- End-of-turn reuse depends on the client and on the arguments. The decode-start point must be `P - 1` to cover
  clients that drop reasoning.
- How often each case occurs in production is **unknown**. The existing counters (`qwen_prefix_chat_requests`,
  class `rewritten`) do not separate them, so section 5.3 adds a counter.

## 3. Prior art and what can be reused

| Piece | State | Reuse |
|---|---|---|
| Sticky sessions phase 1 (`s2/sticky-sessions`, merged; `QWEN_FAST_STICKY_SESSIONS`) | Keeps nothing of a finished request (`test_sticky_sessions.py:1-7`). Resumes at the 2048 checkpoint under the ceiling. | The trim, cap, plan, commit and eviction-coupling wrappers; `_granted_resume`; the capture ledger from R; fail-closed salts; the kill switch; the writer audit and the KV guard's pad sentinel. P0 changes `plan`, `select` and the coordinator call. The "keep the engine, KV slot and end-of-turn state per session" idea (phase 2) was never built. |
| Prefix store (`qwen_prefix_registry`, `_scheduler_patch`, `_model_patch`) | Token-verified LRU keyed by block hash at `pos`, bounded by bytes, with pins, in-flight plans for split prefills, and telemetry classes | Works for any 64-multiple `pos` in lookup and `tokens_match`. `plan`, `trim`, `floor_chunk` and the capture contract assume 2048, and an exactness-class tag is missing (needed by P1/P2). |
| Merged Lever N route (`levern_route.py`) | Sources `COLD`, `CHECKPOINT`, `SCRATCH`, `PARKED`; host park and restore proven byte round-trip | P1/P2 add a fifth source `EXACT` with a head segment. `PARKED` already moves a B=1 scratch to the host and back. |
| Parked engines (`serving_parked_engines.py`; TP4 port in base) | Rebind instead of build (`rebind_device`: history projected from the window taps, `DraftKVHistory` reseeded) | The window carry plugs in here (spliced history). The drafter-equivalence judge and the `drafter` negative control are the template for its gate. |
| `tp4/prefix-tiers` (not merged, 2026-10-11) | Preconverted checkpoints (106,954,752 B; restore `restored_ms` 9.0 unaudited, **measured**); host KV tier with raw block ops (`qwen_kv_read` v2: 13.6 / 22.8 GB/s); tenant quotas; admission classes | The preconverted form suits any new checkpoint kind. The raw block ops can move session-parked blocks (P2). The class telemetry (`returning_served`, `rewritten`, `lost_tokens_*`) measures template divergence in production. |
| The undrop (`QWEN_PREFIX_STICKY_UNDROP`) | Named as "a separate stage" in `docs/tp4-packed-prefix.md:118-121`; not built | That is P0. |
| Agent replay (`prefix_replay.scenario_agent_turns`; plans `agent-turns`, `agent-turns-prefix`, `agent-turns-baseline`; `prefix_agent_turns.py` comparator; `[TURN]` / `[STALL]` lines; job `H3` in `references/tp4-ship-262k-prefix-jobs`) | Exists: eight agents, metering shape, reasoning echoed | Gaps are listed in 5.6. |

## 4. Phase-0 measurement (existing artifacts, no card time)

Parsed from `records.jsonl` and `server.log` by step markers (`lever N step`, `GDN prefill conv engaged`, `lever N
route`, `sticky engine built`, `prefill gate` release). The split columns are:

- **pre:** engine arrival to the first step;
- **prefill:** the route, i.e. the forwards plus each step's setup;
- **post:** the last forward to the gate release (adopt, rebind, first token);
- **API:** the rest of TTFT (send, render, tokenize, emit).

### 4.1 Per request (measured)

| Run / case | P | Q | tail | TTFT | pre | prefill (steps) | post | API |
|---|---|---|---|---|---|---|---|---|
| tier-timed resident 32k | 33,834 | 28,672 | 5,162 | 3.46 s | 0.10 | 1.86 (2) | 1.31 | 0.05 |
| tier-timed resident 128k | 129,616 | 124,928 | 4,688 | 4.25 s | 0.07 | 2.68 (2) | 1.13 | 0.21 |
| tier-timed resident 254k | 249,471 | 245,760 | 3,711 | 4.64 s | 0.07 | 2.86 (1) | 1.23 | 0.36 |
| tier-timed returning (host tier) 32k / 128k / 254k | 35,377 / 131,149 / 250,919 | 30,720 / 126,976 / 245,760 | 4,657 / 4,173 / 5,159 | 3.83 / 4.08 / 6.49 s | 0.71 / 1.13 / 1.27 (restore inside) | 1.77 / 2.06 / 4.11 | 1.12 / 0.57 / 0.61 | 0.06 / 0.20 / 0.38 |
| tier-timed cold twins | 35,377 / 131,149 / 250,919 | 0 | whole | 7.83 / 36.95 / 97.40 s | | 6.66 / 35.68 / 95.11 (3 / 9 / 17) | 0.53 / 0.52 / 0.57 | |
| tier-returning, five hits at about 55-61k | 54,624-60,829 | 49,152-57,344 | 3,485-5,762 | 2.15-5.53 s | 0.06-2.87 (restore) | 1.25-1.87 | 0.60-0.65 | 0.08-0.10 |
| levern-hit beside 6 decoders (fx ship candidate) | 6,379 / 7,904 | 2,048 / 4,096 | 4,331 / 3,808 | 3.88 / 2.45 s | 0.07 | wall 1.82 / 0.95 (route 1.38 / 0.86) | 1.31 / 0.75 | **0.68 / 0.67** (busy queue) |

### 4.2 Per-forward costs (measured medians, solo; n = number of samples)

- **One full 2048-row forward:**
  - 0.31 s at 0-16k (n=521), 0.42 s at 16-48k (1,026), 0.58 s at 48-96k (913);
  - 0.71 s at 96-160k (378), 1.03 s at 160-224k (56), 1.20 s at 224k and above (18).
- **The final tail forward plus end-of-prefill** (logits, host snapshot, slot write):
  - T=128: 0.36-0.49 s up to 160k, 0.71 s at 224k and above;
  - T=1024: 0.42 / 0.73 s at 32k / 128k;
  - T=2048: 0.53 / 0.83 / 1.34 s at 32k / 128k / 254k.
- **Fixed per-forward floor:** about 0.3 s of eager dispatch, so a small tail is not cheap.
- **Lever N step:** about 0.2 s of setup per step (calibrated).
- **Busy:** one owed decode round of 120-160 ms between two steps.
- **Post (engine rebind) is bimodal:** 0.35-0.55 s or 0.92-1.35 s. In the slow mode about 0.6 s passes between `prefill
  slot 0, nothing to adopt` and the rebind ledger (`serving_request_factory.py:547-553`: the window-tap
  `capture.outputs()` and `rebind_refusal`). The cause is not identified.

### 4.3 Estimated TTFT per returning tool turn

The model is `TTFT = pre + Σ forwards + steps × 0.2 + capture + post(1.0) + API`. It is calibrated on the three
resident hits: modelled prefill 1.87 / 2.65 / 2.83 s against measured 1.86 / 2.68 / 2.86 s. All figures are averaged
over `r = P_prev mod 2048`, with `n_out = 300` and a template wrapper of 16 tokens. **All estimated.**

| ctx | new tokens | current (tail) | P0 undrop + window (tail) | P1 at floor128(P-1) | P2 at floor128(E-1) |
|---|---|---|---|---|---|
| 32k | 200 | 2.46 s (3,588) | 2.00 (1,540) **-0.47** | 1.85 (-0.61; +0.36 on turn N if split) | 1.80 (-0.67) |
| 32k | 500 | 2.55 (3,888) | 2.05 (1,840) **-0.50** | 1.85 (-0.70) | 1.85 (-0.70) |
| 32k | 2000 | 3.00 (5,388) | 2.39 (3,340) **-0.62** | 2.41 (-0.59) | 2.30 (-0.71) |
| 128k | 200 | 3.28 (3,588) | 2.52 **-0.76** | 2.32 (-0.96; +0.49 if split) | 2.25 (-1.03) |
| 128k | 500 | 3.41 (3,888) | 2.61 **-0.79** | 2.32 (-1.09) | 2.32 (-1.09) |
| 128k | 2000 | 4.07 (5,388) | 3.16 **-0.91** | 3.16 (-0.91) | 2.84 (-1.23) |
| 254k | 200 | 4.44 (3,588) | 3.20 **-1.25** | 2.81 (-1.63; +0.71 if split) | 2.63 (-1.81) |
| 254k | 500 | 4.64 (3,888) | 3.36 **-1.28** | 2.81 (-1.83) | 2.81 (-1.83) |
| 254k | 2000 | 5.66 (5,388) | 4.27 **-1.40** | 4.06 (-1.61) | 3.67 (-1.99) |

Further cases:

- **Busy** (7 seats decoding): add about 0.65 s of queue to every design, plus 0.14 s per extra Lever N step. P0
  removes one step: -0.56 / -0.85 / -1.33 s at new = 500.
- **Stall on the other seats:** every design shortens it by the same prefill amount, because each step blocks decode.
- **The undrop without the window carry** gains only when `r + Δ >= 2048`: -0.15 / -0.23 / -0.35 s at new = 200.
- **`n_out = 1000`, new = 500:** P0 gives 2.20 / 2.86 / 3.78 s and P2 gives 1.85 / 2.32 / 2.81 s.
- **The floor of every design** is about 1.8 / 2.1 / 2.5 s at the central rebind value, or 1.3-2.2 s in the fast
  rebind mode. It is dominated by the engine rebind and one eager forward.

## 5. Design

### 5.1 Exactness classes

| Class | Meaning | Who has it |
|---|---|---|
| **E0** | A hit is byte-identical to its cold twin (fresh salt, same messages): KV, GDN slot, logits, every token. | Today's hits; P0 |
| **E1** | A hit is byte-identical to its **split twin**: the same conversation replayed under a fresh salt with the same resume points. This proves the snapshot, restore and adoption are faithful, but not that the hit equals a miss. | P1 and P2, unless the probe in 5.3 shows that 128-aligned splits are bit-equal |
| **E2** | Semantic agreement with the cold twin: first-divergence token, teacher-forced top-1 agreement, max abs logit difference, over the agent replay. | Reported for E1 designs against an owner threshold |

Introducing E1 is an **owner decision**: a hit would no longer equal a miss. E1 state must never reach an E0 consumer.
vLLM serves the first block cached per hash, so blocks written by the tail program or by decode must stay
session-scoped, never in the shared cache.

### 5.2 P0: undrop plus drafter-window carry (E0)

**Resume at `Q = floor2048(P_prev)`, the boundary the cap already publishes.**

1. **Undrop** (`QWEN_PREFIX_STICKY_UNDROP=1`, gate-only, default off). Under sticky sessions only, the scheduler graft
   wraps the unitary coordinator's `find_longest_cache_hit` so that it calls the manager with
   `drop_eagle_block=False` (vLLM `kv_cache_coordinator.py:480-496`; audit the other `use_eagle` use at `:615`).
   `plan()` then captures the prompt boundary `floor2048(P)`, which is G1's own plan, instead of `floor2048(P) - 2048`.
   The pin is the stage's scheduler sha.
2. **Window carry** (`QWEN_PREFIX_WINDOW_CARRY=1`). With `Q > P' - 2048`, the window rows `[P' - 2048, Q)` precede the
   resume.
   - Those rows are a suffix of turn N's window rows below `floor2048(P_prev)`. Turn N computed them with full chunk
     programs at absolute positions, which is what the cold twin computes (E0).
   - (As built, section 6: the carried rows are the target TAPS, not the projected history - more bytes, but one splice
     point that every window consumer reads, and exact without assuming the projection is row-invariant.)
   - At turn N's rebind, read the projected history rows `[P_prev - 2048, floor2048(P_prev))` to the host and attach
     them to the checkpoint at `floor2048(P_prev)`: at most 2048 x 5120 bf16 = 21 MB, about 10.5 MB on average. Use
     the history (`HISTORY_SHAPE (1, 1, 2048, 5120)`, `serving_buffer_pool.py:161`), not the raw five taps (105 MB).
   - At turn N+1, splice the stored suffix with this prefill's own projected rows. Projection runs on independent
     32-row chunks with fixed configs (`serving_parked_engines.py:461-479`), so the splice is row-exact. That is still
     checked by digest.
   - Without the carry, the ceiling `Q <= P' - 2048` stays. P0 then gains only when `r + Δ >= 2048`.
3. **Relaxations under the flag only:**
   - `select()` ceiling (`qwen_prefix_scheduler_patch.py:601-603`);
   - `_granted_resume` (`serving_lifecycle.py:324`);
   - `dflash_prefill_window.resume_start` and `validate_prefill_chunks` (the ledger is seeded with the carried rows);
   - `levern_policy.valid_step` and `final_start` (the final step must still contain every window row it computes);
   - `prefix_judge.Oracle` (expected Q).
4. **Exactness argument.** The restored state is a chunk-program capture at an absolute 2048 boundary. The KV below Q
   is chunk-written, and the cap rule is unchanged. From Q, the same chunk programs run on the same bytes (Lever N
   design 3.4). The drafter window equals the cold twin's row for row. Only the boundary moves up one chunk.

### 5.3 Decision gates before any E1 work

- **Split-invariance probe** (card, about 30 min).
  - Mechanism: a gate-only route knob splits a cold prefill at a 128-aligned `R` inside a chunk, as the head segment
    in 1.2. The existing Lever N audit logs its digests (`QWEN_FAST_LEVERN_AUDIT`: GDN slot, logits, KV pages), and
    `levern_compare.py` compares them against the unsplit prefill.
  - Lengths: `R mod 2048` in {128, 1024, 1920}, at about 8k, 128k and 250k.
  - If bit-equal, prefill-side 128-aligned checkpoints are E0, and P1 becomes an E0 design.
  - The Lever N design expects inequality (matmul `M`, bf16 state rounding).
  - Harness: 1-1.5 days.
- **Client telemetry** (no card). Add a bounded counter in the API hook (`qwen_prefix_metrics.count_request`):
  - assistant messages that carry `reasoning`;
  - tool arguments with edge whitespace;
  - `n_out` and tool-result size buckets.
  - Read it on production for a week, together with the `rewritten` and `returning_served` classes.
  - Cost: 0.5-1 day.

### 5.4 P1: decode-start exact checkpoint at `R = floor128(P - 1)` (E1 unless the probe passes)

- **State at R** comes from a fused GDN chunk-op twin that also writes the state after a chosen 128-row sub-chunk, plus
  a slice of the conv input rows `[R - 3, R)`. This is a JIT/programmable-op twin, with no factory edit. The split
  variant is rejected: its extra forward of about 0.25-0.6 s on turn N cancels most of the gain.
- **KV of `[floor2048(P), R)`** is tail-program-written. Under E1 it is session-scoped. That needs a private
  publication path: a salt-derived extra hash key for blocks above the canonical frontier, or blocks parked outside
  vLLM's cache. vLLM has no native support for either.
- **Registry:** positions that are not 2048 multiples, plus an exactness-class tag. A grant of class E1 is admitted
  only by the matching session.
- **Route:** a fifth source `EXACT`. The head segment `[R, ceil2048(R))` runs through the warmed masked bucket at
  `chunk_start = R`, with no new compile, then chunks and the tail. The lifecycle, window and Lever N alignment rules
  go from 2048 to 128 under the flag.
- **Window carry** generalizes P0's.
- **Incremental value over P0:** about 960 tokens average, +0.2 / +0.3 / +0.55 s at 32k / 128k / 254k (**estimated**).
  It works for clients that drop reasoning.

### 5.5 P2: end-of-turn exact checkpoint at `R = floor128(E - 1)` (E1)

- **The state at R comes from decode.** The verify and commit kernels sum in another order, so P2 can never be E0.
  - Getting it means splitting a commit at each 128 crossing and device-copying the state into a per-seat buffer
    (about 19 MB per chip per seat in bf16).
  - The cost is **estimated** at 5-10 ms per round unless the copy is fused into the commit trace.
  - It touches the pinned verify path (`verifier_engine.py` is held byte for byte, so twins are needed).
  - The state's position also depends on whether EOS was consumed as an accepted draft, which must be recorded exactly.
- **Session parking:** on `_free_request`, decode-written blocks `[floor2048(P), R)` are kept, private to the session,
  and adopted by the next admission. This needs reservation accounting and coupling with eviction and the host tier.
- **Drafter carry:** the banks, 5 layers x `(1, 2, 2048, 128)` per chip, plus the history.
- **When it works:** only with reasoning echoed and whitespace-safe arguments (section 2). Otherwise the token check
  falls back to a lower Q.
- **Incremental value over P1:** 0-0.4 s at `n_out` 300-1000 (bucket granularity, **estimated**).

### 5.6 Gates and harness

**P0 (E0):**

- cold-twin pairs IDENTICAL on every continuation, including boundary cases on both sides of `r + Δ = 2048` (window
  carry engaged and not engaged) and previous prompts of 4095, 4096 and 4097;
- the audited attach (`QWEN_PREFIX_AUDIT`, `[PREFIX-AUDIT-CKPT]`), plus a new `[PREFIX-AUDIT-WINDOW]` line: the spliced
  history digest equals the cold twin's;
- the drafter-equivalence judge (acceptance statistics; bf8 proposals are not run-to-run deterministic);
- a negative control (a row-shifted splice must be caught);
- no program growth after warm;
- the kill switch;
- the agent-turns ABAB pair (`prefix_agent_turns.py`: text identical turn by turn, paired continuation TTFT);
- the levern-hit beside six decoders (busy TTFT and stall).

**P1/P2 (E1):**

- split-twin pairs IDENTICAL;
- the E2 report against the cold twin;
- isolation: E1 blocks are never served to another salt or to an E0 hit (fresh-salt and sibling probes);
- fallback: any divergence below R gives a lower Q and never an inexact hit;
- the kill switch.

**Harness gaps** (the agent replay exists):

- the Oracle must model the undrop and the window carry;
- a split-twin mode: earlier turns replayed under a fresh salt with full `max_tokens`, not 1;
- client modes: echo or drop reasoning, and `preserve_thinking=false`;
- whitespace-edged tool arguments in `prefix_agent_corpus`;
- answers allowed to finish (128-token caps cut most calls);
- a Q class in the `[TURN]` line.

### 5.7 Memory

| | Host | Device |
|---|---|---|
| P0 | As designed: window rows of at most 21 MB per checkpoint (projected history), about 67-73 preconverted checkpoints in the 8 GiB store instead of about 80. As built (section 6): the target taps, at most 105 MB (about 52 MB on average), about 40-55 checkpoints. One checkpoint per turn, as today. | none (the carried rows are uploaded for the turn's window build only, as a cold prefill's capture holds them) |
| P1 | +107 MB per turn (the checkpoint at R) | Session-held tail blocks of at most 32 x 2.23 MB = 71 MB (evictable) |
| P2 | +107 MB per turn, plus the drafter state (about 0.13 GB per session if parked on the host) | Decode-written blocks of about 37-48 x 2.23 MB; the per-seat snapshot buffer about 150 MB per chip for 8 seats (free about 4.9 GB per chip, **measured** at the hit) |

### 5.8 Card plan

**Every window is a production outage** (the cards serve production). Tags go one at a time, the CPU suite is green
first, and results are read from the grid of the run (13x10).

| Job | What | Minutes (estimated) |
|---|---|---|
| B0 | Image build from the pushed branch | 60 |
| X0 | Status, rescan, reset | 20 |
| SP0 | Split-invariance probe (5.3), the decision gate for P1 | 30 |
| E0x | Exactness chain on the undrop twin, audited, with boundary cases and the window audit | 45 |
| A1 / A2 | Agent turns, 8 agents: control, then undrop | 2 x 50 |
| G2 | levern-hit beside six decoders on undrop | 15 |
| T1 / T2 | Timed resident hits at 32k / 128k / 254k: control, then undrop | 2 x 35 |
| Z | Reset | 10 |

The total is about 350 minutes (about 6 h) plus about 3 minutes of queue per tag. P1 needs two more windows (op
qualification, then gates; about 6-8 h). P2 needs at least three.

### 5.9 Risks and effort

| Phase | Effort | Main risks |
|---|---|---|
| Phase 0: telemetry, probe harness | 1.5-2.5 eng-days, 1 card job | The probe route is gate-only code in the merged route. |
| **P0** | **5-7 eng-days, 1 window** | vLLM internals behind the drop (pin by sha, real-vLLM CPU test in `qwen-fast-vllm-cpu.yml`); the window splice's orientation and positions (a bad splice only costs acceptance, which the judge measures); Lever N plan rules once the window is not the prefill's own; fewer checkpoints in the store |
| P1 | 12-18 eng-days, 2 windows | E1 (owner decision); the GDN op twin and its qualification; session-private publication, which vLLM does not support; registry generalization off 2048 |
| P2 | 25-35 eng-days, 3+ windows | E1/E2 only; the pinned verify and commit path; per-round cost; session parking and reservation; client and template dependence |

**Recommended order:**

1. Phase 0: telemetry, plus the probe in P0's window.
2. P0.
3. The fixed floor: the bimodal 0.6 s gap before the rebind, the 0.2 s step setup, and the API render and tokenize at
   long P.
4. P1, only if the probe passes or the owner accepts E1 and telemetry shows that clients drop reasoning or give short
   answers.
5. P2, only if telemetry shows that most turns echo reasoning, round-trip, and have long answers.

## 6. As built (2026-10-11): Phase 0 and P0

Nothing in this section has run on a card. The CPU proofs are `test_exact_resume` (32 tests) and `test_tp4_split_probe`
(17 tests); the card pack is `scripts/ci/references/fusion-jobs/ER` (ORDER.txt there).

### 6.1 P0 switches (default off, gate-only: `serving_c2_contract.exact_resume_problems`)

| Switch | What it does | Where |
|---|---|---|
| `QWEN_PREFIX_STICKY_UNDROP=1` | Sticky sessions only. The coordinator's `find_longest_cache_hit` runs with an empty `eagle_group_ids`, for that lookup only, so vLLM's hit keeps its last block. `plan()` captures `floor2048(P)`, plus C0 while the carry is off. | `qwen_prefix_scheduler_patch.py` (install, `wrap`, `plan`) |
| `QWEN_PREFIX_WINDOW_CARRY=1` | Needs the undrop. After every sticky prefill, the window rows `[P - 2048, floor2048(P))` of the capture's five target taps are read to host (every chip's bytes) and filed on the checkpoint this request stored at `floor2048(P)`. A later prompt `P'` may resume there above `P' - 2048` when those rows reach down to `P' - 2048`. | `prefix_window_carry.py`; `qwen_prefix_registry.py` (`Window`, `attach_window`, `window_covers`); `select()`; `serving_lifecycle._granted_resume` / `_file_window`; `dflash_prefill_window` (the ledger counts the carried rows, and `outputs()` puts them in front); `levern_policy.valid_step` |
| `QWEN_PREFIX_WINDOW_CARRY_AUDIT=1` | Needs the carry; gate only. The uploaded rows are read back against the stored bytes (`transport`), and every sticky window's sha256 is logged. | `prefix_window_carry.after_prefill`, `Carried.upload` |

**Why the carried taps are exact (E0).**
- Turn N computed the rows `[P_prev - 2048, floor2048(P_prev))` with full 2048-row chunk programs at absolute
  positions. The cold twin of `P'` runs the same programs on the same tokens at those positions, so its taps there are
  the same bytes.
- The carry moves those bytes host to device unchanged: bfloat16, per chip, `ConcatMeshToTensor` /
  `ShardTensorToMesh` on the mesh's dimension 0.
- Every consumer of the window (the parked-engine rebind's projection, the cold build) sees a complete window of the
  usual shape. The CPU fake device checks this row for row: the spliced window equals the cold window of the same prompt.

**Markers** (`prefix_markers.scan` reads them; `c2_prefix_gate.exact_resume_findings` judges them):

```
[PINDIAG] prefix: window carry engaged undrop=<0|1> carry=<0|1> audit=<0|1> ceiling=...
[PINDIAG] prefix: window carry carried req= Q= P= rows=
[PINDIAG] prefix: window carry stored req= pos= rows= bytes= ms=
[PINDIAG] prefix: window carry skipped req= why=
[PINDIAG] prefix: window carry audit req= P= Q= carried= window_sha= transport= exact=
```

The sticky install line (`install sticky=1 lookahead=16 drop_last=True`) is unchanged: `drop_last` there is vLLM's own
setting.

**Off path.** With the switches unset, the graft makes exactly the decisions of the base commit's graft (loaded from git)
on a sticky DFlash scenario, and the registry keeps its keys.

**Cost (estimated).**
- The export reads at most 2048 rows x 5 taps x 5120 x 2 B, about 105 MB (about 52 MB on average). That is about
  0.04-0.1 s on turn N, by comparison with the 31 ms the 78 MB checkpoint read measures.
- The upload on the carried turn is at most the same bytes.
- The store charges the rows to the checkpoint: the 8 GiB store then holds about 40-55 preconverted checkpoints
  instead of about 80 (107 MB each plus 52 MB of window on average, 105 MB at most).

### 6.2 Phase 0

- **Turn telemetry** (on unless `QWEN_PREFIX_TURN_TELEMETRY=0`; counts only, no tokens, text, salt or tenant leave the
  engine).
  - Every salted admission is classified once against the newest remembered turn of its chain:
    `turn_extends`, `turn_in_answer`, `turn_at_decode_start`, `turn_at_p_minus_1`, `turn_at_p_minus_2`,
    `turn_before_prompt_end`, `turn_earlier`, `turn_first`.
  - Buckets cover the answer length, the new content and the in-answer offset.
  - These are exported as `qwen_prefix_turn_*_total` through the registry's stats file.
  - The API server counts every finished tool call's parameter values, and those carrying more edge whitespace than
    the template's one newline each side: `qwen_prefix_tool_args{kind}`. It wraps vLLM's Qwen3 arg converter and
    leaves the conversion unchanged.
- **Split-invariance probe** (`tp4_split_probe.py`, `C2_FABRIC_PROBE=split`).
  - It runs from the checkout inside an existing image.
  - It uses the image's own model with its first layers up to the first full-attention layer, under the production
    profile's environment.
  - Per length it compares the cold prefill, bit for bit, with:
    - a determinism control;
    - a 2048-aligned split (the harness control);
    - four 128-aligned splits.
  - It compares the GDN states, the logits and the KV rows, by region.
  - `SPLIT_PROBE verdict=IDENTICAL|DIFFERS|FAIL`. Template SP0, about 25 minutes (estimated).

### 6.3 The gate and the pack

- **Plans** (`c2_prefix_gate`, `c2_serving_job.PREFIX_PLANS`):
  - `exact-resume` is judged: strict cold-twin pairs, the oracle with the undrop and the carry, and the window digests
    of every pair equal under the audit.
  - `exact-resume-timed` is the ABAB's arm.
  - Both use `prefix_replay.scenario_exact_resume`: first prompts of about 9k, 33k and 120k tokens, then returning tool
    turns of 150, 400, 900, 1700, 2600 and 300 new tokens.
  - `exact_resume_compare.py` pairs the hits of two timed arms (the answers identical, the Q, the tail and the TTFT
    delta).
- **Pack:** B0, X0, SP0, ERA (the audited attach), ERX (the cold-twin pairs), ERC1/ERL1/ERC2/ERL2 (the timed ABAB),
  optional ERB1/ERB2 (the agent-turn replay at eight seats), and Z.
  - Estimated 350 minutes without ERB1/ERB2, 440 with them.
  - The twins `-fx-all-er` and `-fx-all-er-audit` are placed by WP0 from `scripts/ci/fusion-wp/ER.json`.

