# tp4/sampled-verify: sampling on the TT fast path

Status: **built and CPU-green, never run on a card, no image built.** What exists: the request contract behind `QWEN_FAST_SAMPLED` (default off, byte-identical to today when off),
`scripts/ci/sampled_accept.py` (keyed noise, filter, draw, the coupled accept walk), and the device step (b1) in `scripts/ci/sampled_verify.py`: the per-chip top-64 readback in the packed
block's verify trace, the host merge and draws per packed entry, the sequential TP4 engine's host slice, the first-token graft on the plugin's host sampler, and the engine-side gate that opens
only after the attach proved every block can draw. Two gate-only profiles carry the switch (`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-smp-s7` and `...-smp-s7-audit`; section 7); no shipped profile
does. Every device cost below is an ESTIMATE unless it says MEASURED; the tt-metal sampling sources are not in this repository, so what is said about the upstream sampler is from memory of upstream
and is listed as an open question in section 8. Section 9 is what the first card run has to check.

## 0. The verdict, first

* **Scheme.** Production forces greedy because the verifier commits argmax tokens. Every draft source on the fast path is deterministic (the drafter's top choice, prompt-lookup copies, the rows of a
  tree or octo branch). For deterministic drafts the optimal exact rule is also the simplest: at each verify row draw `y` from the target's filtered distribution with noise keyed **only** by
  `(request seed, absolute position of the token being sampled)`, keep drafts while they equal `y`, emit the first `y` that differs, emit the bonus `y` after a full match. The output is a sample of the
  model's own distribution, **and** it is the same text a plain one-token-at-a-time sampler gives with the same seed, whatever the draft lengths, round splits, tree shapes, admission timing or block packing.
  That keeps every audited "0 mismatches against the reference" gate meaningful (section 4).
* **Cost of exactness.** None in tokens: for a point-mass draft the standard speculative-sampling rule accepts with probability `p(draft)` and resamples from `p` without the draft on rejection, which
  is exactly the distribution of this walk. What sampling costs is acceptance: at a position the walk accepts with probability `p_filtered(draft)` instead of `1[draft == argmax]`, so tokens per round fall
  (by an amount only a card run measures; section 1.4).
* **Device: recommend option (b), each chip returns its shard's top-K_MAX logits and ids and the host filters and draws** (section 3). Staged: (b1) stock `ttnn.topk` for the first card window (correctness, no
  new kernel), (b2) a top-K extension of the existing shard-argmax scan kernel for the timed window. Estimated device cost per 56-59 ms verify trace: b1 +2 to +5 ms, b2 +0.8 to +1.5 ms
  (S1's argmax scan + fold costs 0.29 ms MEASURED); readback 64 KiB a round for 64 rows x 4 chips at K_MAX 64 (today 1.5 KiB), no more time than today's eight reads once the joined `words` layout is read in one
  mesh read; host merge and draw 0.9 ms MEASURED here (numpy, unoptimised, 64 rows). Option (a), TTSampling with per-row k/p/temperature, costs at least b1 and (as I recall upstream; section 8) is capped at k = 32 with a
  device RNG that cannot be keyed by position without a per-row reseed - so the audited reference would have to be the device op itself.
* **Contract.** `QWEN_FAST_SAMPLED=1` (profile-owned, gate-only until a card has judged it) honours temperature / top_p / top_k, takes the model's defaults for what is omitted, keeps temperature 0 greedy,
  clamps `top_k` of 0 or above `K_MAX = 64` to 64 (the one approximation, section 5), gives a request without a seed a random one, **forces presence / frequency / repetition penalties and `min_p` off exactly
  as the greedy contract does, with one log line naming them** (clients may send them today and must keep working; they are not refused), and refuses what it always refused.
  **It fails closed**: the engine-side gate (`validate_request_sampling`) admits a non-greedy request only when the switch is on, `sampled_verify.attach` proved every packed block engaged the sampled readback,
  and the request carries a well-formed keyed seed; until then (and on any image whose runtime predates the attach call) it refuses, ending one request under C2-any.

* **What the device step touched** is the greedy-assumption inventory in section 2: each row now says what was done.

## 1. The scheme

### 1.1 Definition

For a request with seed `s` and spec `(temperature, top_p, top_k)`, the token at absolute position `q` (prompt tokens are positions `0..len(prompt)-1`; the first generated token, the prefill's, is at
`len(prompt)`) is

    y_q = inverse_CDF( F(logits_q ; spec) , u_q ),     u_q = noise(s, q)

where `logits_q` are the target's logits for position `q` given the committed tokens before `q`, `F` is the filtered distribution (section 1.2), and `noise` is a counter-based hash (section 1.3).
A ticket of a round starts at `position` (the position of its seed token, `ticket.tokens[0]`); row `i` consumed the token at `position + i` and samples position `position + i + 1`
(`sampled_accept.noise_positions`). **Naming trap:** in this code base a session's "seed" is the first generated token (`GreedySession.seed`); the PRNG key is called `seed` only in `sampled_accept` and
the request (`SamplingParams.seed`, carried as `extra_args['qwen_sampled_seed']`).

The round: `ys = [y_{position+1}, ..., y_{position+k+1}]` for a ticket of `k` drafts, then

    accepted = number of leading drafts equal to their y;    emitted = ys[:accepted + 1]

(`walk_linear`; `walk_tree` for a tree: from the root take `y`, descend to the child whose token equals it, else emit `y` and stop). `emitted` is `accepted` drafts plus the first non-matching `y`
(or the bonus `y`): exactly the shape `greedy_verify.select_prefix` returns, so **`GreedySession.commit`, `select_prefix`, `tree_verify.select_branch` and `commit_tree` need no change: hand them the
draws where they now get argmax ids.** `test_at_temperature_zero_the_walk_is_greedy_select_prefix` holds this against the real `select_prefix`, and the exactness tests run the real `GreedySession` and
`tree_verify`.

### 1.2 The filter

On the row's candidates ordered by (logit descending, token id ascending) - the first-occurrence rule of `torch.argmax`, of the pinned sampler and of `combine_shards`, so temperature 0 is today's argmax -
keep the first `min(top_k, K_MAX)` (all `K_MAX` if `top_k <= 0`), weight `exp((logit - max) / temperature)`, normalise, keep the smallest prefix whose cumulative probability reaches `top_p` (a token is kept
when the cumulative probability BEFORE it is below `top_p`; the first is always kept), renormalise, and take the first token whose cumulative probability exceeds `u`. This is vLLM's order (temperature,
then top_k, then top_p on the renormalised top-k set). Arithmetic is float64 on the bf16-valued logits; a last-ulp libm difference between machines can flip a draw only when `u` lands within ~1e-16 of a
boundary. Non-finite logits among the candidates are refused (a device fault, never an input); idle segments of a padded block are passed as inactive rows and never read.

### 1.3 The noise

`noise(seed, position)`: three SplitMix64 finalisers, `mix(mix(seed + C1) ^ mix(position + C2))`, top 53 bits as a float64 in `[0, 1)`. A pure function of its two inputs, defined in plain integers
(`test_the_numpy_hash_is_the_scalar_definition`, five known answers pinned in `test_known_answers_pin_the_hash`): a kernel or a C port can implement it exactly. Tests: uniform (chi-square, 64 bins, 200,000
draws, three ways of varying the inputs), uncorrelated across adjacent positions and adjacent seeds, one flipped input bit flips about half the output bits. Not cryptographic; seeds are not secrets. A request
without a seed gets 64 random bits from `secrets` in the contract; the seed is taken modulo 2^64.

### 1.4 Why it is exact, reproducible and lossless

1. **Exact.** By induction over positions, the token committed at `q` is `y_q` computed from the TRUE committed prefix: an accepted draft equals `y_q`, a corrected token is `y_q`. So the committed
   sequence equals plain sampling's, and its distribution is the model's filtered distribution. This needs only that **drafts are a function of the committed prefix** (never of the noise): the
   drafter, lookup and tree/octo branches all are. (A drafter that peeked at `u` would bias the output.)
2. **Reproducible.** `u_q` depends on nothing but `(seed, q)`, so the text does not depend on draft length, the round boundaries, narrowing (`GreedySession.narrow`), aborts, the budget cut, the commit
   cap, the tree shape, which block the request shares, what else is live, or when it was admitted. It is also independent of prefix reuse and exact resume (positions are absolute).
3. **No u is conditioned on.** A later round reuses `u_q` only when an earlier round stopped at a mismatch before `q`; that round never looked at `u_q`.
4. **A chain tree is free.** Every chain's emitted tokens are a prefix of the one sampled continuation (the invariant `tree_verify` already states for greedy), so picking the chain that emits most only
   changes how many tokens a round commits.
5. **Acceptance.** A draft is accepted at a position with probability `p_filtered(draft)`; expected tokens in a round of `k` drafts is `1 + sum_{j=1..k} prod_{i<=j} p_i(d_i)`. Confident positions (code, copied text) behave
   as greedy; flat positions lose. The prompt-lookup gate (`n3m12`...), the draft widths and the octo thresholds were tuned on greedy acceptance and need a re-measure; none of that is a correctness matter.

Tests, all pure CPU (`test_sampled_accept`, 44 tests, ~15 s): the speculative text equals the plain sampled text for every draft length (1, 2, 4, 8, 16), four drafters (an approximate model, prompt-lookup
copies, hash noise, a mostly-right one), four specs including temperature 0, two prompts and seeds up to 2^64 on the real `GreedySession`; with random narrowing, aborts and commit caps; across the budget and a
sampled end-of-text token; through `tree_verify` chain trees (branches win rounds); through real trees with shared prefixes; and with four requests of different specs sharing one `draw_batch` call with padded
idle rows (NaN logits) and random widths and admission rounds. The distribution test runs a Markov toy over 24,000 seeds and compares the (first, +3 tokens) frequencies to the model's by chi-square: the
statistic is the same for every draft length (the text is the same per seed), and two **controls that must fail do**: a biased acceptance rule (statistic 108,000 against a critical 136) and noise keyed by
the round-local row instead of the position (176,000).

## 2. Every place that assumes greedy

"Built" is this branch (the CPU step, then the device step b1); what a row says was done is in its last column. Paths are `scripts/ci/` unless noted.

### 2.1 The request edge and the engine's admission

| Where | What it assumes | Change |
|---|---|---|
| `serving_c2_contract.enforce_request` (the vLLM `InputProcessor.process_inputs` hook) | overwrites every request to temperature 0, top_p 1, top_k 0, min_p 0, no penalties, **seed None** | **built**: with the switch on, `sampled_terms` honours the terms (section 6). The seed is carried in `extra_args` and `params.seed` stays None because the pinned samplers refuse a block while a request seed is active (below) |
| `serving_fast_policy.validate_request_sampling` (called by `serving_lifecycle` at admission and by `serving_request_factory.from_prefill`) | `temperature != 0` refuses; also refuses penalties, `logprobs`, `stop`, structured output, `logit_bias`, `allowed_token_ids`, `bad_words`, `min_tokens` | **built**: `temperature != 0` is accepted only when `sampled_verify.accepts(params)`: the switch is on, `sampled_verify.attach` ran (serving_runtime calls it after every block is built, and it raises unless each block engaged the sampled readback) and the request carries a well-formed keyed seed, a `top_k` of 1..64, a `top_p` and no `min_p`/penalty. Every other refusal is kept; the refusal ends one request under C2-any (`serving_request_quarantine`). Fails closed on an image whose runtime never calls the attach. Switch off: the old condition call for call (`test_sampled_packed_path.FlagOffTests`) |
| `serving_fast_policy.validate_fast_config` | `speculative.draft_sample_method == 'greedy'`, `rejection_sample_method == 'standard'` | **keep**: drafts stay deterministic top choices; the fast path never runs vLLM's rejection sampler |
| `generation_config` | the contract cannot tell an omitted `top_p`/`top_k` from an explicit one: vLLM fills omitted fields before the hook | **built**: `sampled_launch_problems` refuses `--generation-config` other than `auto` and `--override-generation-config` for a sampled profile; vLLM prints a line at boot when it applies the model's generation config as default sampling parameters; the device step's gate should require it (not checked here) |
| `SamplingParams.seed` into the plugin's seed manager | `force_argmax.sample_rows` raises "Seeded sampling is outside the greedy verifier contract" and `verify_trace_t1.shard_sampling_problem` returns "a request seed is active" while `seed_manager.has_active_request_seed()` | **built** (seed moved to `extra_args`); `sampled_verify.of_params` refuses a sampled request whose `params.seed` is set (the seed belongs in `extra_args`), so nothing puts it back |

### 2.2 The first token

| Where | What it assumes | Change |
|---|---|---|
| `serving_lifecycle._sample` calls `original_sample` (the plugin's `sample_tokens`) after the native prefill; `serving_request_factory.from_prefill` takes `state.output_token_ids[0]` as `GreedySession`'s seed | the first generated token is the plugin's, drawn with whatever noise the plugin has (under `decode_only` on-device sampling the prefill step samples on the host) | **built** (`sampled_verify.FirstToken`): the lifecycle wraps the runner's `host_sampler` (`model_runner._get_output_tokens` calls it with the prefill's `[1, vocab]` float logits for every four-card profile: they name no `sample_on_device_mode`, and the install refuses `all`). Armed at admission for a well-formed sampled request, the call of the final prefill chunk returns `sample_logits(logits[0], spec, seed, len(prompt))` computed BEFORE the plugin's sampler runs (it may rescale the logits in place) and then lets that sampler run for its side effects. After `sample_tokens` the lifecycle checks the committed token is that draw (`FirstToken.finish`): an armed request that was not drawn fails the engine. A greedy request disarms whatever an earlier one left. Chunked prefill, sticky and exact-resume prefill change nothing (one call, absolute position). `_terminal_at_seed` and the EOS-at-seed branches work on the drawn token unchanged |
| the contract's comment "the first token is sampled by vLLM from them, so they must agree" | temperature/top_p/top_k written by the contract are what the plugin's own sampler uses for token 0 | **built**: the clamped `top_k` and the request's terms are written to `params`; the graft above replaces the plugin's draw with the keyed one |

### 2.3 The device verify path (the argmax producers)

| Where | What it assumes | Change |
|---|---|---|
| `force_argmax.sample_rows` / `sample_tiles` (a pinned `TTSampling` in force-argmax mode, 32-row tile; 64 rows are two tiles through the same sampler) | `tt_sampling.force_argmax_sampling`; no active seed; `selected_ids` refuses log-probability outputs; the native-row experiment refuses penalties and logprobs | not the sampled path (option b does not use it). Keep as the **top-1 audit reference**: the sampled slice's first candidate must equal its ids |
| `verify_trace_t1.sample_shards` / `served_shards` / `combine_shards` (QWEN_FAST_VERIFY_T1 #8a): per chip `ttnn.argmax` + `ttnn.max` (or the V4a gather), host fold `strictly greater wins, ties to the lower chip, NaN wins` | an id and a max per row per chip; bf16 TILE shard of 62,080 | **built, untouched**: the shard argmax stays in the trace and serves every greedy request; the top-K slice is added beside it (`sampled_verify.device_slice`, merged by `merge_shard_topk`: exact, the global top-K is inside the union of the chunks' top-K). `shard_sampling_problem`'s guards (penalties/logprobs/a seed) stay, and a block that has none of them engaged (`shard_problem`) refuses the sampled readback rather than serve a sampled request from the pinned sampler |
| `tp4_shard_argmax.py` + `tp4_shard_argmax_{scan,fold,fold2}.cpp` (S1, QWEN_FAST_TP4_SHARD_ARGMAX): scan writes one word `(column << 16) \| bits` per row per task; strict greater-than fold; three 64-word pages per chip (ids, values, words) | one maximum per row | unchanged (b1 is beside it). **Still to do for b2** (the timed window): a top-K per (row, task) and a tree fold of sorted lists; output pages become (64 rows x K) words per chip. The joined-word layout and `decode_words` already carry id and value in one uint32 |
| `packed_verifier.PackedVerifierEngine.operation / sample_shards / shard_predictions` and the readback in `verify()` (`predictions = [host[slice(*segment_rows(...))] ...]`, line ~2079) | `host` is 64 ids; segments slice it per entry; the T1 audit compares with the pinned sampler's ids; `sampler_prewarm` / `sampler_in_trace` run the pinned sampler beside | **built**: `__init__` reads the switch and refuses the block when it cannot take the readback; `operation` returns `(logits, ids, values, reference-or-None, *slice tensors)` (`sampled_outputs`: the shard argmax's outputs keep their indices, the slice tensors are last: eight for an M3 or octo block, four for a one-user block); `verify()` calls `sampled_verify.packed_predictions` after the argmax readback: for each SAMPLED entry (the session's `sampling`, set by `serving_request_factory.attach_sampling`) it reads the slice tensors (one `batched_reads_tp.read`, a single fence in `async` mode), merges only that entry's rows, draws with the entry's spec, seed and `ticket.position + 1 + row`, and replaces the entry's rows; greedy entries and idle segments are never read. Every sampled round compares slice[0] with the argmax ids (raises under the T1 audit); `QWEN_FAST_SAMPLED_AUDIT` adds, on `audit_sampling`'s audited rounds (the cadence, **not** token sampling, despite the name), a read of the sampled rows' logits whole and a comparison of the slice and the draw with `topk_slice` / `draw_batch` |
| `verifier_engine.py` (line ~506) and `verifier_engine_tp.py` (`shard_predictions`, line ~593, QWEN_FAST_REQUEST_SHARD_ARGMAX) - the per-request 1/2/4-row engines | `argmax(dim=-1)` of host logits when there is no device sampler; `sample_rows`; the request engine's own shard arm | **built, host only**: `verifier_engine_tp.VerifierEngine.verify` (the four-card class; `verifier_engine.py` is a critical bundle-inventory file and stays untouched, and a sampled profile must be four-card: `sampled_problems`) reads the captured logits back (the vocabulary shard on each chip, or the replicated row), slices them on the host (`topk_slice` per chip, then `merge_shard_topk`) and draws (`sampled_verify.sequential_predictions`) for a session with a `sampling`. No new device op in the 1/2/4-row traces: it is the exceptional path (a narrowed or ineligible round), and the host slice is also what the reference calls. The lone-user lane and the solo packed block are PackedVerifierEngines and take the packed path |
| `packed_verifier` constructor: `verify_trace_t1.cut('shard_argmax')`, `shard_problem`, `QWEN_FAST_PACKED_SAMPLER_PREWARM/IN_TRACE` | the pinned greedy sampler is part of the block's capture contract (the audits-off hang class); a seed or penalties silently keep the pinned sampler | **kept and extended**: the slice launches run in the block's eager warm forward before the capture (so every program is compiled, and a build that refuses k=64 fails the attach there, with `[PINDIAG] sampled verify refused` and the reason, never inside a capture); their outputs are held with the trace's other outputs and released with them; the column slices and pads are freed inside the capture like any intermediate |

### 2.4 The host commit and the rounds

| Where | What it assumes | Change |
|---|---|---|
| `speculative-decoding/harness/greedy_verify.select_prefix` (`target_argmax`) and `greedy_session.GreedySession.commit` / `propose(greedy=True)` | the predictions are argmax ids; the drafter is asked for greedy proposals | **none** for the walk (hand it draws). `propose(greedy=True)` stays: it is the drafter's top choice |
| `serving_packed_step.run_verified_block` -> `commit_entry` -> `session.commit(..., rows[, max_rows=limit])`; `commit_limit`; `narrow_round` | `rows` are the block's per-entry predictions; the extent boundary cap and the budget cut apply | **none** to the logic; the rows become draws. The cap/narrow/budget rules hold (section 1.4, point 2) |
| `serving_fast_request.FastRequest.step` (sequential rounds) | `engine.verify(ticket)` returns predictions, `session.commit` | same: the engine returns draws |
| `tree_verify.select_branch / commit_tree / decide` | each chain judged by `select_prefix` | **none** (tested with sampled draws) |
| `serving_vllm_contract` / `serving_vllm_packed` (`sampled_token_ids`, `logprobs=None`) | vLLM is handed the committed tokens; no logprobs | **none**; logprobs stay refused (the slice could serve them later) |

### 2.5 The draft sources

| Where | Assumption | Change |
|---|---|---|
| DFlash2 / DSpark drafter (`serving_fast_request.runtime`, `dflash_*`) | proposes its top choice from target features of the committed prefix | none: deterministic in the prefix. Acceptance falls, correctness holds |
| `prompt_lookup.RequestLookup` / `lookup_draft` | copies from the history; "lossless greedy is unchanged" | none (a pure function of the history). The `n<N>m<M>` gate was tuned on greedy tau: re-measure |
| `early_draft.py` (H2, QWEN_FAST_EARLY_DRAFT) | drafts the next round inside the step from the committed tokens; `draft_key` confirms nothing changed | none: it reads committed tokens, which now include sampled ones |
| `serving_octo.py` (octo-T8, 8 x 8 rows; "EXACTNESS (owner rule: STRICT): only the target's own argmax is ever committed") | the sentence, and the 8-row cut of tickets | **built**: the octo block is a PackedVerifierEngine (`layout(64)`, 8 rows a user): the same walk per segment; the module docstring now says "only the target's own keyed draw" for a sampled request. 7 proposals per seat lose more acceptance under sampling than 15 do under greedy: a tuning question |
| `levern_route.py` / Lever N, `serving_parked_engines.py` | chunked prefill and parked engines are about state, not tokens | none; the first-token graft above covers the final prefill chunk |
| `serving_fast_lane.py`, `serving_solo_lane.py` | `request_lane(new.sampling_params)` reads `extra_args['qwen_lane']` | none: the contract copies `extra_args` and adds one key (`test_extra_args_keep_the_others...`) |

### 2.6 The reference, the audits and the gates

| Where | Assumption | Change |
|---|---|---|
| `full_request.measure_request`: `gold` is native one-token-at-a-time decode with `logits...float().argmax()`; `check_committed_prefix` compares every committed token | the reference is greedy | **built**: `measure_request(..., sampling=Sampling, prefill_logits=callable)`: `gold` uses `sampled_accept.sample_logits(row.float(), spec, seed, position)` with `position = len(prompt) + len(gold)` and the prefill token at `len(prompt)` (from `prefill_logits`, the same native prefill returning its last row's logits); the verifier's rows are `sampled_verify.sequential_predictions` of the engine's own logits. Same functions, same logits: the gate stays "0 mismatches" (`test_sampled_reference`). The card gate scripts that call it (`full_dflash_request`, `full_dspark_request`, `full-prefix`) pass neither argument and are unchanged |
| `acceptance_report.compare_packed` (`[PACKED] request/position/prefix/emitted/predictions` lines, `--position-keyed`) | a round at a position is a function of the text up to it and the draft | still true with a seed; both arms must send the same explicit seed. `predictions` are now draws |
| `serving_canary_client` (`temperature=0`), the agent replay harness, the tau lab | greedy requests, greedy tau | **built for the smoke and the tau lab** (section 7): `C2_SMOKE_SAMPLING=greedy\|model-default` (+ `C2_SMOKE_SEED`) and `C2_TAULAB_SAMPLING` (+ `C2_TAULAB_SAMPLING_SEED`), the seed a function of the base seed and the prompt, `sampling` reported in `SMOKE_JSON` / the lab's summary; tau is a different number under sampling. The canary client and the agent replay harness stay greedy |
| `QWEN_FAST_VERIFY_T1_AUDIT`, `QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT` | compare ids with the pinned sampler's every row | **kept** (the argmax path still serves greedy rows); **added** beside them: slice[0] against the argmax ids on every sampled round, and under `QWEN_FAST_SAMPLED_AUDIT` the slice-vs-`topk_slice` and draw-vs-`draw_batch` comparison on the audited rounds |
| profile validators (`serving_fast_policy`, `serving_c2_contract.*_problems`) | none name sampling | **built**: `sampled_problems` (also: the four-card mesh, `QWEN_FAST_TP=4`; `QWEN_FAST_SAMPLED_AUDIT` needs the switch), `sampled_launch_problems`; `c2_smoke_check.sampled_problems` judges a sampled run's log (section 9) |

## 3. Device options

Sizes are for the M3 block (64 rows = four T16 users, or the octo 8 x 8), four chips, vocabulary 248,320 = 4 x 62,080, bf16 TILE logits shards. A verify trace is 56-59 ms (docs/tp4-samp-draft.md).

**Today (MEASURED, docs/tp4-samp-draft.md and tp4_shard_argmax.py):** untilize 46 us + `ttnn.argmax` 940 us + `ttnn.max` 787 us or the V4a gather; S1's scan + fold replaces that with 0.29 ms; the in-trace pinned
sampler run inside the trace (the full-row all-gather, twice; the audits' reference and the hang-isolation arms) is 4.3 ms a verify. Readback: eight small reads, about 0.9 ms ESTIMATED from
docs/tp4-round-host.md (40 blocking reads cost 4.4 ms, about 0.11 ms each).

### (a) `TTSampling` with real per-row k / p / temperature

What the repo shows: the pinned sampler is a `TTSampling` (`models/common/sampling/tt_sampling.py` in the image's tt-metal) with a **construction-time** `force_argmax_sampling` mode, `max_batch_size` 32 (a 64-row
block is two calls), `vocab_size`/`padded_vocab_size`, an all-gather configuration (`_get_force_argmax_all_gather_config`), a plugin `seed_manager`, `_penalties_active`, `_log_probs_active`, and the contract's
`SAMPLER_MAX_LOGITS = 65536` per device (62,080 fits). **What is not in the repository** is the generic (non-force-argmax) path and the `ttnn.sampling` op. From upstream as I recall it: per-chip `topk` with
k = 32, an all-gather of the per-chip top-32 values and indices, then one sampling op over 32 users with per-user `k` (uint32), `p` and `temp` (bf16) tensors, driven by an on-device RNG seeded per user
(`manual_seed`). If that is right:

* per-row k/p/temperature is natural (the 32 "users" are rows, set by writing three 32-word tensors per tile each round);
* **k is capped at 32** (the default top_k 20 fits, a client top_k of 50 does not);
* **the noise is a hardware RNG stream, not `f(seed, position)`**: to key it by position every row must be reseeded each round with `hash(seed, position)` (a per-row seed tensor or a reseed op in the
  trace), and the sample is then the device's bf16 arithmetic and hardware generator - the host cannot reproduce it, so the audited reference would be this same op run one token at a time, not
  `sample_logits`;
* the cost is at least (b1): the top-32 per chip plus a small all-gather and the sampling op, twice (two tiles), an ESTIMATE of +2.5 to +6 ms per verify (b1 plus 0.3-0.6 ms), with the per-row seed question unsettled.

Verdict: not recommended; it is only attractive if the op already takes a per-row seed, which section 8 asks to settle.

### (b) each chip returns its shard's top-K_MAX; the host filters and draws (recommended)

* **Output.** Per chip, per row, `K_MAX` candidates as joined words `(local_column << 16) | bf16_bits` (column < 62,080 < 65,536; the layout S1 already defines). **Read size per round: 64 rows x 64 x 4 B =
  16 KiB per chip, 64 KiB for four chips** at K_MAX 64 (32 KiB at 32, 20 KiB at 20); today 1.5 KiB. One mesh read of the `words` tensor (the `tp4_mesh_read` / `batched_reads_tp` layout work) is no slower
  than today's eight reads; the transfer itself is microseconds.
* **Host.** `merge_shard_topk` (256 candidates a row -> top 64; exact) and `draw_batch` (64 rows, vectorised): **0.88 ms MEASURED** here (0.65 merge + 0.23 draw, one numpy thread, unoptimised; 64 rows x 4 chips x 64
  candidates). It can be roughly halved (merge with a partition, skip the second sort); it runs in the shadow of nothing today, so it is on the critical path. It is also exactly what the reference computes.
* **(b1) stock `ttnn.topk`, AS BUILT** (`sampled_verify.device_slice`). The op is `operations.topk(chunk, k=64, dim=-1, largest=True, sorted=True)`, called exactly as the drafter's vocabulary head already calls it on a card
  (`draft_shared_head_tp.local_head_candidates`: k=16 there): on the 32,768-column chunks of the 62,080-wide shard (`operations.slice` of the pre-gather bf16 TILE logits, the short second chunk padded with `-inf` by
  `operations.pad` to 32,768), in pieces of 32 rows (the rows the drafter's head runs it on: 8, 16 or 32). **What is proven on a card is k=16; k=64 on these shapes is the one thing this build asserts without evidence**
  (section 8, question 1): the launches run in the eager warm forward, so a build that refuses k=64 fails the attach there with the op's message under `[PINDIAG] sampled verify refused`. A 64-row block is two pieces x two
  chunks = four launches and eight tensors a chip (values bf16, indices uint16: 64 per row per launch), a 16-row block two launches and four tensors; the column slices and pads are freed inside the capture, the
  outputs are held with the trace's. I did not use `ttnn.sampling` (option a: capped at k=32 as I recall it, device RNG), and did not pad the whole shard to 65,536 for one launch: the 32,768-column chunk is the
  shape this repository has run. The read is `batched_reads_tp.read` over the eight tensors (a blocking read per shard per tensor, 32 reads, with the flag off; one composed read a tensor with `1`; eight non-blocking
  copies and ONE fence with `async`, the s7 profiles' value). The host merges 4 chips x 2 chunks x 64 = 512 candidates a row (`merge_shard_topk`) for the rows of the sampled entries only. The merge is exact up to ties
  at the K-th place of a chunk, which `ttnn.topk` breaks in its own order: the reference keeps the lowest ids, so a tie straddling the cut can differ (it matters only for a row whose nucleus or `top_k` reaches
  rank 64 and whose 64th value is shared; the audit states it: `slice_difference` accepts any ids of that value inside the tie group and counts the row). **ESTIMATE +2 to +5 ms per verify** (a multi-core bitonic network
  over 64 x 65,536; the nearest measured number is the 4.3 ms of the pinned sampler's gather + untilize + argmax). Use it for the correctness window only.
* **(b2) a top-K scan** extending S1: each of the 220 (11 x 10 grid) or 260 (13 x 10) scan tasks keeps a sorted buffer of K candidates per row instead of one maximum, and the fold becomes a merge tree of sorted
  lists. Model: a task sees ~576 elements of a row (18 tiles x 32); inserts into a sorted 64-buffer per row ~ `64 (1 + ln(576/64)) = 204`, ~150 cycles each, 32 rows a RISC-V -> ~1M cycles = **~0.7 ms** for the scan
  (S1's scan is ~0.1 ms of its 0.29, a cycle-count estimate; the cycle-to-time conversion assumes ~1.35 GHz), plus **0.2-0.5 ms** for a two-level merge tree over 110 (130 on the 13 x 10 grid) sorted lists per row. **ESTIMATE +0.8 to +1.5 ms per verify** (1.5-2.5%); measure it on the card-M harness arms as S1 was.
  Cheaper variants exist (per-task K of 16 with a retry when a task's 16th candidate beats the row's 64th) and are not worth designing before b1 says what the host side costs.
* **(b3) optional extra: the shard's `sum(exp(logit - max))` per row.** With it the host can take `top_p` over the full vocabulary (not the renormalised slice) when the nucleus fits in the slice, and can flag a
  row whose nucleus overflows K_MAX; it removes the approximation of section 5 for `top_k = 0` requests. One more reduction pass over the shard (~0.2-0.5 ms ESTIMATE); only worth it if overflowing rows are common.

### (c) full logits to the host - rejected

64 x 248,320 x 2 B = 31.8 MiB a round (about 3 ms of PCIe at best) and a 64-row partition on the host. No.

### Recommendation

**(b), b1 then b2.** It keeps the host arithmetic (float64, vectorised, the same functions the reference calls) as the single definition of the draw, it serves any `top_k` up to K_MAX, it needs no
per-row device RNG, and its device cost sits between S1's 0.29 ms and the 4.3 ms the pinned sampler already costs. Its price is the host's ~0.9 ms and, until b2, a few ms of device time on a trace that
is 56-59 ms: about 5-10% for b1 (device 2-5 ms plus the host's 0.9), 3-4% for b2, before the acceptance loss of section 1.4.

## 4. What the audited reference needs

The gates compare the fast path's committed tokens with a native, non-speculative decode (`full_request.py`'s `gold`, `check_committed_prefix`). To reproduce the same keyed noise the reference needs:

1. **The same function.** `sampled_accept.sample_logits(logits_row, spec, seed, position)` where `position` is the absolute index of the token being sampled (`len(prompt)` for the prefill's, then
   `len(prompt) + len(gold)`), the same `SamplingSpec` (including the `K_MAX` clamp) and the same request seed (an explicit seed on both arms; an unseeded request has a random one the harness cannot know).
2. **The same candidates.** The device returns a slice, the reference derives one: `topk_slice(full_logits)` orders by (logit descending, id ascending) and keeps the lowest ids on ties at the cut; a device
   top-K must break ties the same way **at the K-th place** (a tie straddling rank 64 is rare and only matters when `top_k` is 64 or 0), or the audit must compare slices and the gate must name the case.
3. **The same logits.** Sampling is more sensitive than argmax: a bf16 logit differing in one bit moves a boundary only when `u` lands within it (probability ~ the probability mass of that bit), so it takes
   the same exactness regime the greedy gates already require (the reference's logits for a position equal the verify row's). The audits that read logits back keep their cadence (`QWEN_FAST_AUDIT_FIRST/EVERY`).
4. **The first token from the same call** (section 2.2), or the gate fails at token 0 and nowhere else.
5. A sampled gate has **no per-token expected value to hard-code**: it is equality with the reference run, not with a golden text, plus the CPU chi-square tests of this branch for the distribution.

## 5. K_MAX and the one approximation

`K_MAX = 64`. Why: (1) the model's own defaults are `top_k 20`, so the defaults and every client `top_k` up to 64 are **exact**; (2) 64 KiB a round of readback and a 64-wide sort are small, 128 would double both for
no default-path gain; (3) `ttnn.topk`'s multi-core path supports k up to 64 as far as I recall (open question, section 8) and the stock sampler stops at 32; (4) a request is usually well inside it: at
the model's `top_p 0.95` the nucleus is almost always under 64 tokens.

The approximation: a request that sends `top_k` 0 (or -1: disabled) or above 64 is served with 64, and `top_p` is taken over the **renormalised slice**. The tail beyond rank 64 is dropped, and the
nucleus can only be cut earlier than the full vocabulary's would be (the slice's cumulative probabilities are larger), never include a token outside the true nucleus. It is exact whenever the true nucleus
(or `top_k`) fits in 64, and when it does not, the total variation distance from the exact distribution is at most the probability mass beyond rank 64 at that position. For a request with `top_k 0` AND
`top_p 1.0` at temperature 1 on a flat position that mass can reach a few percent; the CPU test `test_the_k_max_clamp_is_the_one_approximation` pins the clamp's behaviour (top_k 0, -1, 64, 65 and 1000 are the same
sampler; 20 is not). Option (b3) is the exact fix if it matters. The contract also writes the clamped `top_k` back to the request so the plugin's own sampler of token 0 applies the same cut.

## 6. The contract (`serving_c2_contract.py`, behind `QWEN_FAST_SAMPLED`)

| Decision | Choice | Reason |
|---|---|---|
| Flag | `QWEN_FAST_SAMPLED=0\|1`, strict, read per request; **profile-owned** (`apply_environment` drops an inherited value); **gate-only** (`sampled_problems`: needs `QWEN_FAST_ANY_REQUEST=1`, the request contract, `gate_only: true`, the four-card mesh with `QWEN_FAST_TP=4`); `QWEN_FAST_SAMPLED_AUDIT=0\|1` (audited rounds read the logits back and compare the slice and the draw) needs the switch | no sampled verifier has been judged on a card; only the two generated profiles name it (`test_no_shipped_profile_or_image_file_names_the_switch`) |
| Off | the old function, call for call (`test_a_differential_fuzz_against_the_old_function`: 4,000 random requests against a verbatim copy of the old `enforce_request`, same refusals and messages and same mutations; golden table; only the nine legacy attributes touched, no `extra_args` added) | the owner rule |
| temperature | the request's; `None` takes 1.0; **0 is greedy and is coerced exactly as today** | |
| top_p, top_k | the request's; `None` takes 0.95 and 20 (generation_config of the base snapshot, held equal to `sampled_accept.MODEL_DEFAULTS` by a test); `top_k <= 0` or `> 64` -> 64 | section 5; vLLM has normally filled omitted fields already, hence the launch check |
| seed | the client's, else `secrets.randbits(64)`; carried as `extra_args['qwen_sampled_seed']` (other `extra_args` kept; `seed` wins over a client-sent key, which otherwise counts as a seed; a second pass keeps the first seed); `params.seed = None` | the pinned samplers refuse an active request seed (section 2.1) |
| penalties (`presence`, `frequency`, `repetition`) | **forced off exactly as the greedy contract forces them, greedy requests included, with ONE log line naming what was ignored** (`[QWEN-C2] request contract: presence_penalty, ... ignored (forced off, the fast path applies none; QWEN_FAST_SAMPLED=1)`; a second pass through the hook sees neutral values and logs nothing; with the switch off the drop stays silent, as today) | clients may send them today and must keep working; the verifier applies none, so the text is the unpenalised model's, as it is today. Supporting them on the slice is possible later: a row's penalties depend on the drafted prefix, which is known |
| `min_p` | forced off the same way (same log line) | not served; one filter line when wanted |
| n > 1, logprobs, structured output, `logit_bias`, `allowed_token_ids`, `bad_words`, `stop`, `min_tokens`, foreign `stop_token_ids` | refused, unchanged | |
| launch | `--generation-config` other than `auto` and `--override-generation-config` refused for a sampled profile | the model defaults must be what vLLM fills in |
| engine gate | `validate_request_sampling` admits a non-greedy request only after `sampled_verify.attach` (fails closed until then) | section 2.1 |

## 7. What is in the branch, how to run it, and what is left

**Files.** New: `scripts/ci/sampled_accept.py` (the scheme; numpy), `scripts/ci/sampled_verify.py` (the serving side; numpy, imported lazily and only with the switch set), `scripts/ci/sampled_client.py` (the clients' seeded
arm; stdlib), `scripts/ci/make_sampled_profiles.py` (the two profiles), `scripts/ci/sampled_test_support.py` (a fake four-chip mesh and a toy model embedded in the real vocabulary; test only). Changed, each behind the
switch and byte-identical with it off: `serving_c2_contract.py` (the request terms, the penalties' log line, `sampled_problems`), `serving_fast_policy.py` (the gate), `serving_request_factory.py` (`attach_sampling`),
`serving_lifecycle.py` (the first-token wrapper), `serving_runtime.py` (the attach), `packed_verifier.py` (the readback), `verifier_engine_tp.py` (the sequential host slice), `full_request.py` (the keyed gold),
`serving_octo.py` (a docstring), `c2_serving_smoke.py`, `c2_tau_lab.py`, `c2_serving_job.py` and `.github/workflows/qwen-c2-serving.yml` (the seeded arm), `c2_smoke_check.py` (the rules of a sampled run),
`profile_twins.py`, `qwen_c2_profiles.json`.

**Profiles** (gate-only, unqualified; neither replaces a shipped profile, the ship-7 candidate and its audited twin are untouched):
`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-smp-s7` = the ship-7 candidate (`...-fx-traffic-s7`) with the gate marker (`QWEN_C2_GATE_PROFILE=1`) for its owner-waiver id, `gate_only`, and `QWEN_FAST_SAMPLED=1`;
`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-smp-s7-audit` = the ship-7 audited twin (`...-fx-s7-audit`) with `QWEN_FAST_SAMPLED=1` and `QWEN_FAST_SAMPLED_AUDIT=1`. Generated by `make_sampled_profiles.py`
(`--write`, `--check`), placed right after `-fx-s7-audit`, registered in `profile_twins`.

**Shipping.** `sampled_accept.py` and `sampled_verify.py` are in `docker/qwen-c2-overlay.txt` (an overlay-only module goes there and nowhere else; `test_serving_image_copy_closure.C2_ONLY_PLAIN_IMPORTS` and
`test_tp4_closure_literals.SERVED` name `sampled_verify`, which six copied modules import lazily). `sampled_client.py`, `make_sampled_profiles.py` and the test support are not served.

**Tests** (all in the CPU allowlist, `qwen-integration-cpu.yml`; Python 3.11, numpy and torch): `cd scripts/ci && python -B -m unittest test_sampled_accept test_sampled_contract test_sampled_verify test_sampled_packed_path
test_sampled_first_token test_sampled_block_round test_sampled_sequential test_sampled_reference test_sampled_profiles test_sampled_clients`.
  * `test_sampled_packed_path`: the real `GreedySession` driving four requests of different specs and seeds through the packed readback path over a fake mesh (device slice, batched read, merge, draw, commit): the committed
    text equals plain one-token-at-a-time sampling for four drafters, four specs, idle segments, early finishers and shuffled segments; a greedy request in a sampled block keeps the argmax path; a block of greedy
    requests reads nothing; with the switch off the gate is the old function call for call and every hook is inert.
  * `test_sampled_verify`: the request terms, the closed gate, the layout, the topk call shapes (k=64, 32,768-column chunks, 32-row pieces, intermediates freed, a refusing build), the merge against the full-vocabulary
    slice, tie order, the reads per batched-reads mode, the audit catching a corrupted slice and a wrong draw, the sequential engine, the first-token wrapper.
  * `test_sampled_block_round`: the REAL `PackedVerifierEngine.verify` (test_packed_verifier's four-user fixture, switched to the readback) returning the keyed draws of the sampled entries and the argmax rows of the rest, in entry order, with the metrics; the same round without the readback is the old one.
  * `test_sampled_sequential`: the REAL `verifier_engine_tp.VerifierEngine.verify` for a sampled session (the keyed draws of the read-back logits at 1, 2 and 4 rows), a greedy session reading the pinned sampler's ids as before, a malformed output refused.
  * `test_sampled_first_token`: the real `FastServingLifecycle`: the keyed first token, a greedy request, the engine failing closed when the plugin sampled an armed request itself, the restore at close.
  * `test_sampled_reference`: `measure_request`'s keyed gold against the plain sampler, a wrong draw caught at its token.
  * `test_sampled_profiles`, `test_sampled_clients`: the profiles and their parents, the seeded arm in the smoke (against the fake server), the tau lab and the job reader, the smoke check's rules.

**Left for a card or later.** (1) k=64 on `ttnn.topk` and the real tie order (question 1, 5). (2) b2, the scan-kernel top-K, for the timed window: b1 costs a few ms of a 56-59 ms trace. (3) Whether the verify rows' logits
are bit-equal to a native decode's: argmax-equality gates never needed it, sampling does (section 4, point 3). (4) Tuning under sampled acceptance: the lookup gate, the draft widths, the octo thresholds. (5) A
qualification record before the gate-only requirement is lifted. (6) The agent replay harness and the canary client still send greedy requests.

## 8. Open questions (the sampler sources are not in this repository)

1. Does the pinned `ttnn.topk` take k = 64? (Sharpened by the build: the repository runs it at k=16 on (32, 32,768) bf16 TILE chunks, so a (1, 1, 32, 32,768) k=64 launch is what b1 asks; a (64, 65,536) launch is not used. Decides b1's K_MAX and its price; a refusal fails the attach with the op's message.)
2. Does the `ttnn.sampling` op take a per-row seed tensor, or one scalar plus `manual_seed` per user? Is `manual_seed` traceable? What is the RNG (a hardware LFSR) and can a host reproduce it bit for bit? (Decides
   whether option (a) can ever be a keyed, auditable sampler.)
3. Is the generic `TTSampling` path's top-k fixed at 32, and are `p` / `temp` applied in bf16? (Why the host is the better place for the arithmetic.)
4. Under the profiles' `sample_on_device_mode = decode_only` is the prefill's first token sampled on the host, from float logits? (ANSWERED from the plugin source the repo fixtures hold: the four-card profiles name no `sample_on_device_mode`, so `model_runner._get_output_tokens` takes the `not perform_device_sampling` branch: `host_sampler(logits=tt_out[rows, -1, :], sampling_metadata=...)` on the float32 prefill logits; the graft wraps exactly that call, and `FirstToken.install` refuses `all`. What stays unverified on a card is only that the engine process's runner has that attribute.)
5. How does `ttnn.topk` break ties at the K-th place, and can the scan kernel be made to break them by lowest id? (Section 4, point 2.)
6. What fraction of positions in real coding text have a nucleus above 64 tokens at the model's defaults? (Whether (b3) is ever needed; a CPU measurement on any logged logits.)
7. Throughput: tokens per round under sampling against greedy at 8 live seats. Only a card run answers it.

## 9. What the first card run must check

A gate-only attach of `...-smp-s7` (timed) and `...-smp-s7-audit` (audited), `C2_SMOKE_SAMPLING=model-default` (and `C2_TAULAB_SAMPLING=model-default` for the lab), a smoke that includes a decoding test and, for the
timed pair, the control `...-fx-traffic-s7` greedy. In the container log (`c2_smoke_check.sampled_problems` judges: exactly one attach line, no refused line, as many engaged lines as the attach says, and - when sampled requests were sent - a first-token line, a round line and, on the audited twin, a passing audit line; any audit mismatch line fails the run):

| Line | When | Means |
|---|---|---|
| `[QWEN-C2] request contract: sampling honoured (QWEN_FAST_SAMPLED=1; temperature 0 stays greedy, top_k clamped to 64)` | boot | the request contract is on |
| `[PINDIAG] sampled verify engaged site=packed_verify rows=64 users=4 k=64 launches=4 tensors=8 chunk=32768 audit=<0\|1>` | once per block capture: the two M3 blocks, the octo block (`users=8`) and the one-user block (`rows=16 users=1 launches=2 tensors=4`) | the slice is in the verify trace |
| `[PINDIAG] sampled verify attach blocks=4 k=64 audit=<0\|1>` | exactly once, after the blocks | the gate opened; `blocks` equals the engaged lines |
| `[PINDIAG] sampled verify refused: <reason>`, or an attach failure naming `ttnn.topk refused k=64` | instead of the two above | **the finding for question 1**: read the op's message, and the block that refused |
| `[PINDIAG] sampled first token request=<id> position=<n> token=<t>` | once per sampled request | `n` is the request's prompt token count (`usage.prompt_tokens`): token 0 is the keyed draw |
| `[PINDIAG] sampled verify round site=packed_verify n=<k> entries=<e> rows=<r> read_ms=.. merge_ms=.. draw_ms=..` | the first four sampled rounds of each block, then every 256th | the path ran; ESTIMATES: read 1-3 ms (a fence in `async` mode), merge about 1.5 ms (512 candidates a row), draw 0.3 ms for 32 rows; anything several times that is the tuning list |
| `[PINDIAG] sampled verify sequential n=<k> rows=<1\|2\|4> position=<p>` | the first four calls, then every 256th | a narrowed or ineligible round was served by the host slice |
| `[PINDIAG] sampled verify audit site=packed_verify round=<n> rows=<r> slice=exact draws=exact tie_rows=<t>` | the audited twin, on the audited rounds | the device slice and the draws equal the reference path's; `tie_rows` above zero is a finding about `ttnn.topk`'s tie order at rank 64 (not a failure) |
| `[PINDIAG] sampled verify audit mismatch ...` | never | a corrupted slice, a wrong draw or a first candidate that is not the argmax; fails the run (`audit mismatch` rule) |
| `[QWEN-C2] request contract: <terms> ignored (forced off, ...)` | a request that sent penalties or `min_p` | the contract kept the client working |

`SMOKE_JSON` must carry `"sampling": {"mode": "model-default", "base_seed": <int>, "requests": <n > 0>, "terms": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "seed_rule": ...}` (a greedy run carries
`{"mode": "greedy", "base_seed": .., "requests": 0}`); the tau lab's summary carries the same `sampling` entry without the count. Then, off the log: (a) the same seeded prompt twice, on one boot and on two, returns the same text
(the noise is a function of the seed and the position only), and a longer budget's text begins with the shorter one's; (b) the committed text equals the reference path's (`full_request.measure_request(sampling=..,
prefill_logits=..)`, "0 mismatches") on the rows the audited twin read back; (c) the fraction of audited rounds with `tie_rows > 0`, and the acceptance (`tokens per round`) of the sampled arm against the greedy control
(question 7); (d) the verify trace's replay time and `readback_ms` against the control: b1 should cost a few ms of a 56-59 ms trace (section 3), b2 is the work if it costs more.
