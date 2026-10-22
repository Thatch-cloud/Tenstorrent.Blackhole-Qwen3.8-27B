# tp4/sampled-verify: sampling on the TT fast path (the CPU step)

Status: **design, plus the part that settles on a CPU.** Nothing here has run on a card, no image was built, no profile that ships names the switch, and nothing in serving imports the sampler module yet.
What exists: the request contract behind `QWEN_FAST_SAMPLED` (default off, byte-identical to today when off), `scripts/ci/sampled_accept.py` (keyed noise, filter, draw, the coupled accept walk), and
their tests (`test_sampled_accept`, `test_sampled_contract`, in the CPU allowlist). Every device cost below is an ESTIMATE unless it says MEASURED; the tt-metal sampling sources are not in this
repository, so what is said about the upstream sampler is from memory of upstream and is listed as an open question in section 8.

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
  clamps `top_k` of 0 or above `K_MAX = 64` to 64 (the one approximation, section 5), gives a request without a seed a random one, and refuses penalties, `min_p` and what it always refused.
  **It fails closed**: the engine-side gate (`validate_request_sampling`) is unchanged and still refuses a non-greedy request, so until the device step lands no sampled request can reach a verifier that
  would commit argmax for it.
* **What the device step inherits** is the greedy-assumption inventory in section 2: about thirty places, about a dozen of which need a change.

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

"Must change" is for the device step; "built" is this branch. Paths are `scripts/ci/` unless noted.

### 2.1 The request edge and the engine's admission

| Where | What it assumes | Change |
|---|---|---|
| `serving_c2_contract.enforce_request` (the vLLM `InputProcessor.process_inputs` hook) | overwrites every request to temperature 0, top_p 1, top_k 0, min_p 0, no penalties, **seed None** | **built**: with the switch on, `sampled_terms` honours the terms (section 6). The seed is carried in `extra_args` and `params.seed` stays None because the pinned samplers refuse a block while a request seed is active (below) |
| `serving_fast_policy.validate_request_sampling` (called by `serving_lifecycle` at admission and by `serving_request_factory.from_prefill`) | `temperature != 0` refuses; also refuses penalties, `logprobs`, `stop`, structured output, `logit_bias`, `allowed_token_ids`, `bad_words`, `min_tokens` | **must change**, deliberately last: accept `temperature > 0` only when the switch is on AND the attached verifier can draw (a per-block capability the factory can assert), keep every other refusal. Unchanged here so the switch fails closed (`test_the_engine_side_gate_still_refuses_a_sampled_request`). The refusal ends one request under C2-any (`serving_request_quarantine`) |
| `serving_fast_policy.validate_fast_config` | `speculative.draft_sample_method == 'greedy'`, `rejection_sample_method == 'standard'` | **keep**: drafts stay deterministic top choices; the fast path never runs vLLM's rejection sampler |
| `generation_config` | the contract cannot tell an omitted `top_p`/`top_k` from an explicit one: vLLM fills omitted fields before the hook | **built**: `sampled_launch_problems` refuses `--generation-config` other than `auto` and `--override-generation-config` for a sampled profile; vLLM prints a line at boot when it applies the model's generation config as default sampling parameters; the device step's gate should require it (not checked here) |
| `SamplingParams.seed` into the plugin's seed manager | `force_argmax.sample_rows` raises "Seeded sampling is outside the greedy verifier contract" and `verify_trace_t1.shard_sampling_problem` returns "a request seed is active" while `seed_manager.has_active_request_seed()` | **built** (seed moved to `extra_args`). **Must keep**: nothing may put the seed back into the plugin's seed manager while a pinned greedy sampler is in the block |

### 2.2 The first token

| Where | What it assumes | Change |
|---|---|---|
| `serving_lifecycle._sample` calls `original_sample` (the plugin's `sample_tokens`) after the native prefill; `serving_request_factory.from_prefill` takes `state.output_token_ids[0]` as `GreedySession`'s seed | the first generated token is the plugin's, drawn with whatever noise the plugin has (under `decode_only` on-device sampling the prefill step samples on the host) | **must change**: the first token is position `len(prompt)` and must be `sample_logits(prefill_logits, spec, seed, len(prompt))` or text differs from token 0 on. A plugin graft for fast-path requests with a sampled spec: take the last row's logits to the host, `topk_slice`, `draw`. With Lever N the sample is on the final chunk; the position is the same. Sticky/exact-resume prefill changes nothing (absolute position). `_terminal_at_seed` and the EOS-at-seed branches work on the drawn token unchanged |
| the contract's comment "the first token is sampled by vLLM from them, so they must agree" | temperature/top_p/top_k written by the contract are what the plugin's own sampler uses for token 0 | **built**: the clamped `top_k` and the request's terms are written to `params`, so a plugin that samples token 0 itself at least applies the same filter (its noise is not ours until the graft above) |

### 2.3 The device verify path (the argmax producers)

| Where | What it assumes | Change |
|---|---|---|
| `force_argmax.sample_rows` / `sample_tiles` (a pinned `TTSampling` in force-argmax mode, 32-row tile; 64 rows are two tiles through the same sampler) | `tt_sampling.force_argmax_sampling`; no active seed; `selected_ids` refuses log-probability outputs; the native-row experiment refuses penalties and logprobs | not the sampled path (option b does not use it). Keep as the **top-1 audit reference**: the sampled slice's first candidate must equal its ids |
| `verify_trace_t1.sample_shards` / `served_shards` / `combine_shards` (QWEN_FAST_VERIFY_T1 #8a): per chip `ttnn.argmax` + `ttnn.max` (or the V4a gather), host fold `strictly greater wins, ties to the lower chip, NaN wins` | an id and a max per row per chip; bf16 TILE shard of 62,080 | **must change**: a per-chip top-K_MAX of (value, local id), host merge (`merge_shard_topk`: exact, the global top-K is inside the union of the shards' top-K). `shard_sampling_problem` also refuses penalties/logprobs/a seed: keep those guards |
| `tp4_shard_argmax.py` + `tp4_shard_argmax_{scan,fold,fold2}.cpp` (S1, QWEN_FAST_TP4_SHARD_ARGMAX): scan writes one word `(column << 16) \| bits` per row per task; strict greater-than fold; three 64-word pages per chip (ids, values, words) | one maximum per row | **must change** for b2: a top-K per (row, task) and a tree fold of sorted lists; output pages become (64 rows x K) words per chip. The joined-word layout and `decode_words` already carry id and value in one uint32 |
| `packed_verifier.PackedVerifierEngine.operation / sample_shards / shard_predictions` and the readback in `verify()` (`predictions = [host[slice(*segment_rows(...))] ...]`, line ~2079) | `host` is 64 ids; segments slice it per entry; the T1 audit compares with the pinned sampler's ids; `sampler_prewarm` / `sampler_in_trace` run the pinned sampler beside | **must change**: `shard_predictions` returns the merged slices; a new step turns them into `ys` with per-entry spec, seed and positions (entries carry the request; the ticket carries `position`); idle segments are inactive rows. The audit compares slice[0] with the pinned argmax and, on audited rounds, the slice with `topk_slice` of the read-back logits (`audit_sampling` thins audited rounds: it is the audit cadence, **not** token sampling, despite the name) |
| `verifier_engine.py` (line ~506) and `verifier_engine_tp.py` (`shard_predictions`, line ~593, QWEN_FAST_REQUEST_SHARD_ARGMAX) - the per-request 1/2/4-row engines | `argmax(dim=-1)` of host logits when there is no device sampler; `sample_rows`; the request engine's own shard arm | **must change**: the same top-K path for the sequential engines (and the lone-user lane / solo packed block), or a refusal that keeps a sampled request off them |
| `packed_verifier` constructor: `verify_trace_t1.cut('shard_argmax')`, `shard_problem`, `QWEN_FAST_PACKED_SAMPLER_PREWARM/IN_TRACE` | the pinned greedy sampler is part of the block's capture contract (the audits-off hang class); a seed or penalties silently keep the pinned sampler | **must keep/extend**: the new slice op is allocated before capture (the partials-buffer rule of `tp4_shard_argmax.reserve`) |

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
| `serving_octo.py` (octo-T8, 8 x 8 rows; "EXACTNESS (owner rule: STRICT): only the target's own argmax is ever committed") | the sentence, and the 8-row cut of tickets | the same walk per segment; the sentence becomes "only the target's own keyed draw". 7 proposals per seat lose more acceptance under sampling than 15 do under greedy: a tuning question |
| `levern_route.py` / Lever N, `serving_parked_engines.py` | chunked prefill and parked engines are about state, not tokens | none; the first-token graft above covers the final prefill chunk |
| `serving_fast_lane.py`, `serving_solo_lane.py` | `request_lane(new.sampling_params)` reads `extra_args['qwen_lane']` | none: the contract copies `extra_args` and adds one key (`test_extra_args_keep_the_others...`) |

### 2.6 The reference, the audits and the gates

| Where | Assumption | Change |
|---|---|---|
| `full_request.measure_request`: `gold` is native one-token-at-a-time decode with `logits...float().argmax()`; `check_committed_prefix` compares every committed token | the reference is greedy | **must change**: `gold` uses `sampled_accept.sample_logits(row.float(), spec, seed, position)` with `position = len(prompt) + len(gold)` (and the prefill token at `len(prompt)`). Same functions, same logits: the gate stays "0 mismatches" |
| `acceptance_report.compare_packed` (`[PACKED] request/position/prefix/emitted/predictions` lines, `--position-keyed`) | a round at a position is a function of the text up to it and the draft | still true with a seed; both arms must send the same explicit seed. `predictions` are now draws |
| `serving_canary_client` (`temperature=0`), the agent replay harness, the tau lab | greedy requests, greedy tau | add a seeded sampled arm; tau is a different number under sampling |
| `QWEN_FAST_VERIFY_T1_AUDIT`, `QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT` | compare ids with the pinned sampler's every row | compare slice[0]; add a slice-vs-`topk_slice` audit on thinned rounds |
| profile validators (`serving_fast_policy`, `serving_c2_contract.*_problems`) | none name sampling | **built**: `sampled_problems`, `sampled_launch_problems`; the device step adds its own switches beside |

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
* **(b1) stock `ttnn.topk`** per chip on the (64, 62,080) shard, padded to 65,536 as the stock sampler pads, k = 64 if this build supports it (it may cap at 32: then K_MAX is 32 until (b2)). No new kernel; the cost
  is unmeasured. **ESTIMATE +2 to +5 ms per verify** (a multi-core bitonic network over 64 x 65,536; the nearest measured number is the 4.3 ms of the pinned sampler's gather + untilize + argmax). Use it for the
  correctness window only.
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
| Flag | `QWEN_FAST_SAMPLED=0\|1`, strict, read per request; **profile-owned** (`apply_environment` drops an inherited value); **gate-only** (`sampled_problems`: needs `QWEN_FAST_ANY_REQUEST=1`, the request contract, `gate_only: true`) | no sampled verifier has been judged on a card; no shipped profile names it (`test_no_shipped_profile_or_image_file_names_the_switch`) |
| Off | the old function, call for call (`test_a_differential_fuzz_against_the_old_function`: 4,000 random requests against a verbatim copy of the old `enforce_request`, same refusals and messages and same mutations; golden table; only the nine legacy attributes touched, no `extra_args` added) | the owner rule |
| temperature | the request's; `None` takes 1.0; **0 is greedy and is coerced exactly as today** | |
| top_p, top_k | the request's; `None` takes 0.95 and 20 (generation_config of the base snapshot, held equal to `sampled_accept.MODEL_DEFAULTS` by a test); `top_k <= 0` or `> 64` -> 64 | section 5; vLLM has normally filled omitted fields already, hence the launch check |
| seed | the client's, else `secrets.randbits(64)`; carried as `extra_args['qwen_sampled_seed']` (other `extra_args` kept, a client-supplied value of that key overwritten); `params.seed = None` | the pinned samplers refuse an active request seed (section 2.1) |
| penalties (`presence`, `frequency`, `repetition`) | **refused (400), greedy requests included** | the verifier applies none; the greedy contract drops them silently, which returns text the client did not ask for. Alternative if clients send them: force off with a logged warning (the flag-off behaviour). Supporting them on the slice is possible later: a row's penalties depend on the drafted prefix, which is known |
| `min_p` | refused | not served; one filter line when wanted |
| n > 1, logprobs, structured output, `logit_bias`, `allowed_token_ids`, `bad_words`, `stop`, `min_tokens`, foreign `stop_token_ids` | refused, unchanged | |
| launch | `--generation-config` other than `auto` and `--override-generation-config` refused for a sampled profile | the model defaults must be what vLLM fills in |
| engine gate | unchanged (fails closed) | section 2.1 |

## 7. What is in the branch, how to run it, and the device step's list

Files: `scripts/ci/sampled_accept.py` (new; numpy), `scripts/ci/serving_c2_contract.py` (constants, `sampled_enabled / sampled_on / sampled_problems / sampled_launch_problems / sampled_terms`,
`enforce_request(..., sampled=None)`, the environment drop, the boot checks; the only new import is stdlib `secrets`, so the overlay closure tests are untouched), `scripts/ci/test_sampled_accept.py`,
`scripts/ci/test_sampled_contract.py`, `.github/workflows/qwen-integration-cpu.yml` (the allowlist line). Run: `cd scripts/ci && python -B -m unittest test_sampled_accept test_sampled_contract` (Python 3.11, numpy).

The device step, in order:

1. **The slice op** (b1 first), its output pages and the `words` mesh read; allocate before capture; fall back with a logged line, never inside a capture.
2. **The block integration**: `shard_predictions` returns merged slices; a per-entry step turns them into draws with `draw_batch` (spec and seed from the request's `SamplingParams`, positions from the ticket,
   idle segments inactive); the sequential engines and the lone-user lane the same, or refuse sampled requests on them.
3. **The first token**: the plugin graft (section 2.2).
4. **The engine gate**: accept a sampled request only when the verifier can draw.
5. **Reference and gates** (section 4), a seeded sampled canary, and the tau lab on sampled text.
6. **Shipping**: `sampled_accept.py` into `docker/qwen-c2-overlay.txt` (and wherever the packed block's image lists it) - this branch deliberately does not touch an overlay or a Dockerfile; a gate-only
   profile with the switch and its window; a qualification record before the gate-only requirement is lifted.
7. **Tuning**: lookup gate, draft widths, octo thresholds under sampled acceptance.

## 8. Open questions (the sampler sources are not in this repository)

1. Does the pinned `ttnn.topk` take k = 64 on a (64, 65,536) bf16 TILE tensor, and what does it cost? (Decides b1's K_MAX and its price.)
2. Does the `ttnn.sampling` op take a per-row seed tensor, or one scalar plus `manual_seed` per user? Is `manual_seed` traceable? What is the RNG (a hardware LFSR) and can a host reproduce it bit for bit? (Decides
   whether option (a) can ever be a keyed, auditable sampler.)
3. Is the generic `TTSampling` path's top-k fixed at 32, and are `p` / `temp` applied in bf16? (Why the host is the better place for the arithmetic.)
4. Under the profiles' `sample_on_device_mode = decode_only` is the prefill's first token sampled on the host, from float logits? (Decides how small the first-token graft is.)
5. How does `ttnn.topk` break ties at the K-th place, and can the scan kernel be made to break them by lowest id? (Section 4, point 2.)
6. What fraction of positions in real coding text have a nucleus above 64 tokens at the model's defaults? (Whether (b3) is ever needed; a CPU measurement on any logged logits.)
7. Throughput: tokens per round under sampling against greedy at 8 live seats. Only a card run answers it.
