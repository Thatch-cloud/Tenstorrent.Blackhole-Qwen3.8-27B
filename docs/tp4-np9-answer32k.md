# NP9: the per-request output ceiling, 16,384 -> 32,768 tokens

Users in thinking mode and coding agents hit "output token limit reached": the request contract clamps `max_tokens` to the profile's `QWEN_FAST_OUTPUT_BUDGET` (16,384), and
`serving_fast_policy._output_budget` refused any value above 16,384 outright. This note establishes what that number protects on the CURRENT production path (the ship-4 profile,
`c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s4`, on the Swift checkpoint, window 262,144), what it costs to lift, and what changed. Everything here is CPU work: no card ran,
nothing in production moved.

## Verdict

* **Device memory: none.** The policy comment (`serving_fast_policy.py:27-35`, before this change) said the budget is paid per admitted request in verifier captures, one set per 256-position mask
  family. That is true of replay attention. It is not what the serving profile builds: under C2-any beside the extent readers the captures are the sequential widths 1, 2 and 4 whatever the budget,
  there is one 2048 proposal bucket, and the page table is the whole window. A 32,768 budget allocates the same bytes as a 16,384 one (table below, held by `test_np9_answer32k`).
* **What a longer ceiling does cost is KV admission**, not KV memory: the reservation holds `prompt + max_tokens` blocks per running request for the request's whole life. The pool does not
  change (16,499 usable blocks); the number of long-prompt requests that run at once does, and only for requests that name a `max_tokens` above 16,384 (those were clamped to 16,384 before).
  A request that omits `max_tokens` still gets 16,384 on Swift (8,192 on the profile and the base): the default is unchanged.
* **The change is the policy bound plus data.** `QWEN_FAST_OUTPUT_BUDGET` may now be any multiple of 256 up to 32,768; no profile of `qwen_c2_profiles.json` changes (the ship-4 profile and its
  waiver are byte-identical, held by digest); the production change is one key in the Swift catalog entry, `"request": {"default_max_tokens": 16384, "output_budget": 32768}`.
* **What only a card can show** is that a seat runs 30,000 tokens end to end (nothing in the smoke set runs longer than the 12,288-token coding set) and that the text does not depend on the
  ceiling. The card templates are `scripts/ci/references/fusion-jobs/NP9` (see the end).

## What the 16,384 cap protects, item by item

Line numbers are the tree this branch was cut from (origin `tp4/swift`) unless marked as new.

| Candidate | What the code does | Evidence | Depends on the output budget? |
| --- | --- | --- | --- |
| The hard refusal | `budget > 16384` raised in `_output_budget`; `validate_request_sampling` refuses `max_tokens > OUTPUT_BUDGET`; `MINIMUM_MODEL_LEN = 4096 + OUTPUT_BUDGET` | `serving_fast_policy.py:39-49` (was `:35-45`), `:257`, `:349` | This was the whole cap in code. Now `MAXIMUM_OUTPUT_BUDGET = 32768` (`:36`). The window (262,144) holds `4096 + 32768`. |
| Verifier captures per 256-position mask family | `capture_bucket_rows(verifier_rows, remaining, max_verify_rows)`: widths below 8 once, each wider width once per family the budget can touch | `verifier_engine.py:78-97`; called with `policy['output_budget']` at `serving_runtime.py:668` | **No on the serving shape.** `capture_rows` is `M3_SEQUENTIAL_CAPTURE_ROWS = 4` beside the M3 blocks (`packed_shapes.py:107`, chosen at `serving_runtime.py:639-653`), so the widths are (1, 2, 4), all below 8: `capture_bucket_rows(16, B, 4) == (1, 2, 4)` for every B from 256 to 32,768. The replay shape (16 rows) is where it grows: 133 entries at 16,384, 261 at 32,768 (`test_np9_answer32k.CaptureShapeTests`). |
| The pool refusing a replay width | Under the extent readers the pool allocates no per-family tables and refuses any bucket of 8 or more rows | `serving_buffer_pool.py:620-633` | A budget can never reach a replay capture on an extent profile: the attach would raise. |
| C2-any needs sequential captures | `QWEN_FAST_ANY_REQUEST=1` raises at attach unless the captures are narrower than a replayed block | `serving_runtime.py:659-666`, `serving_request_factory.py:85-87` | The shape the serving profile has, by construction. |
| Per-request budget | Under C2-any each request's budget is its own `max_tokens` within the ceiling and the room the page table leaves, `min(max_tokens, ceiling, min(capacity, 262,112) - prompt)` | `serving_request_factory.py:61-80`, used at `:529` | The ceiling is the only place `OUTPUT_BUDGET` enters; it is read from the environment, so it follows the profile or the checkpoint. |
| Proposal (draft) bucket | One 2048-row bucket for every request on an extent profile (W6a); without it the ladder 256..2048 is bounded by 2048 | `serving_request_factory.py:95,113-129,664-668`; `dflash_proposal_inputs.py:7-13` | No. `proposal_contexts(p, 16384) == proposal_contexts(p, 32768)` for every prompt that admits both; the only budget term is the bound `max_new_tokens <= 262144 - position - 32`, which the clamp never exceeds. |
| Per-request page table | `page_width = max(68, ceil(max_model_len / 64))` = 4096 pages, whatever the request | `serving_runtime.py:554`; `serving_page_binding.py:7-17` (`pages.shape[1] * 64 >= position + budget - 1`) | No. Prompt + answer never exceeds 262,112, so the table always holds the clamped budget (held at five prompt lengths from 1 to 253,920). |
| K64j extent replay, per-bundle `cur_pos` | Each round writes the positions word and every bundle's `cur_pos = [E - 1] * B` for its own family, over the pool-lent full-width table; `admits(position)` is any start with `128 <= start` and `start + rows <= C` | `extent_attention_replay_tp.py:150-166,316-340`; `extent_attention_replay.py:60,89-121`; `packed_verifier.py:1339-1390` | No. Everything is a function of the position, not of how many tokens remain. A longer answer walks more families with the same trace and the same lent tensors. |
| Octo-T8 (the third block) | Eligible when six or more seats are live; its attention is the K64j program at flags 0x21 | `serving_octo.py` (no reference to a budget or to remaining tokens) | No. |
| Engine reuse (parked engines) | Built at attach on a synthetic request (position 1, budget 16); a rebind asks `capture_widths(position, capacity, 16, remaining, capture_rows)` and needs the widths to be a subset of the captured (1, 2, 4) | `serving_parked_engines.py:636-637,1186-1200`; `verifier_engine_tp.py:409-430` | No. `remaining` only trims the widths below four rows, at the end of an answer. |
| Drafter end limit | The drafter's last proposal reaches 32 positions past the last token: prompt + answer <= 262,112; the clamp's window is `max_model_len - drafter_headroom_tokens` | `serving_c2_contract.py:1365` (`DRAFTER_END_LIMIT`), `:1451` (the clamp), `serving_request_factory.py:56-58,75-80` | Unchanged. `prompt + granted <= 262,112` at every prompt (`test_the_drafter_end_limit_still_holds_for_every_prompt_and_every_ceiling`). |
| Prompt cap and answer room | `min_answer_tokens` 8,192, `max_prompt_tokens` 253,920 = 262,144 - 32 - 8,192 | `serving_c2_contract.py:1368-1390` (`prompt_room`) | Unchanged: the room is `max_model_len - headroom - min_answer_tokens`, not `- budget`. A checkpoint raise that would shorten it (a profile with no `min_answer_tokens`) is refused by name. |
| KV reservation | `r = ceil((prompt + max_tokens + 32) / 64) + 1` blocks, held from admission to the end of the request; a request whose `r` exceeds the whole pool is a 400 at the edge | `serving_kv_reservation.py:55,80-84`, `:156-189` (`hold`); `serving_c2_contract.py:1458-1463`; pool `real_pool_blocks` (`:1475`) = ceil(1,055,488 / 64) + 8 = 16,500, less the null block = 16,499 | **Yes: this is the cost.** The largest `r` over all prompts is 4,097 blocks at either ceiling (`prompt + answer <= 262,112`), so a single request is never refused and four full-window requests always fit. See the seat table. |
| Trace region (640 MiB) | The traces of the captured widths, the proposal bucket, the packed blocks | `qwen_c2_profiles.json`, ship-4 `additional-config.tt.trace_region_size` = 671,088,640 | No capture is keyed on tokens, so its size is not a function of the budget (it moved from 512 to 640 MiB for the third block, not for an answer length). |
| DRAM admission | Engine build 800 MB, prefill transient by prompt length | `serving_prefill_admission.py:393-420` (`engine_build_peak`, `dram_need`) | No: prompt-based. |
| Lever N | `QWEN_FAST_LEVERN_SHORT_TOKENS = 16384` | `levern_policy.py:326` | Unrelated: the tokens left to prefill, not to generate. |
| Decode-time prefix publishes | The scheduler graft publishes only whole blocks below `floor2048(P)` | `docs/sticky-sessions-writer-audit.md` | No: a long answer publishes nothing. |

So the number 16,384 was a ceiling someone chose for the replay shape's memory (and kept when the extent shape made it free), not a limit of the packed-any / octo / engine-reuse path.

## What changed

| File | Change |
| --- | --- |
| `scripts/ci/serving_fast_policy.py:27-49` | `MAXIMUM_OUTPUT_BUDGET = 32768`; `_output_budget` accepts a multiple of 256 from 256 to 32,768; the comment now states the real cost model. Unset is still 256. |
| `scripts/ci/serving_c2_contract.py:1543-1590` (new) | `output_budget_problems(profile, budget)`: a budget above 16,384 needs `QWEN_FAST_ANY_REQUEST=1` and `QWEN_FAST_EXTENT_REPLAY=1` (the shape where it is free), the request contract, and a window of at least `4096 + budget`. `request_limits` applies it to a profile that names such a budget (none does). |
| `scripts/ci/serving_c2_contract.py:1574-1590` (new) | `checkpoint_budget_problems(profile, limits, budget)`: a checkpoint's budget only RAISES the profile's and must leave the longest admitted prompt where it is (253,920), else it is refused by name. |
| `scripts/ci/serving_c2_contract.py:329-365` | `checkpoint_limits(limits, environ, profile=None)` lays `output_budget` over `limits['budget']` (and checks `default_max_tokens` against the new ceiling). |
| `scripts/ci/serving_c2_contract.py:289-320` | `configure_checkpoint` exports `QWEN_FAST_OUTPUT_BUDGET` for the checkpoint in EVERY process (the worker imports `serving_fast_policy` and reads it there), after `apply_environment` wrote the profile's value; idempotent for a child that boots again. |
| `scripts/ci/serving_c2_contract.py:1829` | `boot` passes the profile and re-reads `budget` after the checkpoint's limits, so the API server's `install_request_contract` and its log line carry 32,768. |
| `scripts/ci/qwen_c2_checkpoint.py` | `REQUEST_KEYS` gains `output_budget` (multiple of 256, at most 32,768, at least `default_max_tokens`); `OUTPUT_BUDGET_STEP`, `OUTPUT_BUDGET_MAX`. |
| `scripts/ci/qwen_c2_checkpoints.json` | The Swift entry: `"request": {"default_max_tokens": 16384, "output_budget": 32768}`. **This is the whole production change; deleting the key restores 16,384.** The base never carries a `request` block. |
| `scripts/ci/c2_serving_smoke.py`, `c2_smoke_check.py` | Opt-in `long_answer_solo_4k`, `long_answer_solo_200k`, `long_answer8_4k`, `long_answer8_96k`, `long_answer_prefix`; `long_answer_problems`. |
| `scripts/ci/make_np9_jobs.py`, `references/fusion-jobs/NP9`, `fusion-wp/NP9.json` | The card-window templates and the CPU allowlist entry. |
| Tests | `test_np9_answer32k.py`, `test_np9_pack.py` (new); `test_qwen_c2_checkpoint.py`, `test_tp4_fusion_jobs.py` (updated for the Swift entry and the new package folder). |

Why the checkpoint and not a twin profile: a twin of the ship-4 profile is a new traffic profile with a waiver of its own (`APPROVED` by the owner for ship-4 only); the catalog's
`request` block is the pattern already approved for `default_max_tokens`, it reaches Swift alone, and it leaves the ship-4 profile and its waiver untouched. A future profile may still name
32,768 itself (`QWEN_FAST_OUTPUT_BUDGET`) and the same guards apply.

## The KV reservation trade-off (the pool is 16,499 usable blocks, eight seats)

`r(prompt, max_tokens) = ceil((prompt + max_tokens + 32) / 64) + 1`; a request is admitted only while the running requests' `r` plus its own fit the pool, and a request that does not fit
waits (held, never preempted). `max_tokens` is the clamped one.

**Longest prompt at which N identical requests all run at once** (`test_the_numbers_of_the_trade_off`):

| Seats at once | max_tokens 8,192 | 16,384 (the old ceiling and the Swift default) | 30,000 | 32,768 (the new ceiling) |
| --- | --- | --- | --- | --- |
| 4 | any (253,920) | any (253,920) | any (253,920) | any (253,920) |
| 5 | 202,848 | 194,656 | 181,040 | 178,272 |
| 6 | 167,648 | 159,456 | 145,840 | 143,072 |
| 7 | 142,560 | 134,368 | 120,752 | 117,984 |
| 8 | 123,680 | 115,488 | 101,872 | 99,104 |

**Seats that fit (of eight) for identical requests** (`test_the_seats_at_each_prompt_length`):

| Prompt tokens | max_tokens 8,192 | 16,384 | 32,768 |
| --- | --- | --- | --- |
| 4,096 | 8 | 8 | 8 |
| 32,768 | 8 | 8 | 8 |
| 98,304 | 8 | 8 | 8 |
| 100,000 | 8 | 8 | 7 |
| 115,488 | 8 | 8 | 7 |
| 120,000 | 8 | 7 | 6 |
| 131,072 | 7 | 7 | 6 |
| 200,000 | 5 | 4 | 4 |
| 253,920 | 4 | 4 | 4 |

Reading it: the eighth request of a full house waits for blocks when the running ones are long AND ask for a long answer; at 99,104-token prompts or shorter all eight 32,768-token requests
run, between 99,104 and 115,488 an eighth 32,768-token request waits where a 16,384 one runs, and from 200,000 on the answer length does not matter (four seats either way). Who pays:
**only a client that names `max_tokens` above 16,384** (the clients that hit the old limit); one that names 16,384 or less, or omits it, reserves exactly what it did. The
reservation is a worst case (the request's blocks are allocated as it decodes, and returned when it ends): a request that stops at 3,000 tokens held 32,768 blocks' worth of admission
for its life but used 3,000 tokens' worth of KV. If that proves too costly in traffic the lever is the default or a per-tenant cap, not the ceiling.

**Answer room by prompt** (`enforce_request`, `test_the_clamp`): omitted `max_tokens` on Swift: 16,384 up to a 245,728-token prompt, then the room left (250,000 -> 12,112; 253,920 ->
8,192). An explicit 32,768: granted whole up to a 229,344-token prompt, 22,112 at 240,000, 12,112 at 250,000, 8,192 at 253,920; 253,921 is refused (prompt cap, unchanged).

**Time.** At about 35 tokens a second per seat with eight live (the octo evidence in the ship-4 waiver) a 32,768-token answer is 15 minutes; alone it is under 10. Anything between the
server and the user with a request timeout under that (a gateway, a client) is outside this repository and has to allow it.

## Exactness

The ceiling cannot change which token is committed; it changes where a request stops.

1. Decoding is greedy and every committed token is the verifier's argmax (the request contract forces `temperature = 0`, `serving_c2_contract.py:1443-1450`); `max_tokens` is not an input of
   any sampling or device program.
2. It reaches a round only through `remaining = max_new_tokens - emitted`: `capture_widths` (`min(rows, remaining)`, which matters below four rows), the `BUDGET_CAP` commit limit, and the
   drafter bound `max_new_tokens <= 262144 - position - 32`. All three act on the LAST few tokens of an answer. A request at `max_tokens` 32,768 and the same prompt at 16,384 run the same
   rounds, bit for bit, until the shorter one's final rounds; the first 16,384 minus a handful of tokens agree. The card test `long_answer_prefix` asks one prompt at both and holds the long
   answer to begin with the short one (all but its last 512 characters).
3. Every position it reaches is one a request can reach today: prompt + answer <= 262,112 either way (a 253,920-token prompt with 8,192 tokens, the 262k ladder gates). What is new is the
   walk: from a short prompt, a decode that crosses 100+ 256-position families and some 470 pages of vLLM allocation, each round the same path as every other (the extent's word and
   `cur_pos` for its own family, the append-only page binding validated at its refresh). The K64j records cover the extent program per family and per start, and one trace over more than 50 families.
4. The GDN state is constant-size and the drafter's history window is 2,048 rows: neither grows with the answer.

Not claimed: that a 30,000-token decode has run on the four cards on this stack (the longest smoke answers are 6,000 tokens for the stall shapes and 12,288 for the coding set).

## Card window (templates only; no tag is pushed from this branch)

`scripts/ci/references/fusion-jobs/NP9`, generated by `scripts/ci/make_np9_jobs.py`, image placeholder `tp4-np9-1` (B0 builds it from the pushed branch head with the K64j-OQ graft), Swift
pinned by revision, the ship-4 profile. `ORDER.txt` has the read rules and stop rules.

| Job | Shape | Tests | Estimate |
| --- | --- | --- | --- |
| N1 (stop) | one request | `long_answer_prefix` (4k prompt at 16,384 then 30,000, text agreement), `long_answer_solo_4k`, `long_answer_solo_200k` (200,000-token prompt, 30,000 tokens: `r` = 3,596 blocks) | 75 min |
| N2 | eight concurrent, 4k prompts, 30,000 tokens each | `long_answer8_4k` (535 blocks a seat, 4,280 in all: nothing held) | 70 min |
| N3 | eight concurrent, ~96,000-token prompts, 30,000 tokens each | `long_answer8_96k` (1,971 blocks a seat, 15,768 of 16,499: all eight resident, no `kv reservation hold` line) | 80 min |

Every job names `concurrent8_steady` (the profile's required eight-seat check). `c2_smoke_check` holds: every user `finish == "length"` with exactly 30,000 tokens (`ignore_eos`, so a
server whose ceiling clamped the request reads as a fault), the container log shows `request contract installed: greedy, max_tokens <= 32768` (the checkpoint's budget reached the API
server), and `long_answer_prefix` `prefix_equal`. The harness had no long-output test; these five are new, opt-in (named in `C2_SMOKE_TESTS`), and their seat counts are held at or under
eight by `test_w2ln_smoke_shapes`. `LONG_ANSWER_TOKENS` and `LONG_ANSWER_PREFIX_TOKENS` override the budgets for the CPU test's fake server; the cards run 30,000 and 16,384.

## Risks

* **Never run 30,000 tokens on the cards on this stack.** N1 is the gate; stop if a request ends short, the text differs, or a stall or refusal line appears.
* **Admission trade-off for long-prompt traffic**: eight seats of 100,000-token prompts that all ask for 32,000+ tokens no longer all run at once (seven do). Product decision, not a fault.
* **The Swift catalog entry reaches every Swift boot of an image built from this branch**, including the Swift window's own jobs (`references/fusion-jobs/SW`), whose tests ask for at most 3,000 tokens
  and are unaffected. The base is untouched.
* **Image rebuild**: `serving_fast_policy.py` (both copies the overlay installs), `serving_c2_contract.py`, `qwen_c2_checkpoint.py` and `qwen_c2_checkpoints.json` are all in the overlay and
  the copy lists already; nothing new needs listing. An image from before this change given the new catalog refuses the Swift boot (unknown `request` key): the pair ships together.
* **Time-outs outside the repository** (gateway, client) for a 10 to 15 minute response.
* **Revert**: delete `"output_budget": 32768` from the Swift entry. No profile, no waiver and no image default moves either way.
