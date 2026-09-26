# Serving C2 to real coding traffic: the plan

2026-09-26, revision 2. This revision applies the feasibility and traffic reviews. It is read-only analysis: nothing was edited, committed, dispatched or run on the rig. The only reads beyond local files were two `gh run view` calls. Written to `docs/c2-serve-for-real-plan-2026-09-26.md` (UTF-8).

**What changed in revision 2**
- **Hardware.** M+A are development cards again, so hardware no longer sets the calendar (§1, §4).
- **Total effort.** It rises from 38-68 to 65-121 engineer-days of required work:
  - S1 grows for a first bring-up, the contract cap and a source check;
  - S2 grows for its kernel, readers and mask kernel;
  - a new production-readiness track R collects the traffic gaps.
- **S1** is an internal milestone.
- **D1** moves to S2.
- **Lever N v2** moves to S3.
- **Two optional stages for agent traffic**, E (engines prebuilt per slot) and P (prefix reuse), are costed but kept out of the core total.
- **The exactness gate** gets a flip policy.
- **The `general` bar** is now measured: 18.6-19.3 tok/s.
- Rejected and partly rejected review points are in the Appendix.

**Citation roots**

- **Unprefixed paths** are in `D:/Programming/Tenstorrent.Blackhole-Qwen3.8-27B/.worktrees/c2-serving` at aaf35537, branch `serving/c2-coding`.
  - aaf35537 changes only `.github/c2-serving-job.env` and `.github/workflows/qwen-c2-serving.yml` over 25b7bd42 (`git show --stat aaf35537`).
  - Among the pre-existing `scripts/ci` files, only these three changed since t32-score-reuse HEAD 0cfd3a2b (`git diff --stat 0cfd3a2b aaf35537 -- scripts/ci`):
    - `packed_verifier.py` (6 lines);
    - `serving_fast_policy.py`;
    - `serving_request_factory.py`.
  - Workflow line numbers are HEAD's. The `unserve` step moved them down by 16; the copy list is at `:205-208` at HEAD and at `:189-192` at 25b7bd42.
- **Caveat on three files.** `frozen_combined_runtime.py`, `frozen_recipe_context.py` and `frozen_target_replay.py` differ between the image's bundle (77d6995a) and HEAD, by 38, 71 and 50 changed lines (`git diff --stat 77d6995a HEAD`).
  - They are also rewritten at staging (`frozen_ladder_stage.py:8-40`, `frozen_combined_adapters.py:30-120`).
  - So HEAD line numbers for those three files are **not** what the image runs.
- **Short names:**
  - `T32:` means `git show 0cfd3a2b:<path>` in `.worktrees/t32-score-reuse`;
  - `SKILL` is `the tt-qwen-serving-deploy operator skill (local, not in the repo)`;
  - `MEM` is the auto-memory directory;
  - `JOB` is `the operator's local scratch directory (not in the repo)`;
  - `LEVN` is the main checkout's `docs/lever-N-prefill-decode-interleave.md`;
  - **B1-B5** are the five blocker analyses this plan merges (any length, long answers, packed at mixed positions, request lifecycle, validation).

**Which citations were re-read**

- **Revision 1:**
  - `serving_request_factory.py`, `frozen_combined_runtime.py`, `verifier_engine.py`, `model_batch.py`, `attention_request_plan.py`, `attention_mask_replay.py`, `packed_verifier.py`, `serving_packed_step.py`, `packed_shapes.py`, `serving_runtime.py`, `serving_fast_policy.py`, `serving_c2_contract.py`, `serving_worker_hook.py:425-482`, `serving_lifecycle.py`, `serving_startup.py:142`, `qwen_c2_profiles.json`;
  - the C2 Dockerfile and workflow;
  - the `sdpa_decode_qwen` reader and README, and `apply_factory_qwen.py`;
  - the real-text and feasibility docs;
  - SKILL, LEVN, MEM goal, and the JOB plans.
- **Added in revision 2:**
  - the commit, tag and run of the `unserve` action (aaf35537; run 36191246394 and its step log);
  - the priority probe log (run 36106529648);
  - the tag history of `c2-serving-job.env`;
  - `serving_worker_hook.py:20-85`, `serving_packed_step.py:550-575,817-866`, `test_serving_packed_step.py:1038-1073`;
  - `target_t16_attention_gate.py:1-80`, `frozen_combined_adapters.py:30-120`, `frozen_target_replay.py:183-190`, `frozen_ladder_stage.py:1-40`;
  - `attention_replay.py:50-100`, `pooled_attention_replay.py:195-200,388-396,452-460`;
  - `reader_decode_qwen.cpp:25,45,78-86,120-190,293,366-375` and `apply_factory_qwen.py:70-80`;
  - README lines `:152-164`, and the `sdpa_decode_qwen/` and `sdpa_decode_slice/` file lists;
  - `dflash_proposal_inputs.py:1-13` and `dflash_proposal_trace.py:100-130`;
  - `serving_lifecycle.py:1-30,238-300,425-478`;
  - `c2_serving_smoke.py` and `c2_platform_replay.py:160-180`;
  - `docker/qwen-fast-serving.Dockerfile:1-20`;
  - T32 files: `dflash_packed_proposal_coordinator.py:592-650`, `coding_request.py:12-24`, `coding_functional_eval.py:40-55`, `docs/combined-prefix-cache.md` (all), `docs/gotchas.md:150-160`, `docs/real-text-2026-09-24.md:120-135,200-230,286-296`, `docs/four-streams-131k-feasibility-2026-09-23.md:505-560`;
  - SKILL in full, LEVN `:118-200`, and `MEM/goal-200tps-concurrent.md:1-30`.

Citations marked **(c)** are carried from B1-B5 and were not re-read. Numbers are tagged **(m)** measured, **(d)** derived from measurements, or **(e)** estimated.

---

## 0. METERING RESULT (added 2026-09-26 after revision 2)

- **Metering 2026-09-26 (spark-8c4d, 14 days, run 36202702533):** 6,612 requests; prompts average ~42,200 tokens (278.8M) of which 91.7% hit the prefix cache (255.6M cached); answers average ~740 tokens (4.92M); up to 16 running, queue depth up to 18; TTFT median ~2 s. => Prefix reuse (stage P) is NOT optional: without it every TT turn re-prefills ~42k tokens (15-20 s). D-f is answered: P (and E) before S3. max_num_seqs 4 is tight.
- **No September regression (run 36202676747):** `thatch-serving-tt:863112a` decodes 19.6 tok/s single stream in the same smoke shape (C2 `general`: 20.2); the 2026-09-04 36 ms/token does not reproduce. Mesh p300/FABRIC_1D vs p150_x2 and the fast-path env make no difference to stock decode (runs v14-v17, all 20.1-20.3).


## 0b. RISK RESULTS (added 2026-09-26; skeptic-verified reports in tmp/risks/verified-*.md)
- **R4 vLLM semantics** (verified-r4-vllm.md): an abort that takes a packed round 2->1, 3->1 or 4->1 kills the engine (CONFIRMED in code, both draft paths) - client disconnects will hit it: D1/D2 must cover aborts, and `discard_stale_ticket` cannot rescue a `refuse_round`-failed session. The one-fresh-prompt-per-step cap is `lever_n_model_patch.patch_scheduler` (allowed=1), not `lever_n_scheduler_patch.patch_one_in_flight`; C2 was measured on the stock TTScheduler. `ignore_eos` fix: only `serving_request_factory.py:154-155` (`:94` already exempts it). Check `generation_config.json`'s EOS ids equal the profile's `eos_ids` (a one-sided set strands a request).
- **R21 parsers** (verified-r21-parsers.md): tool calls survive multi-token deltas (2,295 streams, 0 crashes); K1 (a real `</think>`/`<tool_call>` binding to a later piece-spelled lookalike) appears at >=16 tokens/delta. Fix M: re-chunk at marker tokens via a wrapper on `DelegatingParser.parse_delta` installed from the boot hook (with the listed hardening); I1 truncation fix only with `finish_reason: length`. Belongs in S1 (sequential spec decode already emits multi-token deltas) - R track.
- **Q4 prebuild** (verified-q4-review.md): DROP bc09236b - prebuild moves the build into the 4th user's TTFT instead of saving it; REUSE at 131k x 4 releases rather than rebinds; empty-string REUSE breaks flags-off equivalence. Amend S4: no Q4 prebuild item.

## 1. Verdict

**Effort (e).** Getting all of C2 onto real coding traffic, safely enough to leave unattended, is **65-121 engineer-days** of required work.

| Stage | What it is | Eng-days |
|---|---|---|
| **S1** | lift the host pins | 13-25 |
| **S2** | K64j: the packed block at any position | 25-45 |
| **R** | production readiness, run in parallel | 17-33 |
| **S3** | Lever N and 1-live packing | 8-14 |
| **S4** | hardening | 2-4 |

- **Optional levers for agent traffic:**
  - **E**, engines prebuilt per slot: 8-15 days;
  - **P**, conversation prefix reuse: 15-30 days.
- **Only if chosen:**
  - a stock-decode lane for short outputs: 5-10 days;
  - structured output: 6-12 days.
- **Hardware runs:**
  - card B: 6-12;
  - TT-Sim: 2-4;
  - M+A: **32-56 planned**, or up to about 86 at the C2 campaign's 35% run-failure rate (B5 §1.2 (c)).

**Hardware is not the limit any more.**
- M+A were given back for development on 2026-09-26:
  - `MEM/goal-200tps-concurrent.md:11`;
  - commit aaf35537;
  - run 36191246394 (tag `experiment/c2-serving-v13`, `success` at 2026-09-25T21:23Z).
- The step log shows no `thatch-inference-*` container to stop, and `fuser` found nothing holding M or A.
- The planned M+A work is about 16-56 h, plus about 10 h of load testing and soak: about **2-8 rig-days** in total.
- Two open items:
  - whether the placement was also removed through the admin API. That is **UNVERIFIED**; the step's own comment says the platform may reload it until then (`qwen-c2-serving.yml:101-102`);
  - how long the pair stays in development (D-d).

**Calendar (e): engineering sets it now.**
- **One engineer:** 13-24 weeks for the core.
- **Two engineers**, one on the kernel and one on serving: about 9-16 weeks. The serving side is the long pole: S1, S2's serving items and R come to about 40-77 days.
- **Agent-paced**, as in the C2 campaign (confidence low):
  - S1 bring-up in 1-3 weeks;
  - S2 in **4-8 weeks**;
  - R alongside S2;
  - S3 and S4 2-4 weeks after that.
- **Why confidence is low.** The C2 campaign's pace does not transfer to this work.
  - It ran about 115 gate tags from v117 to v231 between 09-23 19:25 and 09-25 13:46.
  - That was bound by hardware iteration on small, isolated changes.

**What beats `general`, and when**

- **S1 is an internal milestone, not a release.**
  - It lets the fast path take any request, but decode is 4-row sequential.
  - It loses to `general` at two or more live users.
  - Its packed block practically never engages (§2.2).
- **S2 is the first stage whose steady decode beats `general` at one to four live users.**
- **End to end, S2 is about level with `general` for coding agents (e, §3).**
  - Every turn re-prefills its whole context: neither path has a prefix cache.
  - The fast path's prefill plus build is slower than `general`'s prefill.
  - Every arrival stalls all other decoders for about 4 s.
  - The model's result:
    - S2 wins by about 18% on 1,000-token turns (e);
    - it is level or slightly behind on 300-token tool-call turns (e);
    - a clear end-to-end win, about 3x on turn latency (e), needs Stages E and P too.
- **No production cutover before R is done.** R covers:
  - a loop guard;
  - cancellation;
  - context-limit errors;
  - observability;
  - a known-answer canary;
  - deploy, drain and heal;
  - edge admission control.

**Critical path**
- **Kernel:** the K64j probe, then K64j with a new flag bit (0x10 is taken, §2.3), then its card-B byte gates.
- **Then:** the extent readers and a new mask-refresh kernel, then packed integration, then the S2 M+A ladder.
- **Must come before the S2 M+A gates:**
  - S1's code;
  - S1's first bring-up of the fast path inside the serving image. That has never completed a request there (§2.0).

| Stage | What ships | Eng-days | Card B / Sim / M+A runs | Per-user decode (live users 1 / 2-3 / 4) |
|---|---|---|---|---|
| `general` (the bar; not serving now) | stock TT decode, any shape, context 65,536 | - | - | 18.6-19.3 (m, run 36106529648; 15.3 m under CI load, SKILL:84) / ~14 (e) / 12-13 (m, SKILL:85) |
| `exact` (C2) | prompts of exactly 131,072 tokens, 256 out | - | - | never served a request in the serving image / - / 28.2-28.6 (m, gate container) |
| **S1 C2-any** (internal) | fast path takes any request; 4-row sequential spec decode | 13-25 | 0 / 0 / 8-14 | 22-33 (d) / 7.6-12 (d) / ~6 (d) |
| **S2 C2-packed-any** | K64j extent replay; all C2 features live | 25-45 | 6-12 / 2-4 / 13-20 | 22-33 / 27-37 (e) / 27-35 (e; 27.3-28.6 m at 4 x 131k) |
| **R readiness** (parallel; gates any cutover) | loop guard, cancel, errors, parsing, metrics, canary, deploy/heal, edge queue | 17-33 | 0 / 0 / 4-8 | unchanged; safe to leave unattended |
| **S3 arrivals** | Lever N v1 and v2 in C2; 1-live packing | 8-14 | 0 / 0 / 5-10 | ~30-36 (e) / same / same; decoders no longer freeze during prefill |
| **S4 hardening** | Q4 prebuild, soak | 2-4 | 0 / 0 / 2-4 | same, fewer stalls |
| *E (optional)* | engines prebuilt per slot, repointed per request | 8-15 | ? / 0 / 3-6 | removes the ~4 s stall at each arrival |
| *P (optional)* | conversation prefix reuse per slot | 15-30 | 0 / 0 / 5-10 | removes most re-prefill (e) |

---

## 2. Stages

### 2.0 Starting point

**What pins C2 to the benchmark shape**

1. **Per-request T16 gate (policy).**
   - `from_prefill` builds every engine with `target_attention_t16=True` (`serving_request_factory.py:195-198`).
   - The staged gate admits only `position == staged_combined_context()` and `1 <= remaining <= 256` (`frozen_combined_runtime.py:74-84`).
   - It is reached through `verifier_engine.py:162-169`: `target_t16_attention_gate.validate_request_option` and `qualify`. Staging rewrites both to the frozen runtime's (`frozen_combined_adapters.py:66-70`).
   - The refusal is raised inside `execute_model`, which kills the engine. Smoke v3 died this way on a 54-token warmup (SKILL:67-73).
   - `frozen_combined_runtime.py` cannot be edited: its bytes are pinned against the native cache's `adapter_sha256` (`serving_native_install.py:57`). The fix has to go in the callers.
2. **The output budget is a server constant, not the request's (policy).**
   - The session, the drafter and the page check all take `OUTPUT_BUDGET` (`serving_request_factory.py:98,146,155,169`).
   - `GreedySession.max_new_tokens` is the whole budget (`serving_worker_hook.py:25-54`, docstring).
3. **One family for the packed block (structural).**
   - `capture_position = capacity - 256` and `replay_capacity = (capture_position//256+1)*256` (`packed_verifier.py:490-493,533`).
   - Every round is re-checked per ticket:
     - `packed_verifier.py:1036-1044`;
     - `serving_packed_step.py:226-235`;
     - the gate is `attention_mask_replay.validate_ticket`, which needs `capacity-256 <= start` (`attention_mask_replay.py:30-32`).
   - Outside the family the round falls back to the 4-row sequential step. That is graceful, not fatal.
4. **Paths that kill the engine, or fail a request (lifecycle).**
   - **Every in-engine refusal kills the engine:** `serving_lifecycle.py:296,473` sets `failed` and re-raises (SKILL:60).
   - **Correction to revision 1.** `refuse_round` does not kill the engine (`serving_packed_step.py:845-866`).
     - It fails that round's requests and returns their outputs.
     - Before the fix, the survivors' failed sessions made the next `prepared_ticket` raise, which did kill the engine (`test_serving_packed_step.py:1049-1051`, run 35564623068).
     - `discard_stale_ticket` (`serving_worker_hook.py:55-85`, called at `:478`) now stops that stale round from forming.
     - Whether B4's 2→1 abort case still reaches `refuse_round` is **UNVERIFIED**.
   - **`ignore_eos` followed by a later EOS** (B4 5b (c)).
5. **Whole-prompt prefill.**
   - `no-enable-chunked-prefill` is set in every profile (`qwen_c2_profiles.json:34,89,140`).
   - A 131k prompt freezes every decoder for 55-63 s (T32 `docs/real-text-2026-09-24.md:409-413`).
   - Lever N is not in the image (B4 §3.6 (c)).
6. **Idle segments cost as much as live ones, and one live user is never packed.**
   - `MAX_IDLE_SEGMENTS = 2` (`packed_verifier.py:992`).
   - `PADDED_MIN_USERS_DEFAULT = 2` (`packed_verifier.py:396`).
   - Padded rounds take 167-173 ms, the same as a full round (B3 §1 (m)).
7. **The fast path has never completed a request inside the serving image.**
   - The 13 `experiment/c2-serving-v*` tags:
     - v1-v3 ran the `coding` profile;
     - v1's geometry was refused (SKILL:104-108);
     - v3 died on the warmup (SKILL:71-73);
     - every tag from commit 4635377f on runs `general` (`git log -p -- .github/c2-serving-job.env`).
   - `exact` was never smoked in the image.
   - `c2_serving_smoke.py` compares nothing against references.
   - **The image differs from the gate:**
     - 8 CPUs, not 16 (SKILL:234);
     - a read-only root with tmpfs caches (SKILL:146-150);
     - the kernel cache through the `/models/.qwen-c2` symlink (`docker/qwen-c2-serving.Dockerfile:64`);
     - an argv rewritten by the contract;
     - the boot hook's `os._exit` (SKILL:195-196).
   - Every C2 (m) rate below was measured in the 16-CPU gate container.
8. **The T16 gate is also the per-request source check.**
   - Under staging, `verifier_engine.py:167-169` runs `frozen_combined_runtime.qualify_target` → `qualify` (`frozen_combined_runtime.py:93-123`), which hashes every pinned component source.
   - That is the check that refused image v42 (`probe_frozen_pins.py:3-9`).
   - Attach separately runs the windows, down and norm qualifications (`dflash_combined_request.py:57-59`).
   - Whether any other serving-path caller of the frozen `qualify` remains is **UNVERIFIED**. Staging also rewires `dspark_8k_build.py` and `dspark_8k_admission.py` to call it (`frozen_combined_adapters.py:52-56,113`).
   - The staged geometry is already marked `requested_geometry_qualified=False, requires_fresh_full_request_audit=True` (`frozen_ladder_stage.py:21-25`).

**A correction to SKILL:110-113.** It says "the output budget is structural". That holds only for per-request engines that capture 8- and 16-row buckets.
- Next to the M3 block, the engines are trimmed to widths 1, 2 and 4 (`packed_shapes.py:79-96`, `serving_runtime.py:224-234`).
- `capture_bucket_rows(16, N, 4)` is `(1, 2, 4)` for every N (`verifier_engine.py:78-95`).
- So under C2, a larger budget costs no verifier device memory beyond the KV blocks, which are already sized at 4 × 2052 (`qwen_c2_profiles.json:27`).
- **What a budget does still cost: draft proposal traces.**
  - One is captured per draft bucket, from `min(pos,2048)` to `min(2048, pos+budget)` (`dflash_proposal_inputs.py:7-13`, `dflash_proposal_trace.py:106-130`).
  - A 131k prompt gets one; a 60-token prompt with a 16,384 budget gets four (256, 512, 1024 and 2048).
  - Their DRAM and build time are unmeasured (R3).

**Reference bars**
- **`general`, measured in the agent's 8-CPU shape:**
  - 18.57 tok/s single-stream (median) before the CPU-priority change and 19.26 after, at load averages of 22-30 (run 36106529648, `c2_decode_probe.py`);
  - 15.3 tok/s single-stream and 12-13 tok/s each at four users, measured earlier with 12 CI pods active (SKILL:82-85);
  - 6.8 s time to first token at 22k (SKILL:84).
- **C2 single-stream:** 32.5 tok/s at 131k (v160).
- **What serves Qwen coding traffic now, if anything, is UNVERIFIED.** The `unserve` step found no container on M+A.

### 2.1 Work that starts now

1. **Before the first M+A run: confirm the placement is gone.**
   - Check the `Qwen/Qwen3.8-27B` placement through the admin API. A placement left behind can reload onto M+A in the middle of a gate (`qwen-c2-serving.yml:101-102`).
   - This needs an admin session, which the user mints (SKILL:169-170).
   - It also moves the stale `MEMORY.md` index line ("PAUSE") to lifted. I have not edited it.
2. **A first bring-up that needs no code change.**
   - Run the `exact` profile in the agent's container shape, at 4 × 131,072 real-text prompts, through the serving endpoint.
   - It passes when the texts are byte-identical to v235.
   - It retires §2.0 item 7 for the benchmark shape.
   - In the same session, run a paired `c2_decode_probe.py` on `general` (R14).
3. **Metering data**, from the Thatch side (source **UNVERIFIED**):
   - context sizes;
   - output lengths per turn;
   - a histogram of live users;
   - how often clients send `stop`, `ignore_eos`, `tool_choice` required or named, `response_format`, and FIM/autocomplete.
   - This feeds D-a, D-e, D-f and G8's workload.
4. **S1's CPU work.**
   - Tests on Python 3.11, named in `qwen-integration-cpu.yml` (MEM ci-tests-need-allowlisting).
   - Run through a PR to main, because tags never trigger the suite (MEM cpu-suite-needs-python-311).
5. **S2's kernel track.**
   - The K64j probe, then the graft and its byte gates.
   - They run on card B (`qwen-card-b.yml`, harnesses under `optimisation/ttnn-op/<dir>/*.sh` using `qual_card`, B5 §1.1 (c)), or on card M now that it is free (`run_card_m.sh` exists in `sdpa_decode_qwen/`).
6. **S2's TT-Sim lanes**, on the dev PC's WSL copy (B5 §2.3 (c)).
7. **Tooling:**
   - the real-text matrix with a prompt length per user;
   - the exactness policy (§2.2 item 5);
   - the extended platform replay;
   - the trace-driven load generator.

### 2.2 Stage 1: "C2-any", the fast path takes any request and decodes sequentially (internal milestone)

**Deliverable.** A C2 serving image with a new profile `c2`:
- **Geometry** 131,328, so the page width stays the reviewed 2052. `ordered_cache` admits only 1..1024 or exactly 2052 (SKILL:99-108).
- **8,208 KV blocks,** the same as `exact` (`qwen_c2_profiles.json:27`). That is 4 × 2052, so no admitted request can outgrow the cache and no preemption is needed.
- **An output ceiling of 16,384,** the policy maximum (`serving_fast_policy.py:35-48`), with `max_tokens` clamped per request to capacity − prompt (`serving_c2_contract.py:187-189`).
- **A prompt cap of about 123,136** (131,328 − 8,192), so every admitted request keeps at least 8k of answer room (decision D-c, R item R2).
  - **Correction to revision 1:** the cap needs a contract change.
  - `room = max_model_len - budget` (`serving_c2_contract.py:173`), and `max_prompt_tokens` can only lower it. With a 16,384 budget, the largest prompt today would be 114,944.
- The fast path admits everything the edge contract admits.

**Code touched**

1. **Lift the per-request pin in the caller** (B1 Phase 0).
   - In `from_prefill` (`serving_request_factory.py:195-198`), when the engine's `capture_rows < 8`, build it with `attention_replay=False, target_attention_t16=False`.
   - `capture_rows < 8` always holds next to M3 (`M3_SEQUENTIAL_CAPTURE_ROWS = 4`, `packed_shapes.py:86`).
   - Rows below 8 never replay (`model_batch.py:442`) and are captured once each (`attention_request_plan.py:53`), so their device programs do not change.
   - The T16 gate (`verifier_engine.py:162-169`) and the width-cap guard (`:171-172`) stop firing.
   - **Alternative,** if CPU tests show pool storage keyed on `attention_replay`: keep replay on and relax the unpinned width-cap guard instead.
2. **Replace the source check that step 1 removes.**
   - Add an attach-time check that hashes the same pinned component set, from the frozen evidence reports, once per process.
   - Refuse to attach on a mismatch.
   - Decide R11 (what "qualified" means off the frozen geometry) now, not at S2.
3. **Budget per request.**
   - Use min(request `max_tokens`, ceiling, capacity − prompt) at `serving_request_factory.py:98,146,155,169`.
   - Keep `real_remaining_budget` (`serving_worker_hook.py:25-54`, used at `:449`) as the backstop.
4. **Lifecycle** (B4). D1, the survivor narrowing, **moves to S2**: in S1 the packed block practically never engages, and `refuse_round` fails requests, not the engine (§2.0 item 4).
   - **D2:** a host-side refusal ends one request as FINISHED_ABORTED instead of re-raising at `serving_lifecycle.py:296,473`.
   - **D4:** four edge and policy fixes:
     - `ignore_eos`: pass `eos_ids=()` when it is set (`serving_request_factory.py:154-156`);
     - `max_tokens` exhausted at the first token is terminal. `serving_lifecycle.py:439-449` checks only EOS and then builds a bridge at `:450`. **This is on the path of every placement:** the agent's warmup is `max_tokens` 1 (SKILL:174);
     - stop strings, allowed once D1 lands in S2 (`serving_c2_contract.py:164-165`, `serving_fast_policy.py:162`);
     - `QWEN_FAST_FAULTHANDLER=0` (`docker/qwen-c2-serving.Dockerfile:71`).
5. **Gate tooling and the exactness policy.**
   - A prompt length per user, and long answers (`lever_n_m3native_gate.py:43-55`, `real_text_prompts.py:36-47` (c)).
   - **Exactness policy, fixed before G4:**
     - the reference is a solo run on the same image with the same arithmetic flags;
     - compare full answers, not prefixes;
     - a first divergence triggers one re-run of both arms. If it reproduces, the arm FAILS, unless the arms differ in an arithmetic flag. Then it is NOT COMPARABLE and needs a reference under the same arithmetic (T32 `docs/real-text-2026-09-24.md:293`).
   - **Why this is needed:** near-tie tokens flip within 0-758 characters when arithmetic changes (T32 `docs/real-text-2026-09-24.md:222-227`).
   - `c2_platform_replay.py` gets these steps:
     - a long prompt;
     - a streamed answer of thousands of tokens;
     - four arrivals at random times;
     - survival after a refusal;
     - `max_tokens=1`;
     - reload and restart.
6. **Image plumbing.** The C2 layer overlays only nine `scripts/ci` files (`.github/workflows/qwen-c2-serving.yml:205-208`, `docker/qwen-c2-serving.Dockerfile:49-51`).
   - `serving_worker_hook.py`, `serving_lifecycle.py` and `serving_packed_step.py` would need adding.
   - `greedy_session.py` arrives by a third route: `COPY bundle/speculative-decoding` (`docker/qwen-fast-serving.Dockerfile:8`). It is outside both copy lists and outside the closure test (MEM serving-image-bundle-provenance).
     - Unchanged since 77d6995a today.
     - D1 changes it in S2.
   - Either extend the overlay and the Dockerfile, or build a new fast-serving base covering all three routes.
   - Read the launched argv (MEM read-the-launched-argv), and check traceback line numbers against the bundle, not HEAD.
7. **First bring-up of the fast path in the serving image** (§2.0 item 7).
   - Budget for the gate-container vs agent-shape differences: CPUs, read-only root, cache symlink, contract argv and `os._exit`.
   - Measure rates in the agent shape (R9).
8. **Dev-run automation:** reset, and the fabric re-measure before any placement (SKILL:159-165, 201-202).

**Engineer-days: 13-25**

| Item | Days |
|---|---|
| Phase 0 caller switch, budget per request, profile | 1.75-3.25 |
| Contract prompt-cap formula, its tests, the SKILL:99-102 invariant | 0.5 |
| Attach-time source check | 0.5-1 |
| D4 edge and policy fixes | 1-1.5 |
| D2 request quarantine | 2-4 |
| Exactness policy in `real_text_compare` | 0.5-1 |
| Real-text matrix: a length per user, long answers, short-prompt ladder | 2.5-5 |
| Extended platform replay | 1-1.5 |
| First bring-up in the serving image | 2-4 |
| Image plumbing (three delivery routes) | 0.5-1.5 |
| Dev-run reset and re-measure automation | 0.5-1 |
| CPU CI hygiene | 0.5 |

**M+A iterations: 8-14**

| Gate | Runs | What passes |
|---|---|---|
| G1 provenance | (every run) | The `QWEN_` string diff of the `.so`, the closure test, and the launched argv line (SKILL:62-63) |
| Bring-up (was "G3 control") | 2-5 | First, `exact` unchanged in the agent shape at 4 × 131,072 real text: byte-identical to v235. Then `c2` with Phase 0 at the same shape: byte-identical again. Rates are recorded against the 16-CPU gate's. |
| G4, part 1: lengths | 2-3 | Real text at user lengths {60, 255, 2047, 2048, 2049, 4k, 32k, ~60k, ~120k}, with 2-4k-token answers and EOS on. Concurrent equals sequential on the same image under the exactness policy. Acceptance per step is recorded on coding prompts (R8). Proposal-bucket count, DRAM and build time are recorded at short prompts (R3). |
| Lifecycle | 1-2 | Drops at 4, 3 and 2 live users; a triple drop; EOS mid-round; a 5th request into a freed slot; `ignore_eos`; `max_tokens=1`; cancel during prefill and during build (R3 in track R). The engine stays alive, survivors are exact, slots are released, the ledger stays flat. |
| G5 memory | (in the above) | `[PINDIAG] dram after engine` at 4 × max prompt × ceiling. Only 0.887 GB is free per chip at v238 (B2 §2 (m)). |
| Smoke plus G7 replay | 1-2 | The agent's sequence plus the new steps. Restart time on the fast-path profile is recorded; it is currently unknown. |
| Contingency | 1-2 | |

**What S1 does** (internal; not for production unless D-a says otherwise)
- **It admits every request the contract admits:**
  - prompts up to about 123k, against 65,536 on `general` (`qwen_c2_profiles.json:134`);
  - answers up to 16,384 tokens;
  - greedy decoding, with sampling coerced (`serving_c2_contract.py:177-186`);
  - tool calls (`tool_choice` auto) and reasoning parsed at the edge (SKILL:55-56).
- **Structured output** and `tool_choice` required or named still return 400 (`serving_c2_contract.py:158-160`).
- **The packed block practically never engages.** It needs every live user inside [131072, 131328) at once (`serving_packed_step.py:226-235`, `packed_verifier.py:1036-1044`). Under a 123k prompt cap, that takes at least 8k generated tokens from all four users together.
- **So decode is 4-row sequential speculative decode:**

  | Live users | Per-user decode |
  |---|---|
  | 1 | **22-33 tok/s** (d): 2.76-4.0 tokens per 122-127 ms step. The three sources disagree: 2.93 (B3 §5, v235/v238, small n); 3.75-4.0 (`JOB/variable-user-packed-round-plan.md:100`); and 2.76 as the floor (the all-rounds mean at 4 × 131k, which includes pair users at 1.0-1.5 tokens per step: T32 `docs/real-text-2026-09-24.md:130,209`). |
  | 2 | 10-12 (d) |
  | 3 | 7.6-8 (d) |
  | 4 | about 6 (d) |

- **Time to first token** rises by the build of 3.4-4.1 s (T32 `docs/four-streams-131k-feasibility-2026-09-23.md:529-531`), plus a slower prefill. The fast path's first-token time at 32k is 14-16 s (`:510,513`), against `general`'s ~3.2k tok/s (SKILL:84).
- **Each arrival stalls every other live decoder** for the build (the same lines).

**Verdict on S1.**
- For a lone user it breaks even with `general` after about 180-840 output tokens at a 30k context (d).
  - Basis: 19 tok/s against 22-33 tok/s, and a 4-6 s prefill-plus-build penalty.
- It loses at two or more live users.
- It is S2's prerequisite and the first proof that the fast path serves in the image.

### 2.3 Stage 2: "C2-packed-any", the packed block at any position (all C2 features live)

**Deliverable.** An image carrying graft K64j and `QWEN_FAST_EXTENT_REPLAY=1`, where the 64-row block serves 2-4 live users at arbitrary, different positions. This is where these C2 features start working for real traffic:
- the packed M3 verify;
- K64i;
- K5-A;
- the pair fold;
- Q4.

**Design** (B1 Phase 1 = B2 (c) = B3 P2, merged)

1. **K64j: a new mode flag over K64i.**
   - **The flag cannot be 0x10.** 0x10 is the card test's N3 "unknown flag" negative control (`sdpa_decode_qwen/README.md:164`).
     - Use another bit, proposed 0x20; whether it is unused in the stage-1, 3 and 4 binaries is **UNVERIFIED**.
     - Or move the N3 control to a different value.
   - **Today:**
     - The non-causal Qwen reader fixes `cur_pos = St*32-1` (`reader_decode_qwen.cpp:124-127`).
     - Only the causal branch reads a per-user `cur_pos` and honours the `UINT32_MAX` "skip this user" value (`:128-168`).
     - The chunk split is `get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos)`, then `get_workload_for_core` (`:178-190`), and the compute kernel mirrors it (`sdpa_flash_decode_qwen.cpp:169`).
   - **Under the new flag:**
     - Take the `cur_pos` read path while staying non-causal, with the tail mask applied on the final chunk.
     - Relax the factory refusal `"modes are non-causal, full-window and take no cur_pos tensor"` (`apply_factory_qwen.py:71-72`).
     - Make the same edits in:
       - the stage-3 kernels (`sdpa_decode_qwen/stage3/`);
       - the K64i slice reader and writer (`sdpa_decode_slice/reader_decode_qwen_slice.cpp:193`, `writer_decode_qwen_slice.cpp`);
       - their sha-pinned generators (`make_qwen_kernels.py`, `make_slice_kernels.py`) and source tests.
   - **Why it is exact.** Each user gets `cur_pos = E-1`, with `E = (start//256+1)*256`. That is the value today's call at capacity E bakes in.
     - `k_chunk_size=256` is fixed in every replay config (`attention_replay.py:53-54`, `pooled_attention_replay.py:199-200`).
     - The factory requires `St % Sk_chunk_t == 0` (`apply_factory_qwen.py:77`).
     - So the split depends on `cur_pos` and on `max_dynamic_chunk_size` (`reader_decode_qwen.cpp:45`); the served value of the latter is **UNVERIFIED**.
     - The compile-time `St` still feeds the tail-mode assert (`:82`), the mask strides for a full-width mask (`:293`), and the offsets outside paged mode (`:366-375`).
   - **Scope of the claim (corrected from revision 1).**
     - Card-level tail-vs-legacy was checked only at 2,304, 33,024 and 131,328 keys (README:152).
     - The staged `validate_ticket` admits 4096..16640 or exactly the selected capacity (`frozen_target_replay.py:183-190` over `attention_mask_replay.py:18,26`). So model-level runs cover only the 33,024 (32k campaign) and 131,328 families.
     - S2 serves about 500 families, from 256 to 131,328.
     - "Identical to a call C2 already proved exact" therefore holds for the arithmetic by construction, not by evidence. K1 below widens the card evidence, and G4 is the first model-level evidence.
2. **Mask refresh v2: a new device kernel and module.**
   - `attention_mask_replay.py` and `.cpp` are pinned: they are hashed by `attention_mask_replay.py:8-10` and listed in `target_t16_attention_gate.SOURCES` (`target_t16_attention_gate.py:8-10`).
   - The new kernel computes E on the device from the positions word.
   - It writes a narrow 256-column tail relative to E−256, plus the `cur_pos` word.
   - The narrow mask equals the wide one on card M (README:156). It frees about 0.2 GB per chip (e), against the 0.887 GB free today.
3. **Boundary cap.**
   - A ticket with `start mod 256 > 240` crosses E, so accept at most `E − start` tokens.
   - Rows past E cannot see their own keys. They are handled like rejected drafts: their K/V are overwritten and their GDN state is discarded.
   - This touches about 6% of a user's rounds and about 1% of tokens (e).
   - No 512-column mask is needed; N3 refuses those anyway (README:164).
4. **Idle segments** get `cur_pos = UINT32_MAX`, so the SDPA skips them. That saves about 10.75 ms per idle segment at 131k (d: 43 ms of SDPA over 4 users, `JOB/next/next-cuts.md:32`).
5. **Extent readers, re-implemented rather than subclassed.**
   - `attention_replay.py` is pinned (`target_t16_attention_gate.py:8`; `probe_frozen_pins.py:3-9`).
   - Its `validate` → `validate_ticket` runs on every `stage` and `refresh` (`attention_replay.py:62-67,69-72,91-92`).
   - The pooled reader validates per segment (`pooled_attention_replay.py:392-394,455-458`).
   - The new readers reimplement bundling and mask setup in three forms (single, pooled, packed), over one full-width table set.
   - The `cur_pos` words are allocated at attach, before any trace (construction-order rule).
   - The family checks become capacity checks at:
     - `packed_verifier.py:533-538,1040`;
     - `serving_packed_step.py:229-235,607`;
     - `model_batch.py:65-80`;
     - the pool's per-family tables (`serving_buffer_pool.py:197-226` (c)).
   - Tickets with `start < 128` stay sequential (`attention_head_fold.py:32-48` (c)).
   - A new admission module pins the new sources, the K64j `.so` markers and the card-B evidence.
6. **D1, survivor narrowing** (moved from S1).
   - Narrow a lone survivor's 16-row ticket to its engine's widest capture, instead of `refuse_round` (`serving_packed_step.py:563-565,845-866`).
   - Add `GreedySession.narrow` in `speculative-decoding/harness/greedy_session.py`. This file ships by the bundle route (§2.2 item 6).
   - The test at `test_serving_packed_step.py:1038-1073` flips.
7. **Image.**
   - A new opgraft replaces K64i (`docker/qwen-c2-serving.Dockerfile:13-27`).
   - `QWEN_FAST_RUNTIME_BINARY_SHA256` (`:79`) and the `_ttnncpp.so` sha test (`:26`) change with it.
   - A new kernel-cache key, warmed by one gate run (SKILL:142-145). A cold graft cache doubled time to first token at v122 (T32 four-streams:516).
   - Diff the `QWEN_` strings of the `.so` (MEM graft-so-drops-image-patches).
8. **Also in S2:**
   - **A load generator driven by metering traces:**
     - contexts that grow across turns;
     - tool-time gaps;
     - a short/long output mix taken from metering;
     - streaming, tools and cancels, with sampled solo replays.
     - **Pass criteria:** p50 and p95 turn latency and time to first token no worse than `general` on the same trace, with zero engine deaths.
   - **An agentic quality gate** that replaces the single-function V8:
     - multi-turn tool-loop tasks with thinking on, through the product's own agent harness;
     - `c2` (greedy) against `general` at the clients' default sampling (`request_contract: false`, `qwen_c2_profiles.json:153`);
     - plus the loop rate.
     - Today's harness runs thinking off (T32 `scripts/ci/coding_request.py:17-21`) on single functions (T32 `coding_functional_eval.py:40-55`).

**Fallback if K64j fails its byte gates:** "P1" (B3 §4).
- Every segment runs at maximum capacity, with a full causal mask and tail mode off.
- It needs no kernel change, provided share and slice work without tail mode (**UNVERIFIED**).
- The round is about 181 ms at any length, about 25 tok/s per user (e).
- It is not bitwise-equal to a solo run, so it needs the user to accept a "deterministic" definition of exact (decision D-b).
- It costs about 8-13 days in place of the kernel items.

**Engineer-days: 25-45**

| Item | Days |
|---|---|
| Card-B probe: runtime vs compile-time split, `UINT32_MAX` skip, dynamic chunk | 1.5-2.5 |
| K64j: factory, reader, compute and writer; stage-3 and slice variants; generators, source tests and build; flag-bit move | 5-9 |
| Mask refresh v2: a new device kernel, generator and sim test | 2.5-4 |
| Extent readers, re-implemented; narrow masks; full-width tables; `cur_pos` words | 3.5-5.5 |
| Packed block and step: capacity checks, boundary cap, idle skip | 1.5-3 |
| D1 survivor narrowing | 1.5-3 |
| Admission module, evidence pins, copy lists, CPU allowlist | 0.5-1.5 |
| TT-Sim lanes: extent@E == legacy@E at 2304, 8448 and one mid family | 1-2 |
| Card-B harness and analysis | 1.5-2.5 |
| Exactness classification for cross-path flips | 0.5-1 |
| Trace-driven load generator with pass criteria | 3-5 |
| Agentic quality gate, with the loop rate | 3-6 |

**Hardware iterations**

- **Card B (or card M): 6-12 runs.** Assume one fix loop per gate.
  - **P0 probe.**
    - The existing causal `cur_pos`-tensor path at `E-1`, against the non-causal call at capacity E, byte-compared for single-row users.
    - Pages poisoned with NaN beyond the extent must not change the output.
  - **K1.** The new flag is bitwise-equal to the per-family tail call:
    - capacities 2304, 16896, 33024, 65792, 98560 and 131328, widened from three so the card evidence spans the families G4 will serve;
    - starts +0, +7, +127, +240 and +255;
    - seeds 0-4;
    - combined with share and slice;
    - one trace replayed while the extent changes across more than 50 families.
  - **K2.** Equals native B1 per row before the boundary.
  - **K3 negative controls:**
    - `cur_pos` off by one chunk must change the output;
    - a -inf planted in a non-final chunk must not;
    - skipped users;
    - the N3 unknown-flag refusal on its new value.
  - **K4.** The mask kernel's output equals the host prediction.
  - **Timing.**
- **TT-Sim: 2-4 runs**, local, several hours each.
- **M+A: 13-20 runs.**

  | Run | Count | Purpose |
  |---|---|---|
  | Kernel-cache warm | 1 | Warms the new graft's cache |
  | G3 control, paired ABAB | 2 | Flag off vs on at 4 × 131,072. Byte-identical to v235, and the round no slower. |
  | G4 mixed lengths, packed | 3-4 | Spread across families, e.g. {1.5k, 20k, 60k, 120k} and {5k, 40k, 90k, 128k}. Concurrent equals solo under the policy. |
  | G4 long answers | 1-2 | N ≈ 4-8k, crossing 16-32 boundaries per user. Cap events are counted. |
  | G4 staggered arrivals | 1 | |
  | G5 memory ledger | (in the above) | |
  | G7 replay | 1 | |
  | G8 load test | 1 | 1-2 h on the metering-driven trace, against the pass criteria |
  | Quality gate | 1-2 | `c2` vs `general`, with the loop rate |
  | Contingency | 2-6 | |

**"Concurrent equals solo" compares two arithmetic paths.**
- **Solo** is the trimmed 4-row engine: widths 1, 2 and 4, no replay (`packed_shapes.py:79-96`, `model_batch.py:442`).
- **Concurrent** is the K64j non-causal tail.
- **Agreement so far is empirical and short:**
  - v177 matched v158 byte for byte over 4 × 256 tokens;
  - C2 matched v160 (T32 `docs/real-text-2026-09-24.md:130,249` (c)).
- **Card-level E4** (G8 vs G4 unfolded) is "recorded, not failed on" (README:161).
- **So a flip on answers of thousands of tokens is possible without any bug.**
  - Under the policy, a reproducible flip between these two paths is NOT COMPARABLE, not a failure.
  - It is closed by the card-B bitwise gates (K1-K4) and by a bounded investigation:
    - where the first divergence falls;
    - whether it is a near-tie. Logging logit margins is **UNVERIFIED** as available.
  - A flip between two arms on the **same** path fails.

**What users get** (steady decode). Every S1 request shape, with 2-4 live users packed at any position ≥128. All projections assume the 4.54 tokens per packed round measured on real text at 131k (B3 §1); acceptance on coding text at other lengths is unmeasured (R8). For the end-to-end view, see §3.

| Live users | Context | Round | Per-user decode |
|---|---|---|---|
| 4 | 4 × 131k | about 166 ms | 27.3-28.6 tok/s (m, gate container; unchanged from C2 by construction) |
| 4 | 4 × 32k | about 131-134 ms | about 32-35 tok/s (e). Basis: C2 less the 131k-to-32k gap in the measured trace, 169.5 vs 134.1 ms (T32 four-streams:510,515). |
| 4 | mixed {2k, 16k, 64k, 128k} | | about 32 tok/s (e, B3 §5) |
| 2-3 | 131k | about 146-162 ms, from 167-173 ms today, less 10.75 ms per skipped idle segment | 28-31 tok/s (e) |
| 2-3 | short contexts | | up to about 37 tok/s (e) |
| 1 | any | | 22-33 tok/s, as S1 (`PADDED_MIN_USERS_DEFAULT = 2`, `packed_verifier.py:396`) |

### 2.4 Stage 3: random arrivals (Lever N v1 and v2 in C2, and packing a lone user)

**Deliverable.** A chunked-prefill C2 profile, short prompts admitted during a long prefill, and 1-live packed rounds.

**Code touched**

1. **Lever N v1 in the image.**
   - Bake in the five graft files, which today are mounted only by the gate arm (`lever_n_m3native_run_arm.sh:103-112` (c)):
     - `model.py`;
     - `qwen36_vllm.py`;
     - the plugin's `platform.py`, `scheduler.py` and `lane_scheduler.py`.
   - Pin them to the P8/v235 tree.
   - Profile flags:
     - `--enable-chunked-prefill`;
     - `--long-prefill-token-threshold 2048`;
     - `--max-num-batched-tokens 4096`;
     - `TT_M1_FORCE_CHUNKED_PREFILL=1` (`lever_n_m3native_gate.py:1698` (c)).
   - Add `long-prefill-token-threshold` to `OWNED_FLAGS`; `serving_c2_contract.py:37-46` lacks it.
2. **Lever N v2, "scratch parking"** (LEVN:125-130), moved here from S4.
   - Without it, a new arrival waits behind any long prefill (LEVN:173). For agent traffic, that is the common case.
   - Add shortest-prompt-first ordering among waiting prefills.
   - Gate: LEVN gate 1 with a short prompt admitted mid-prefill (LEVN:195-197).
3. **1-live packing.**
   - A third idle segment needs its own K/V target. Page 0 holds only two 32-row tile rows (`packed_verifier.py:985-992`), so reserve a KV scratch block in the plugin.
   - Set `QWEN_FAST_PADDED_BLOCK_MIN_USERS=1` (`packed_verifier.py:394-396`).
   - With S2's idle skip, a 1-live padded round is about 120-135 ms at 131k with 4.54 tokens (e). The sequential step is 122-127 ms with 2.76-4.0 tokens.
   - This also makes D1 a last resort rather than the normal 2→1 path.
4. **Optional: A7, mixed narrow rounds.** Today one user within 16 tokens of `max_tokens` sends the whole round sequential (`serving_packed_step.py:229-230`).

**Engineer-days: 8-14** (A7 adds 1-2)

| Item | Days |
|---|---|
| Lever N v1 in C2 | 2-4 |
| Lever N v2 scratch parking | 2-4 |
| Shortest-prompt-first | 0.5-1 |
| 1-live packing | 3-5 |
| A7 (optional) | 1-2 |

**M+A: 5-10 runs**
- Lever N exact at 4 × 32k and 4 × 131k under the C2 flags. The run must include a decoder that finishes mid-prefill (the v121 slot-move trigger) and the v119-style negative control.
- A short prompt admitted mid-prefill (v2).
- A 1-live padded run exact against solo, plus paired timing.
- G7 and G8 re-run.
- Contingency.

**What users get**
- One live user packs at about 30-36 tok/s (e).
- **During another user's prefill, decoders keep producing tokens instead of freezing.**
  - On the older fast path they ran at about 3 tok/s during a 131k prefill (v128), with a worst stall of 4.1-4.5 s instead of 44.8 s (T32 four-streams:529-531,551-552 (m)).
  - It has never run with EARLY_DRAFT, PRESTAGE, PADDED_BLOCK, QUAD_DRAFT or K5.
- A short prompt no longer waits behind a long one (v2).
- The new prompt's time to first token rises about 20%: prefill went from 68.3 to 82.6 s at v128 (T32 four-streams:552).
- A cancel takes effect within one 2048-token chunk (with R3).

### 2.5 Stage 4: hardening for unattended traffic

**Deliverable and code touched**

1. **Q4 rebuilds.**
   - Q4 retires whenever a slot's device changes (T32 `dflash_packed_proposal_coordinator.py:597-603`) and rebuilds at 0.29-0.61 s each (`JOB/next/next-cuts.md:107`).
   - After repeated failures it disables itself for the process, leaving only a marker line (`:605-616`).
   - Prebuild it during prefill, and export "Q4 disabled" as a metric (R5).
   - Another agent's uncommitted Q4-prebuild change sits in the t32 working tree; coordinate with it.
   - This is unnecessary if Stage E lands, because stable slots stop the retirements.
2. **Overnight soak and restart drills.**
   - Hundreds of admissions and releases.
   - The restart sequence.
   - No `llrt.cpp:594` signature (SKILL:185-192).

The diagnostics trim, formerly item 3, moved to R5, and Lever N v2 moved to S3.

**Engineer-days: 2-4. M+A: 2-4 runs**, including one overnight soak window.

### 2.6 Track R: production readiness (new; runs in parallel with S1-S2; gates any production cutover)

**R1. Loop guard.**
- **Why:**
  - The contract coerces sampling to greedy without telling the client (`serving_c2_contract.py:177-186`).
  - A missing `max_tokens` becomes the whole ceiling (`:187-189`).
  - Reasoning is on in serving (SKILL:86).
  - Qwen3's guidance warns against greedy decoding in thinking mode because of endless repetition. Whether that applies to Qwen3.8 is **UNVERIFIED**.
  - One loop holds a slot for up to 16,384 tokens at 6-33 tok/s, which is 8-45 min (d).
- **Fix:**
  - an n-gram repetition stop;
  - a wall-clock cap per request;
  - a default `max_tokens` of about 8k when the client omits it;
  - a response header saying sampling was coerced.
  - Measure the loop rate with thinking on, using `general` at temperature 0 as the proxy.
- **Cost:** 2-4 days. 1-2 runs, which can be on `general`.

**R2. Context-limit errors.**
- **Why:**
  - The refusal is custom text (`serving_c2_contract.py:175-176`). Whether clients' compaction logic recognises it is **UNVERIFIED**.
  - Near the cap, `max_tokens` is clamped silently (`:187`). A reasoning step plus a tool call can then end with `finish_reason=length` in the middle of the XML.
- **Fix:** use vLLM's standard wording and error code, refuse when room < 8k, and cap prompts at about 123k (§2.2, D-c).
- **Cost:** 0.5-1 day, CPU only.

**R3. Cancellation.**
- **Why:**
  - `cancelled=lambda: False` (`serving_startup.py:142`).
  - A whole-prompt prefill cannot be interrupted.
  - `_sample` builds the 3.4-4.1 s engine after the prefill sample without checking for an abort (`serving_lifecycle.py:450`). Whether vLLM can deliver the abort before `_sample` in the same step is **UNVERIFIED**.
- **Fix:**
  - an abort registry shared between scheduler and lifecycle, on the `PREFILL_GATE_KEY` pattern (`serving_lifecycle.py:8-25`);
  - skip the build for an aborted request;
  - "cancel during prefill" and "cancel during build" in the S1 lifecycle gate.
- **Cost:** 1-2 days, about 0.5 run.

**R4. Streaming and tool parsing with multi-token deltas.**
- **Why:**
  - Every tool-call test is non-streamed and single-turn (`c2_serving_smoke.py:109-112`, `c2_platform_replay.py:169-173`).
  - C2 emits about 3-16 tokens per round, so the `qwen3` reasoning parser and `qwen3_xml` tool parser (SKILL:56) see multi-token deltas in production for the first time. How they behave under vLLM 0.25.1 is **UNVERIFIED**.
- **Fix:**
  - CPU tests that feed recorded 1-16-token deltas into the pinned parsers and require the same result as non-streamed parsing;
  - replay steps: a streamed tool call, parallel tool calls, a tool-result follow-up, reasoning followed by a tool call, and `stream_options.include_usage`, on `c2` against `general` at temperature 0.
- **Cost:** 1.5-3 days.

**R5. Observability, including the diagnostics trim formerly in S4.**
- **Why:**
  - The agent's containers log nowhere, and the engine's log dies with its tmpfs (SKILL:203-205).
  - Meanwhile the image turns on `PHASE_LOG`, `PACKED_AUDIT`, `CARRY_LOG`, `SEQ_PUBLISH_LOG`, `PHASE_TIMING` and `FAULTHANDLER` (`docker/qwen-c2-serving.Dockerfile:70-81`) in an 8-CPU container.
- **Fix:**
  - a rotated log ring on `/models/.qwen-c2/logs`, the only persistent mount (SKILL:230-231), audited for prompt content first;
  - worker counters for: rounds by kind (packed, sequential, padded), acceptance, build time, stall time, Q4 disablement, boundary caps, refusals by reason, and engine deaths;
  - alerts on restarts;
  - the diagnostic flags off after a paired ABAB, keeping `MEMORY_LEDGER`.
- **Cost:** 3-6 days, 1-2 runs.

**R6. Known-answer canary.**
- **Why:**
  - Buffers allocated while a trace exists can be overwritten by its replay (`serving_buffer_pool.py:3-19` (c)).
  - Arbitrary lengths and churn add allocation paths.
  - The exactness gates run only before a deploy.
- **Fix:**
  - a fixed prompt, greedy, hash-compared every N minutes and after every deploy or restart;
  - on a mismatch, alert and fall back to `general`.
- **Cost:** 2-3 days.

**R7. Deploy, drain, heal and cutover.**
- **Why:**
  - The profile defaults to `general` (`qwen_c2_profiles.json:2`).
  - The agent forwards only allow-listed environment variables (SKILL:47-48).
  - ADR-0106 is not built (SKILL:31-35).
  - So switching `general` ↔ `c2`, including a rollback, needs a new base or layer: a release of about 90 minutes (SKILL:20).
  - Deploys do not drain first (`release_first`, SKILL:18,172).
  - A wedge needs a CI reset plus a fabric re-measure (SKILL:201-202).
  - M+A are the only linked pair (SKILL:39-40), so a "canary" is a cutover. With TT serving currently down, the first `c2` placement is a cutover from nothing on TT.
- **Fix:**
  - `QWEN_C2_PROFILE` on the agent allow-list, which is a Thatch.Server drop-in PR (SKILL:234-236), or one base per profile;
  - drain: 503 with `Retry-After`, then wait for in-flight requests up to a limit;
  - automated heal (reset, `tt-measure-links`, reload), or at least an alert and a runbook, with time-to-recover measured;
  - shadow replay of sampled real requests on M+A before the cutover, replacing revision 1's G10 canary.
- **Cost:** 4-8 days across repos. 2-3 restart and wedge drills.

**R8. Edge admission control** (mostly Thatch.Server).
- **Why:**
  - `max-num-seqs` is 4 (`qwen_c2_profiles.json:24`); whether vLLM's waiting queue is unbounded is **UNVERIFIED**.
  - The fourth user's first token measured:
    - 56-59 s at 4 × 32k (T32 four-streams:510,513);
    - 249-343 s at 4 × 131k (`:515-517`).
  - Whether any stream bytes go out before the first token is **UNVERIFIED**.
  - The health monitor's probe and timeout are **UNVERIFIED** (SKILL:186-187 names only "a health-monitor recovery"). A probe that sends a chat completion would restart the engine in the middle of a burst.
- **Fix:**
  - a queue cap with 429 and `Retry-After`;
  - a maximum queue wait;
  - per-tenant concurrency caps;
  - SSE keepalive comments while a request is queued or prefilling;
  - check the monitor's probe.
- **Cost:** 3-6 days.

**R totals: 17-33 engineer-days; M+A: 4-8 runs.**

### 2.7 Optional stages for agent traffic (decide on metering, D-f)

**E. Engines prebuilt per slot.**
- **Why:**
  - The verifier and drafter are built per request, inside `_sample` on the engine thread (`serving_lifecycle.py:450` → `serving_request_factory.py:140-200`).
  - The build is 959.92 ms of feature setup plus 2,689.68 ms of verifier setup at 4K (T32 `docs/combined-prefix-cache.md:35-36`), and 3.4-4.1 s at 32k (T32 four-streams:529-531).
  - It stalls every other decoder.
  - Agent turns are short, so arrivals are frequent.
- **Design:**
  - capture once per slot at attach, and point the capture at each new request;
  - K64j's runtime-position word is the likely enabler.
- **Open questions, added to card-B P0:**
  - whether that word is enough;
  - whether the drafter's proposal traces can be repointed the same way, given buckets per position (`dflash_proposal_inputs.py:7-13`).
  - Both are **UNVERIFIED**.
- **Also:** stable slots stop the Q4 and pair-trace retirements (S4 item 1).
- **Cost (e, reviewer):** 8-15 days, 3-6 M+A runs, after K64j.

**P. Conversation prefix reuse per slot.**
- **Why:**
  - No profile has a prefix cache (`qwen_c2_profiles.json:28,83,138`).
  - The fast path refuses one (`serving_fast_policy.py:109-112`).
  - The TT model declares `supports_prefix_caching: False` (T32 `docs/gotchas.md:157`).
  - So every agent turn re-prefills its whole context.
- **Evidence that it can work:**
  - A hybrid-state prefix cache passed a 4K offline screen: run 35198371713, cached prefill 683.94 ms against 1,239.17 ms cold, a 1.81x speed-up (T32 `docs/combined-prefix-cache.md:3-21`).
  - Its remaining gates are longer contexts, device miss/evict/cleanup checks, a split of the restore cost, and serving ownership (`:204-211`). "Serving remains disabled."
- **Design:**
  - each slot keeps its last prompt's KV pages leased after the request finishes;
  - keep a GDN checkpoint and the DSpark feature history at the last 2048-token boundary;
  - match by token prefix, and evict when a fifth conversation arrives;
  - report the hit to the scheduler as computed tokens.
- **Whether the checkpoints fit** is **UNVERIFIED**: 0.887 GB is free per chip. One candidate home is the 4 unused native GDN slots (`NATIVE_GDN_SLOTS = 8`, `serving_fast_policy.py:49`, against `max-num-seqs` 4); that is also **UNVERIFIED**.
- **Cost (e, reviewer):** 15-30 days, 5-10 M+A runs.

**Only if chosen (D-e):**
- **A stock-decode lane for short outputs** (`max_tokens` ≤ ~64: autocomplete, probes).
  - Mixed steps are refused (`serving_worker_hook.py:334`), so this is new routing.
  - 5-10 days.
- **Structured output** by grammar-masked argmax in greedy verify.
  - The lifecycle refuses structured requests at `serving_lifecycle.py:243,430`.
  - 6-12 days, 2-4 runs.

### 2.8 Totals and the critical path

| | Eng-days | Card B | Sim | M+A planned |
|---|---|---|---|---|
| S1 | 13-25 | 0 | 0 | 8-14 |
| S2 | 25-45 | 6-12 | 2-4 | 13-20 |
| R | 17-33 | 0 | 0 | 4-8 |
| S3 | 8-14 | 0 | 0 | 5-10 |
| S4 | 2-4 | 0 | 0 | 2-4 |
| **Core total** | **65-121** | **6-12** | **2-4** | **32-56** (up to ~86 at a 35% failure rate: 79 of 225 m3native runs failed, B5 §1.2 (c)) |
| E (optional) | 8-15 | ? | 0 | 3-6 |
| P (optional) | 15-30 | 0 | 0 | 5-10 |

**Device time**
- About 16-56 h of M+A runs at 30-60 min each, up to about 86 h with failures.
- Plus about 10 h of load testing and soak.
- Back to back, that is about **2-8 rig-days**, spread over the programme.

**Critical path to S2**
1. The K64j probe (card B or M; can start now).
2. K64j with the new flag bit, and its card-B byte gates.
3. The extent readers and the mask-refresh v2 kernel.
4. Packed integration and D1.
5. The new base image and the kernel-cache warm.
6. The S2 M+A ladder.

S1's code and its first bring-up in the image must land before step 6. R must be done before any production cutover.

**Calendar (e)**
- **One engineer:** 65-121 days, about 13-24 weeks.
- **Two engineers:** about 9-16 weeks.
  - The kernel engineer takes K64j, the mask kernel, the readers, and then E or S3's 1-live packing: about 15-26 days of S2.
  - The serving engineer takes S1, S2's serving items and R: about 40-77 days.
- **Agent-paced** (low confidence, see §1):
  - S1 bring-up 1-3 weeks;
  - S2 4-8 weeks;
  - S3 and S4 2-4 weeks after that.

---

## 3. The first stage that beats `general` for real coding requests

**Nothing in the TT stack is serving right now** (run 36191246394), so the bar is `general` as measured, not a live deployment.

**On steady decode, Stage 2 is the first stage that beats `general` at every concurrency:**

| Live users | `general` | Stage 2 |
|---|---|---|
| 1 | 15.3-19.3 tok/s (m) | 22-33 (d/e) |
| 2-3 | about 14 (e) | 27-37 (e) |
| 4 | 12-13 (m) | 27-35 (e) |

- Context goes to about 123k, against 65,536.
- The (m) rates for C2 come from the 16-CPU gate container. S1's bring-up measures them in the 8-CPU agent shape (R9).

**End to end for coding agents, Stage 2 alone is roughly level with `general` (e).**
- Neither path has a prefix cache (§2.7 P), so every turn re-prefills.
- The fast path's prefill plus build runs at about 2.0-2.3k tok/s: 14-16 s at 32k (T32 four-streams:510,513).
- `general` prefills at about 3.2k tok/s: 6.8 s at 22k (SKILL:84).
- Each arrival also stalls all other decoders for about 4 s (§2.2), and whole-prompt prefill freezes them until S3.

**Worked example 1:** a 40k context, an 800-token answer, one user.

| Stage | Prefill | Decode | Total |
|---|---|---|---|
| `general` | about 12.5 s | about 41-52 s | about 54-65 s |
| S2 | about 16.5-19 s | about 24-36 s | about 41-55 s |

**Worked example 2:** the traffic review's model (e).
- The workload: four agents, 30k contexts, 300-token tool-call turns, 5 s of tool time, prefill serialised on both paths.

| Profile | Exclusive device time per turn | Decode of 300 tokens | One agent turn every |
|---|---|---|---|
| `general` | ~10 s | ~22 s at 12.5-15 tok/s | ~62 s |
| Stage 2 | ~14.5 s (prefill plus build) | ~10 s | ~70 s, including rebuilds when users change |
| Stage 2 + E + P | ~1.5 s (a 3k-token suffix) | ~10 s | ~16-21 s |

- At 1,000-token turns, Stage 2 alone wins by about 18%.
- The third row's basis is the 4K prefix-cache speed-up (T32 `docs/combined-prefix-cache.md:9-21`), extrapolated.

**How soon**
- Stage 2 is about 4-8 weeks agent-paced (e), and must be paired with R before any cutover.
- A clear end-to-end win for agent traffic needs E and P as well: roughly another 23-45 days (e).
- Which comes first, S3 or E+P, should follow metering (D-f).

**Stage 1 wins in a narrow case:** a lone user with an answer over about 180-840 tokens at a 30k context (d, §2.2), and on context length. It is not a release candidate.

---

## 4. Hardware access

1. **M+A are development cards now.**
   - `MEM/goal-200tps-concurrent.md:11`: "LIFTED 2026-09-26 ... two-card runs are allowed".
   - Commit aaf35537 and run 36191246394 (`unserve`, success). Its log shows no serving container and no holder of M or A at 21:23Z.
   - Before the first gate, confirm through the admin API that the placement is removed (`qwen-c2-serving.yml:101-102`). This needs an admin session, which the user mints (SKILL:169-170).
   - The `MEMORY.md` index line "PAUSE: no two-card tests" is stale. It was not edited, because this pass is read-only.
2. **Run gates back to back.**
   - Batch several arms per session.
   - Reset between arms only when needed.
   - Re-measure the fabric only before a production placement: the agent gate needs it, the m3native gate does not (SKILL:159-160).
   - Put the paired `general` baseline probe (R14) in the first session.
3. **If TT production has to come back before S2 is ready (D-d):** batched nightly windows again, as in revision 1.
   - Each window takes 30-60 min of fixed overhead (release, reset, arms, reset, fabric re-measure, place with `release_first`, verify).
   - Each needs a user-minted admin session.
   - Serial windows turn each S2 fix loop into a day, so S2 would stretch by about 1.5-3 weeks (e).
4. **Card B is still the kernel lane.**
   - With M free, the K1 card test can also run on card M (`sdpa_decode_qwen/run_card_m.sh`).
   - A switch power-cycle to recover card B also drops card A (B5 §1.2, topology memory (c)). Run card-B work under the watcher.
5. **Longer term:** a second linked pair is the only way to develop and serve on TT at the same time. Card B cannot be cabled into M+A without coupling its resets to a serving card (B5 §2.3 (c)).

---

## 5. Risks and unknowns

| # | Risk or unknown | Why it matters | Experiment that retires it | When |
|---|---|---|---|---|
| R1 | K64j's runtime `cur_pos` does not reproduce the compile-time split. The split also depends on `max_dynamic_chunk_size` (`reader_decode_qwen.cpp:45,178`), whose served value was not checked. `St` still feeds `:82,293,366-375`. The writer's c_8 handling was not read. | S2's exactness rests on it | Card-B P0: the causal `cur_pos` path at E-1 against the non-causal call at capacity E, byte-compared. Then K1 at six capacities. On failure, fall back to P1 (D-b). | Now |
| R2 | The `UINT32_MAX` skip exists only in the causal branch (`:128-168`). Compute and writer handling are unread. | Idle-segment savings; possible hangs | Card-B K3 with skipped users, watcher on | Now |
| R3 | Short-prompt paths were never exercised. History under 2048 (the pair and quad folds need 2048, `quad_draft.py:88-89` (c)). Up to four proposal buckets per request (`dflash_proposal_inputs.py:11-13`), with unmeasured DRAM and build time. Prefill buckets under 256 and 32. Prompts that are not a multiple of 64. Prompts under 128. | Engine death, divergence or out-of-memory on small requests | CPU tests on the staged tree, then S1 G4's length ladder, with bucket count, DRAM and build time recorded | S1 |
| R4 | vLLM 0.25.1 behaviour: a speculative request returning zero tokens; stop strings implemented as an engine abort; the size of the waiting queue; whether bytes stream before the first token | D1, D4, R8 | A CPU probe of the pinned vLLM source | Now |
| R5 | Lever N has never run with the C2 flags; v121 crashed on a slot move (T32 four-streams:532-535). v2 is new code. | S3 correctness | The S3 gates, including a decoder finishing mid-prefill and a short prompt admitted mid-prefill | S3 |
| R6 | Trace and allocator stability over hundreds of admissions (`serving_buffer_pool.py:3-19` (c)) | Silent corruption, or engine death under load | S2 G8 with ledger lines; S4 soak; R6's canary in production | S2, S4, R |
| R7 | DRAM at the maximum shape: 0.887 GB free per chip at v238; proposal buckets; P's checkpoints | New words, tables and caches must fit | S1 G5; K64j's narrow masks free about 0.2 GB per chip (e) | S1, S2 |
| R8 | Acceptance on real coding text at short and mid contexts is unmeasured; 4.54 tokens per round is 131k real text only | Every rate projection scales with it | Acceptance per step recorded in S1 G4 and S2 G4 on coding prompts | S1 |
| R9 | The agent container has 8 CPUs against the gate's 16 (SKILL:234); K5's host tax is 2-5 ms per round (`JOB/next/next-cuts.md:71`); CI load skews rounds (MEM control-plane) | Rates below the gate's | S1 bring-up in the agent shape; the shelved CPU-quota ABAB (MEM goal:14-15) | S1 |
| R10 | Prefill and the per-request build dominate agent turns | End-to-end gain much smaller than the decode gain (§3) | G8 on a metering-driven trace; then E and P (§2.7). Evidence exists for P (T32 `docs/combined-prefix-cache.md`). | S2 |
| R11 | Qualification policy. The staged tree records `requested_geometry_qualified=False` (`frozen_ladder_stage.py:21-25`), and S1 removes the per-request T16 qualification (§2.0 item 8). Do card-B byte gates plus real-text concurrent == solo replace per-geometry full-model audits? | If audits are required, add 3-6 M+A runs and hours of sim time | The user's decision (D-b), **before S1** | Before S1 |
| R12 | Image provenance. A graft `.so` drops image patches. There are three delivery routes (both copy lists, and the bundle's `COPY bundle/speculative-decoding` at `docker/qwen-fast-serving.Dockerfile:8`). The C2 overlay is at `qwen-c2-serving.yml:205-208`. The runtime binary sha is pinned (`Dockerfile:26,79`). Three frozen files differ between bundle and HEAD. | A run that looks healthy but is not what was intended | G1: the `QWEN_` string diff, a closure test covering all three routes, and reading the argv line (SKILL:62-63) | Every image |
| R13 | Restart time on the fast-path profile, and the in-place restart wedge (SKILL:185-192). Healing needs a person (SKILL:201-202). | Outage length on every death | S1 G7 records the restart time; R7 drills | S1, R |
| R14 | The baseline: 18.57-19.26 tok/s after the CPU-priority change (run 36106529648), 15.3 before under CI load | Stage-ship decisions depend on it | A paired `c2_decode_probe.py` on `general` in the first M+A session | Now |
| R15 | Structured output and `tool_choice` required are refused (`serving_c2_contract.py:158-160`) | Clients that use them break | Metering sample (§2.1 item 3) | Before cutover |
| R16 | The boundary cap costs more than estimated | Rate | Count capped rounds in S2 G4 | S2 |
| R17 | How often a single live user is the state in real traffic | Decides S3's 1-live packing and E/P priority | Live-user histogram from metering | Now |
| R18 | Greedy decoding with thinking on loops (Qwen3 guidance; **UNVERIFIED** for Qwen3.8) | Slots held 8-45 min; four loops take the server | R1's loop rate in the agentic quality gate | S2, R |
| R19 | Near-tie flips between the solo and concurrent arithmetic paths on long answers | Every flip becomes an investigation loop | The exactness policy (§2.2 item 5); K1-K4 bitwise gates | S1, S2 |
| R20 | The `Qwen/Qwen3.8-27B` placement still exists and reloads onto M+A mid-gate | A lost run, or a wedge | An admin API check (§4 item 1) | Now |
| R21 | The reasoning and tool parsers under multi-token deltas | Broken tool calls in streams | R4's CPU tests and replay steps | R |
| R22 | The admission build (3.4-4.1 s) stalls every live decoder on each arrival | Stalls under frequent short turns | Measured in G8; retired by E | S2, E |
| R23 | Wrong tokens in production go undetected | Silent quality loss | R6's known-answer canary | R |

---

## 6. Decisions needed from the user

- **D-a:** is S1 internal, as recommended, or a non-default profile? And does TT serve `general` again in the meantime? That is the D-d question.
- **D-b:** confirm what "exact" means for arbitrary lengths. The proposal:
  - real-text concurrent == solo on the same image under the flip policy (§2.2 item 5);
  - plus card-B byte gates at six capacities, standing in for per-geometry audits;
  - decided before S1, because S1 already leaves the frozen geometry.
  - If the answer is "bitwise vs solo" and K64j fails its gates, P1 is not acceptable, and the fallback becomes a K64j redesign.
- **D-c:** the output ceiling and prompt cap. The proposal:
  - ceiling 16,384;
  - a default `max_tokens` of about 8k when the client omits it;
  - a prompt cap of 123,136 with at least 8k of answer room, which needs the contract change in §2.2.
- **D-d:** how long do M+A stay development cards? If TT production must return before S2, the programme goes back to nightly windows and S2 stretches by about 1.5-3 weeks (e).
- **D-e:** what the coding clients send:
  - `stop` (safe only after D1, in S2);
  - `ignore_eos`;
  - `tool_choice` required or named;
  - `response_format`;
  - autocomplete/FIM.
  - The last three decide the optional structured-output and short-output work (§2.7).
- **D-f:** after S2, which comes first: E and P (end-to-end agent latency), or S3 (fairness during long prefills)? Decide on metering: the context sizes, turn lengths and live-user histogram.
- **D-g:** is greedy-only acceptable with thinking on, given R18? And are R1's guard settings (n-gram stop, wall-clock cap, default `max_tokens`) acceptable?

---

## Appendix: review points rejected or only partly applied

Every other point in both reviews was verified and applied above.

1. **Feasibility review 1, "roughly 100+ M+A runs in three days".** Partly applied.
   - The tags support the pace: 199 `lever-n-m3native` tags, about 115 of them from v117 (09-23 19:25) to v231 (09-25 13:46).
   - But tags are not full runs: v157 to v186 is 29 tags in 74 minutes.
   - Used only as a pace indicator, not as a run count.
2. **Feasibility review 2, "every (m) rate in the plan was measured in the 16-CPU gate container".** Partly applied.
   - True for C2's rates.
   - `general`'s were measured in the agent's 8-CPU shape (SKILL:82-85; run 36106529648).
3. **Feasibility review 3, "the chunk split depends only on `cur_pos`".** Partly applied.
   - The split also takes the compile-time `max_dynamic_chunk_size` (`reader_decode_qwen.cpp:45,178`), not checked.
   - `St` still appears at `:82,293,366-375`.
   - R1 is kept, at lower weight.
4. **Feasibility review 4, "a build script with about 15 verification steps (README:188-219)".** Not verified.
   - Those lines describe the K1 probe, not the build script.
   - The cost increase is applied anyway, on the verified pins and file lists.
5. **Feasibility review 5, "that engine uses stock causal SDPA (`model_batch.py:442`)".** Partly applied.
   - `:442` shows only that rows below 8 do not replay. Which SDPA the non-replay path calls is not shown there.
   - The substance, two arithmetic paths needing a flip policy, is applied.
6. **Feasibility review 6, "`refuse_round` does not kill the engine".** Applied with a nuance.
   - After `refuse_round`, the next `prepared_ticket` historically raised and killed the engine (`test_serving_packed_step.py:1049-1051`, run 35564623068).
   - `discard_stale_ticket` (`serving_worker_hook.py:55-85,478`) now stops that round forming.
7. **Feasibility review 8, "the build was measured only at 32k and 131k (four-streams:523-525)".** Corrected.
   - The figures are at `:529-531` (32k, v118) and at 4K (T32 `docs/combined-prefix-cache.md:35-36`).
   - The substance is applied.
8. **Feasibility review 10, "`verifier_engine.py:167-169` calls `frozen_combined_runtime.qualify` ... nothing else on the serving path calls it".** Partly applied.
   - The direct call is `target_t16_attention_gate.qualify`, which staging rewrites to `frozen_combined_runtime.qualify_target` (`frozen_combined_adapters.py:66-70`).
   - Staging also rewires `dspark_8k_build.py` and `dspark_8k_admission.py` to the frozen `qualify` (`:52-56,113`).
   - Attach runs other qualifications (`dflash_combined_request.py:57-59`).
   - "Nothing else" is **UNVERIFIED**. The recommendation, an attach-time source check in S1, is applied.
9. **Feasibility review 13, "T32's real-text doc :127 has 2.76".** Partly applied.
   - The figure is at `:130`, and it is the all-rounds mean at 4 × 131k including pair users (`:209`), not a lone user's rate.
   - It is used only as the floor of the widened 2.76-4.0 range.
10. **Feasibility review, revised S2 calendar "4-8 weeks agent-paced".** Adopted as an estimate.
    - Its basis is not given, and this plan could not derive one either.
    - Marked low confidence.
11. **Traffic review 2, "Stage 2 barely moves agent turns", and the Stage P cost of 15-30 days.** Applied as (e) only.
    - The model and the costs are the reviewer's and were not re-derived.
    - The underlying facts were verified: no prefix cache, prefill rates, the 4K cache result.
    - Stage P is optional, pending metering (D-f), and kept out of the core total.
12. **Traffic review 3, Stage E.** Applied as optional. Its feasibility (repointing verifier and drafter captures per request) is **UNVERIFIED**.
13. **Traffic review 5, "the waiting queue is vLLM's unbounded default" and "no stream bytes before the first token".** Not verified in vLLM 0.25.1; moved into R4.
    - The Thatch-side controls are applied regardless.
14. **Traffic review 9, "nothing counts packed, sequential or padded rounds ...".** Partly applied.
    - The image does log these as lines (e.g. the padded-round lines, `packed_verifier.py:397-400`), but the logs are lost (SKILL:203-205).
    - No metrics endpoint was found, though the search was not exhaustive.
15. **Traffic review 11, "Replace the canary with shadow replay".** Applied, with a correction.
    - TT serving is currently down (run 36191246394), so the first `c2` placement is a cutover from nothing on TT, not from `general`.
16. **Traffic review 14, "break-even 180-380 output tokens".** Replaced by this plan's own derivation of 180-840 tokens at a 30k context.
    - It uses the widened tokens-per-step range and adds the prefill penalty.
17. **Traffic review, suggested order "Stage 2 (K64j) and Stage E, which share the runtime-position work".** Partly applied.
    - That E can reuse K64j's position word is **UNVERIFIED**, so E stays after S2, as an option.
    - Its open questions are added to card-B P0.
18. **Traffic review 12, "the quality gate is not agentic".** Applied.
    - The claim that C2 "silently changes model behaviour" relative to `general` stands only for sampling: `general` honours client sampling (`qwen_c2_profiles.json:153`).
    - Whether that changes quality is exactly what the gate measures.
