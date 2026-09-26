# Conversation prefix reuse on Tenstorrent (Qwen3.8-27B, TP2 on M+A): design

Revision 2, 2026-09-26. This is read-only analysis. No repository was edited, nothing was dispatched, and nothing ran on the rig.

Revision 2 applies two skeptic reviews after checking each point against the code:
- the exactness and device-memory review (points **F1-F10**);
- the serving review (points **S1-S11**).

Appendix A maps every point to where it landed, and lists the points rejected in whole or in part, with reasons. Appendix B lists what this revision re-read and re-ran.

Files written outside this document, all throwaway, under `JOB/prefix/`:
- `calc.py`, the prefill-cost fit;
- `ts_engine.py`, a `git show` dump of Thatch.Server's `py/serving/engine.py`;
- `sim_evict.py`, `sim_pool2.py` and `sim_lpm.py`, the serving reviewer's agent-traffic simulation. This revision re-ran all three and reproduced the reviewer's numbers.

Revision 1 merged four lens reports (vLLM, model, fast path, traffic). §5 records where they disagreed.

Written to `docs/tt-prefix-reuse-design-2026-09-26.md` (UTF-8).

## Citation roots

| Short name | What it is |
|---|---|
| `PLAN` | `docs/c2-serve-for-real-plan-2026-09-26.md`, revision 2 plus the metering note (§0) |
| `SKILL` | `the tt-qwen-serving-deploy operator skill (local, not in the repo)` |
| `MEM` | `the operator's session notes (local, not in the repo)` |
| `JOB` | `the operator's local scratch directory (not in the repo)` |
| `IMG` | `JOB/matmul-attr/img/`, the model sources a serving image carries (`models/demos/blackhole/qwen36/tt`). Its `gdn/tp.py`, `layer.py`, `attention/tp.py`, `mlp.py` and `model_config.py` have the sha256 values pinned as the graft's originals (`C2/docker/qwen-c2-graft/source.sha256`, re-checked here). So IMG is the C2 base tree for those five files. `model.py` (md5 e4ba08d9) and `qwen36_vllm.py` (md5 b5230935) are not pinned anywhere, so their provenance is still **UNVERIFIED**; the C2 image does not graft them. |
| `GRAFT` | `C2/docker/qwen-c2-graft/graft/`. These are the bytes the C2 image actually serves for the five grafted files (`C2/docker/qwen-c2-serving.Dockerfile:29-44`). Where GRAFT and IMG differ, GRAFT is what runs. |
| `PLG` | `JOB/risks/vllm-tt-plugin/src/vllm_tt_plugin/`, the TT plugin at bf77cd63 (checked with `git log`), unpatched. C2's `serving_plugin_patch.py` patches only capacity, platform, worker and entrypoints, and all four are no-ops when the fast path is off. |
| `V` | `JOB/risks/vllm-0.25.1/vllm-0.25.1/vllm/`, the vLLM 0.25.1 source |
| `C2` | `D:/Programming/Tenstorrent.Blackhole-Qwen3.8-27B/.worktrees/c2-serving` at aaf35537 |
| `T32` | `D:/Programming/Tenstorrent.Blackhole-Qwen3.8-27B/.worktrees/t32-score-reuse` at 0cfd3a2b |
| `LEVN` | `D:/Programming/Tenstorrent.Blackhole-Qwen3.8-27B/docs/lever-N-prefill-decode-interleave.md`, the working copy on branch `lever-n-prefill-decode-interleave` |
| `TS` | `D:/Programming/Thatch.Server` at `origin/main` 183ae356, read with `git show` |
| `ADM` | TS submodule `services/administration` at 535f2378 (the gateways), read with `git show` |
| `SIM` | `JOB/prefix/sim_evict.py` (+ `sim_pool2.py`, `sim_lpm.py`). It models the design's mechanics, not the hardware: the pool, cap, trim and checkpoint LRU; FCFS admission of at most 4 running; decode frozen during prefill; the `calc.py` prefill fit plus 0.3 s fixed and 0.5 s restore/capture; decode from 20 tok/s at one user down to 12.8 tok/s each at four. Agents send a 740-token answer, then a tool result averaging 2k tokens, then a gap averaging 15 s or 60 s, and compact past 60k. Queue wait counts in TTFT. Three seeds. |
| `(c)` | Carried from a lens report and not re-read here. Mostly the fast-path admission timings from `JOB/base/m3native-gate-35669428795/gate/server.log`. |

Numbers are tagged **(m)** measured, **(d)** derived from measurements, **(e)** estimated, or **(s)** from SIM (a model of the mechanism, not a hardware measurement).

**The prefill cost fit** is `t(N) = N/3369 + 1.419e-9·N²` s (d; `JOB/prefix/calc.py`).
- It is anchored on run 35419117141: 79,368 tokens in 32.5 s (c).
- It predicts 7.2 s at 22k, against the 6.8 s measured on `general` (SKILL:84).

**A "busy agent"** is a coding agent that sends its next turn a mean 15-60 s after the previous answer, as in SIM.

---

## 1. Verdict

### What to build first

**Build `general-prefix` v1**, which is two things:
- vLLM's ordinary attention prefix cache;
- GDN checkpoints owned by the model at 2048-token boundaries.

It is exact by construction. It ships as a new profile of the C2 serving image, in front of the `general` path that serves arbitrary traffic.

**It pays only behind two platform changes.** Without them it behaves worse under load than today:
- **A per-pair session cap** of about 4 busy agents (5 when tool gaps are about 60 s). Each session is pinned to its backend at its first turn.
- **A TT-specific health probe** that does not queue behind real work.

**1. vLLM keeps owning the attention KV pages.**
- vLLM already hashes and ref-counts them and evicts them LRU.
- The TT stack reports only one attention KV spec (`PLG/worker.py:367-374`), so vLLM sees a non-hybrid cache.
  - `has_mamba_layers` is therefore False (`V/v1/kv_cache_interface.py:939-940`).
  - So no Mamba-aligned splitting happens (`V/v1/core/sched/scheduler.py:295-299`).
- The plugin runner already turns a hit into `start_pos = num_computed_tokens` (`PLG/model_runner.py:1063-1083,1784-1797`).
- The scheduler and the model run in one process. At DP=1 the plugin runs with world size 1, so vLLM picks the uniproc executor (`V/config/parallel.py:915-916`).
  - Revision 1 cited `PLG/platform.py:357`. That line is in the lane-folding helper `_collapse_parallel_config_to_single_process`, not the DP=1 path.

**2. Scheduler wrappers make every hit exact and every grant current.**
- **The cap.** Publish, meaning hash into the cache, only prompt blocks below `floor(prompt_len/2048)·2048`.
  - Those blocks are always written inside a full 2048-token prefill chunk.
  - vLLM does not de-duplicate blocks, and it returns the first block it cached for a hash (`V/v1/core/block_pool.py:47-72`).
  - vLLM also caches blocks that decode and the prompt tail wrote. It does so at allocation (`V/v1/core/kv_cache_manager.py:452-462`) and after every output (`V/v1/core/sched/async_scheduler.py:51-74`). Both calls reach `coordinator.cache_blocks`, which the cap wraps.
- **The trim.** Cut each hit down to the largest 2048-multiple that has a saved GDN checkpoint and whose stored token ids match. The model then never writes into shared blocks.
- **Per-step commit (F2, S6).** vLLM calls `get_computed_blocks` on every attempt to admit a waiting request, not once per admission (`V/v1/core/sched/scheduler.py:684-729`). A later budget break (`:833-840`), allocation failure (`:917-924`) or TT decode fallback (`PLG/scheduler.py:140`) can discard the attempt. So grants are staged per `schedule()` call and committed only for requests that the returned output actually schedules.
- **Coupled eviction (F8, S9).** A checkpoint is dropped when vLLM evicts its boundary block (`V/v1/core/block_pool.py:574`). The checkpoint registry therefore stays a subset of the cached KV.
- **Fail-closed tenancy (S4).** A request with no `cache_salt` gets no hit and publishes nothing.

**3. The model saves and restores GDN state at 2048-token boundaries.**
- **What it saves.** At each prefill's last full-chunk boundary, before the tail, the model snapshots the B=1 prefill scratch to host: the fp32 recurrent state and `conv_carry`. It is keyed by vLLM's block hash at that boundary.
- **What a hit does.** The model restores that snapshot in place into the scratch the chunk trace was captured against. The restore must compile no program after the traces are parked (F3). It then runs the same chunk loop from the boundary. The loop is guarded for a hit that leaves no full chunk to run (F1).
- **Why the output is byte-identical to a cold prefill.** From the boundary on, the arithmetic is the same (§2.0.4).
- **Parts already exist:**
  - Lever N M1's resumable loop edits (T32 `scripts/ci/lever_n_model_patch.py:75-121`).
  - The in-place restore helper (`IMG/model.py:1360-1377`). It lacks `conv_carry`, which is required.
  - T32's guarded eager resume loop (T32 `scripts/ci/prefill_prefix_resume.py:52-81`).
  - The 4K hardware screen that proved hybrid restore on the eager path (T32 `docs/combined-prefix-cache.md:7-30`).

**4. No per-slot state and no session id in the engine.**
- A conversation's next turn finds its own prefix by hash. It can land in any of the 4 GDN slots.
- Checkpoints live in host RAM, so they cost no device DRAM.

### Where it stops working (s)

SIM, one pair, `general` pool of 262,144 tokens, three seeds.

| Busy agents / tool gap | Reuse | TTFT p50 / p90 (s) | Turns/h | Without reuse: TTFT p50 (s), turns/h |
|---|---|---|---|---|
| 1 / 15 s | 0.88 | 2.1 / 4.2 | 66 | 13.7, 55 |
| 4 / 15 s | 0.88 | 2.3 / 5.2 | 188 | 21.7, 120 |
| 4 / 60 s | 0.88 | 2.2 / 4.6 | 129 | 16.3, 98 |
| 5 / 15 s | 0.88 | 5.2 / 22.6 | 205 | 30.9, 125 |
| 5 / 60 s | 0.88 | 2.3 / 7.1 | 154 | 19.5, 110 |
| 6 / 15 s | 0.83 | 23.7 / 40.5 | 203 | 62.7, 125 |
| 6 / 60 s | 0.83 | 2.8 / 20.8 | 169 | 28.5, 120 |
| 8 / 15 s (60 s) | 0.29 (0.42) | 112 / 141 (43 / 98) | 144 (153) | 131 (78), 125 |
| 12-16 / 15-60 s | 0.00-0.01 | 212-367 / 250-395 | 123-125 | about the same |

**Why it collapses past 5 busy agents.**
- vLLM admits FCFS from the queue head (`V/v1/core/sched/scheduler.py:644-650`).
- A waiting request holds no blocks.
- Eviction pops the LRU head (`V/v1/core/block_pool.py:556-561`), and a finished request frees its blocks tail first (`V/v1/core/single_type_kv_cache_manager.py:402-410`).
- Once more conversations are live than the pool holds, the next one admitted is always the one whose blocks are oldest. That is the one just evicted: cyclic LRU, 0% reuse.

**What does not fix it:**
- Admitting the longest cached prefix first, or protecting waiting conversations from eviction: 0.48-0.59 reuse at 8 agents, 0.00-0.23 at 12-16 (s).
- Doubling the pool (G2): reuse 0.85-0.88 up to 12 agents, but TTFT stays at 1.5-2.5 minutes, because four decode slots saturate at about 205-211 turns/h (s).

**What fixes it:**
- The session cap (§2.2, platform need 2).
- More decode slots. That is a decode-throughput lever outside this design.

**TT is a bounded, opt-in backend, not a Spark replacement.**
- The Spark peaked at 16 running plus 18 queued (PLAN:69).
- Its SGLang recipe has 16 slots and a pool of about 997k fp8 tokens (TS `py/serving/sglang_engine.py:162-186`). That is why its 91.7% does not carry over.
- TT serving was pulled from production on 2026-09-26 so that M+A could be used for development (MEM `goal-200tps-concurrent.md`: "LIFTED"; C2 aaf35537). G1 reaches production as a new placement.

### Effort for v1

| Piece | Eng-days (e) | M+A runs |
|---|---|---|
| P0: CPU probes and a template/traffic study | 1-2.5 | 0 |
| Engine work (G1) | 15-27 | **9-12 planned**; up to ~19 at the C2 campaign's 35% run-failure rate (PLAN:93) |
| Platform work (TS + ADM), with TT on its own model id | 4.5-9 | counted above (the replay run) |
| Extra if TT shares the Spark's model id (affine, length-aware routing) | +4-8 | — |
| **G1 total** | **21-39** (25-47 on a shared id) | 9-12 |

- No card-B or TT-Sim runs: the model is TP2.
- Agent-paced, about 4-7 weeks (e). The M+A turnaround and platform PRs across two repositories set the pace.

### Expected time to first token (TTFT) on real traffic

| Case (metered shape: 42.2k prompt, about 3.5k new tokens) | Today on `general` | With v1 |
|---|---|---|
| A continuation turn, 4 or fewer busy agents per pair | 13-20 s (s: p50 13.7-21.7 s including queueing) | **about 2-3 s** (e: re-prefill 1.4-1.9 s, restore and capture 0.3-1.0 s, fixed about 0.3 s); s: p50 2.1-2.3 s, p90 4.1-5.2 s |
| 5 busy agents | s: p50 19.5-30.9 s | s: p50 2.3-5.2 s, p90 7-23 s |
| 6 busy agents | s: p50 28.5-62.7 s | s: p50 2.8-24 s, p90 21-41 s. Tool-gap length decides. |
| 8 or more busy agents | minutes | minutes; reuse 0-0.42. Do not admit them: the session cap. |
| A miss (first turn, compaction, or an evicted conversation) | 7-20 s | unchanged |

- The Spark's median is about 2 s (PLAN:69).
- Under the cap, about 88% of prompt tokens are reused (s) if chat-template prefixes are stable. Stability is **UNVERIFIED** (P0b).
- **The input that decides TTFT is agent duty cycle, not the number of live conversations.**
- **At four busy agents on one pair:**
  - prefill device time per round falls from about 60 s to about 10 s (d);
  - turns per hour rise 1.3-1.6x (s);
  - each arrival freezes the other decoders for about 2-3 s instead of about 15 s, because `general` prefill is whole-prompt.

### Then C2

- **C2's version of P (C1)** reuses all of the above. It adds:
  - admission of a hit in the fast-path lifecycle;
  - a rule that the drafter window lies inside the re-prefilled range;
  - support for shared pages, including page-table padding that never points at a shared page (F9).
- **Cost:** 11-22 engineer-days and 5-9 M+A runs.
- **When:** after S2, paired with Stage E. Without E, the 3.0-4.1 s per-request engine build dominates every hit turn: TTFT about 5-6 s instead of about 2.5 s (§2.5).

### P goes first

`general-prefix` precedes S1/S2 in value and runs alongside them in time (§3). It also raises the bar C2 must clear for agent traffic.

### Not recommended

| Option | Why not |
|---|---|
| vLLM's native hybrid (`MambaSpec`, align-mode) prefix cache | 35-70 days (e). It forces about 832-token blocks, which conflict with the 2048-token chunk trace and with every 64-token paged kernel. It needs GDN state indexed by block inside the decode trace. |
| Session parking with the prefix cache off | It re-implements vLLM's cache and uses private scheduler methods, and it still needs the same model work. It is the fallback only if P0a shows the engine cannot start with the prefix cache on. |
| Reusing state that decode produced ("tier B") | Not byte-exact against a cold run. It needs a user decision on the exactness policy (D-P1). |
| Serving more than about 5 busy agents per pair by growing the KV pool | Queueing, not reuse, sets TTFT there (s). |

---

## 2. Increments, in order (each is shippable)

### 2.0 The shared mechanism (built in G1, reused by C1)

#### 2.0.1 Components

**1. `qwen_prefix_registry`**, one module shared through a fixed `sys.modules` key.
- **Precedent:** `PREFILL_GATE_KEY`, `C2/scripts/ci/serving_lifecycle.py:13-25`.
- **Holds:**
  - an LRU keyed `BlockHash -> Checkpoint(pos, rec[48], carry[48], token_ids[0:pos], nbytes, pins)`. It is bounded by bytes: default 8 GiB, knob `QWEN_PREFIX_STORE_GIB`. Pinned entries are never evicted;
  - optionally, a preallocated host slab of N × 154 MB with checkpoints copied into its slots. This gives an exact byte bound and avoids churning about 1.5 MiB tensors through the allocator (S9, e);
  - **staged grants** for the current `schedule()` call, and **committed grants** per request `(req_id, Q, key)`;
  - planned captures per request;
  - the set of blocks cached in the current scheduler step;
  - counters: grants, trim loss (h−Q), registry hit without KV, KV hit without checkpoint, token mismatches, capture failures, restore and capture ms, bytes, pins;
  - `clear()`.

**2. Scheduler graft.** The class that actually runs at DP=1 is `TTScheduler`, an `AsyncScheduler` subclass (`PLG/scheduler.py:7,31`; `PLG/platform.py:1078-1081`).
- **Delivery:** an AST patch of the plugin's `scheduler.py`.
  - Lever N's `patch_scheduler` is the precedent, and it is proven to execute.
  - Do not use a `scheduler_cls` subclass: `platform.py:1081` assigns `scheduler_cls` unconditionally in the non-lane branch.
- **Install point:** `__init__`, when `QWEN_PREFIX_REUSE=1`.
- **At install, refuse to start unless every one of these holds** (F5, F6):
  - `scheduler_config.async_scheduling` is False. The platform forces it off for Qwen (`PLG/platform.py:1037-1050`; `IMG/qwen36_vllm.py:57-61`). Under async scheduling, blocks hashed at allocation in step t are unwritten when step t+1 reads them.
  - Chunked prefill is off after the platform rewrite, and `max_num_batched_tokens ≥ max_model_len` (`PLG/platform.py:67-105`). This excludes Lever N, which turns vLLM chunking on at a 2048-token budget (LEVN:109-115,133-135): under Lever N, `start_pos > 0` also means "continue my own suspended scratch", and a hit restore between two of another request's chunks would destroy that request's scratch.
  - The KV coordinator is `UnitaryKVCacheCoordinator`. `HybridKVCacheCoordinator` calls `manager.cache_blocks` directly and would bypass the cap (`V/v1/core/kv_cache_coordinator.py:602-628`).
- **Wrappers installed on the instance:**
  - **a. `self.kv_cache_manager.get_computed_blocks`.** vLLM calls it on every attempt to admit a waiting request. It returns a 64-aligned hit capped at `num_tokens-1` (`V/v1/core/kv_cache_manager.py:206-246`). The wrapper:
    1. calls the original;
    2. returns no hit if the request has no `cache_salt` (`V/v1/request.py:154`) or the kill switch is set;
    3. takes `Q`, the largest `2048k ≤ h` that meets three conditions:
       - `block_hashes[Q/64-1]` has a registry entry;
       - none of `blocks[0:Q/64]` was cached in this step. This is the TT analogue of vLLM's own Mamba rule (`V/v1/core/single_type_kv_cache_manager.py:1196-1203`);
       - the entry's stored token ids equal `request.all_token_ids[0:Q]`. This is the token check, moved here from the model (S7): a mismatch only lowers Q, where in the model it would kill the engine;
    4. trims `blocks` to `Q/64` and **stages** the grant. The stage is idempotent across repeated attempts and takes no pin;
    5. plans captures: at `B = floor2048(num_prompt_tokens)` when `B > Q`, plus the **gap boundary** `floor2048(h)` when `floor2048(h) − Q ≥ 2048`. The gap boundary is where a shared system prompt or tool block becomes reusable by other conversations after one miss.
  - **b. `self.kv_cache_manager.coordinator.cache_blocks(request, n)`.** vLLM calls it at allocation (`V/v1/core/kv_cache_manager.py:452-462`) and after every output (`V/v1/core/sched/async_scheduler.py:67-73` through `kv_cache_manager.py:620-629`). The wrapper:
    - passes `min(n, request.num_prompt_tokens // 2048 * 2048)` for a salted request, and 0 for an unsalted one;
    - records the block ids it newly caches, for the same-step rule;
    - is safe with repeated calls, because `num_cached_block` makes repeated capped calls no-ops (`V/v1/core/single_type_kv_cache_manager.py:337-341`).
  - **c. `self.schedule` (the outer `TTScheduler.schedule`, `PLG/scheduler.py:105`)** (F2, S6):
    - on entry, clears the staged grants and the same-step set;
    - on exit, commits only the grants of requests in the returned `SchedulerOutput`, new or resumed, and pins their checkpoints;
    - drops every other staged grant.
  - **d. `self.kv_cache_manager.block_pool._maybe_evict_cached_block`** (`V/v1/core/block_pool.py:574-593`) (F8, S9). After the original runs, if the evicted hash no longer maps to any cached block, drop the registry entry keyed by it.
    - Tail-first freeing makes a conversation's boundary block leave before its earlier blocks.
    - The rare orphan left when an earlier block goes first ages out of the LRU and is counted.
- **Also:**
  - hook `reset_prefix_cache` (`V/v1/core/sched/scheduler.py:2196`) so it also calls `registry.clear()`;
  - hook `_free_request` (`:2099`) to drop the request's plans, staged grant and pins. This also covers a waiting request that is aborted;
  - **runtime kill switch** (S7, e): the wrapper polls a file flag on the persistent mount (`/models/.qwen-c2/prefix-reuse.off`) at most once per second.
    - When the flag is present, it grants nothing and clears the registry. Hits off is always safe.
    - Re-enabling needs an engine restart, so no block published under a suspect build is ever served.
    - It needs no platform change or release. The operator writes `~/hf-cache/hub/.qwen-c2/` on the host (the image symlinks `/experiment-cache` there; `C2/docker/qwen-c2-serving.Dockerfile:64`).
  - log `[PINDIAG] prefix: install` with the scheduler class, the plugin path, the KV dtype and the block size. Log `[PINDIAG] prefix: grant h=… Q=… plan=[…]` from inside the commit (MEM "graft mounted is not graft executed").

**3. Runner patch.** An AST patch of `PLG/model_runner.py`, in the style of `C2/scripts/ci/serving_plugin_patch.py`.
- Pass the row request ids, which exist at `:1269`, into the `submit_prefill` kwargs (`:1784-1797`).
- The model reads them from `**kwargs`. `prefill_forward` already takes kwargs (`IMG/qwen36_vllm.py:203`).
- **The worker asserts `cache_config.block_size == 64` after executor init (F7).** The uniproc executor calls `update_block_size_for_backend` after `load_model` (`V/v1/executor/uniproc_executor.py:69`). For a hybrid model that can re-align the block size (`V/platforms/interface.py:590-630,750`).
  - It is skipped only when `_find_non_ssm_backend` finds no vLLM attention layer (`:574-588`), which is **UNVERIFIED** for the TT model.
  - A block size other than 64 breaks the 64-token page tables and the chunk page arithmetic.

**4. Model graft** (`qwen36_vllm.py` and `model.py`):
- **Capability flag.** `supports_prefix_caching` = `QWEN_PREFIX_REUSE == "1"`, read at import (`IMG/qwen36_vllm.py:57-61`).
  - The platform reads it in `check_and_update_config` (`PLG/platform.py:1134-1147`). So the env must exist in the API-server process too. The contract's `.pth` hook sets profile env in every python process of the image (`C2/scripts/ci/serving_c2_contract.py:1-23,92-101`).
  - **The flag and the graft ship in one stage.** With the flag on but no graft, today's model ignores `start_pos` (`IMG/qwen36_vllm.py:271-295`) and silently rewrites shared blocks.
- **Entry.** `_prefill_forward_tp_batched` (`:271-295`) forwards `start_pos` and the request ids to `prefill_paged_slots` (`IMG/model.py:1719-1783`).
- **Per row, in the existing loop.** Rows run strictly in sequence, each ending in a blocking `to_torch` (`IMG/model.py:1756-1774`).
  - **With a committed grant:**
    1. Assert that the grant's request id is the row's and that the grant's `Q == start_pos` (F2). Re-check the stored token ids as a last guard against registry corruption.
    2. Restore `rec_state` and `conv_carry` in place into the bound B=1 scratch. `conv_states` are not needed: every resumed chunk and the tail rewrite all K of them (`GRAFT/gdn/tp.py:660-686`, `capture_state=True` at `GRAFT/layer.py:327`). How to restore (F3):
       - **Preferred:** `ttnn.copy_host_to_device_tensor` from a host tensor into the existing device tensor. This is the traced loop's own input path, which allocates nothing and runs no program (`IMG/model.py:2515-2568`).
       - Whether its spec check accepts a per-chip host tensor into the scratch is **UNVERIFIED**. The scratch was created with a replicate mapper and holds different per-chip contents (`IMG/gdn/tp.py:238-252`).
       - **Fallback:** `from_torch` + `ttnn.copy`, the pattern of `_restore_gdn_scratch` (`IMG/model.py:1360-1377`) plus the carry. Warm that exact helper once, zero-filled, in `warmup_model_prefill` before the chunk trace is captured (`IMG/qwen36_vllm.py:365-399`).
       - Why the warming matters: today the helper runs only inside `capture_prefill_trace_bucket`, before parking (`IMG/model.py:1480,1499`). A compile after parking is the documented second-request hang (`IMG/model.py:2181-2184`, #48536).
       - The chunk's own `ttnn.copy(final_state, rec_state)` (`GRAFT/gdn/tp.py:660-662`) may already compile the same program. That depends on the fused op's output memory config, which is **UNVERIFIED**.
    3. Run the chunk loop from `Q/2048` without a reset.
  - **Otherwise:** today's cold path.
  - **A row with `start_pos > 0` and no committed grant is an assertion.** It is a last guard: the scheduler already prevents it, and the alternative is a silent rewrite of shared blocks.
- **Loops.** Adopt Lever N's loop edits (`chunk_from`, `do_reset`) in both `_prefill_traced_chunked_tp` (`IMG/model.py:2482-2598`) and `_prefill_chunked_eager_tp` (`:2442-2480`). Source: T32 `lever_n_model_patch.py:75-121`.
  - **Guard the eager tail (F1).** The eager loop sets `last_hidden = None`, runs `for c in range(...)`, then calls `ttnn.deallocate(last_hidden)` when `tail_real > 0` (`IMG/model.py:2449-2466`). Stock code never reaches that call with None, because `num_full == 0` takes the short path (`:2310`).
    - A hit with `Q = floor2048(P)`, where the new tokens do not cross the next boundary, leaves the resumed loop empty and deallocates None.
    - Guard it the way T32's resume helper does (T32 `prefill_prefix_resume.py:61-81`).
    - The traced loop has no `last_hidden` and is safe (`:2530-2598`).
    - Whether `ttnn.deallocate(None)` raises is **UNVERIFIED**; the CPU test's fake ttnn raises on None.
  - **Do not adopt** Lever N's entry edit that builds RoPE only at `start == 0` (T32 `lever_n_model_patch.py:139-146`). The stock entry builds RoPE for the whole prompt on every call (`IMG/model.py:2302`), and a resumed row needs exactly that.
    - The two edits target one anchor, so the stage refuses a tree that carries Lever N's edit (F5).
- **Capture hook.** Right after the loop drains and before the tail (`IMG/model.py:2588-2597`; eager `:2465`), for each planned boundary:
  1. `synchronize`;
  2. `to_torch` of the scratch's `rec_state` and `conv_carry`. This is the D2H pattern every prefill already runs (`:1766-1773`);
  3. `registry.put`.
  - A gap boundary inside the traced loop adds one synchronize. The loop otherwise syncs every 8 chunks (`:2528,2582`).
  - **A capture failure (for example `MemoryError`) skips the checkpoint and counts it. It never fails the request (S7).**
- **Unchanged:** the tail, the logits, the end-of-prefill snapshot and `_write_gdn_slot` (`:1785-1812`).
- **Markers.** Log `[PREFIX] row=… Q=… L=… restored_ms=… captured=[…] ms=…` from inside the branch that ran.
- **Audit mode (`QWEN_PREFIX_AUDIT=1`) is program-free (F3).**
  - It reads the KV caches with `to_torch`, **one tensor at a time**, and slices blocks `[Q, L)` on host.
  - One K or V cache is about 143 MB packed per chip and about 0.54 GB after `to_torch` to fp32. All 32 tensors on 2 chips at once would need about 34 GB of host RAM.
  - It compares the unpacked values, which are what every reader sees, and the slot bytes. A device-side slice would compile after parking.
  - Alternative: run the audit arm with traces off.

**5. Profile `general-prefix`** in `qwen_c2_profiles.json`. It is `general` (`C2/scripts/ci/qwen_c2_profiles.json:115-155`) plus:
- `enable-prefix-caching: true`;
- `enable-chunked-prefill: true`, replacing `no-enable-chunked-prefill`.
  - This only satisfies vLLM's align-mode assertion (`V/model_executor/models/config.py:579-582`).
  - The TT platform then switches chunking off again for `qwen3_5` (`PLG/platform.py:81-86`), sets `long_prefill_token_threshold` to 0 (`:105`) and raises `max_num_batched_tokens` to `max_model_len`. So prefill stays whole-prompt, as the stock model needs.
- env `QWEN_PREFIX_REUSE=1`.

The serving contract already owns all four prefix and chunking flags, so it drops the platform's `--no-enable-prefix-caching` (`C2/scripts/ci/serving_c2_contract.py:37-46`). TS injects that flag for TT by default (`TS py/serving/engine.py:1295-1300`).

The profile keeps `general`'s default `trace_mode` (`"all"`, `PLG/worker.py:198`), so prefill takes the traced chunk loop (`PLG/model_runner.py:1795`; `IMG/model.py:2334-2345`).

#### 2.0.2 Data flow

```
turn N  (prompt P_N)  sched: h = vLLM hit; Q = best checkpoint <= h with matching tokens (0 on a first turn);
                             staged; committed at end of schedule() only if scheduled; plan B_N = floor2048(P_N)
                      model: assert grant.Q == start_pos | reset | restore(Q) -> chunks [Q/2048, B_N/2048)
                             -> CAPTURE(B_N) -> tail -> logits, slot write
                      vLLM : only blocks [0, B_N) are hash-published (cap); decode and tail blocks never are
                      finish: blocks return to vLLM's free queue, still hashed (LRU-evictable);
                             evicting block B_N/64-1 drops checkpoint B_N (coupling)
turn N+1 (P_{N+1})    sched: h >= B_N if the prompt extends turn N's up to B_N -> Q = B_N -> num_computed = B_N
                      runner: start_pos = B_N (existing path), request ids (patch)
                      model: RESTORE(B_N) -> chunks from B_N/2048 (possibly none: guarded) -> CAPTURE(B_{N+1})
                             -> tail -> logits, slot write
```

- **Re-prefilled per hit turn:** `(P_N mod 2048) + (P_{N+1} − P_N)`, about 1k + Δ.
- **When a turn's prompt diverges earlier,** for example because the template strips older reasoning when a new user message arrives, the trim falls back to an older turn's checkpoint below the divergence point. That works only if the checkpoint is still in the LRU and its KV is still cached.

#### 2.0.3 Memory (d)

**GDN checkpoints.** The per-chip GDN state is 48 layers × fp32 `rec_state [1,24,128,128]` plus bf16 `conv_carry [1,3,5120]` (`IMG/gdn/tp.py:238-252`, same in `GRAFT/gdn/tp.py:300-309`). Nv=48/TP2 and qkv_dim_tp=5120 are confirmed by `T32/scripts/ci/test_verify_trace_t1_graft.py:68`.

| Checkpoint form | Size |
|---|---|
| Host, rec + carry | 73.4 MiB per chip, **about 154 MB per checkpoint** for both chips |
| Device, tile-padded (for G3) | 87 MiB per chip |

- LEVN:125-130 says "~38 MB/device". That is the bf16 recurrent state (`QWEN35_GDN_STATE_BF16=1`); fp32 is the default.
- An 8 GiB host budget holds about 55 checkpoints: for example about 8 turn boundaries for each of about 6 resident conversations. Under the session cap, 55 is not binding (s: 200 changes nothing).
- The container runs with `--memory 80g` at 11-20 GiB RSS (SKILL:82-87).

**KV is bf8 on the served image (F4).**
- The image bakes `QWEN_SDPA_BF8=1` into ENV (`C2/docker/qwen-c2-serving.Dockerfile:86`). The contract only adds profile keys (`serving_c2_contract.py:92-101`), and `general`'s env does not override it.
- `allocate_kv_caches` switches the KV dtype to `bfloat8_b` (`IMG/model.py:2684-2686`), and attention casts Q, K and V to bf8 (`GRAFT/attention/tp.py:137-138,846`).
- KV is therefore 16 layers × K,V × 2 heads × 256 × 1.0625 B ≈ **17 KiB per token per chip**. Revision 1 assumed bf16 at 32 KiB.
- The 262,144-token pool is about 4.25 GiB per chip. A 42k conversation holds about 0.7 GiB per chip.

#### 2.0.4 Exactness proof (what the gates test)

Let S = 2048, the chunk trace size (`IMG/model.py:2287`, captured at warmup `IMG/qwen36_vllm.py:365-399`). A cold prefill of T[0,L) does three things:
- runs chunks c = 0..n−1, n = ⌊L/S⌋, through one fixed chunk program: the trace on `general`, eager on C2;
- runs the tail [nS, L) through the masked bucket at `chunk_start = nS`;
- writes logits and the slot.

The GDN input to chunk c is (rec_state, conv_carry), read at `IMG/gdn/tp.py:496-506,561` (`GRAFT/gdn/tp.py:573,641`).

- **L1 (determinism).** Chunk c's outputs are a function only of: the tokens of chunk c, RoPE rows [cS, (c+1)S), the GDN input state, and the KV of [0, cS). The same program with the same inputs gives the same bytes.
  - This assumes run-to-run determinism and that paged SDPA does not depend on physical block ids. Both are strongly supported: M1 chunked equals whole (run 35416319586), and the 4K screen matched after priming with a different suffix.
  - Neither is proven across different page placements, so the G1 audit tests it.
  - **"The same program" means the same path.** In the served graft, `gdn_prefill_conv_exact` runs on every GDN call that has a `valid_len` (`GRAFT/gdn/tp.py:162-222,575-581`). That is every eager full chunk, because the eager loop passes `valid_len=chunk_size` (`IMG/model.py:2461-2463`), and every tail. Traced chunks pass `valid_len=None` and use native conv1d.
  - So traced and eager chunks are not byte-identical to each other. A KV block or checkpoint is canonical only within one path.
  - One engine uses one path, fixed at warmup: it has either captured the chunk trace or not (`IMG/model.py:2334-2345`). The registry dies with the process. If checkpoints are ever persisted, key them by (path, arithmetic flags).
  - **bf8 KV (F4).** K and V are cast to bf8 before the paged fill in both arms, and SDPA reads the prior KV back from the bf8 cache in both.
    - bfloat8_b shares one exponent per 16 values along a tile row. For a KV tile that is within one token, so the bytes do not depend on block placement.
    - That exponent granularity comes from tt-metal format knowledge and is **UNVERIFIED** in the repository.
- **L2 (canonical KV).** Under the cap, a block returned by a hit was written by chunk c of some prefill whose tokens [0,(c+1)S) equal T's. The hash is a sha256 chain over all prior tokens plus salt and extra keys (`V/config/cache.py:95`; salt on the first block, `V/v1/core/kv_cache_utils.py:560-568`). By L1 and induction over c, those bytes equal what T's cold run writes.
- **L3 (canonical checkpoint).** A checkpoint at Q = kS was captured after chunk k−1 of a prefill that matches T on [0,Q), before any tail. So it equals T's cold state at Q.
- **Theorem.** Restore the state at Q, then run chunks k..n−1 and the same tail with the same full-prompt RoPE table. Every input is then equal. So KV [Q,L), the final GDN slot and the logits are byte-identical to a cold prefill.
  - At a 2048-multiple the SDPA `qk_chunk` and the tail bucket are the cold run's (`GRAFT/attention/tp.py:855-866`; the tail bucket depends only on the tail length, `IMG/model.py:1943-1948,2120`).
  - Sampling: `general` samples on host, because device sampling is refused on TP2 (`IMG/qwen36_vllm.py:63-76`). Each request's generator advances once per step (`PLG/model_runner.py:934-960`). So the tokens match too.
- **What breaks it:**
  - resuming at a point that is not a multiple of S. The split changes: conv1d versus FIR, masked-bucket shapes, SDPA `qk_chunk` (64 at a 64-aligned offset), and GDN 128/32 sub-chunk alignment;
  - serving decode or tail blocks. The cap prevents that, on both publish paths;
  - omitting `conv_carry`. The existing helper does (`IMG/model.py:1344-1377`; T32 `docs/combined-prefix-cache.md:51-56`);
  - staging RoPE only at start==0;
  - **a stale grant whose Q differs from the runner's `start_pos` (F2).** The model would restore 8192 while KV is cached only to 4096. The error would then **propagate**: the wrong blocks are hash-published and a checkpoint is captured from the bad run. Per-step commit and the `Q == start_pos` assertion prevent it;
  - **async scheduling, or co-installed Lever N chunking (F5).** Install assertions prevent both;
  - **a block size other than 64 (F7).** A worker assertion prevents it;
  - arithmetic flags differing between capture and use. That is impossible within one engine, because the registry lives in the process and dies with it.
  - A restore that compiles after parking does not make results inexact; it hangs the engine (F3, R20).
- **How the gate measures it.**
  - Compare against a cold arm on the same engine with a different `cache_salt`, which forces a miss and different physical pages.
  - Compare full answers (PLAN §2.2 item 5's flip policy), GDN slot bytes, and an audit of the KV blocks [Q,L).
  - **Chain it:** a conversation of 6 or more turns (a hit of a hit, and so on) against a salted cold run at every turn (F2).
  - **Run it on both paths:** traced on `general`, and eager with `trace_mode: decode_only` (F1), so the loop C1 reuses is proven first.

### 2.1 P0: CPU probes and a prefix-stability study (do first)

| Item | What it retires | Eng-days |
|---|---|---|
| **P0a. Config and scheduler probe** on Python 3.11 (MEM cpu-suite-needs-python-311), with the pinned vLLM 0.25.1 and plugin bf77cd63. See the check list below. | R1, R2, R12, R22, R24 | 0.5-1 |
| **P0b. Prefix-stability and traffic study.** See the list below. | R8, R9 sizing | 0.5-1.5 |

**P0a checks.** Build a `VllmConfig` for `Qwen3_5ForConditionalGeneration` (`IsHybrid`, `V/model_executor/models/qwen3_5.py:389`) with prefix caching and chunking on. Then run the TT `check_and_update_config` and check:
1. the align assertion passes;
2. the platform then turns chunking off;
3. `validate_block_size` passes: `long_prefill_token_threshold` is 0 and `disable_chunked_mm_input` is False (`V/config/vllm.py:2186-2199`; called at `V/v1/engine/core.py:318`);
4. `validate_mamba_block_size` passes (`V/config/vllm.py:2213-2225`; the old trap is at T32 `docs/gotchas.md:140-159`);
5. a scheduler built on the TT single-spec KV config has `has_mamba_layers=False` and a `UnitaryKVCacheCoordinator`;
6. `get_computed_blocks` returns hits.

Then drive the wrappers with real `KVCacheManager` objects:

7. the same-step rule holds with two same-step arrivals that share a prefix;
8. a turn that contains the previous answer verbatim, where every hit block must have been written by a full prefill chunk (the cap's positive control);
9. **decode drive (F6):** `update_from_output` runs through at least 2k decode tokens, and no hash beyond `floor2048(P)` ever enters the pool's hash map;
10. **allocation failure after a grant (F2):** use a tiny pool. The attempt leaves no committed grant, and the next step's lower hit gets a fresh one. Repeated attempts for one waiting request stage one grant and double-count nothing (S6);
11. **abort while waiting** clears the plan and the grant (S6);
12. **eviction coupling (F8):** evicting a boundary block drops its checkpoint;
13. **fail-closed salt (S4):** an unsalted request gets no hit and publishes nothing; a salted one does;
14. **the install assertions** refuse async scheduling, chunked prefill and a hybrid coordinator (F5, F6).

**Block size is checked at bring-up, not in P0a (F7).** It is fixed after `load_model`, which needs the device.

**P0b steps.**
1. Render replayed coding conversations with snapshot `1d4bf0f2…`'s chat template, covering tool calls, a new user message, and reasoning on.
2. Find where prompt N+1 first diverges from prompt N.
3. From the Spark's `usage_events` (`cached_prompt_tokens` per request), get the partial-hit histogram.
   - `usage_events` carry key and tenant but no conversation id (S10). So inter-turn gaps, per-key duty cycle (what the session cap needs) and live conversations per key can only be inferred.

**Cost:** 1-2.5 days, 0 M+A runs. The P0a checks become the scheduler graft's unit tests.

**If P0a fails,** for example because some other hybrid path is keyed on `mamba_cache_mode`, fall back to "session parking":
- keep vLLM's prefix cache off;
- hand the finished request's blocks to a store in `_free_request`;
- return them from a wrapped `get_computed_blocks` (the traffic lens's b1).

The model work is identical; the scheduler work grows by about 2-4 days.

### 2.2 G1: `general-prefix` v1 (the first production deliverable)

**Code touched**
- **New scripts:**
  - `scripts/ci/qwen_prefix_registry.py`;
  - `qwen_prefix_scheduler_patch.py`, the AST stage for `PLG/scheduler.py`;
  - `qwen_prefix_runner_patch.py`, the AST stage for `PLG/model_runner.py` and the worker's block-size assertion;
  - `qwen_prefix_model_patch.py`, the AST stage for `IMG/model.py` and `IMG/qwen36_vllm.py`. It reuses `lever_n_model_patch.patch_tp_replay` and adds the eager-tail guard.
- **Anchor checks.** Each stage refuses to patch if its anchor bytes differ, using the `source.sha256` pattern of the C2 graft. Line citations and sha probes for the five grafted files point at GRAFT (`graft.sha256`), not IMG (F10).
- **Profile.** `qwen_c2_profiles.json` gets the `general-prefix` profile.
- **Image plumbing.** `docker/qwen-c2-serving.Dockerfile` and the workflow copy lists get the new files. They must go in both copy lists, plus the closure test (MEM serving-image-bundle-provenance).
- **Tests** go into `qwen-integration-cpu.yml` (MEM ci-tests-need-allowlisting).
- **Metrics export.** The registry lives in the EngineCore child process (SKILL:211-212), so its gauges cannot register directly in the API server's Prometheus. Export them through vLLM's `SchedulerStats` or a tmpfs file (S5).

**Engineer-days (e)**

| Item | Days |
|---|---|
| Registry (LRU by bytes, pins, staged and committed grants, token ids, eviction coupling, optional slab, kill-switch flag, counters, clear) and tests | 1.5-2.5 |
| Scheduler graft (cap, trim with token check, fail-closed salt, per-step commit, eviction hook, install assertions, reset and free hooks, markers), with CPU tests on real `KVCacheManager` objects (P0a 7-14) | 2.5-4.5 |
| Runner patch (request ids, block-size assertion) | 0.5-1 |
| Model graft: per-row start and grant assertion; restore (rec and carry) with no post-park compile; resumable traced and eager loops with the tail guard; capture hooks (replay and gap) that skip on failure; program-free audit; markers | 3.5-5.5 |
| Profile, image plumbing, `QWEN_` string diff and launched-argv check (MEM read-the-launched-argv) | 1-2 |
| Cross-process metrics export (registry bytes and pins, restore and capture ms, trim loss, orphans, mismatches) | 1-2 |
| Host tests (a fake ttnn that raises on `deallocate(None)`, T32 fixtures), CPU CI hygiene | 1-2 |
| A real-text multi-turn agent corpus. It is a hidden dependency: the gateway drops content by default (ADM `crates/thatch-administration/src/gateway.rs:28-33`), and synthetic prompts hide context bugs (MEM). | 1-2 |
| Hardware harness: chained multi-turn replay; salted cold-versus-hit comparator; eviction, allocation-failure, preemption and abort drivers; overload and probe drill; warm-reload drill; TTFT and hit-rate report | 2.5-4 |
| Measurement and write-up | 0.5-1 |
| **Engine total** | **15-27** (16-29.5 with P0) |

**Assumptions**
- Agent-paced, with one fix loop per gate.
- The image's `model.py` and `qwen36_vllm.py` match IMG. The anchor probe checks this.
- The wrappers work against pinned vLLM 0.25.1 with no upstream change.
- The traced restore behaves like the eager restore.
- No SGLang changes.

**M+A iterations: 9-12 planned** (up to ~19 at 35% failures). Card B 0, TT-Sim 0.

| Gate | Runs | What passes |
|---|---|---|
| Bring-up | 1 | See the list below. |
| Exactness, traced (`general`) | 1-2 | See the list below. |
| Exactness, eager (`trace_mode: decode_only`) (F1) | 1 | The exactness subset on the eager loop, including tail-only hits. This proves the loop C1 reuses. |
| Lifecycle | 1-2 | See the list below. |
| Timing | 1 | See the list below. |
| Overload and probe drill (S2) | 1 | 8 and 16 busy agents, with the health monitor on under TT's probe policy and the session cap on. No reload happens; new sessions past the cap get 429. |
| Soak | 1 | 1-2 h on a metering-shaped trace. The registry, RSS and ledger stay flat. |
| Platform replay and fabric re-measure (SKILL:199-218) | 1 | The thin-layer image under the agent's container config: release-first load, in-place reload, stop/start. Then the fabric gate. |
| Contingency | 1-2 | |

**Bring-up checks:**
- the anchor probe: sha of the image's `model.py`, `qwen36_vllm.py` and the five grafted files;
- the launched argv shows the prefix cache on and chunking back off;
- `Automatic prefix caching is enabled` (`PLG/platform.py:1152-1155`);
- `[PINDIAG] prefix: install` count greater than 0, with the scheduler class, the plugin path, KV dtype `bfloat8_b` and block size 64;
- `[TP chunk-replay]` present, which confirms the traced loop;
- **program-cache entries unchanged across the first hit (F3)**;
- DRAM free per chip is logged, for G2;
- with `QWEN_PREFIX_REUSE=1` but grants disabled, output is byte-identical to `general`.

**Exactness checks** (traced gate):
- salted cold versus hit at 4k, 16k, 42k and 60k, on real coding text with long answers, compared in full;
- **a chained conversation of 6 or more turns, against a salted cold run at every turn (F2)**;
- GDN slot bytes and a KV [Q,L) audit;
- a turn containing the previous answer verbatim;
- a changed suffix, a prefix that diverges early (falls back to an older checkpoint), and a shared system prompt across two conversations (gap capture);
- previous-prompt lengths 2047, 2048 and 2049, and a **tail-only hit** (new tokens do not cross the next boundary).

**Lifecycle checks:**
- forced KV eviction under 4 × 60k;
- checkpoint LRU eviction and eviction coupling;
- **allocation failure after a grant, on a tiny pool (F2)**;
- preemption and resume;
- abort during a hit prefill, and abort while waiting;
- `reset_prefix_cache`, and the kill switch;
- 4 mixed arrivals in one step, including the same-step rule;
- **a warm-cache in-place reload drill (S8)**.

**Timing checks:**
- a metering-shaped multi-turn replay at 1, 4, 5 and 6 busy agents, under normal CI load with the pod count recorded (S9; SKILL:92);
- recorded: TTFT p50/p90, turn time, restore and capture ms, host RSS;
- the hit rate from the model's `L−Q`, not vLLM's `prefix_cache_hits`, which counts before the trim;
- trim loss h−Q, and orphan counts.

**Platform and routing needs (TS and ADM)**

1. **Give TT its own model id (catalogue entry) for an opt-in set of agents (D-P3).** 0.5-1 day.
   - **Why a shared id is likely today:**
     - The Spark's SGLang table is keyed on `Qwen/Qwen3.8-27B` among others (TS `py/serving/sglang_engine.py:235-240`), and TT is placed under `Qwen/Qwen3.8-27B` (SKILL:29). The id each live placement advertises is **UNVERIFIED**.
     - `next_warm_node` takes the first node in registry order (TS `crates/thatch-spine/src/serving_backend.rs:810-815`).
     - The registry lists nodes with no `ORDER BY` (`crates/thatch-store-postgres/src/registry.rs:182`), and every heartbeat rewrites the row (`:212-216`). So order can change at heartbeat cadence.
     - The spine runs 2-4 replicas with no session affinity (`crates/thatch-store-postgres/src/billing.rs:1611-1613`).
   - **Why a shared id hurts:**
     - A conversation that alternates between the Spark and TT misses on both.
     - The Spark serves 262,144 context (`sglang_engine.py:180`) and TT 65,536. A long conversation that lands on TT fails with "maximum context length", which the spine does not reroute (`serving_backend.rs:697-709`).
   - **Why a separate id fixes it:** agents that choose TT know the 64k limit, and a node-local session cap is enough.
   - Whether the model library allows a second id for the same weights is **UNVERIFIED**.
2. **Session cap in `serving.server` on the TT node: required.** 1-2 days.
   - The node admits at most about 4 active sessions (5 when tool gaps are long). It returns 429 with `Retry-After` for a new session key over the cap, and always admits known keys.
   - The session key is a hash of (`cache_salt`, messages up to the first user message).
   - Counting sessions rather than requests matters: a per-request cap bounces a conversation whenever it is in a tool gap.
   - This overlaps PLAN R8, edge admission control.
   - How the spine treats a 429 (surface it, or reroute it) is **UNVERIFIED**. `is_reroutable_inference_error` reroutes everything except request-invalid signatures (`serving_backend.rs:697-709`).
3. **A TT health-probe policy: required (S2).** 0.5-1.5 days.
   - **What the probe does today:**
     - It is a real generation ("." with 8 tokens) through the same vLLM FCFS queue (TS `py/serving/manager.py:282-353`).
     - Defaults are every 30 s with a 120 s timeout, reload after 2 failures, and `os._exit` after 2 failed recoveries (`py/serving/server.py:2940-2953,3029-3062`). The TT drop-in overrides none of them (`deploy/systemd/thatch-node-agent.service.d/tenstorrent.conf`).
     - The probe also needs one of the 4 slots. Four 2k-token answers at 12.8 tok/s take about 156 s, so it can fail with no queue at all.
   - **The loop:** a reload wipes the pool and the registry. Every live conversation then misses serially, which fails the probe again.
   - **Fix:**
     - use a liveness signal that does not queue: engine step progress from `/metrics`;
     - or raise the TT probe timeout above the worst 4-slot decode;
     - add a grace period after reload.
   - Whether a gateway timeout cancels the TT request is also checked here (R27).
4. **Per-tenant `cache_salt` in the ADM admin-http gateway: required (S4, D-P2).** 1.5-3 days.
   - **Why it goes in that gateway:**
     - The spine receives only `key_id` and the body (ADM `crates/thatch-administration/src/gateway.rs:43-52`).
     - `tenant_id` exists only in the admin-http gateway (ADM `crates/thatch-admin-http/src/gateway.rs:744,767,947`).
     - The gateway forwards the client body with only `force_stream_usage_reporting` added (`:1405-1416`). So today a client-supplied `cache_salt` would reach vLLM, which accepts it (`V/entrypoints/openai/chat_completion/protocol.py:425,913-919`).
   - **Fix:**
     - set `cache_salt = HMAC(secret, tenant_id)` and overwrite any client value. A tenant id is a displayed ULID, so it must not be the salt itself;
     - strip the field for SGLang in `serving.server`, whose TT/SGLang proxy sends no field an engine has not been shown to accept (TS `py/serving/server.py:407-415`). Whether SGLang accepts `cache_salt` is **UNVERIFIED**;
     - the scheduler's fail-closed rule covers any path without a salt: the probe, warmup, direct calls.
5. **Metrics (S5).** 0.5-1 day.
   - TS's vLLM mapping has no prompt-token or cached-token metrics (TS `py/serving/metrics.py:55-70`). The SGLang mapping has them (`:139-142`), which is where the 91.7% came from.
   - Map `vllm:prompt_tokens` and `vllm:prompt_tokens_cached` onto `thatch_serving_prompt_tokens_total` and `thatch_serving_cached_prompt_tokens_total`.
     - `prompt_tokens_cached` is set from `PrefillStats` after the wrapper returns (`V/v1/core/sched/scheduler.py:774-780`), so it is correct after the trim.
   - Never map `vllm:prefix_cache_hits`. It is recorded inside `get_computed_blocks` before the trim, once per attempt (`V/v1/core/kv_cache_manager.py:238-244`).
   - The registry's own gauges come through the engine's export.
6. **Cached-token reporting on TT, and its price (S11, D-P5).** 0.5 day, plus one rig check that the flag parses on the image.
   - Set `enable_prompt_tokens_details` true for TT, per model, in the catalogue entry, or in operator env (precedence: defaults < per-model < catalogue < operator, TS `py/serving/engine.py`, `_build_command`). The TT defaults set it False (`TS py/serving/engine.py:1295-1300`), and the serving contract does not own that flag.
   - The meter bills from `cached_tokens` (`TS py/serving/engine.py:571-583`), so turning it on changes TT billing. The cached-token price on TT needs an explicit decision.
7. **Later, only if TT shares the Spark's model id:** rendezvous hashing with bounded load, deterministic across spine replicas and aware of prompt length (anything above about 60k goes to the Spark). 4-8 days.
8. **Profile selection and rollback.** The agent cannot pick a profile until ADR-0106 is built (SKILL:31-35). So `general-prefix` ships as the image's `default` profile (`qwen_c2_profiles.json:2`). That is about a 90-minute release (SKILL:15-20). Rollback is the previous tag. The runtime kill switch (§2.0.1 item 2) covers the time in between.

**Platform total:** 4.5-9 days on a separate id; 8.5-17 on a shared id.

**What ships:** the TTFT and throughput effect in §1 for at most about 4-5 busy agents per pair, with exactness gated byte-for-byte.

### 2.3 G2: KV capacity (config plus gate; after G1's DRAM reading)

- **Raise the pool with `QWEN36_MAX_TOKENS_ALL_USERS`** (`IMG/qwen36_vllm.py:79-100`). At bf8 (F4), each extra 131,072 tokens costs about 2.1 GiB (2.3 GB) per chip. That holds about 3 more resident 42k conversations.
- **Budget per chip,** with free DRAM on `general` **UNVERIFIED**:
  - about 34.4 GB total;
  - weights about 10 GB (c);
  - KV about 4.6 GB;
  - trace region 1 GiB (`qwen_c2_profiles.json:148`);
  - GDN slots and scratch about 0.5 GB.
  - That suggests roughly 18 GB of headroom (e). Revision 1 said 10 GB, assuming bf16 KV.
- **The bf8 lever is already spent.** `QWEN_SDPA_BF8=1` is on in the served image, and the references are already bf8.
- **What it buys (s).** Doubling the pool lifts 8-agent reuse from 0.29-0.42 to 0.88 and 12-agent reuse to 0.85-0.86. But the 8-agent TTFT p50 is still 17-64 s and the 12-agent p50 is 87-137 s, because the four decode slots are saturated.
  - So G2 keeps more idle conversations resident (bursty or long-gap traffic). It does not raise the busy-agent cap.
- **Cost:** 1-3 days, 1-2 M+A runs. Platform: nothing, unless the session cap is re-tuned.

### 2.4 G3: device checkpoint store and device-side slot write (only if G1's timing shows it matters)

- **The store.**
  - Allocate an N-entry device pool (87 MiB per chip per entry) in `_allocate_kv_caches_tp` (`IMG/model.py:2740-2763`), before any trace is parked. Compiling after parking clobbers the trace (`IMG/model.py:2181-2192`).
  - Warm the fp32 and bf16 copy programs at warmup.
  - Capture and restore then become device copies taking milliseconds, instead of about 0.15-0.5 s each (d).
  - The host LRU stays as the second tier.
  - This is the mechanism the 4K screen proved: preallocated checkpoints, no device storage allocated at restore (T32 `docs/combined-prefix-cache.md:58-61,136-141`).
- **Also move `_write_gdn_slot`'s host round trip onto the device.** Every prefill pays it today (`IMG/model.py:1766-1812`), so this helps misses too. The gain is **UNVERIFIED**; the Lever N M1 cost of about 1 s per extra step bounds it from above (T32 `docs/lever-n-plugin-contract-2026-09-19.md:214-216`).
- **Sharing.** This is exactly Lever N v2's "scratch parking" (LEVN:125-130). Build it once and let S3 reuse it.
- **Cost:** 3-6 days, 1-3 M+A runs.

### 2.5 C1: prefix reuse in the C2 fast path (after S1 and S2; ship together with E)

**Carried over unchanged from G1:**
- the registry, the scheduler cap, trim, commit and coupling, the runner patch;
- the model graft. C2's prefill takes the eager loop, because its trace mode is `decode_only` (`qwen_c2_profiles.json:38`) and no chunk trace is captured (`IMG/qwen36_vllm.py:365-370`). G1 patches and gates both loops. The eager loop is also the one the 4K screen proved.

**C2-specific work**

| Item | Where | Days (e) |
|---|---|---|
| Profile and policy: allow the prefix cache in the fast-path profiles | `serving_fast_policy.py:109-113`, `qwen_c2_profiles.json`. The contract already owns the flag. | 0.5-1 |
| **Lifecycle admission of a hit.** Accept `computed == B` when B is 2048-aligned and within the C2 rule. Build the capture with `capture_factory(P, start=B)`. Keep D2 quarantine (S1) for the misuse cases. **This ships with the flag, or the first hit kills the engine.** | Today `_execute` refuses `num_computed_tokens != 0` (`serving_lifecycle.py:263-272`) and every in-engine refusal re-raises (`:296,473`). The capture is built at `:277`. | 1.5-3 |
| **The C2 rule `B ≤ P − 2048`,** so the drafter window `[P−2048, P)` lies in the re-prefilled range. `DFlashDevice` pins `feature_start == P−2048` and window-sized features (`dflash_device.py:494-507`). The window is then captured exactly as a cold run captures it, so no feature history needs retaining. The rule also always leaves at least one chunk, so C2 never takes the tail-only path. Cost: at least one chunk re-prefilled, about 0.85 s at 42k (d). | In the scheduler trim | 0.5-1 |
| `PrefillWindowCapture(start=B)` and a chunk ledger counted from B | `dflash_prefill_window.py` (c: `:90-105,148-149,173`) | 1-1.5 |
| **Shared read-only pages.** A common system prompt leaves two live slots holding the same physical blocks. The per-request checks require unique pages only within one request (`serving_runtime.py:356-359`, `serving_page_binding.py:69-73`). Whether any cross-slot check refuses sharing, and whether K64i and ordered-cache readers are safe over shared pages, is **UNVERIFIED**. | | 1.5-3 |
| **Page-table padding (F9).** C2 fills unused page-table entries with the request's first block (`serving_runtime.py:360-361`, `serving_page_binding.py:93-94`). Under prefix caching `blocks[0]` is the most-shared page, typically a common system prompt; one stray write through a padded entry would corrupt every conversation sharing it. Pad with vLLM's null block 0 instead, which is never hashed and which the model already avoids writing (`IMG/model.py:1978-1979`), or with a dedicated scratch page. Audit each writer's page-index source (tree scratch, K64i, ordered cache) as part of R16. The verifier refresh already bounds its rows to allocated blocks (`serving_page_binding.py:76-85`). | | 0.5 |
| One-in-flight and packed-step interplay; a restore while another prompt's chunked prefill is suspended. With S3 this needs the joint Lever N contract: two grant kinds (restore versus continue), and hits counted toward the one-in-flight limit (F5). | `serving_one_in_flight.py`; `serving_lifecycle.py:255-296` already admits first chunks; LEVN:112-115 | 0.5-1.5 (+1-2 if S3 has landed) |
| Image plumbing (three delivery routes), `QWEN_` string diff, pinned-source check | MEM qualification-pins-model-sources | 0.5-1.5 |
| Gate tooling beyond G1's (template-faithful turns under the C2 contract, TTFT and stall) | | 1-2 |
| Observability and contingency | | 2.5-5 |
| **C1 total** | | **11-22** |

**Pinned-source check.**
- The `model.py` and `qwen36_vllm.py` grafts must not break the fast path's frozen qualification, or the `exact` profile dies at attach.
- Lever N's grafts of these two files ran under the fast path in the m3native gates (`lever_n_m3native_run_arm.sh:103-112`, c), which suggests they are outside the pinned set.
- That is **UNVERIFIED** for the served tree.

**M+A: 5-9 runs**
- H1: hit equals cold solo at 8k, 32k, ~60k and ~120k, with new-token counts 100, 2047, 2048, 2049 and 6k. Includes negative controls: restore skipped, and a foreign checkpoint.
- H2: 2-4 concurrent conversations under S2 packing, hits mixed with cold arrivals, and two live users sharing a system prompt (shared pages and padding).
- H3: eviction with 6-8 conversations, and restart.
- H4: TTFT and stalls on a metering-shaped trace.
- Contingency.

**Exactness.** The §2.0.4 argument, plus the drafter:
1. Under `B ≤ P−2048`, the five-tap window is recomputed by the same prefill chunks.
2. The adopted slot, the drafter history and the verifier's initial GDN snapshot therefore equal the cold run's.
3. So proposals and verification are identical.

It is judged under PLAN §2.2 item 5's flip policy: same image, same arithmetic flags.

**Effect per hit turn at 42k (d, c: fast-path lens §5)**

| Configuration | TTFT |
|---|---|
| C2 S2 today | about 18-19 s: prefill 15 s plus build 3.0-4.1 s |
| S2 + C1 | about 5-6 s: re-prefill of about 4.5k (1.8 s), build 3-4 s |
| S2 + E + C1 | about 2.5 s |

**Capacity.** The C2 `exact` profile has 8,208 blocks, about 525k tokens (`qwen_c2_profiles.json:27`): about 12 resident conversations (e). Decode slots remain 4, so the session cap from G1 carries over, at about 4-6 depending on C2's faster decode (e).

**Routing needs** are the same as G1's.

---

## 3. Composition with the C2 plan's S1/S2: P changes the stage order

**What the plan says now.**
- P and E are optional stages after S2 (PLAN:84-86,158-159,742-757).
- The metering note upgrades P to "NOT optional" and answers D-f as "P (and E) before S3" (PLAN:69).
- The plan's P is per-slot: "each slot keeps its last prompt's KV pages leased… evict when a fifth conversation arrives" (PLAN:752-755).

**What changes.**

**1. P splits into P-general (G1) and P-C2 (C1). G1 moves to "now".** It does not depend on S1, S2, K64j or E, and it is the cheapest large win on real agent traffic, within the session cap.

| Path | TTFT | Decode | Turn or throughput |
|---|---|---|---|
| `general` today, one agent, 42.2k turn, 740-token answer | 13-20 s (s: p50 13.7 s) | 38-48 s at 15.3-19.3 tok/s (PLAN:151) | about 53-68 s |
| **`general` + G1 (hit), one agent** | **about 2-3 s** (s: p50 2.1, p90 4.2) | 38-48 s | **about 40-51 s** |
| C2 S2, no P (PLAN §3; c) | about 18-19 s | 22-34 s at 22-33 tok/s | about 40-53 s |
| C2 S2 + E + C1 (hit) | about 2.5 s | 22-34 s | about 25-37 s |
| 4 busy agents, `general` + G1 | s: p50 2.2-2.3 s, p90 4.6-5.2 s, queueing included | 12-13 tok/s each (SKILL:85) | s: 129-188 turns/h against 98-120 without reuse (1.3-1.6x) |

**2. G1 raises the bar C2 must clear for agent traffic.**
- S2 alone is about level with `general` + G1 for a lone agent, and loses on TTFT and arrival stalls.
- So cutting agent traffic over to C2 should wait for **S2 + E + C1**.
- S2 alone still wins where decode dominates: long answers, or 2-4 live users (27-35 against 12-13 tok/s per user).
- This sharpens D-a and D-f.

**3. Order after S2: E then C1, or both together, then S3.**
- C1 without E leaves a TTFT of 5-6 s, so C1 and E ship as a pair.
- S3's urgency for agent traffic drops: on hit turns decoders freeze for about 2-3 s instead of 15-19 s. S3 then matters mainly for first turns and compactions.
- S3 (Lever N) and prefix reuse cannot share a process until the joint contract exists (F5). Budget it in S3: +1-2 days.
- G3's device store is Lever N v2's scratch parking, so S3's v2 item (2-4 days, PLAN:584-585) shrinks by about 1-2 days if G3 is built first.

**4. Small scope changes inside S1 and S2** (no new stage):
- **S1 lifecycle (D2, PLAN:297).** Write the quarantine so that a `computed == B` admission can be added without restructuring `_execute` (`serving_lifecycle.py:255-296`).
- **S1 attach-time source check (PLAN:289-292).** It must not pin `model.py` or `qwen36_vllm.py`, or it must pin the grafted bytes. Otherwise G1's graft kills the fast-path profiles.
- **S1 image plumbing (PLAN:317-323).** The three delivery routes are shared with G1's graft plumbing. Do it once, in whichever lands first.
- **S2 extent readers.** Do not assume unique physical pages per user, because C1 shares prefix pages read-only, and do not pad page tables with a real page (F9). Raise both in the S2 reader design so C1 does not reopen them. Whether today's readers already tolerate sharing is **UNVERIFIED**.
- **The plan's E note** that "K64j's runtime-position word is the likely enabler" (PLAN:733-734). The fast-path lens reads C2's sequential engines as already using runtime positions (c: `attention_batch.py:91-105`). If that holds, E can start before K64j lands.

**5. Hardware contention.**
- G1's 9-12 M+A runs fit before the S2 M+A ladder, which waits on K64j's card-B work (PLAN:785-791).
- Run them in the same window as S1's bring-up runs. Share the `general` paired probe (R14), and combine resets where possible.
- M+A are development cards since 2026-09-26: the user pulled TT serving (MEM `goal-200tps-concurrent.md`, "LIFTED"; C2 aaf35537). The MEMORY.md index line still says "PAUSE"; it is stale.
- G1's production step is therefore a new placement, with its replay and fabric gate (SKILL:199-218).

**Revised order**

| When | Work |
|---|---|
| Weeks 0-7 | P0 → G1 (+ platform items 1-6), in parallel with S1's CPU work, the K64j probe and S2's card-B/TT-Sim track |
| Then | S1 bring-up → S2 (with R in parallel) |
| Then | E + C1 |
| Then | S3 (Lever N, joint contract with P, reusing G3) → S4 |

G2 and G3 slot in after G1's measurements.

**Totals for P (e)**

| Increment | Eng-days | M+A runs |
|---|---|---|
| G1 | 21-39 (including P0 and platform; 25-47 on a shared id) | 9-12 |
| G2 + G3 | 4-9 | 2-5 |
| C1 | 11-22 | 5-9 |
| **All of P** | **36-70** | **16-26** |

- The plan's reviewer figure for P was 15-30 days and 5-10 runs (PLAN:757). It covered C2 only and was never derived (PLAN:972-975).
- E stays separate: 8-15 days, 3-6 runs.

---

## 4. Risks and the experiments that retire them

| # | Risk | What breaks | Experiment that retires it | When |
|---|---|---|---|---|
| R1 | The engine will not start with the prefix cache on for `qwen3_5` on TT: align assertion ordering (`V/config/vllm.py:878` vs `:1384`), `validate_block_size`, or another path keyed on `mamba_cache_mode` | G1 as designed; fall back to session parking (§2.1) | P0a CPU probe; then G1 bring-up reads the argv and the platform's "Automatic prefix caching is enabled" line | P0, G1 run 1 |
| R2 | Silently inexact hits from KV that decode or the tail wrote. vLLM caches up to `num_tokens` at allocation (`kv_cache_manager.py:452-462`) and after every output (`async_scheduler.py:67-73`), and returns the first block for a hash (`block_pool.py:62-72`). | Wrong tokens, no error | The cap on `coordinator.cache_blocks`, which both paths reach; install assertion on the unitary coordinator; P0a checks 8-9; the G1 exactness gate repeats it on hardware | P0, G1 |
| R3 | Restoring into the **traced** scratch is unproven. The 4K proof was eager-only, and `native_resume_scope` rejects traced mode (T32 `docs/combined-prefix-cache.md:108-114`). | G1 exactness | G1 traced exactness run, chained turns included | G1 |
| R4 | Run-to-run non-determinism, or dependence on physical page ids | Hits correct in exact arithmetic but not byte-equal | Audit mode: two salted cold runs of one prompt on different pages, byte-compare KV and slot. The M1 evidence is weak (completions of about 10 tokens, T32 `docs/lever-n-plugin-contract-2026-09-19.md:205-210`). | G1 bring-up |
| R5 | The graft is mounted but not executed: wrong scheduler class, or a different plugin tree (`/opt/vllm-tt-plugin` versus `/opt/qwen-fast-plugin`, **UNVERIFIED**) | No hits, or hits without restores | `[PINDIAG] prefix: install` and `[PREFIX]` counts above 0 from inside the branches; the install line logs the scheduler class and plugin path; anchor sha probe | G1 bring-up |
| R6 | IMG's `model.py`/`qwen36_vllm.py` are not the served ones. The five grafted files are served from GRAFT, not IMG (F10). | Patch anchors miss | Stage scripts refuse on anchor mismatch; the bring-up probe dumps the served files' sha | G1 bring-up |
| R7 | Capture plus restore costs more than 0.15-0.5 s each (d) | TTFT rises toward 3-4 s | The timing run logs `restored_ms` and `captured ms`; above about 0.5 s each, pull G3 forward | G1 timing |
| R8 | Prompt N+1 does not extend prompt N: reasoning stripped when a new user message arrives, tool calls re-rendered, client compaction or pruning. Qwen3.8's template behaviour is **UNVERIFIED**. | Lower hit rate; older-checkpoint fallback more often | P0b render study, plus the Spark's partial-hit histogram. Retain per-turn checkpoints (LRU), so a divergence costs only back to the last turn boundary below it. | P0, G1 timing |
| R9 | **Capacity and queueing.** More than about 5 busy agents per pair: reuse falls to 0.29-0.42 at 8 and 0 at 12-16 (cyclic LRU), and TTFT reaches minutes (s). The Spark's peak is about 34 in flight. | Hit rate and TTFT collapse under burst | The per-pair session cap (platform need 2), counted in sessions, with the backend fixed at a session's first turn; timing run at 1/4/5/6; overload drill at 8/16. G2 only keeps idle conversations resident. | G1 timing, overload drill |
| R10 | **Routing.** The Spark and TT probably serve one id; registry order is unstable; spine replicas have no affinity; the 262,144 versus 65,536 context mismatch turns long conversations that land on TT into client errors (`serving_backend.rs:697-709`) | Misses on both backends; errors to users | Separate TT model id (D-P3); if shared, affine, length-aware routing (4-8 days) and a two-node staging test | Before production |
| R11 | Cross-tenant timing side channel through a shared cache (about 2.5 s against 15 s) | Leaks whether another tenant's prefix exists | Per-tenant HMAC salt injected by the admin-http gateway, overwriting any client value; fail-closed wrapper; test that an unsalted or foreign-salted request never hits | Before production |
| R12 | Same-step hits on blocks allocated in this step. Rows run strictly in sequence (`IMG/model.py:1756-1774`), but whether row order follows admission order is **UNVERIFIED**. | A row reads KV not yet written | The same-step rule makes row order irrelevant; P0a check 7; G1 lifecycle "4 mixed arrivals in one step" | P0, G1 |
| R13 | Pins and memory leaks: host RSS growth in an 80g container; allocator fragmentation | OOM, or a slow leak | Registry byte gauge; optional preallocated slab; soak with RSS and ledger flat | G1 soak |
| R14 | Preemption, abort or allocation failure around a hit prefill | Stale pin; state written into the wrong slot | Pins exist only for committed grants and last one step; `_free_request` hook; lifecycle run (preempt and resume, abort during a hit and while waiting, allocation failure after a grant, `reset_prefix_cache`) | P0, G1 lifecycle |
| R15 | C2: the first hit kills the engine, because the lifecycle refusal re-raises (`serving_lifecycle.py:263-272,296`) | Engine death | Ship the flag and the lifecycle change in one commit; CPU test of a hit admission; H1 | C1 |
| R16 | C2: shared physical pages refused or unsafe (page binding, cache owner, packed and K64i readers), and padding with a shared page (F9) | Engine death or corruption of every conversation sharing the prefix | CPU review of the uniqueness checks and of each writer's page-index source; pad with the null block; H2 with two live users sharing a system prompt | S2 design, C1 |
| R17 | C2: the model graft trips the frozen qualification pins | `exact`/`c2` die at attach | Check `runtime_component_sources` for `model.py`/`qwen36_vllm.py`; attach `exact` with G1's graft present (byte-identical to v235) | G1 bring-up (cheap), C1 |
| R18 | C2 without E: TTFT stays at 5-6 s on hits | C1's value halves | E's feasibility probe: replay a width-4 trace captured at position X at position Y (c: fast-path lens §8 q7) | Before C1 |
| R19 | Near-tie flips when comparing hit against cold under different arithmetic | Endless investigation loops | Always compare on the same engine and flags (salted cold arm); PLAN §2.2 item 5 flip policy | G1, C1 |
| R20 | **The restore compiles a program after the traces are parked (F3)** | Second-request hang (#48536) | Restore by `copy_host_to_device_tensor`, or warm the exact restore helper before trace capture; bring-up requires program-cache entries unchanged across the first hit; program-free audit | G1 bring-up |
| R21 | **A tail-only hit on the eager loop deallocates None (F1)** | Engine death on C2 and on any trace-less `general` | Guard the deallocate; CPU test with a fake ttnn that raises on None for `chunk_from == num_full`, `tail_real` in {1, 2047}; the eager exactness gate | P0/G1 CPU, G1 eager gate |
| R22 | **A stale grant (F2)**: a grant recorded on an attempt that did not schedule, then used after the hit shrank | Wrong KV and state, then hash-published and checkpointed, so later hits inherit the error | Per-step commit; the model asserts `Q == start_pos` and the request id; P0a check 10; the chained-turn gate; the kill switch for recovery | P0, G1 |
| R23 | **Health-probe metastable reload (S2)**: overload fails the probe, the reload wipes the cache, and every conversation misses | Repeated reloads; `os._exit` after two failed recoveries | TT probe policy (platform need 3); the session cap; overload drill with the monitor on | G1 overload drill |
| R24 | **Async scheduling or co-installed Lever N (F5)** | Unwritten blocks read; another request's scratch destroyed | Install assertions (async off, chunked off, budget ≥ max_model_len); the model stage refuses Lever N's RoPE edit; joint contract in S3 | P0, G1 bring-up, S3 |
| R25 | **Block size changed after `load_model` (F7)** | Page tables and chunk page arithmetic wrong | Worker assertion `block_size == 64` after executor init; logged in the install line | G1 bring-up |
| R26 | **No production observability (S5)**: containers run `--log-driver none` (SKILL:219-221), and TS maps no vLLM cache metrics | Hit rate, orphans and failures invisible outside gates | Metrics mapping (platform need 5); engine export of registry gauges | G1 |
| R27 | **Gateway timeout does not cancel the TT request (S6).** Qwen3.8-27B gets the default 120 s budget, because 27B is under the gateway's 30B threshold (ADM `crates/thatch-admin-http/src/gateway.rs:36,43,835-863`). Whether a timeout cancels the TT request is **UNVERIFIED**. | Orphaned requests hold slots when the queue exceeds 120 s, and client retries double the load | Check tunnel cancellation on gateway timeout; the session cap keeps queues under the budget | G1 platform |
| R28 | **Billing change (S11)**: cached-token reporting changes what TT bills | Wrong invoices, or a silent discount | D-P5 before turning `enable_prompt_tokens_details` on | Before production |
| R29 | **Restarts wipe warm state (S8)**: every deploy, probe reload, OOM or engine death sends all live sessions through serial cold re-prefills. The registry cannot survive a restart: the container is read-only and `/models` is its only persistent mount (SKILL:246-247); that is acceptable. | 60-100 s before the last of 4-5 agents gets a first token (e) | Session cap; probe grace after reload; warm-cache reload drill | G1 lifecycle |

**Decisions needed from the user**
- **D-P1, exactness policy for hits.** This design assumes "byte-identical to a cold run on the same engine". Accepting state that decode produced (GPU-like; about 0.7 s more saved per turn) would be "NOT COMPARABLE" under PLAN's flip policy. Not recommended.
- **D-P2, tenant isolation.** The design now defaults to isolation: a per-tenant HMAC salt injected in the admin-http gateway, and fail-closed for any request without one. Confirm this. Also decide whether to apply the same rule to the Spark, whose SGLang radix cache probably shares prefixes across tenants today (inferred, **UNVERIFIED**).
- **D-P3, placement.** Recommended: TT on its own model id for an opt-in set of agents, with a session cap of about 4 per pair. The alternative is a shared id with affine, length-aware routing (+4-8 days).
- **D-P4, host memory budget** for checkpoints. The default is 8 GiB of the 80g container.
- **D-P5, cached-token price on TT,** once `enable_prompt_tokens_details` is on (the meter bills from it).

---

## 5. Where the lens reports disagreed, and the resolution

1. **Trim versus clamp-and-rewrite.**
   - The vLLM lens trims hits to a checkpoint in the scheduler.
   - The model lens clamps to a 2048-multiple and lets the model recompute `[Q, P′)` into shared blocks, relying on identical bytes.
   - **Resolution: trim.** The scheduler and model share one process (uniproc executor at world size 1, `V/config/parallel.py:915-916`), so no shared block is ever written.
2. **The publish cap.**
   - The vLLM lens's design lacks it and is inexact in the common case. Turn N−1's tail and decode blocks are inserted first and returned first for turn N+1's hit below `B_N` (`block_pool.py:47-72`).
   - **Resolution:** adopted the model lens's cap, on `coordinator.cache_blocks` so that it catches both publish paths (F6).
3. **DSpark feature history for C2.**
   - The vLLM and traffic lenses would retain it.
   - C2's drafter is DFlash2, which uses a 2048-row window pinned at `P−2048` (`dflash_device.py:494-507`).
   - **Resolution:** the rule `B ≤ P−2048` removes the need. DSpark history belonged to the older T16/DSpark combined runtime.
4. **Checkpoint size: ~38 MB, 75 MiB, 87 MiB or 100.7 MB per chip.**
   - **Resolution:** the host checkpoint is rec + carry, 73.4 MiB per chip; on device it is 87 MiB tile-padded.
   - 38 MB is the bf16 recurrent state.
   - The larger figures include the 4 `conv_states`, which a restore does not need (§2.0.1 item 4).
5. **Session parking (traffic lens b1) versus vLLM's prefix cache.** Parking is kept as the P0a-failure fallback, because it avoids R1. Otherwise vLLM's cache wins: shared system prompts, partial matches at older turn boundaries, and vLLM's LRU and accounting, at a lower scheduler cost.
6. **Host versus device checkpoint store.** Host for v1:
   - no pre-trace allocation constraint;
   - no DRAM risk.

   Revision 1 also argued that the request-time pattern was "already on every prefill". Revision 2 corrects that (F3): the pattern every prefill runs is `_write_gdn_slot` → `write_slot` into the batched buffers (`IMG/model.py:1785-1812`, `IMG/gdn/tp.py:796-837`), not a copy into the B=1 scratch. The host restore must therefore use a no-program copy, or a copy warmed before parking (§2.0.1 item 4).

   The device store is G3, and is justified only by measured cost.
7. **The plan's per-slot P (PLAN:752-755)** is superseded. No slot holds anything between turns, and capacity is set by the KV pool and, beyond about 5 busy agents, by decode slots.

## 6. Still UNVERIFIED (each has an experiment in §4)

- Which image `IMG/model.py` and `IMG/qwen36_vllm.py` came from, and which plugin tree `general` imports.
- That the engine starts with the prefix cache on for `qwen3_5` on TT (P0a), and that the block size stays 64 after `load_model` (whether the TT model exposes any vLLM attention layer).
- Traced-scratch restore exactness, and independence from physical page placement.
- Whether `copy_host_to_device_tensor` accepts the restore; whether the fallback's copy program is already in the program cache.
- Whether `ttnn.deallocate(None)` raises.
- bf8 exponent granularity (per token within a KV tile).
- Capture and restore cost.
- Qwen3.8 chat-template prefix stability.
- Free DRAM on `general`.
- Row order versus admission order within one prefill call (moot under the same-step rule).
- Whether `enable_prompt_tokens_details` parses on the TT image.
- Which model id each live placement advertises; whether the model library allows a separate TT id; how the spine treats a node's 429.
- Whether a gateway timeout cancels the TT request.
- Whether SGLang accepts `cache_salt`, and whether the Spark shares its cache across tenants.
- C2's shared-page safety, its writers' page-index sources, and its pinned-source set.
- E's feasibility.
- Which path Lever N's run 35416319586 exercised, and whether Lever N's own final tail-only step has F1's defect.

Every (d) timing comes from the one-parameter-pair fit or from carried fast-path admission logs. Every (s) figure comes from SIM, a model of the mechanism with its own assumptions (decode rate by slot count, exponential gaps and tool results, compaction at 60k). It is not a hardware measurement.

---

## Appendix A. Review points: where each landed, and what was rejected

**Exactness and device-memory review (F1-F10).** Every point was checked against the code (Appendix B).

| Point | Verdict | Where |
|---|---|---|
| F1 eager tail-only hit deallocates None | Applied | §2.0.1 item 4 (loops), §2.0.4, §2.2 eager gate, R21 |
| F2 a hit is not tied to `start_pos`; bad hits propagate | Applied, in an equivalent form (see below) | §1 item 2, §2.0.1 items 2c and 4, §2.0.4, P0a 10, chained gate, R22 |
| F3 restore copy may compile after parking | Applied, extended with a no-program restore | §2.0.1 item 4, bring-up check, program-free audit, §5 item 6, R20 |
| F4 `general` already runs bf8 KV | Applied, one part partially (see below) | §2.0.3, §2.0.4, §2.3, install line |
| F5 synchronous scheduling and Lever N collision | Applied; the joint contract is deferred to S3 | §2.0.1 item 2 (install assertions), item 4 (RoPE anchor), §2.5, §3 item 3, R24 |
| F6 second publish path through `AsyncScheduler._update_request_with_output` | Applied | §1 item 2, §2.0.1 item 2b, P0a 9, R2 |
| F7 block size can change after `load_model` | Applied | §2.0.1 item 3, bring-up, R25 |
| F8 registry and block cache evict independently | Applied | §2.0.1 item 2d, P0a 12, counters |
| F9 C2 pads page tables with `blocks[0]` | Applied | §2.5 table, §3 item 4, R16 |
| F10 provenance, graft citations, R12, sampling citation | Applied; R12 downgraded to "mitigated", not "resolved" (see below) | Citation roots, §2.0.4, R6, R12 |

**Serving review (S1-S11).** SIM was re-run: the 1-8-agent table with and without reuse, the doubled pool and the admission policies. It reproduced the reviewer's figures.

| Point | Verdict | Where |
|---|---|---|
| S1 pool thrash at 8+ busy agents; the cap must count sessions | Applied | §1 "Where it stops working", TTFT table, §2.2 platform 2, §2.3, R9 |
| S2 health probe turns overload into a reload loop | Applied; the priority-policy option is not adopted (see below) | §1, §2.2 platform 3, overload drill, R23 |
| S3 affinity is the likely default; context mismatch | Applied: separate TT id for v1, affine routing later at 4-8 days | §2.2 platform 1 and 7, R10, D-P3 |
| S4 tenant isolation; salt in the wrong layer | Applied in engineering; the policy framing partially (see below) | §1 item 2, §2.0.1 item 2a/2b, §2.2 platform 4, R11, D-P2 |
| S5 observability does not exist in production | Applied | §2.2 (metrics export, platform 5), R26 |
| S6 attempt-level grants, pins and cancellation | Applied | §2.0.1 items 2a, 2c and hooks, P0a 10-11, R14, R27 |
| S7 engine-death blast radius and kill switch | Applied; the kill switch is a different mechanism (see below) | Token check moved to the scheduler, capture failures skip, file-flag kill switch, §2.2 platform 8 |
| S8 restarts wipe warm state | Applied | Warm-reload drill, probe grace, R29 |
| S9 uncoupled LRUs, host memory, CPU-load sensitivity | Applied; the slab is optional | §2.0.1 items 1 and 2d, timing under CI load, R13 |
| S10 estimates | Applied, summed with F1-F10's additions and de-duplicated | §1 effort, §2.2 tables, §3 totals |
| S11 billing change; platform-wide isolation | Applied | §2.2 platform 6, D-P2, D-P5, R28 |

**Rejected in whole or in part, with reasons**

1. **F2's "look up the checkpoint with the row's own `block_hashes[start_pos/64 − 1]`".**
   - *Replaced by an equivalent check.* The model has no block hashes without another runner change.
   - The committed grant carries its key. The model asserts `grant.Q == start_pos` and that the request id is the row's. That catches the same stale-grant case.
2. **F4's "the KV audit must compare packed bf8 tiles, including the exponent section".**
   - *Partially adopted.* The audit compares the values `to_torch` unpacks, which are what every reader sees, plus the slot bytes.
   - A packed-byte comparison needs a raw-tile read API that this revision could not confirm in ttnn (**UNVERIFIED**). It will be used if one exists.
3. **F5's "hits must count toward Lever N's one-in-flight limit".**
   - *Deferred, not rejected.* G1 refuses to run beside Lever N at all, so the rule belongs to the S3 joint contract (+1-2 days there).
4. **F10's "R12 is resolved".**
   - *Downgraded to mitigated.* Rows do run strictly in sequence, but row order follows the input batch's row indices, not admission order (**UNVERIFIED**).
   - So the same-step rule is still needed, and it makes row order irrelevant.
5. **S2's option of `--scheduling-policy priority` with the probe at top priority.**
   - *Not adopted as the fix.* By the reviewer's own caveat it still needs a free slot.
   - It also changes admission order for all traffic on an engine whose queue behaviour SIM and the gates characterise under FCFS.
   - A non-queuing liveness signal, or a longer TT timeout plus a grace period, is kept instead.
6. **S4's framing that isolation is a "hard requirement, not decision D-P2".**
   - *Partially adopted.* The design now defaults to isolation and fails closed, which is the reviewer's engineering.
   - No user statement or ADR making it a platform-wide requirement was found. So D-P2 remains as a confirmation, plus the question of extending it to the Spark.
7. **S7's kill-switch mechanisms: allow-list `QWEN_C2_PROFILE`, or ship `general` and `general-prefix` side by side.**
   - *Replaced.* A file flag on the persistent `/models` mount, polled by the scheduler wrapper, needs no platform change and no release.
   - Allow-listing the profile env stays PLAN R7's item.
8. **S9's "use a preallocated host slab".**
   - *Made optional.* The fragmentation risk is an estimate. The byte-bounded LRU with pins already bounds memory, and the soak gate measures RSS.
9. **S10's G1 total of 17-32 days (21-40 on a shared id).**
   - *Revised to 21-39 (25-47).* This revision sums both reviews' engine additions, removing the overlap (per-step pins appear in both), and keeps the harness corpus and platform items. The reviewer's figure counted only the serving additions.
10. **S1's "rewrite R9: the fix for more than 5 agents is decode slots plus the session cap, not G2".**
    - *Applied.* Raising decode slots (`max-num-seqs` on `general`) is left out of scope as a decode-throughput lever; this design neither sizes nor gates it.

**Corrections found in this revision (not raised by either review)**
- `PLG/platform.py:357` is in `_collapse_parallel_config_to_single_process`, the lane path. The one-process claim for DP=1 rests on `V/config/parallel.py:915-916`.
- `supports_prefix_caching` is read at config time (`PLG/platform.py:1134-1147`). So `QWEN_PREFIX_REUSE` must be set in the API-server process as well. The contract's `.pth` hook does that.
- `general` takes the traced chunk loop because its default `trace_mode` is `"all"` (`PLG/worker.py:198`; `PLG/model_runner.py:1795`). C2's `decode_only` captures no chunk trace, so it takes the eager loop (`IMG/qwen36_vllm.py:365-370`).
- The preferred restore is `copy_host_to_device_tensor`, which needs no program at all.
- TT serving was pulled on 2026-09-26, so G1's production step is a new placement.

## Appendix B. What this revision re-read and re-ran

- **F1:** `IMG/model.py:2300-2345,2442-2480,2482-2598`; T32 `scripts/ci/lever_n_model_patch.py:60-150`, `scripts/ci/prefill_prefix_resume.py:50-90`.
- **F2, S6:** `V/v1/core/sched/scheduler.py:636-650,680-790,820-930`; `PLG/scheduler.py:1-206`.
- **F3:** `IMG/model.py:1290-1377,1719-1812`; `IMG/gdn/tp.py:490-610,790-840`; `IMG/qwen36_vllm.py:360-405`.
- **F4:** `C2/docker/qwen-c2-serving.Dockerfile:25-95`; `C2/scripts/ci/serving_c2_contract.py:1-130`, `qwen_c2_profiles.json`; `IMG/model.py:2678-2690,2738-2765`; the `QWEN_SDPA_BF8` sites in `IMG` and `GRAFT` `attention/tp.py`.
- **F5:** `PLG/platform.py:60-107,1036-1085,1086-1158`; LEVN (grep: `:36,53-57,94-115,125-135`).
- **F6:** `V/v1/core/sched/async_scheduler.py:40-80`; `V/v1/core/kv_cache_manager.py:200-248,440-465,612-632`.
- **F7:** `V/v1/executor/uniproc_executor.py:60-75`; `V/platforms/interface.py:565-640,750-825`.
- **F8:** `V/v1/core/block_pool.py:40-75,545-600,685-700`.
- **F9:** `C2/scripts/ci/serving_runtime.py:350-366`, `serving_page_binding.py:60-100`; `IMG/model.py:1970-1982`.
- **F10:** `sha256sum` of IMG's five files against `C2/docker/qwen-c2-graft/source.sha256`; grep of `GRAFT/gdn/tp.py`, `GRAFT/layer.py`.
- **vLLM config:** `V/model_executor/models/config.py:560-600`; `V/config/vllm.py:2183-2228`; `V/config/parallel.py:860-920`; `cache_salt` in `V/entrypoints/openai/chat_completion/protocol.py`, `V/v1/request.py`, `V/v1/core/kv_cache_utils.py`.
- **Plugin runner and worker:** `PLG/model_runner.py:1055-1090,1262-1275,1780-1800`; `PLG/worker.py:360-380`; the `trace_mode` grep.
- **Serving review, TS:** `py/serving/engine.py:565-590,1275-1305`; `py/serving/manager.py:280-356`; `py/serving/server.py:400-420,2931-2960,3020-3065`, the `_health_monitor_from_env` grep; `deploy/systemd/node.env.example:110-130`; the drop-in grep; `py/serving/sglang_engine.py:160-190,232-242`; `crates/thatch-store-postgres/src/registry.rs:176-218`; `crates/thatch-spine/src/serving_backend.rs:690-712,800-818`; `crates/thatch-spine/src/tunnel.rs:78-90`; `crates/thatch-store-postgres/src/billing.rs:1606-1616`; `py/serving/metrics.py:50-75,128-150`.
- **Serving review, ADM:** `crates/thatch-admin-http/src/gateway.rs:30-48,830-866,1405-1420`, the `tenant_id` and `spawn_blocking` greps; `crates/thatch-administration/src/gateway.rs:28-55`.
- **Other sources:** SKILL in full; `MEM/goal-200tps-concurrent.md:1-30`; PLAN:60-72,740-760; T32 `docs/combined-prefix-cache.md:1-70,100-145,195-211`, and the grep of T32 `docs/` for prefix-cache material; `C2/scripts/ci/serving_fast_policy.py:100-116`, `serving_lifecycle.py:255-300`.
- **SIM re-runs:**
  - `run(4|6|12, 15 s)` at one seed;
  - 1/4/5/6/8 agents × 15/60 s, with and without reuse, at three seeds;
  - `sim_pool2.py` and `sim_lpm.py` in full.
  - All matched the reviewer's figures.
- **Side effect reported by the serving reviewer, not verified here.** Stopping a slow sim run, the reviewer ran `taskkill //F //IM python.exe`, which killed every `python.exe` on the host (PIDs 34896, 41860, 37544). Two of them may have belonged to other sessions or workflows and may need re-running.
- **Housekeeping in this revision.** One `git show` dump of the admin-http gateway landed in the Git-Bash `/tmp` and was deleted. Nothing else was written outside `JOB/prefix/` and this document.
