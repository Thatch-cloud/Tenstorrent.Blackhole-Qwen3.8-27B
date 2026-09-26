# C2 serving for real: S1, S2 and G1 results to date

2026-09-27, a living record: update it at each checkpoint. It covers the programme that takes the C2 fast
path from a benchmark shape (four users at exactly 131072 tokens) to real coding traffic through the
Thatch platform: any prompt up to about 123k tokens, answers up to 16k, up to four users arriving at random,
exact output, and faster per user than `general`.

Every hardware run below is a `qwen-c2-serving.yml` run on the M+A pair, dispatched by an
`experiment/c2-serving-vN` tag. Card-level qualification runs are `qwen-card-b.yml` (`card-b-vN`) on
card B, or the `cardm` action on card M. Run ids are GitHub Actions run ids in this repository.

The designs are in this directory:

- **The plan:** `docs/c2-serve-for-real-plan-2026-09-26.md` (revision 2): S1 any request, then S2 packed at
  any position, then R readiness, then cutover.
- **S2:** `docs/s2-design-2026-09-26.md` (revision 2): K64j extent readers, the gate order, the exactness
  policy.
- **G1 prefix reuse:** `docs/tt-prefix-reuse-design-2026-09-26.md`: the vLLM prefix cache plus
  model-owned GDN checkpoints, and what the platform must provide.

## Short answer

| Stack | 1 user | 4 users at once | State |
|---|---|---|---|
| `general` (baseline) | 18.6-19.3 tok/s | 12-13 tok/s each | what the TT image serves by default |
| S1 (C2, any prompt length) | 24.5-37 tok/s, mean 30.3 | 7-10 tok/s each (rounds are sequential) | every S1 gate passed |
| C2 packed, all four at exactly 131072 | - | 29.4 tok/s each (162 ms rounds, 68.7 aggregate) | the benchmark shape only |
| S2 (packed at any position) | - | ~28-29 tok/s each, estimated from 164-167 ms rounds, G3 shape only | kernel side qualified; serving gates in progress |
| G1 (`general` plus prefix reuse) | general's decode | general's decode | exactness and lifecycle passed; TTFT timing running |

- **S1 helps a lone user and hurts at four.** Without a packed block, four live users take turns, so each
  gets 7-10 tok/s against general's 12-13.
- **S2 is the fix for four users.** Its real-traffic rate, with four users at mixed lengths (G4 mixed),
  is not measured yet.
- **G1 does not speed up decode.** It cuts time to first token on long conversations, where the design
  expects about 2-3 s instead of 13-20 s. Measured so far, in bring-up: a 4033-token prompt 1.20 -> 0.70 s,
  and an 8962-token prompt 2.81 -> 2.28 s.
- **Real traffic reordered the programme** (2026-09-26). The metering sample was 42k prompts, 92% of them
  prefix hits. That makes prefix reuse (G1) the first production deliverable, with S1 and S2 in parallel.
  One TT pair saturates at about 4-5 busy agents (four decode slots), so TT is a bounded opt-in backend
  beside the Spark.

## 1. S1: any request (branch `serving/c2-coding-s1`, head 1a9e1878)

**What it is.**
- The `c2` profile builds per-request engines with captures of 1, 2 or 4 rows.
- It has no packed block: `QWEN_FAST_PACKED_STEP=0` and `QWEN_FAST_PADDED_BLOCK=0`.
- Admission allows one fresh prefill per step (`serving_prefill_admission`).
- Parser M is in.
- It caps prompts at 123136 tokens and answers at 16384.

**Bring-up.**
- Exact, run 36210542605: 4 x 131072 real text, IDENTICAL to v235 inside the serving image, in the agent's
  container shape. 162 ms rounds, 29.4 tok/s per user.
- `c2-gate`, run 36211140225: IDENTICAL to v235 on any-request engines, with 1.32 GB free per chip after
  four engines.

**Defects found on the way, each with its fix:**
- **v23 (run 36211578069), the concurrent arm died.** The platform's `TTScheduler` replaces the fast path's
  one-in-flight scheduler class and batched three fresh prefills.
  - Fix bc522a67: wrap `_schedule_prefill_only` with the one-fresh-prompt rule.
  - Fix 9aefc6e3: the gate requires the admission line.
- **v26 (run 36218104858), the fourth engine ran out of DRAM.** Attach leaves 4.17 GB per chip and an engine
  takes 0.80-1.35 GB. The 3.84 GB packed block could never engage under `c2` anyway.
  - Fix 71b2ccdc: `c2` drops the block.
  - Fix 559d2ef9: admits single gate/up without a block.
- **Short prompts ran at 3-7 tok/s.** The feature history was rebuilt at a new shape every round while
  there were fewer than 2048 rows: about 1 s of host time per round, and per-shape JIT.
  - Fix: skip the ramp write under the steady condition (afeff689, cherry-picked as 5866eb14).
- **The installed-vLLM CPU job stopped at a stale test** because the shell ran with `-e`. The modules after
  it never ran, but the job read green.
  - Fix bbfeb96a; v8 (run 36222171688) ran every module.

**Gates, all PASSED:**
- **G4, run 36220649220 (v28).** 9/9 lengths from 60 to 120000 with 4096 out: concurrent == solo, IDENTICAL.
- **Lifecycle, run 36224065261 (v30).** Drop during the build, drops at 4, 3 and 2 live, a cancel during a
  60k prefill, a triple drop, `max_tokens=1`, freed seats: all IDENTICAL.
- **G5 memory, run 36225292333 (v31).** 4 x 123k prompts: floor 4.91 GB per chip. 4 x 60-token prompts
  with 16k answers: 2.74 GB.
- **G7 platform replay, run 36227190700 (v32)** on thin layer `thatch-serving-tt:e570ee2`.
  - The runtime's health monitor did two false recovery reloads during it. An unloaded engine counted as a
    failed probe, and the counter was not reset by a successful recovery.
  - Fixed in Thatch.Server #2645 (6e89e32f). The replay now fails on any unrequested recovery.
- **S1 image:** `tt-vllm@sha256:b41de8a85278414bbf6f73953704f5e5d5057012bf75a7da94ac34f4e4597e5c`
  (`s1-bbfeb96`).

## 2. S2: packed at any position (branch `s2/c2-packed-any`)

### 2.1 Kernel side (K64j), qualified

K64j adds flag 0x20: a runtime extent with `cur_pos = E - 1` and `E = (start // 256 + 1) * 256`. It replaces
K64i's compile-time family. The graft's `_ttnncpp.so` sha256 is `152951c1...` (34 QWEN strings, K64i's 30
kept).

| Check | Run | Result |
|---|---|---|
| P0 pass 1 (watcher) | 36218136852 | 70/70 |
| P0 pass 2 | 36218529407 | 638 comparisons + 228 liveness, 114 families, 0 failures. Runtime extent 96 us at E=2304 to 1193 us at E=131328, 8-10% under compile-time |
| K1/K3 watcher | 36223057469 | extent 80/80, mixed 15/15, trace 20/20, fence 12/12, skip 8/8, refusals 7/7 |
| K1/K3 full, seeds 0-4 | 36223820488 | 2387 comparisons, 0 failures |
| **K2 full (decides the exactness policy)** | 36239779235 | 1980/1980 tickets, 29730/29730 rows bitwise; X7 500/500, Z 900/900 over 15 families |
| CB2b reader, full | 36260236826 | 3762 comparisons over 56 families, 0 failures; `extent_attention_replay.py` sha 5633fc3a |

K2 passing bitwise means the strict policy stands: concurrent == solo, and a reproduced divergence fails.
No relaxation decision was needed.

**A provenance trap found here** (run 36255706983): the image serves the frozen recipe's staged
`attention_mask_replay.py` (6a31981c), not the checkout's (3e431742). The evidence stood, because the two
differ only in `validate_ticket`, which the extent path never calls. The fix was to pin the served copy in
the test (578ce33f).

### 2.2 Serving gates (image `s2-c74b29e`)

| Gate | Run | Result |
|---|---|---|
| M1 build + warm (a1, v55) | 36263302227 | PASS |
| M2 warm-off + exact bring-up (a2, v56) | 36269458906 | PASS: IDENTICAL to v235, zero new kernel-cache entries |
| M3 G3 control (a3, v57) | 36270917139 | FAIL on timing only: every arm IDENTICAL to v235; pair ratios 1.026 and 1.003 |
| M3 G3 control (a3, v59) | 36272682217 | FAIL on timing only: 1.042 and 1.014 |
| M3 G3 control, fixed gate (v61) | 36276533585 | queued |

**The G3 timing failure is the audit, not the feature** (attribution over 144 paired four-live rounds; the
arms are token-identical, so round k pairs with round k):

- Flag on minus flag off, whole round: +4.0 ms median. The audit's own logged ms: +1.8.
- Rounds with a slow audit (median 8 ms) were 4.4 ms slower than their twins beyond the audit's own time.
  Staging, the readbacks and the draft all slowed in those same rounds. Rounds with a quick audit (1.4 ms)
  were only +0.6 ms.
- The feature's own work is about 0.6 ms a round (0.4%). It adds no fence. Verify-time staging writes the
  same buffers as flag-off at this shape.
- The gate's statistic, median round minus median audit, under-corrects. The audit time is bimodal, and its
  cost reaches beyond the ms it logs.

**Fix a25f67b6 (gate only, no image rebuild).**
- The control pairs' flag-on arms run without `QWEN_FAST_EXTENT_AUDIT`. They are judged on the raw median
  four-live round, still at 1.02.
- An audit line in a timing arm refuses the pair.
- One extra audited arm, `control-audit`, judges exactness and a clean, complete audit, and is never timed.

**Next, in order:** forced-cap (M4), G3b below C (M5), lifecycle and lifecycle-arrival (M6), G4 mixed, short,
boundaries and staggered (M7-M10), G5 churn (M11), then the G7 replay (M12).

## 3. G1: `general` plus prefix reuse (branch `prefix/g1`, image `g1-35fdb0c`)

**Design.** vLLM's prefix cache covers attention. The model owns GDN recurrent-state checkpoints every 2048
tokens:
- a hit is trimmed back to the last checkpoint boundary;
- the hit rate is read from the model's L - Q, not from vLLM's `prefix_cache_hits`, which counts before the
  trim;
- tenancy fails closed: only a signed `qps1.<tag>.<mac>` `cache_salt` reuses, and an unsalted request gets
  no hit and publishes nothing.

**P0.** The chat template keeps past reasoning by default, so append-only turns extend the prefix exactly.
`reasoning_effort` high, minimal and max fail the request in vLLM 0.25.1.

| Step | Run | Result |
|---|---|---|
| Bring-up (v46) | 36245488518 | 3/3 cold/hit IDENTICAL, baseline IDENTICAL, 0 programs compiled on capture or hit; 15.48 GB free per chip beside a 262,400-token KV pool |
| Exactness, traced + audit (v47) | 36246961161 | PASS (10/10; boundaries 2047/2048/2049 right) |
| Exactness, eager (v47) | 36246961161 | FAIL: a real device hang, see below |
| Exactness, eager, fixed (v50) | 36255601083 | PASS 10/10 |
| Bring-up + traced exactness on the fixed image (v53) | 36261168369 | 20/20 identical; chain reuse reaches Q=53248 at L=63479 (84%) |
| Lifecycle (v54) | 36262994507 | PASS: evict 19/20 identical + 1 not-comparable-by-batching, solo 16/16; store 8/8; tiny 4/4 |
| Timing at 1/4/5/6 busy agents (v60) | 36274183527 | running |
| Platform replay through a `:tt` thin layer | - | next |

**The eager hang.**
- **Symptom:** "MMIO per-op timeout: 4B load took 49571 us (budget=2 ms)" in `forward_prefill_paged`.
- **Cause:** under `trace_mode decode_only` the plugin warmed no prefill. The first request then compiled
  133 programs after the decode traces were parked, which is the known second-request hang.
- **Fix 35fdb0cd:** warm the eager prefill before parking, and pad the eager page table to the warmed width.
  Any post-warmup compile now fails a gate row.

**`general` batched decode is not batch-invariant** (lifecycle v48, run 36251045616).
- Concurrent cold controls differ from solo cold runs even with no prefix reuse, so a concurrent hit cannot
  be judged against a solo run.
- The judge (062857f8) requires solo pairs to be IDENTICAL. It judges a concurrent hit against a
  batch-matched cold control, and reports NOT_COMPARABLE only when the same run proves the baseline itself
  diverged under batching.

## 4. The platform (Thatch.Server), merged 2026-09-26

- **#2624:** vLLM prompt and cached-token metrics.
- **#2627:** the TT health probe reads the engine's step counters first, with a 300 s grace after a reload.
- **#2628:** session cap: 4 sessions, 600 s idle TTL, 429 over the cap; `THATCH_SERVING_SESSION_CAP=0`
  disables it.
- **#2629:** cached-token reporting on TT, billed at the Spark's cached rate.
- **#2637 (74f52046) and #2638 (b3a1baed):** TT serves as `Qwen/Qwen3.8-27B:tt`. The image bakes the alias,
  and the registry and spine accept `<ref>:<tag>`. The publish lane refuses to move `:latest` to an untagged
  image without `allow_untagged_rollback`.
- **#2645 (6e89e32f):** fixes the health monitor's false recoveries.
- **Prod spine promoted** to `sha256:cf59368d...` (promote run 36228884857, pin PR #2644). The rollback
  digest is `sha256:f983a3eb...`.
- **Still needed for G1 in production:** the admin-http gateway must mint the per-tenant signed salt and
  strip any client-sent `cache_salt` (decision D-P2). The same key must be provisioned to the gateway and to
  the node's `/models/.qwen-c2/prefix-salt.key`. PRs are in preparation.

## 5. Open items

- **G1:** read v60's timing, then publish the base, build the thin layer from Thatch.Server main, and run
  the platform replay asserting `:tt`.
- **S2:** G3 on the fixed gate, then M4-M12.
- **D-P2:** the gateway salt, key provisioning, then the submodule pin and rollout.
- **Fleet verify:** the nightly check is hollow while its conformance API key is unset.
- **Cutover:** needs a fresh go-ahead and an admin session.
