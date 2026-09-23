# Four concurrent streams at 131k, each at single-stream rate: where it stands

2026-09-23. Target set by the user: **four concurrent streams, each with a 131,072-token
window, each at single-stream throughput or better.** The old 200 tok/s-per-user target
is a closed verdict and is not the subject of this note.

Provenance: a read-only workflow (run `wf_8d61a40d-bfd`, 13 agents: six area readers,
each adversarially re-checked against its own cited sources, then a completeness critic).
Of the claims checked, **197 were confirmed, 28 corrected and 2 refuted**; neither
refutation is load-bearing. The three scaling runs below were measured the same day.
`docs/batch-spec-ceiling-2026-09-19.md` - which the critic identified as the most
relevant unread source - supplies the physical floor.

## Short answer

**Not reachable on the current m3native packed design. Not ruled out by physics.**

> **Update, v101 (run 35809096034): four streams at a 131k window now fit and run** - 4/4
> token-exact over 256 tokens, 1.30 GB free per chip. Capacity is solved; the per-user rate
> is not: 17.2 tok/s each against 47.5 single-stream (36%). See the last section.

The DRAM-bandwidth floor for four users at 131k is **75.8 ms per round** at bf8
(65.2 ms at bf4) - `docs/batch-spec-ceiling-2026-09-19.md:9-10`, every non-bandwidth
overhead set to zero. The current design is predicted at roughly **355 ms** for that
round, about **4.7x above the floor**. So the walls below are overwhelmingly walls of
efficiency and structure, not of bandwidth. The one partly-physical wall is DRAM
capacity, and even that is mostly fast-path overhead rather than the KV cache itself.

## The single-stream bar

| context | path | tok/s | round | source |
|---|---|---|---|---|
| 32k | m3native, current image | **48.0** | 125.0 ms median | run 35783241354 (v84), token-exact |
| 32k | m3native, older image v41 | 44 | ~138 ms | `docs/sideways-target-2026-09-22.md` (superseded) |
| 131k | frozen/combined lineage | 53.69 | 186.2 ms cycle, 10.0 tokens/block | run 35173979225 |
| 131k | **m3native** | **unmeasured** | - | run 35787268942 (v86) in flight |

The 53.69 figure is **not** the bar for this target: it uses a different drafter (DSpark
full-history against DFlash2 with a 2,048-row window), a coding prompt against the
synthetic periodic one, an offline harness against vLLM streaming, and different token
accounting. Its 10.0 tokens per block is the 131k row; 9.31 is the 16k row.

## Measured scaling at 32k (identical image, flags and K64 graft; no Lever N)

| users | median round | per-user rate | verify trace | token-exact | run |
|---|---|---|---|---|---|
| 1 | 125.0 ms | 48.0 tok/s | (unpacked step, 98.3 ms) | 1/1 | 35783241354 |
| 2 | 164.0 ms | 40.8 tok/s | 103.3 ms | 2/2 | 35783217176 |
| 4 | 267.5 ms | ~20-23 tok/s | 172.4 ms | 4/4 | 35728931019 |

**Two users reach ~85% of single-stream; four do not.** At 32k, parity means taking the
four-user round from 267.5 ms to ~125 ms, a 2.14x gap. `gate_passed` is false on the
1- and 2-user runs only because the gate requires the native-graft markers, which engage
only for the 64-row block.

Round composition at four users (`scripts/ci/round_breakdown.py`, top-level phases only):
packed_verify 179.8, prepare_proposals 39.5 (containing propose_pair 6.1), packed_commit
35.9, propose 7.7, unaccounted 4.6 ms. Two corrections from the verifiers: packed_commit
is mostly **draft-history publication** (prepare_history ~31.7 ms per round), not GDN
commit (~1.8 ms); and prepare_proposals is mostly device time waiting on two traced
two-user draft passes, with ~6.1 ms of host enqueue.

## The five walls at 131k

**1. DRAM capacity.** Four users at 131k are about **6.1 GB per chip short** at bf8
(measured base: v83 logged 32.88 of 33.64 GB allocated after four engines, 0.76 GB free;
bf8 KV adds 6.85 GB per chip from 2,068 to 8,212 blocks). bf4 KV still leaves ~1.8-2.0 GB
short. Break-even today is ~43.9k tokens per user at four users. **This is mostly fast-path
overhead, not KV:** the plain path fits 4 x 163,840 (run 35410320811, 11.41 GB per chip of
bf8 KV beside the weights), which bounds its non-KV at <= 21.69 GB per chip against
30.58 GB on the fast path - at least **8.9 GB per chip of extra non-KV allocation** on the
fast path, of which ~2.9 GB is per-request engines. About 12 GB per chip of overhead has
never been itemised. One user at 131k fits with ~6.7-6.8 GB free; two are ~0.5-0.9 GB short.

**2. Per-user round cost.** Task #39, one weight pass for four users, is **done** and
engaged in v83 (native_m3 and native_attn; the replaced binders log zero calls). The
shared weight matmuls are ~34-43 ms of the 172 ms trace, near the DRAM floor. Almost all
remaining four-user cost is per-user: SDPA over the user's own K/V (~40 ms at 32k, growing
with context), GDN recurrence and launches (~37 + 25 ms), in-trace data movement and
fold DMA (~26 + 14.5 ms), the draft passes and draft publication. Levers that need no
kernel rewrite - re-enabling GDN_USER_BATCH (-14 ms, blocked by a warmup hang), batched
draft publication (-20 to -27), one four-user draft pass (-10 to -20), a 12-row verify
width (-15 to -25; positions 11-15 are never accepted on the synthetic prompt) - are
estimated to reach **175-203 ms** at 32k, ~30-35 tok/s per user. Short of 125 ms.

At 131k the attention term grows. Carrying the frozen lineage's measured slope
(a = 0.2225 ms per 1k tokens per user) gives a four-user round of ~355 ms (333-377)
against ~147 ms for one user: **~0.41 of single-stream per user**, worse than 0.47 at
32k. With every context-independent per-user cost set to zero, shared cost plus four
users' attention is already 178 ms > 147 ms; parity needs the attention slope at or
below ~0.15 even then. That slope is **not measured on m3native** - see "next".

**3. Packed-block structure.** The packed block admits a user only at positions within
the last 256 of max_model_len (`attention_mask_replay.py:30-32` via `packed_verifier.py`),
and output is capped at 256 tokens. Four real 131k conversations at different fill levels
fall back to the sequential step, which measured 297 ms at three users (v83).

**4. Prefill and admission.** Four simultaneous fresh 131k prompts cost at least 4T under
any schedule. T is 61.7 s measured on the combined path and estimated at 69-72 s on
m3native (an unsourced ~310 s also circulates; neither is measured on m3native). Under
the per-user metric completion_tokens / (wall_s - ttft_s), earlier users sit frozen
through later users' prefills; at 32k that already reduces users 1-3 to 5.0 / 7.1 /
10.2 tok/s. A single-stream TTFT for all four is physically impossible with simultaneous
arrival, because the total prefill work is fixed; it is reachable only with staggered
arrivals or faster prefill.

**5. Qualification.** A 131k request on the baked image runs on the retained 32768
component evidence: `qualify()` returns requested_geometry_qualified=False and nothing
reads it. The m3native "token-exact" check covers a ~64-token (240-character) prefix, not
the full output.

## What is reachable now

- **One user at 131k** fits in DRAM and, on a reading of the baked code, is not refused.
- **Two users at 32k at ~85% of single-stream** (40.8 against 48.0 tok/s), measured.

## Corrections this work recorded

- The single-stream bar on the current image is **48.0 tok/s at 125 ms**, not 44 at 138.
- The documented 131k refusal (`docs/lever-n-131k-attach-arm.md:46-54`) describes the
  repo's pre-stage source, not the image that runs; on image 67a28229 the T16 gate
  compares against the request context taken from the environment.
- 65536 numerics are qualified on hardware (run 35658618872), not failing.
- `round_breakdown.py` double-counted the nested propose_pair phase; fixed.
- Tokens per round are counted four ways across the evidence (5.89, 6.1, 6.89, 7.15-7.17);
  predictions using the highest figure over a median round run ~15% high.

## Next

Run 35787268942 (v86): one user at 131,072 on the exact v84 configuration, changing only
the geometry. In one run it answers the five unknowns every 131k prediction above rests
on - whether the m3native path reaches decode at 131k at all, the attention slope on this
kernel, tokens per round at 131k, prefill time, and whether non-KV DRAM stays flat - and it
produces the single-stream 131k bar the target has to meet. If the step implied slope is
at or above ~0.15 ms per 1k tokens per user, four-user parity at 131k is structurally
excluded on this kernel family; below it, parity becomes a question of per-user overhead
and capacity rather than of attention.

## v86 result: the real 131k pin, and three unknowns answered

Run **35787268942 (v86)**: one user at 131,072 on the exact v84 configuration.

It **attached and served** (`ready: true`), **prefilled all 131,072 tokens**, and died on
the first decode KV write:

```
ordered_cache.py:53 validate_shapes
ValueError: Paired position vector and page-table rows required
  <- attention_batch.OrderedCacheWriter <- attention/tp.py _decode_from_prep
  <- layer.forward <- model._forward_decode
```

The clause is `not 1 <= pages[1] <= min(1024, cache[0])`: the native ordered-cache writer
refuses a page table wider than **1,024 entries**. At 64-token pages that is exactly
**65,536 tokens**. 131,328 needs 2,052. The same cap refuses 65,792 (1,028 pages), so
**the m3native path has never been able to decode above ~65k**, whatever the 65k
numerics qualification says. `test_ordered_cache.py` asserts 1,025 is rejected on
purpose, so the cap was deliberate; whether it reflects a kernel limit or only audit
coverage is under investigation (workflow `wf_8cc3f3d3-0a4`) before anything is changed,
because a wrong KV write corrupts output silently rather than failing.

**This pin was missed by the static reading** ("no Python validator in the request path
refuses 4 x 131072"). A sweep afterwards of scripts/ci, the grafted model files and the
image's own model.py / qwen36_vllm.py found it to be the **only** target-side context cap.
The DSpark 8192 caps belong to a drafter m3native does not use; the 2048 caps are
DFlash2's draft window by design; `frozen_context_geometry.py:13` already grows past
1024 pages.

Answered by v86 even though decode failed:

| unknown | result |
|---|---|
| does m3native attach and prefill at 131k? | **yes** - ready, and prefill completed |
| prefill time at 131k on m3native | **64.4 s** (21:36:03.6 -> 21:37:08.1), close to the combined path's 61.7 s; the unsourced ~310 s is wrong |
| does non-KV DRAM grow with context? | **no** - 24.87 GB allocated after attach against a predicted 23.13 + 1.71 = 24.84 GB; 8.77 GB free at one user |
| attention slope, acceptance, single-stream 131k bar | **still unmeasured** - they need a decode step |

The vLLM invocation was verified: `--max-model-len 131328`, `--num-gpu-blocks-override
2052`, `--max-num-seqs 1`. The docker `-e` flags are not echoed in the job log, but the
prefill of 131,072 positions is itself proof `QWEN_FAST_MAX_POSITION` reached the
container: without it `dflash_prefill_window` refuses positions above 65,504.

## v88: 131k decode works, and context length is not the wall

v87 (run 35790454545) still refused - its traceback raised from `ordered_cache.py` line
54, the OLD file's raise, because the file was in neither image copy list and image v93
shipped the frozen bundle's version. Image v94 overlays it (build log: `COPY
scripts/ci/ordered_cache.py` at step 41/51), and `test_serving_image_copy_closure` now
fails on any module changed since the bundle but overlaid by neither list.

Run **35791589709 (v88)**: one user at 131,072, the v84 configuration, image v94. It
**decoded** - the first m3native decode at 131k - with only the harmless shutdown
`empty_cache` error.

| | 32k (v84, 35783241354) | 131k (v88, 35791589709) |
|---|---|---|
| per-user rate | 48.0 tok/s | **47.5 tok/s** |
| median round | 125.0 ms | 128.5 ms |
| decode step | 98.3 ms | 106.7 ms |
| tokens per round (completion / chunks) | 6.74 | 6.10 |
| mean acceptance (vLLM metric) | 6.47 | 6.50 |
| TTFT | 17.2 s | 68.4 s |
| DRAM after engine, chip 0 | - | 26.93 GB, 6.71 GB free (predicted 26.98) |

> **Corrected below ("Where the 4 x 131k verify goes"):** the 0.085 slope compares decode steps across two
> images (v84 67a28229, v88 f369d08c) whose commit and draft phases also moved. The verify trace itself grows
> 0.2245 ms per 1k tokens per user - the frozen lineage's 0.2225 was right.

**The m3native attention slope is ~0.085 ms per 1k tokens per user**: +8.4 ms of step over
98,304 extra tokens. That is **2.6x smaller** than the 0.2225 every earlier 131k
prediction borrowed from the frozen lineage, and **below the ~0.15 threshold** at which
four-user parity at 131k would be structurally excluded. Acceptance does not fall.

**Output at 131k is byte-identical to the 32k run over all 960 characters (256 tokens).**
Those decode positions, 131,072-131,327, are exactly page-table entries 2,048-2,051 - the
entries the new width admits. A misread there would corrupt the most recent K/V and
almost certainly change the continuation. This is strong evidence the writer is correct
at width 2,052. It is **not** the independent qualification the review required: the
prompt is synthetic and periodic, and the check is not independent of the kernel. The
host-predicted hardware probe is still owed.

### What this changes

The previous section predicted ~355 ms for four users at 131k against ~147 ms for one,
using the borrowed slope. With the measured slope:

- single-stream at 131k is **~47.5 tok/s** - the bar is the same as at 32k;
- attention adds only ~8.4 ms per user from 32k to 131k, so a four-user 131k round is
  projected at roughly 267.5 + 4 x 8.4 = **~301 ms**, about **20 tok/s per user, ~0.42
  of single-stream**. (Projection: the packed four-user path uses the native batch-64
  attention, whose context scaling is not yet measured separately.) **Wrong twice, see below:**
  the K64 graft covers attn_decode_prep and nlp_concat_heads_decode only - SDPA is still the
  per-user replay reader - and v101 measured 340 ms, not ~301.

**So the 131k target reduces to the 32k problem plus capacity.** The wall is the ~40-52 ms
per added user of context-independent work - GDN recurrence and launches, in-trace data
movement, draft passes, draft publication - plus the ~6.1 GB per chip DRAM shortfall for
4 x 131k, which is fast-path overhead rather than KV. The DRAM-bandwidth floor (75.8 ms
for four users at 131k) says the hardware has the headroom; the per-user work is
overhead-bound, not bandwidth-bound, so the path to parity is making it overlap rather
than add.

## The KV writer is independently verified at width 2,052

Run **35793799919** (`experiment/ordered-cache-hw-probe-v1`, image v94) - the weight-free
hardware probe the review required. The expected cache is predicted on the HOST from the
page table alone, never read back from a native writer, so a misread shared by the
ordered and native writers can no longer pass.

- provenance: the image's baked `ordered_cache.py` and the checkout's are the same file
  (sha256 `c74799af`), tt-metal `9f9cd4fd`, kernel hashes equal to `HASHES`;
- geometry: 2,064 zeroed BF8 blocks, 16 rows each with its own permutation of block ids,
  every row hitting page entries 0, 1, 1023, 1024, 1025 and 2047-2051;
- **82 checks on hardware, all exact**: 58 full-cache comparisons plus zero-baseline,
  page-upload, pages-unchanged and input-unchanged checks, on both chips;
- the predicted set grows step by step - 16 blocks up to 95 (1,024 control), 157 (2,052
  eager) and 166 (2,052 traced, page table rewritten in place between replays) - with
  **0 predicted-block mismatches and 0 stray non-zero blocks** anywhere in the cache.

So the writer is correct at width 2,052, eager and traced, both chips, including the four
tail entries - independently of the native page-table read. Together with v88's
byte-identical 131k decode, the 131k single-stream measurement stands on a verified
writer.

Still owed from the review, and lower risk now: an in-model permutation-invariance check -
the same 131,072-token prompt under two physical block maps must give bit-identical tokens.

Housekeeping: `ordered_cache.py`'s comment still says 2,052 is not yet qualified on
hardware. Updating it changes the file's sha256 and so forces an image rebuild for the
baked-copy check; it will be corrected with the next functional change to that file.

## Increment 0: GDN_USER_BATCH, and the flag interaction that hid it

`M3NATIVE_GDN_USER_BATCH` had hung in warmup in v77 and v79, which were blamed on the
Lever N model graft. v89 (run 35798375001) - v34's exact flags, no Lever N, image v94 -
hung identically, so Lever N was not the cause. `GDN_STATE_COPY_BATCH` was dead code in
v34's image (e41ef884) and is live now: v34 effectively ran `GDN_USER_BATCH` alone, v83
runs `GDN_STATE_COPY_BATCH` alone, and v89 was the first run with both live.

v91 (run 35799760385) runs `GDN_USER_BATCH` without `GDN_STATE_COPY_BATCH`. It **passes
the gate, 4/4 token-exact**:

| | v83 (STATE_COPY_BATCH) | v91 (USER_BATCH) | delta |
|---|---|---|---|
| verify trace (packed_phase) | 172.4 ms | **157.8 ms** | **-14.6** |
| packed_commit | 35.9 ms | 40.1 ms | +4.2 |
| median round | 267.5 ms | **257.0 ms** | **-10.5** |
| TTFT | 13.6 / 26.5 / 39.6 / 52.5 | 13.6 / 26.6 / 39.7 / 52.8 | - |

The -14.6 ms matches the review's -14 ms estimate. The two flags help different buckets
and hang together; resolving that interaction would recover ~4 ms more. **v91 is the new
best four-user configuration and the baseline for the next increments.**

## Four streams at 131k now fit, and run token-exact (v101)

**Run 35809096034 (v101): four concurrent streams, each with a 131,328-token window and a
131,072-token prompt, passed the gate - 4/4 byte-identical to single-stream over all 256
tokens, against references recorded at 131k.** The KV cache holds 525,568 tokens (9.149 GB
per chip, the ledger's P0 line). Peak DRAM after the fourth user is 32.61 of 33.91 GB per
chip: **1.30 GB free**, against a projected 1.26.

What it cost to get there, all measured by the memory ledger (L0) on image A
(`sha256:0648ca9a...`, commit 0552008b) and each behind its own default-off flag:

| lever | flag | GB per chip | evidence |
|---|---|---|---|
| A2 trace region 512 -> 256 MiB | `M3NATIVE_TRACE_REGION_BYTES` | +0.27 usable | v92: total 33.64 -> 33.91, 4/4 exact, round 255.0 vs 257.0 ms |
| C1 no packed `w_gate_up` | `QWEN_FAST_SINGLE_GATEUP` + mlp/layer graft | -3.209 | ledger P0 item gone |
| A1/C1 no block-stream copy | (implied by C1) | -3.220 | ledger P1 item gone |
| B1 bf8 draft projections | `QWEN_FAST_DRAFT_BF8` | -0.811 (1.877 -> 1.066) | ledger P5 item |
| **total** | | **-7.24 + 0.27** | attach 29.89 -> 22.64 GB at 32k |

The ledger's attach accounting closes: every allocator delta P0..P7 matches its walked
items, and the P7 residual is 0.016-0.023 GB.

### How C1 got through (three runs)

C1 changes target prefill arithmetic: the fused all-gather + SwiGLU matmul is replaced by
the model's own unfused w1/w3 path. `tp_common.py`, where the switch lives, cannot be
grafted - `mlp_down_grid_gate` pins its sha256 and v95 (run 35802496949) died at attach -
so the flag is applied at the three call sites in `mlp.py` and `layer.py` instead. The
unfused 2D prefill branch had never run at TP2 and did not fit L1: v100 (run 35807762937)
found its L1 gate/up outputs (35.7 MB at 2048 rows, N=8704) in the way; with DRAM outputs,
v102 (run 35808212287) found the program's own circular buffers reaching 1,559,424 bytes
against persistent L1 buffers. Under the flag the branch now runs 1024-row slices (the
model's own builder per slice, DRAM outputs, joined on rows). v103 (run 35808667002, 4 x
32k, every flag) and v101 are both **4/4 exact over 256 tokens**: the unfused prefill
produces the same tokens as the fused one on these prompts.

### References are now full-length and context-exact

Every tracked reference used to be 64 tokens, so "token-exact" had covered a quarter of
each stream. v97 and v98 (runs 35803057593, 35803058152) ran bases 1000..1003 one at a time
on one server, 256 tokens each, at 32k and at 131k. Each extends its old plain-path
reference exactly, and the 131k and 32k outputs are byte-identical (the prompt is token ids
base + i % 64). Re-judged against them, v91 and v92 are exact over all 256 tokens too.

### Throughput at 4 x 131k: the fit is not the rate

| | 1 x 131k (v98) | 4 x 32k (v99) | 4 x 131k (v101) |
|---|---|---|---|
| verify trace | - | 157.8 ms | **246.3 ms** |
| median round | - | 250.0 ms | **340.0 ms** |
| tokens per user per round | - | 7.15 | 6.81 (bf8 draft) |
| per-user decode, steady state | 45-53 tok/s | 22.2 tok/s | **17.2 tok/s** |
| aggregate | ~48 tok/s | ~89 tok/s | **~69 tok/s** |
| TTFT, 4th user | - | 52.9 s | **275.9 s** |

At 131k each concurrent stream decodes at about **36% of the single-stream rate** (17.2 vs
47.5 tok/s); the four together deliver about 1.45x one stream. The verify trace grows by
88.5 ms from 32k to 131k (the attention reads the planner priced), and prefill is serial:
~68 s per 131k prompt, during which the others' decode stalls, so the fourth user waits
276 s for its first token.

The bf8 draft costs acceptance: 7.15 -> 6.83 tokens per user per round at 32k (-4.5%, over
the planner's 3% bar). It is 0.81 GB of the 1.30 GB margin, so it cannot simply be dropped
at 131k without leaving ~0.5 GB. v104 (C1 without B1, 4 x 32k) attributes the rest of
v103's slower round.

### What remains for "single-stream rate or better"

Capacity was the gate for four streams at 131k, and it is open. The rate is the remaining
wall and it is the one this note's short answer described: the round at 4 x 131k is
340 ms against a DRAM floor of 75.8 ms, so it is structure, not bandwidth - per-user
proposal/commit work that does not overlap (81 ms of the 340: proposals 36.4, commits
36.2, propose 8.7), a verify that reads each user's KV separately, and serial prefill. The
levers that would move it are the ones the planner listed as beyond the evidence-backed
plan: a second command queue / sub-device overlap of the per-user phases, a cross-user
SDPA kernel, and prefill-decode interleave (Lever N) for TTFT. WY-form GDN is faster but
not token-exact, which is a user decision.

## Where the 4 x 131k verify goes: each user's KV is read four times per layer

A 9-agent diagnosis with adversarial verification (workflow wf_a79442c7-ffb) settled why the
verify trace grows 88.5 ms from 4 x 32k to 4 x 131k (157.8 -> 246.3 ms; matched pairs v103 ->
v101 +88.47 and v104 -> v105 +88.53; the trace does not depend on the capacity flags).

**Mechanism, from the code.** `packed_verifier.py:494-500` builds the block with
`replay_group_rows=4`, so `attention_head_fold.parallel_groups` splits each user's 16 query rows
into a batch-3 and a batch-1 bundle of 4-row groups. `attention_replay.py:49` gives every batch
entry the same user's page table, and `attention_parallel.py:17-19` calls
`paged_scaled_dot_product_attention_decode` with `is_causal=False`, a dense full-capacity mask and
no `cur_pos`, so every entry walks the whole capture family (context + 256 keys). The sdpa_decode
factory assigns cores per batch entry with no KV sharing - K multicast exists only on the MLA path.
**Each user's KV is streamed four times per layer.** The device-profiled bench shows it at 32k:
batch-3 428 us against batch-1 191 us for the same work per core.

**The packed path does not scale worse than single-user.** The single-user verifier uses the same
reader: its trace grows +22.06 ms per user (65.61 -> 87.67 ms, v97 -> v98, one image), and
4 x 22.06 = 88.2. The single-user ROUND grew only 3.5 ms (v84 -> v88) because commit and draft
phases shrank at the same time on a different image. An earlier reading of mine - that the packed
path scaled ~25x worse than one user - compared a whole round across images with a trace on one,
and is withdrawn.

| part of the +88.5 ms (bytes-proportional; not yet profiled at 131k) | ms |
|---|---|
| first read of the added KV (unavoidable) | ~18-20 |
| reads 2-4 of the same KV (redundant) | ~54-59 |
| dense bf16 mask, read by every batch entry, non-zero only in the last 256 keys | ~9-17 |

SDPA is ~39.6 ms of the round at 4 x 32k (619 us per user-layer) and ~128 ms at 4 x 131k. One read
per user would cost ~23-26 ms at 4 x 131k.

**Two cheaper routes are closed.** v107 (run 35812188001) tried 8-row groups
(`QWEN_FAST_REPLAY_GROUP_ROWS=8`: one batch-2 bundle, two reads): the SDPA program's static
circular buffers grow to 1,835,968 B against 1,572,864 B of L1 and the engine dies at warmup, even
with the tree-scratch patch. And dropping the bf8 draft at 4 x 131k (v105, run 35809967020) fits
with 0.49 GB free, but its largest free block (285 MB) falls below the ~342 MB proposal reserve, so
the second proposal pair falls back every round (37 `fallback=dram_reserve` lines, ~+21 ms):
capacity margin shows up as speed. v101's configuration stays.

**Next: the kernel.** Read each k-chunk once for all of a user's row groups, and read the mask
only in the last chunk, in sdpa_decode itself (kernel work authorised). The served sources are
dumped by `scripts/ci/probe_sdpa_decode_sources.py` (cpu-probe-v24, run 35812394787; factory
tree-scratch patched 3e0a69af). Expected: up to ~70-100 ms off the 340 ms round at 4 x 131k.
