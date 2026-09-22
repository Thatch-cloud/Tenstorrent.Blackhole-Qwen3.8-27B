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
