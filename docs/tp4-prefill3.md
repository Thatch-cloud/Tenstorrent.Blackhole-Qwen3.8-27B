# tp4/prefill-3: the next prefill speedups after ship-5 (traced masked tails, split attention for short tails)

**Date:** 2026-10-21. **Base:** `tp4/next-1` (758264ac: the ship-5 stack, `...-fx-traffic-s5`, with `tp4/prefill-2` merged in). **Status:** both levers built behind default-off flags and CPU-tested; **nothing here has run on a card.**
Every number is labelled **M** (measured, with its source) or **ESTIMATE** (with its arithmetic). Read `docs/tp4-prefill-2.md` first: this document starts where its sections 7 and 8 stop.

## 0. Summary

| lever | flag | what is built | exact? | owner decision needed | ESTIMATE |
| --- | --- | --- | --- | --- | --- |
| **PF3-T traced masked tails** | `QWEN_FAST_PREFILL_TAIL_TRACE` (+ `_AUDIT`, `_BUCKETS`, `_ARENA_MB`) | the whole lever: `scripts/ci/prefill_tail_trace.py`, the route hooks, six gate-only profile twins, the window pack, the readers, 95 CPU tests (56 + 12 + 18 + 9) and two cost-model tests | class E0 **if the card agrees** with two claims (C1, C2 of section 2); an in-process audit settles them | none to run the gates; a ship profile needs an owner traffic waiver naming the flag | 110-170 ms off a tail pass of a bucket of 1,024 rows or fewer; 60 ms a prompt on the production population, 4.6 percent of its route time today, 7.5 percent after exact resume |
| **split attention for short tails** (K-split SDPA) | `QWEN_FAST_SDPA_PF_KSPLIT` | see section 6 (the kernel design, the graft, the host plumbing, the waiver text) | **no: tolerance class**, the numerics waiver of section 6.3 | **the numerics waiver** | see section 6 |

**The trace-headroom finding (M, section 3):** the shipped stack leaves **251.9 MB** of its 640 MiB trace region free at the end of a run (261.0 MB after the first packed round); four tail traces are estimated at 82-218 MB. The audited stack leaves 114.8 MB, so its two audited twins trace two buckets each. **The DRAM, not the trace region, is the cost:** the captures' intermediates are holes that need an arena (320 MB a chip, ESTIMATE) and the attach refuses by name when any of them falls outside it.

## 1. What was read

`docs/tp4-prefill-2.md` (the cost model, the candidates, the two designs), `docs/tp4-next1.md` (the s5 stack and its card results), `docs/tp4-traced-publish.md` section 5 (the trace-capture-order hazards this repo has hit), the stage-1 and merged Lever N route (`levern_route.py`), the model's masked-bucket path (`fixtures/qwen36_model.py`, `prefill_masked_bucket`, `_forward_prefill_chunk_masked_tp`), the GDN and attention grafts (`docker/qwen-c2-graft/graft`), the drafter window capture (`dflash_prefill_window.py`, `target_features.py`), NP1's scratch page (`idle_scratch.py`), and the container logs of the s5 window runs (aggregates only, section 3).

## 2. PF3-T: the masked tail pass as a trace replay

**Why.** Below the 2,048-row bucket the tail pass is host-bound: ~3,155 ops dispatched eagerly at ~100 us (**M**, `docs/tp4-prefill-2.md` 3.2: bucket 128 costs 324 ms at no context, bucket 1,024 381 ms) against 15-150 ms of device work. A replayed trace pays the device time.

**Why it was not a trace, and what replaces each obstacle** (no new kernel anywhere):

| per-call host value | stock | PF3-T |
| --- | --- | --- |
| `valid_len` in the GDN conv op and the fused chunk op (runtime arguments, baked by a capture) | the real row count | every GDN layer runs with `valid_len = bucket`; the pad rows are made **inert through the gates**: a persistent per-row vector (0 for a real row, `PAD_GATE` = -30000 for a pad row) is added to `a` and `b`, so `softplus(a + dt_bias)` is exactly 0 (a decay of exp(0) = 1) and `sigmoid(b)` exactly 0 (a write strength of 0) on a pad row: the recurrence over a pad row is the identity. A real row's `a`, `b` are untouched (x + 0 = x) |
| the conv state (the FIR's one-hot, a host write) | rows `valid_len-3 .. valid_len-1` of `[carry; input]` | two persistent one-hot matrices select the same rows with a HiFi4 / fp32-accumulate matmul pair and an add |
| the K/V fill width (`paged_fill_cache` over the real blocks only) | `page_table[blk0:blkN]` | a fixed-width chunk page table: the request's real blocks, then **NP1's KV scratch page** for every block past the last real one (a block past vLLM's pool that no request owns; the lever refuses without `QWEN_FAST_SOLO_PACKED=1`) |
| page tables, RoPE tables, chunk start, tokens | built per call | persistent buffers, `copy_host_to_device_tensor` before each replay (the chunk trace's own method); the SDPA is the bucket's own program, flexible chunk start |
| the logits row selector | a one-hot built per call | **stays eager** after the replay (five small programs the stock warm already compiled) |
| the drafter's window features | `LayerOutputCapture` wraps the forward of five tap layers and snapshots their outputs | the trace **retains** the five tap outputs in persistent buffers; on a replay the tap layers' forward is answered from them, so `PrefillWindowCapture`'s snapshot, ownership and ledger checks run unchanged |

**What is the stock program and what is not.** Same layers, same SDPA program (the bucket's), same graft conv op (a runtime argument, the same program), same logits ops. New device ops per GDN layer: two adds (the gates), a matmul pair and an add (the conv state); per trace six copies (output, taps). The stock path is untouched when the flag is unset (the route does not import the module).

**The two claims a card settles** (class E0 if both hold):

* **C1**: the fused chunk op with `valid_len = bucket` over gate-neutralised pad rows leaves the same recurrent state, byte for byte, as the op with `valid_len` = the real count. (Adding exact zeros and multiplying by exact ones leaves bits alone; the op's own masking has not been read against that on a card. The toy harness proves the *logic* of the lever, not the kernel.)
* **C2**: the one-hot matmul pair writes the same conv state as the op's `valid_len` state, its -0 and denormal canonicalisation included (the graft op reproduces the FIR's one-hot matmul canon bit for bit; this matmul is the same arithmetic).

`QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT=1` proves both on real tails inside the attach: it reads the scratch, replays the trace, reads the scratch, the real output rows and the five taps, **restores** the scratch, runs the **stock** pass, and requires every GDN recurrent state, conv carry and conv window, every real output row and every tap row byte equal; the stock result is what is served. It runs the first 8 traced tails, every 24th, and the first two of every bucket. The first mismatch names itself: `layer <i> recurrent state` is C1 false, `layer <i> conv window` is C2 false, `tap layer <i> rows` the tap hand-off, `the output rows` the attention or K/V path. The K/V of the real rows are held by the levern audit digest of the pair, not by this audit.

**The warm also refuses** a model whose KV block is not 64 rows, a GDN layer not bound to the persistent scratch the traces bake, an audit without the prefix graft's restore path, and a first capture while the conv op's own FIR audit has calls left; and it replays every trace once over the tail with the most pad rows (one real row past the previous bucket) and requires the output and every GDN state finite, because `PAD_GATE` has not met the real ops until then (the compile pass and the captures run with an all-zero pad vector).

**The holes (the real risk).** A replay writes its intermediates into the holes its capture freed, so a buffer allocated after the capture may be overwritten (`docs/tp4-traced-publish.md` 5, the 8-seat hang class). The lever's rules: every persistent buffer (inputs, output, taps) is allocated **before any capture**; the captures run inside an **arena** (a ballast tensor freed for the captures and allocated again after the last, required to come back at the same address); the warm records every tensor the captures free and **fails the attach by name** when any of it lies outside the arena (an allocator with free gaps below the arena would let a capture use them). The census counts DRAM tensors only (the L1 intermediates of the projections are not holes of a DRAM arena; the engaged line says `l1_skipped=`), and a freed tensor whose address cannot be read counts as uncovered. A tensor freed by garbage collection is not seen by the census: that residual is why the audited A1/A2 attaches (decode audits beside tails) are gates. The arena is DRAM the lever holds for the life of the process.

**Flags.**

| flag | values | meaning |
| --- | --- | --- |
| `QWEN_FAST_PREFILL_TAIL_TRACE` | unset / `0` / `1` | unset: off and nothing imported; `0`: the CONTROL (the stock tail, plus the `[PF3] tail mode=eager` timing line); `1`: the traced tail. Needs `QWEN_FAST_LEVER_N=1`; `1` needs `QWEN_FAST_SOLO_PACKED=1` |
| `QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT` | `1` | the in-process audit above; needs `1` |
| `QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS` | `128,256,512,1024` (default), any strictly ascending subset | which buckets are traced (the 2,048-row bucket is device-bound and its full chunks are not tails) |
| `QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB` | 1-2048, default 320 | the arena per chip. **ESTIMATE** (the 1,024-row bucket scales the admission's 600 MB prefill transient by 1,024 / 2,048); the ladder twins settle it |

What stays eager per call: vision tokens, an untraced bucket, another page-table width than the warm's, another tap set than the drafter's (or a feature capture that is not a `LayerOutputCapture`), a chunk size other than 2,048, GDN layers not bound to the persistent scratch, and everything after a latch (a staging failure latches the lever off and serves the request on the stock pass). Markers: `[PINDIAG] prefill tail trace engaged ... uncovered=0` once, `... fell back reason=..`, `... audit n=.. exact=True`, `[PF3] tail mode=<traced|eager> bucket=.. valid=.. start=.. ms=..`.

## 3. The trace-region headroom on the s5 profile (M, from the 2026-10-19 window's container logs)

The s5 audited attach (A1) was the first run with the ledger's trace lines. Aggregates only; every chip reads the same:

| run | region | used after the attach (P7) | used after the first packed round (P12) | used at most | least free |
| --- | ---: | ---: | ---: | ---: | ---: |
| the shipped candidate (the unaudited timed leg and the long-answer leg) | 671.1 MB (640 MiB) | 69.9 MB | 410.1 MB | 419.2 MB | **251.9 MB** |
| the audited twin (A1) | 671.1 MB | 205.6 MB | 545.8 MB | 556.3 MB | **114.8 MB** |

Reading it: the 340 MB between P7 and P12 is the eight parked engines (`parked engines built k=8 of 8 ... trace_used=410124288`, 42.5 MB each); the 9.1 MB after that is twelve more captures (a pair 1.4 MB, a quad 1.6 MB, the octo pass 1.7 MB each), which is the "7.5 MB at the 262k shape" of the churn test seen from the ledger; the audits add 136 MB at the attach (in-trace audit ops).

**What the tail traces need (ESTIMATE).** A traced body is the stock forward (~3,155 ops) plus 48 x 5 gate/state ops and six copies: ~3,400 ops. A trace holds dispatch commands, not data; the repo's own captures put it at ~6 KB an op (the pair and quad draft traces, 1.4 and 1.6 MB for a few hundred ops) to ~12 KB (a parked engine's full-model trace, 42.5 MB); prefill ops run on up to 130 cores, so the estimate takes 6-16 KB: **20-54 MB a trace**.

| stack | free | traces | need (ESTIMATE) | verdict |
| --- | ---: | --- | ---: | --- |
| shipped candidate, `-pf3-s5` | 251.9 MB | 4 (128, 256, 512, 1,024) | 82-218 MB | fits at both ends (34-170 MB to spare) |
| audited twin, `-pf3-s5-audit` / `-audit-mid` | 114.8 MB | 2 each | 41-109 MB | fits at both ends, 6 MB to spare at the top: hence two buckets a twin |

`python3 -B scripts/ci/pf3_report.py --headroom <container log> --traces 4` prints this fit from any attach's ledger; the pair report (`--control --arm`) prints the trace bytes the lever actually took (arm minus control after the attach), which replaces the estimate at gate H0.

**DRAM.** The trace region is not the constraint; the arena is: 320 MB a chip (ESTIMATE) off ~4.9 GB free after the first packed round on the shipped stack, and off the audited twin's ~2.3 GB (where the s5 audit run already held prompts of 2,048 tokens or more for the admission need of 1,124 MB: the audited twins will hold more, and run those prompts one at a time; audit-only, as before).

## 4. Expected effect (ESTIMATE; `python3 -B scripts/ci/prefill_cost_model.py tails`)

The model of `docs/tp4-prefill-2.md` (the eager floor of the bucket's production fit minus its device work, 80 percent recovered) less 8 ms of replay host cost (nine staging copies ~3 ms, the eager select/norm/lm_head ~5 ms):

| tail bucket | saving a tail pass | prompts with it (M, the 1.85-hour production window) |
| ---: | ---: | ---: |
| 128 | 168 ms | 61 |
| 256 | 126 ms | 66 |
| 512 | 128 ms | 126 |
| 1,024 | 110 ms | 243 |
| 2,048 | 0 (not traced) | 526 |

Over the population: **60 ms a prompt**, **4.6 percent** of the window's route time today and **7.5 percent after exact resume P0** (shipped in s5; P0 took 39 percent of the route time). Against the brief's cases: a returning turn (p50 first token 1.46 s in the s5 window) gains 4 percent on average and up to 11 percent on a short tail; a cold 59k prompt (12-15 s) gains about 1 percent, a cold 254k prompt (75 s) 0.2 percent: **this lever is for returning turns and short prompts, not for the cold long prompt**. Prompts under 2,048 tokens (no full chunk) trace too (`chunk_start` 0 resets the state eagerly before the replay).

## 5. Card gates for PF3-T (the pack: `scripts/ci/references/fusion-jobs/PF3`, `make_pf3_jobs.py`; one image `tp4-pf3-1`, Swift, ARC runners at zero from H0 to R2)

| job | what | stop rule | minutes (estimate) |
| --- | --- | --- | ---: |
| B0 | build the window image from the pushed head with the K64j-OQ graft (while production serves) | build fails | 40 |
| X0 | status, rescan, reset | cards / fabric / grid not as measured | 20 |
| **H0** | **fit**: `-pf3-s5` (arena 320 MB), `warmup,coding,concurrent8_steady`. First prefill trace in the served stack: compile pass (conv op engaged, hooks), four captures in the arena, the **hole census** (`uncovered=0`), the arena back at its address; records the trace bytes and the DRAM the arena holds | any attach error naming the arena, census, conv op or hooks; `uncovered` above 0; a stall | 40 |
| **A1** | **exactness, buckets 128 and 1,024**: `-pf3-s5-audit`, `warmup,coding,parked_turns,prefill_ladder_solo,concurrent8_steady,agent_turns` (4k / 32k / 128k / 254k cold rungs, forty parked turns, six agent sessions of 64k-200k: about seventy returning-turn tails) | one audit mismatch, one fall-back | 100 |
| **A2** | **exactness, buckets 256 and 512**: `-pf3-s5-audit-mid`, `warmup,coding,parked_turns,concurrent8_steady,agent_turns` | as A1 | 80 |
| **AP1** | **exactness on returning turns**: `c2_prefix_gate.py --plan tier-timed` on `-pf3-s5-audit` (cold, device hit, host-tier restore at ~32k / 128k / 254k; the tail of each audited) | plan not PASS; a mismatch | 90 |
| L1, L2 | arena ladder: 160 MB, 80 MB (the attach is the measurement; a failed rung is the answer) | none (soft) | 30 + 30 |
| T1-T4 | timed ABAB: control `-pf3-s5-ctl`, arm `-pf3-s5`, same seven tests; `pf3_report.py --control T1 --control T3 --arm T2 --arm T4` and `prefill_ladder_report.py` (answers EQUAL between arms) | none (soft); GO needs a saving and no loss in a rung | 4 x 70 |
| R1, R2 | returning-turn pair on the tier-timed plan, the saving at 125k and 245k of context | none (soft) | 2 x 90 |
| Z | status, reset | | 10 |

Total 850 minutes; **the gates block (B0 X0 H0 A1 A2 AP1) is 6 h 10 min** and answers whether the lever is exact and fits; the byte/token exactness at **4k, 64k, 128k and 254k** is: the audit (byte, in process) on every tail of the 4k / 32k / 128k / 254k ladder rungs and of the agent sessions (64k-200k) in A1 and A2, and the answer-text equality of the ladder and of `agent_turns` between the control and the arm in T1-T4. The timed prefill ladder is `prefill_ladder_solo` in T1-T4 (TTFT per rung, the answers equal). Shapes with a controlled tail length (a prompt of exactly C + 40 tokens) are not in the pack: the server's token count cannot be fitted closer than 1 percent, so the tails come from the production-shaped tests, and the per-bucket audit rule guarantees each traced bucket is audited twice.

**A cheaper first step, if the window is short:** H0 alone (40 minutes) answers the question that can sink the lever (the census and the fit) before any exactness run.

## 6. Split attention for short tails (K-split SDPA)

(Section filled by the K-split work; see `docs/tp4-prefill3-ksplit.md`.)

## 7. Risks

1. **Holes** (section 2): the census cannot see a tensor freed by garbage collection; a replay can silently corrupt a later persistent allocation. Mitigation: the arena, the audited attaches with decode audits running beside tails, the ladder. Residual: unknown.
2. **C1 / C2 may be false** on the real ops (the toy cannot say). The audit names which; the lever then stays a gate instrument.
3. **A collective inside a trace bakes one semaphore handle** (`docs/tp4-traced-publish.md` 5, item 5): the chunk trace of earlier bring-ups did this with the same all-gather matmuls, but never beside the packed traces.
4. **The sink page** is NP1's: pad K/V (finite garbage) lands in the block the lone-user round's idle segments also use; their outputs are discarded, but the page-writer audit of the audited twins may flag writes it did not make: read the first audit line before blaming the lever.
5. **Arena DRAM** (320 MB, ESTIMATE) comes out of the admission's headroom; the audited twins already hold prompts. The shipped stack has ~4.9 GB free.
6. **The 8 ms replay overhead and 80 percent recovery are model constants**, not measurements; H0/T1-T4 replace them.
7. **Host writes or an untraceable op inside the capture** (the fused chunk adapter's mask for a `valid_len` argument, the graft conv op's `generic_op` under a trace, the conv op's first-calls audit against the FIR): any of them is a `TT_FATAL` at the first capture, i.e. at H0, loudly and cheaply; the warm checks that the conv audit has run out before the first capture.
8. The tail's output buffer is persistent: nothing may deallocate the return value of `_forward_prefill_chunk_masked_tp` for a traced bucket (`test_prefill_tail_trace` pins the callers).

## 8. Files

| file | what |
| --- | --- |
| `scripts/ci/prefill_tail_trace.py` | the lever (overlay module; `docker/qwen-c2-overlay.txt`) |
| `scripts/ci/levern_route.py` | install / warm / uninstall behind the four flags |
| `scripts/ci/make_pf3_profiles.py`, `fusion-wp/PF3.json` | six gate-only twins of the ship-5 candidate; the manifest (overlay, CPU workflow, smoke rule) |
| `scripts/ci/make_pf3_jobs.py`, `references/fusion-jobs/PF3` | the window pack |
| `scripts/ci/pf3_report.py` | the pair reader and the trace-headroom reader |
| `scripts/ci/prefill_cost_model.py` | `tails`: the per-bucket and population pricing |
| `scripts/ci/test_prefill_tail_trace.py`, `test_pf3_profiles.py`, `test_pf3_pack.py`, `test_pf3_report.py` | CPU tests (a recording fake ttnn with an address-level allocator, a toy hybrid model) |
