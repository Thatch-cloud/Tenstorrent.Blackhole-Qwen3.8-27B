# Decode speed and 262k context on two p150a — plan

Where the 55 ms decode step actually goes, what serving the full 262,144-token
window through vLLM needs (config only), and the levers for making it fast — ranked
by expected ms/step, with effort and risk. Written against the state of the repo on
2026-09-02 (`main` + `speculative-decoding`), tt-metal `v0.77.0-rc1` and current
`main` (b87f414), and `vllm-tt-plugin` at `bf77cd6`.

> **Status:** partially executed. §0.5 is the silicon record as of 2026-09-02 and is
> kept verbatim. **Everything after it was rewritten on 2026-09-02 to agree with
> §0.5** — the first version of this plan estimated the step as "~34 ms of dispatch
> overhead over a 21 ms bandwidth floor", and §0.5 showed that to be wrong: the step
> is 97% device time, and ~47 of its 54 ms is *kernel* time. The levers below are
> re-derived from the per-module profile. *M* = measured on this rig; *E* = arithmetic
> on measured numbers, to be replaced by the per-op rows §3 asks for.

---

## 0.5 Measured on silicon, 2026-09-02

Rig changed under this section: the third p150a (board `…409a`) was removed, and a
**second QSFP-DD cable** was added between the remaining two. The pair is now
**M** (`d1:00.0`, board `…40b6`, gen5 **x16**) and **A** (`f3:00.0`, board `…408e`,
gen5 **x4**, behind a c-payne MCIO switch). Baseline reproduces: **17.79 tok/s** at
128 ctx against the README's 18.19 (−2.2%, within session drift).

### The step is 97% device-side, not host-bound

`QWEN36_DEBUG_DECODE_TIMING=1`, traced path, B=1, 128 ctx:

| phase | ms | share |
| --- | ---: | ---: |
| `exec_sync` (device) | 54.45 | 96.9% |
| `readback` | 1.07 | 1.9% |
| `update` | 0.69 | 1.2% |
| **total** | **56.21** | → 17.79 tok/s |

**This falsifies §1.2's host-sampling estimate.** The whole host round-trip is
**1.76 ms**, not ~6.6 ms. The 6.6 ms figure came from the spec-decoding branch's
*draft-phase* readback — a different code path, measured before the device-side
untilize fix. The traced demo path already reads efficiently.

Consequences: **at B=1 lever B's ceiling is 1.76 ms, not 3–7 ms** — demote it *for B=1*.
(Measured 2026-09-02 at B=8: the host round-trip is **~8.7 ms**, so B is re-promoted for
the batched endpoint — see lever B. This sentence should not have been generalised past
B=1, and the original 3–7 ms estimate was closer to right for the batched case.) And §1.2's
~34 ms of fixed overhead is confirmed but is **entirely inside the device step**
(54.45 ms against a ~21 ms bandwidth floor ⇒ ~33 ms), which makes **lever A the
lever** and §3's op-level profile the deciding measurement.

Corollary: because decode is device-bound it is *insulated from host CI load* —
17.79 held at load average 21. The 30× inflation `docs/speculative-decoding.md`
warns about applies to host-dispatch-bound phases, not the traced decode path.

### The second cable changed nothing, because the model never asks for the links

Fabric now discovers **4 ethernet links** (`chip0 ch8-11 ↔ chip1 ch4-7`, all
`link UP (QSFP)`, retrain 0), and `p150_x2` (4 channels) now validates where it
previously died on `TT_FATAL: Expected 4 eth links`. Three interleaved A/B pairs,
`p150_x2` vs `p300`, 128 ctx:

| pair | `p150_x2` | `p300` | Δ |
| --- | ---: | ---: | ---: |
| 1 | 17.95 | 17.94 | +0.01 |
| 2 | 17.66 | 17.66 | 0.00 |
| 3 | 17.67 | 17.95 | −0.28 |
| **mean** | **17.76** | **17.85** | **−0.5%** |

That A/B was **inert**, and the reason is a hardcoded table. Measured directly with
a no-model probe (`get_num_links` under each descriptor):

```
p300     links_any=2  axis0=2  axis1=2
p150_x2  links_any=2  axis0=2  axis1=2      <- descriptor changed nothing
```

`models/common/modules/tt_ccl.py::get_num_links` does **not** query the control
plane despite its docstring saying so. It is a static lookup:

```python
link_dict = { ..., "P300": (2, 2), "P150x4": (2, 2), "TG": (4, 4), ... }
device_links = link_dict[_determine_device_name(mesh_device)]
```

**This is the README's "two Blackhole devices are a P300" gotcha, with teeth.**
`_determine_device_name` keys purely off device count, so two cabled p150a inherit
a real p300's *on-package* link count of 2 — no matter how many QSFP-DD cables are
fitted. Four links were up and nothing in the stack would ask for them.

Separately, of the four hardcoded `num_links` sites in `tp_common.py`, all are
**prefill-only**, and `matmul_reduce_scatter_decode` (`num_links=1`) has **zero call
sites** — it is dead code at this version. Decode's collective is
`tt_transformers/tt/ccl.py::tt_all_reduce`, which *does* auto-size:

```python
if num_reduce_scatter_links is None: num_reduce_scatter_links = tt_ccl.get_num_links(cluster_axis)
```

— so decode automatically follows whatever the table says, and the table said 2.

### Correcting the table is worth 1.35 ms/step

With `get_num_links` overridden to 4 (announced per run, controls verified silent),
`traced_128`, `p150_x2`, interleaved:

| pair | 2 links | 4 links | Δ |
| --- | ---: | ---: | ---: |
| 1 | 17.69 | 18.09 | +2.3% |
| 2 | 17.68 | 18.14 | +2.6% |
| **mean** | **17.685** | **18.115** | **+2.4%** |

Controls agree to 0.06%, treatments to 0.28%. **56.55 → 55.20 ms, −1.35 ms/step** —
inside §4 E's original 1–3 ms estimate.

At `traced_8k`, two interleaved pairs, with the fused prefill all-gathers also raised
(`QWEN_CCL_AG_LINKS=4`):

| pair | TTFT | decode |
| --- | --- | --- |
| 1 | 2.27 → 2.18 s | 17.39 → 18.16 |
| 2 | 2.28 → 2.16 s | 17.92 → 18.19 |
| **mean** | **2.275 → 2.170 s (−4.6%)** | **17.655 → 18.175 (+2.9%)** |

TTFT is consistent across both pairs; the 8k decode controls are noisier (3% spread)
than the 128 ones, so treat +2.9% as agreeing with the firmer +2.4% rather than as a
second, larger result. No hangs, no numerical differences, `1 passed` throughout.

**So §4 E was right, and the earlier "decode CCL is latency-bound, expect ~0" call
was wrong.** The cable pays; it just needed the link table corrected to be reachable.

Filed upstream as
[tenstorrent/tt-metal#55125](https://github.com/tenstorrent/tt-metal/issues/55125).

### §3's op-level profile is blocked by the profiler, not by the model

The device step total is confirmed twice — `exec_sync` 54.45 ms host-side, and
`DEVICE TRACE FIRMWARE/KERNEL DURATION` **54.21 ms** across 110 cores from
`TT_METAL_TRACE_PROFILER=1`. The 0.24 ms difference is host enqueue. But the
**kernel-vs-op-to-op split inside the trace could not be obtained** at
`v0.77.0-rc1`, for three interacting reasons:

| attempt | outcome |
| --- | --- |
| `TT_METAL_DEVICE_PROFILER=1` on `traced_128` (50 tokens) | 1,100 × `Profiler DRAM buffers were full, markers were dropped` — ~125k ops against a 12k-marker buffer. Captured ~85% of programs, `METAL TRACE ID` empty. |
| Patched demo `profile_128` (3 tokens) | Still overflows. |
| `TT_METAL_TRACE_PROFILER=1` | No overflow, but profiles each trace replay as **one unit** — total only, no per-op rows. |
| `+ TT_METAL_PROFILER_MID_RUN_DUMP=1` | `TT_FATAL: Cannot dump data mid-run if only profiling trace runs` — the two modes are mutually exclusive. |
| `QWEN35_TP_DECODE_EAGER=1` (§3's own fallback) | Runs, but decode drops to **2.39 tok/s** — the trace hides ~7.5× of host dispatch, so eager op-to-op gaps say nothing about the traced ones. |

Also note `OP NAME` is blank in every `cpp_device_perf_report.csv` row; tracy's
`-r` post-processing fills it, and that step aborts whenever markers were dropped
(`AssertionError: Device data missing: Op N not present`). Two further gotchas
cost a run each: **tracy re-joins and re-splits its trailing args**, so
`-k "traced_128 and not 128k"` word-splits into `ERROR: file or directory not
found: and` with `rc=0` and an empty report — pass the exact node id instead; and
`--timeout` must be given explicitly, because `pytest.ini`'s 300 s default is far
too short under profiling.

**What the host-side op census gives instead**, and it points the same way. Op
counts across the traced 3-token run (`tracy_ops_data.csv`):

| op | count | | op | count |
| --- | ---: | --- | --- | ---: |
| `SliceDeviceOperation` | 24,510 | | `TilizeWithValPadding` | 6,908 |
| `CopyDeviceOperation` | 17,376 | | `MatmulDeviceOperation` | 6,288 |
| `BinaryNgDeviceOperation` | 16,000 | | `ReshapeView` | 5,064 |
| `UntilizeWithUnpadding` | 10,180 | | `AllGatherAsync` | 3,910 |

The step is dominated by **slices, copies and small elementwise ops** — matmuls are
sixth. That is precisely the population §4 A's fused kernel removes (the composed
GDN recurrence is slices, copies, typecasts and small matmuls), and it is
qualitative support for A being the right target even without the timing split.

### §3 resolved by per-module profiling — and it overturns §1.2

The per-layer unit tests drive one module eagerly, which fits the 12k-marker buffer.
All three ran with **zero overflow**, and tracy's `-r` post-processing then succeeds,
so `ops_perf_results_*.csv` arrives with `OP CODE` populated. One warm forward per
module (bounded by the collective each TP layer ends in; the cold first call and
weight setup excluded), averaged over both devices:

| module | ×N | ops/layer | kernel/layer | ops × N | kernel × N |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN | 48 | 71 | 0.4201 ms | 3,408 | **20.16 ms** |
| full attention | 16 | 44 | 0.2813 ms | 704 | **4.50 ms** |
| MLP | 64 | 7 | 0.3458 ms | 448 | **22.13 ms** |
| **total** | | | | **4,560** | **46.80 ms** |

Devices agree to within 2% on every module. MLP is verifiably one decode forward
(`max_batch_size=1`; 7 ops = tilize, copy, gate 94.7 µs, up 90.5 µs, silu·mul,
down 122.2 µs, reduce-scatter 24.8 µs); attention's block is confirmed decode by the
`NLPCreateQKVHeadsDecodeDeviceOperation` inside it.

```
measured device step (DEVICE TRACE KERNEL DURATION)  54.21 ms
sum of kernel time from per-module profiles          46.80 ms
implied op-launch / gap inside the trace              7.41 ms   (~1.63 us over 4,560 ops)
```

**§1.2's central claim is wrong.** It estimated "2,500–3,000 device ops per token…
at Blackhole-typical single-digit-µs op-to-op gaps that is **20–30 ms**". The actual
op count is higher (~4,560 for the layers alone) and the actual gap is **≈7 ms** —
and 7.41 ms is an *upper* bound, because Σkernel omits the LM head, embedding and
final norm (~1–2 ms more kernel). **The decode step is kernel-bound, not
dispatch-bound.**

Cross-check from the eager arm: eager decode ran at 2.39 tok/s = 418 ms/step. With
Σkernel = 46.8 ms that is 371 ms of host dispatch over 4,560 ops ≈ **81 µs/op**,
against **1.63 µs/op** traced. A ~50× reduction in per-op overhead is exactly what a
metal trace is for, and the two numbers are consistent with one story.

**Consequence for lever A.** It removes ~900 of 4,560 ops (the composed recurrence,
~19 ops × 48 layers). At 1.63 µs/op that is **1.5 ms of launch overhead**, plus
whatever kernel time the fused op saves over the composed one — the recurrence ops
in a GDN layer total roughly 50–80 µs of kernel, and the fused kernel still has to do
the arithmetic, so call it another 1–1.5 ms. **Lever A is worth ~2–3 ms, not 5–10 ms.**

**Where the time actually is.** 46.8 ms of kernel against §1.1's ~21 ms bandwidth
floor means kernels run at roughly **45% of bandwidth efficiency at M=1**. The
largest single term is MLP: 22.13 ms against a 12.2 ms floor. Per-op, the bf8
down-projection hits 92/122 = 75% of its floor while the bf4 gate hits only
44/94.7 = 46%. **Matmul efficiency at M=1 — especially for bf4 weights — is now the
biggest identified lever, and it is not in this plan.**

---

## 0. Short version

> **Ship list (2026-09-03).** Everything below is measured on this rig and gated where a
> gate was specified. **Composed on the endpoint, 8 streams, ITL with first token dropped,
> interleaved: A-only 68.3 ms → A + C + D + shard-greedy 61.2 ms (−10.4%), 14.6 → 16.3
> tok/s per user (+11.6%); GSM8K for that stack 57/60 (the C gate ran with A and D on, and
> shard-greedy is byte-identical to host argmax).** Per-lever detail in the lever table. On the existing `tt-vllm:qwen38-fused-decode` image the endpoint
> takes the first three from **environment variables alone**:
>
> ```
> -e QWEN_GDN_FUSED_DECODE=1                          # lever A: +15.5%/user, +18.9% aggregate at 8 streams
> -e QWEN35_GDN_DECODE_BF16=1 -e QWEN35_GDN_STATE_BF16=1   # lever C: -1.8 ms B=1 / -3.5 ms B=8, GSM8K 57/60 = fp32
> -e QWEN_SDPA_BF8=1                                  # lever D: -10.8 ms ITL, -23% TTFT at 262k, GSM8K 57/60 = bf16
> -v ~/ttcache:/ttcache -e TT_METAL_CACHE=/ttcache    # readiness 510-615 s -> 165-270 s
> ```
>
> Two more need a mounted file: lever E (`get_num_links` override, −1.35 ms; upstream
> #55125) and the in-place state write (−0.42 / −3.26 ms; `wrap-3c/`). B0's shard
> readback now reaches the endpoint too (row 3d: −4.4 ms ITL at 8 streams, greedy-only,
> needs the `wrap-3e/model.py` mount and `sample_on_device_mode: decode_only`).
> Measured negatives, do not revisit without new evidence: H (core count), I (packed
> gate|up), J (bf4 read rate), L (DRAM-sharded in-projections), CPU offload.


1. **The step is device-bound and kernel-bound.** *M:* `exec_sync` 54.45 ms of a
   56.21 ms step; host round-trip 1.76 ms. Σ kernel time from the per-module profiles
   is 46.8 ms; the launch gap inside the trace is ≤ 7.4 ms (≈ 1.6 µs/op over ~4,560
   ops). Against §1.1's 21 ms weight-streaming floor the kernels run at **~45%
   efficiency**, and that shortfall — not dispatch, not sampling — is where the time is.
2. **Two mechanisms account for the shortfall** (§1.2). The MLP matmuls read weights
   at 265–277 GB/s for bf4 and 387 GB/s for bf8 — "time tracks tiles, not bytes",
   which is what a per-transaction DRAM/NoC limit on tile-sized pages looks like. And
   the GDN layer spends ~11 ms/step in ~68 small ops per layer whose kernel time
   averages ~3.5 µs each: a per-op kernel floor, not arithmetic. Those two are levers
   **J** and **K** in §4, and they are new — neither was in the original plan.
3. **More dies buy the same 1.3–1.4×.** TT's own 4-die targets (26.0 tok/s at 128,
   14.67 at 256k) still fit: ~23 ms of the step is per-layer fixed cost (small-op
   floors, launch, CCL) and ~31 ms scales with weights per device (§1.3).
4. **262k through vLLM is a config change** (§2, unchanged): `QWEN36_MAX_TOKENS_ALL_USERS`
   decouples the KV pool from `max_model_len × max_num_seqs`.
5. **The fused GDN kernel and speculative decoding both exist on the `speculative-decoding`
   branch, and the kernel is only used for speculative verify.** Applying it to plain
   decode is worth ~4–5 ms, not the 5–10 ms first claimed and not the 2–3 ms §0.5
   re-estimated — see the arithmetic under lever **A**. It is the first slice of **K**,
   the full-GDN-layer fusion, which is the largest lever on this list. Speculative
   decoding comes after K, for the reason §5 gives.

6. **End state on this hardware (2026-09-03).** With the bf8 ceiling at ~385 GB/s the
   honest floor is ~28 ms. If K (−11.0, measured ceiling), J (−3.5, only if bf4 reaches
   bf8's rate), I (−1) and the proportional launch saving (~−3) all land, the step is
   ~35 ms: **~28 tok/s on two p150a, 1.5× from 18, all software.** The same work on four
   p150a gives ~26 ms, ~38 tok/s. Beyond that needs a faster path to DRAM than 385 GB/s,
   which §3.2's microbenchmark either finds or rules out.

7. **A is done, and the endpoint is a B=8 problem (2026-09-02).** The fused T=1 op
   already existed (`QWEN_GDN_FUSED_DECODE=1`); measured **−2.55 ms at B=1** (+4.7%) and
   **−9.5 ms at B=8** (+12.9%, 101 → 114 tok/s aggregate). Two consequences: the
   composed small-op graph's *batch penalty* (16.5 ms at B=8 vs 9.6 fused) means K is
   worth more at B=8 than its 11 ms B=1 ceiling suggests; and the host round-trip is
   **8.7 ms at B=8** (11% of the step), so lever B is back — with a cheap first half,
   B0, that is hours not days. Everything from here should be A/B'd at **B=8 first**,
   B=1 second, because B=8 is what the endpoint serves.
8. **B0 is done and it was one assert (2026-09-02).** `"shard"` decode mode — per-shard
   on-device argmax, two tiny tensors read back — already existed in the demo and was
   blocked at TP=2 by an assert written for the TopK sampler it never uses. Removing it:
   **−9.7 ms at B=8 (+15%)**, host round-trip 10.85 → 1.18 ms, output byte-identical.
   A + B0 together: **12.63 → 15.94 tok/s per user, 101 → 127.5 aggregate (+26%)**, from
   a flag and three lines. **Corrected 2026-09-02:** the endpoint gets *A* but not *B0*.
   A is in the model layer, so `QWEN_GDN_FUSED_DECODE=1` on the existing
   `tt-vllm:qwen38-fused-decode` image is enough — **measured on the endpoint at
   +15.5% per user (12.39 → 14.31) and +18.9% aggregate (93.2 → 110.9)**, tracking the
   demo to within 0.4%. B0 lives in `text_demo.py`'s own decode loop and does *not*
   reach the endpoint; the plugin has its own separate guard
   (`qwen36_vllm.py:63 _validate_device_sampling_request`), so exposing it means lifting
   the shard-greedy reduce into the plugin — bounded work, not three lines, and still
   the top item (§7 row 3d).
9. **Endpoint end state at B=8** (*E*, anchored on the measured endpoint ITL): with A
   the endpoint is at **69.9 ms ITL, 14.3/user, 110.9 aggregate** (*M*). B0 exposed
   through the plugin (−9.7 ms, §7 row 3d) → ~60 ms, ~16.6/user, ~133 aggregate. K at
   its B=8 value (≥ the 9.6 ms fused batch penalty; §3.3 measures it), J and I →
   ~46 ms: **~21.5 tok/s per user, ~170 aggregate on two cards**, against the README's
   12.6 / 101 and the endpoint's own control of 12.4 / 93.2.

The order that follows: **§2 (262k config) → §3.3 per-module profile at B=8 → B0 into
the plugin (§7 row 3d) → A's three leftovers → K (T-aware) → §3.2 microbench → J → §5
(re-run MTP)**. A and B0 themselves are done; what is left of them is getting B0 to the
endpoint and collecting A's last ~1.5–2 ms.

---

## 1. Where the step goes

### 1.1 The bandwidth floor (estimate, from the config)

Qwen3.8-27B: 64 layers, hidden 5120, MLP 17408, 48 GDN + 16 full-attention layers,
24 q-heads × 4 kv-heads × 256, vocab 248,320. At bf8 weights with bf4 on MLP gate/up:

| Component | Params | Bytes streamed per token |
| --- | ---: | ---: |
| MLP gate + up (bf4) | 11.4 B | 6.4 GB |
| MLP down (bf8) | 5.7 B | 6.1 GB |
| GDN projections (bf8) | 5.6 B | 5.9 GB |
| Full attention (bf8) | 1.7 B | 1.8 GB |
| LM head (bf8) | 1.3 B | 1.4 GB |
| **Total** | **26.9 B** | **21.5 GB** |

| Mesh | Per device | Floor per step at 512 GB/s | Ceiling |
| --- | ---: | ---: | ---: |
| TP=2 (2× p150a) | 10.8 GB | 21.0 ms | ~48 tok/s |
| TP=4 (4× p150a) | 5.4 GB | 10.5 ms | ~95 tok/s |

512 GB/s is the spec figure. The *practical* ceiling for tile-granular reads is still
unmeasured directly (§3.2), but two independent bf8 matmuls of different shapes now
bracket it tightly: MLP down 387 GB/s and the GDN in-projection **382 GB/s**. Two
unrelated shapes landing within 1.3% of each other looks like a ceiling rather than a
coincidence. If ~385 GB/s is the real limit, the honest floor is **~28 ms, not 21**, the
kernels run at ~59% rather than ~45%, and roughly a third of the apparent headroom in
§0 was never there. §3.2's DRAM microbenchmark is what would confirm it.

**Confirmed 2026-09-03 with tt-metal's own `test_dram_read`** (raw reads, 8 cores — one
per p150a DRAM bank, 16 KB pages, no matmul; steady-state after the cold first run):

| transfer | size | raw DRAM read | best matmul (§3.2) |
| --- | ---: | ---: | ---: |
| bfp4 tiles | 23.9 MB | 285–324 GB/s | 286 |
| bfp8 tiles | 45.2 MB | 377–397 | 364 |
| bf16 tiles | 85.0 MB | **415–428** | 391 |

**The memory path itself tops out at ~430 GB/s, 84% of the 512 GB/s spec**, with the
same transfer-size dependence the matmuls show; the matmul reader is within ~10% of it
at every width. There is no factor of two hiding in the kernel. The honest floor for
this document is **~26–28 ms** (10.8 GB/device at 390–430 GB/s), not 21. (Note: the
benchmark's default `--num-banks 12` is a Wormhole count and fails on Blackhole with
`No core coordinate found at (0, 33, TENSIX, LOGICAL)`; pass `--num-banks 8`.)

### 1.2 What the device step is made of (measured 2026-09-02, B=1, 128 ctx)

Per-module kernel time from §0.5, decomposed with the per-op figures it reports. Rows
marked *E* are arithmetic on those measurements, not separate measurements.

| Component | Kernel ms/step | How it decomposes | |
| --- | ---: | --- | --- |
| MLP matmuls (gate 94.7 + up 90.5 + down 122.2 µs × 64) | **19.7** | gate 265 GB/s, up 277 GB/s, down 387 GB/s | *M* |
| MLP reduce-scatter (24.8 µs × 64) | 1.6 | | *M* |
| MLP tilize / copy / silu·mul | 0.9 | 22.13 − 19.7 − 1.6 | *M* |
| GDN in/out projections (117.6 + 49.1 µs/layer, 44.9 + 16.7 MB) | **8.0** | 382 and 340 GB/s → 369 GB/s combined | *M* |
| GDN all-reduce (28.2 µs dev0 / 16.4 dev1 → 22.3 mean) | **1.07** | | *M* |
| **GDN remaining 68 ops/layer** | **11.0** | 229.7 µs/layer → **3.38 µs per op** | *M* |
| Attention projections at 387 GB/s (0.89 GB/device) | 2.3 | | *E* |
| Attention SDPA + ~40 small ops | 2.2 | 4.50 − 2.3 | *E* |
| Σ per-module kernel | **46.8** | | *M* |
| LM head (0.68 GB/device) + embedding + final norm | ~2.0 | not in the per-module profiles | *E* |
| Launch gap inside the trace | ~5.4 | 54.21 − 46.8 − 2.0 → ~1.2 µs/op | *E* |
| **Device step** | **54.2** | `DEVICE TRACE KERNEL DURATION` | *M* |

Three things to read off this table.

**Matmuls: time tracks tiles, not bytes.** Gate (bf4, 576-byte tiles) and down (bf8,
1,088-byte tiles) process the same 43,520 tiles per layer; bf8 moves 1.9× the bytes in
1.3× the time. In tiles: gate ≈ 460 M tiles/s, down ≈ 357 M tiles/s, and §0.5's
core-count sweep showed gate flat from 39 to 91 cores. An aggregate rate that is
insensitive to bytes *and* to core count is the signature of a per-transaction limit —
each interleaved tile is one DRAM page and one NoC read, and small pages amortise
per-request cost badly. The practical consequence is that the bf4 weights are being
read at ~52% of spec while bf8 reaches 76%, so **the bf4 layers are where the matmul
headroom is** (lever J), and further quantisation would make it worse (§0.5 H).

**GDN: the small-op kernel floor is the biggest single addressable term.** **11.0
ms/step** across 3,264 ops averaging **3.38 µs** each regardless of how little data they
touch (a firmware launch, CB setup and a handful of NoC reads have a floor). 60 of the
68 are under 5 µs and account for 66% of the non-matmul mass — it really is a floor, not
a few fat ops. The recurrence is 25 of the 71 (lever A, **5.04 ms**); the whole layer's
non-matmul work is lever K (**11.0 ms**).

**Where that 11 ms actually sits — layout churn.** The largest single category is
`ReshapeViewDeviceOperation`: **13 ops, 76.8 µs/layer = 3.69 ms/step**, and three of
them cost 12.1–12.5 µs each:

```
53  Matmul   [1,24,32,128] -> [1,24,32,128]    4.65 us     <- k@h
54  Reshape  [1,24,32,128] -> [1,1,32*24,128]  12.13 us    <- 2.6x the matmul it serves
56  Reshape  [1,24,32,128] -> [1,1,32*24,128]  12.50 us
63  Matmul   [1,24,32,128] -> [1,24,32,128]    4.71 us     <- q@h
64  Reshape  [1,24,32,128] -> [1,1,32*24,128]  12.16 us
```

These fold the 24-head axis from Z into Y. In tiled layout Y is the tile-row axis, so
this is a **physical relayout, not a view** — which is why a "reshape" costs 2.6× the
matmul it follows. The layer bounces between heads-in-Z and heads-in-Y six times
(idx 49, 50, 51, 57, 58, 59 do the inverse at 2.3–4.9 µs). The three big folds alone are
**1.77 ms/step in three ops**. A fused recurrence never materialises the intermediate
layouts at all, which is a large part of why A and K are worth what they are.

**Launch gap is real but small.** ~5 ms at ~1.2 µs/op is the trace doing its job (the
eager arm measured 81 µs/op). Reducing op count helps it, but only in proportion.

### 1.3 Calibration: Tenstorrent's own numbers for this model on 4 dies

`models/model_targets.yaml` on tt-metal `main` carries CI targets for Qwen3.6-27B on
`bh_quietbox_2` (2× p300c, four Blackhole dies, 128 GB, the supported `P150x4`
TP=4 path, on-device sampling):

| Context | TT target, 4 dies (t/s/u) | TT TTFT | Yours, 2× p150a | Your TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 26.0 | 0.17 s | 18.19 | 0.19 s |
| 4,096 | 18.13 | 2.03 s | 18.01 | 1.04 s |
| 8,192 | 18.01 | 4.62 s | — | — |
| 65,536 | 17.1 | 44.2 s | 16.19 | 27.3 s |
| 131,072 | 16.65 | 77.6 s | 15.80 (~104k) | 51.8 s |
| 262,144 | 14.67 | 286 s | 12.76 | 220 s |

The decomposition in §1.2 splits the 54 ms into a part that halves with TP and a part
that does not:

```
scales with weights/device : MLP matmuls 19.7 + GDN proj 7.6 + attn proj 2.3 + LM head ~1.7  ≈ 31 ms
per-layer fixed            : small-op floors ~14.5 + launch ~5.4 + CCL ~3.2                   ≈ 23 ms
TP=2: 23 + 31   = 54 ms  (18.5 tok/s)  ✓ measured 54.2
TP=4: 23 + 15.5 = 38.5 ms (26.0 tok/s) ✓ TT's target
```

So the 4-die target is consistent with the *same* per-op efficiency; four dies do not
fix either mechanism in §1.2, they halve one of them. And **TT's 4-die prefill is slower
than your 2-card prefill at every context length above 128** — 286 s vs 220 s at 256k —
so do not budget on TTFT improving with the upgrade.

### 1.4 The long-context decode penalty

At 256k the step grows 55 → 78 ms (+23 ms). The paged KV for 16 attention layers is
64 KiB/token at bf16, so 262,144 tokens is 16 GiB, **8 GiB per device at TP=2**, and
reading it once per token is a **16.8 ms floor at spec** (≈ 22 ms at the 387 GB/s the
matmuls actually achieve — i.e. the SDPA decode kernel is essentially at the practical
ceiling). The lever is `QWEN_SDPA_BF8=1`, which makes the KV cache bf8 and halves the
read; expected gain at 256k roughly 8–10 ms/step (12.8 → ~14.5 tok/s), which is where
TT's 4-die 256k target sits. The README advises against it at B=1 on precision grounds;
validate PCC / GSM8K at long context before adopting.

---

## 2. Serve 262k through vLLM

### 2.1 How the plugin sizes the KV pool

`vllm-tt-plugin` does not profile device memory. `get_num_available_blocks_tt()` asks
the model class for `get_max_tokens_all_users`, and for this model (`qwen36_vllm.py`,
identical between `v0.77.0-rc1` and `main`):

```python
override = os.environ.get("QWEN36_MAX_TOKENS_ALL_USERS")
if override:                       return int(override)
if max_model_len is not None:      return int(max_model_len) * int(max_num_seqs or 1)
```

then adds `block_size × max_num_seqs` of headroom and divides by `block_size`. So the
current command allocates a pool of `4096 × 8 = 32,768` tokens. Naively raising
`--max_model_len 262144` with `--max-num-seqs 8` asks for a **2.1 M-token pool = 134 GB
of KV** and the allocator dies. That is also the real reason B=8 at 64k OOMs without
`QWEN_SDPA_BF8=1`: `65536 × 8 = 524k` tokens is 32 GiB of bf16 KV, 16 GiB/device, on
top of ~13.75 GiB of weights. The pool, not the batch, was the problem.

### 2.2 Memory budget per device (TP=2, 32 GiB card)

| Item | bf16 KV, 262k pool | bf8 KV, 524k pool |
| --- | ---: | ---: |
| Weights (README: 27.49 GiB / 2) | 13.75 GiB | 13.75 GiB |
| Paged KV (16 layers, 2 kv-heads/device, 256 dim) | 8.0 GiB | 8.4 GiB |
| GDN recurrent state, fp32, B=8 (48 × 8 × 24 × 128 × 128 × 4 B) | 0.56 GiB | 0.56 GiB |
| Trace region (`trace_region_size`) | 1.0 GiB | 1.0 GiB |
| Prefill chunk activations, CCL buffers, L1-spill | ~1–2 GiB | ~1–2 GiB |
| **Total** | **~24.5 GiB** | **~25 GiB** |

Both fit. The single-stream 262k demo run in the README already exercised the bf16
column on this hardware.

### 2.3 The command

Diff against the README's serve command:

```diff
   -e QWEN36_BATCHED_DECODE_MODE=host \
+  -e QWEN_GDN_FUSED_DECODE=1 \
+  -e QWEN36_MAX_TOKENS_ALL_USERS=262144 \
   ...
-    --max_model_len 4096 --max-num-seqs 8 --no-enable-prefix-caching \
+    --max_model_len 262144 --max-num-seqs 8 --no-enable-prefix-caching \
+    --max-num-batched-tokens 262144 \
```

Notes on each:

* `QWEN_GDN_FUSED_DECODE=1` is lever A on the endpoint: **+15.5% per user, +18.9%
  aggregate at 8 streams, measured by ITL** (§4 A). It needs the image that carries the
  op (`tt-vllm:qwen38-fused-decode`); the README's serve command should carry it too.
* `QWEN36_MAX_TOKENS_ALL_USERS=262144` fixes the pool at one full-context user, or any
  mix whose lengths sum to 262k (e.g. 4 × 64k). Without it the pool is
  `max_model_len × max_num_seqs`.
* `--max_model_len 262144` also becomes `max_seq_len` for `create_tt_model`, which sizes
  the RoPE tables and the page-table width the prefill trace is captured against
  (`warmup_model_prefill` sizes it to the full KV, rounded to a multiple of 32 blocks —
  4,104 blocks at `--block-size 64`).
* `--max-num-batched-tokens 262144`: the plugin disables chunked prefill for every model
  except Gemma 4 and bumps this to `max_model_len` itself if you don't; setting it
  explicitly keeps the startup log honest.
* Keep `--max-num-seqs 8` if you want short requests to keep flowing between long ones;
  decode bucketing already runs the narrowest power-of-two trace. Use `--max-num-seqs 1`
  if the endpoint is purely for single long-document requests — it removes the
  preemption case in 2.4.
* `-e QWEN_SDPA_BF8=1` plus `QWEN36_MAX_TOKENS_ALL_USERS=524288` is the "two concurrent
  256k users, or 8 × 64k" configuration. Same precision caveat as §1.4.
* `VLLM_RPC_TIMEOUT=100000` is 100 s. A 262k prefill is 220 s. Raise it to at least
  `600000` before the first long request rather than after.

### 2.4 What to expect, and what to test

* **TTFT is ~220 s at 262k and the engine is busy for all of it.** Prefill is
  model-owned and runs per user; with `max_num_seqs 8` other users' decode stalls while
  a long prompt prefills. Clients need a first-byte timeout above 5 minutes, and a
  streaming client is the only sane way to consume it.
* **No prefix caching** (`supports_prefix_caching: False`; the GDN state is not
  block-addressable), so a multi-turn conversation over a 262k document re-prefills the
  whole document every turn. 262k is a one-shot document mode, not a chat mode, until
  that changes upstream.
* **Preemption.** If a 262k request lands while the pool is partly occupied, vLLM will
  preempt (recompute) to make room. Recompute means another full prefill. Watch
  `vllm:num_preemptions_total` on `/metrics` during the first mixed-load test; if it moves,
  either drop to `--max-num-seqs 1` for this endpoint or bound long requests at the
  proxy.
* **Warmup** captures the prefill chunk trace once against the full-width page table;
  readiness may exceed the README's 510 s. Bump the probe.

Acceptance test, in order: (1) start with the new command and confirm the log line
`Getting max_tokens_all_users=262144 for number of blocks in KV cache`; (2) one request
at ~4k to confirm decode is unchanged (expect ~18 tok/s); (3) the 262k Frankenstein/War
and Peace prompt from `demo/sample_prompts` through the endpoint with `stream=true`, check
TTFT ≈ 220 s and decode ≈ 12.8 tok/s; (4) GSM8K at `max_tokens=2048` as in the README to
confirm nothing regressed; (5) if enabling bf8 KV, repeat (3) and (4) and record the delta.

**Measured 2026-09-03 — §2.3 works as written, config only.** Serve command as in §2.3
plus `QWEN_GDN_FUSED_DECODE=1` (lever A, shipped) and the kernel-cache mount:

| acceptance | expected | measured |
| --- | --- | --- |
| (1) KV pool log line | `max_tokens_all_users=262144` | present; **ready after 210 s** (README budgeted 510+) |
| (2) short request, single stream | decode unchanged | TTFT 0.61 s, ITL 62.4 ms → 16.0 tok/s |
| (3) 261,000-token prompt, streamed | TTFT ≈ 220 s, decode ≈ 12.8 | **TTFT 222.3 s**, ITL 82.6 ms → **12.11 tok/s** |
| preemptions during (3) | 0 | `vllm:num_preemptions_total` = 0 |

TTFT reproduces the README's demo figure to 1%. Decode is ~5% under the demo's 12.76
because a single vLLM stream carries the scheduler/sampler/detokenise cost per token
that the demo loop does not (the same ~2% gap seen at B=8 grows at B=1). The prompt was
built from the demo's own cached Gutenberg texts, tokenised with the model tokenizer
and cut to 261,000 tokens (`~/prompt262k.txt`). GSM8K (4) is outstanding; (5) bf8 KV
is queued as lever D.

---

## 3. Measure before optimising — what is still missing

§0.5 settled the big split (kernel vs launch vs host) and documented why a per-op
profile of the traced step is not obtainable at `v0.77.0-rc1`. The per-module profile is
the workaround that worked, and two more measurements of the same kind decide the size
of the top two levers.

**1. The per-op rows of one GDN layer — DONE 2026-09-03.** Both devices, warm
`forward_decode` (71 ops between the last two collectives), from the existing
`mp-gdn-out/*/ops_perf_results_*.csv`:

| | measured | this section predicted |
| --- | ---: | ---: |
| in-projection matmul (bf8, 44.9 MB) | 117.6 µs → **382 GB/s** | ~115 µs |
| out-projection matmul (bf8, 16.7 MB) | 49.1 µs → 340 GB/s | ~43 µs |
| all-reduce | 28.2 dev0 / 16.4 dev1 | ~25 µs |
| other 68 ops | **229.7 µs, mean 3.38 µs** | ~3.5 µs |
| 60 of 68 under 5 µs | 150.6 µs = 66% of non-matmul mass | — |
| **lever K ceiling** | **11.0 ms/step** | 8–12 ms |
| **lever A ceiling** (25-op recurrence, 105.0 µs/layer) | **5.04 ms/step** | 4–5 ms |

Devices agree to 0.6% on the block total. **Every link of §1.2's inference chain holds
to within 5%**, including the two borrowed constants: GDN's bf8 projections really do
run at 369 GB/s combined against the 387 GB/s assumed from the MLP, and the all-reduce
at 22.3 µs against the 25 µs assumed by analogy. The *E* rows elsewhere in this document
can be trusted at about that level.

The distribution also settles the fusion question the section posed: the mass is **not**
in a few large ops — 60 of 68 are under 5 µs — so K keeps its full scope, and the
per-op floor is real rather than an artefact of averaging. See §1.2 for the layout-churn
breakdown that came out of the same rows.

**2. A matmul microbenchmark for the bf4 shape.** `ttnn.linear` at M=32 (one tile) ×
`[5120, 8704]`, one warm run each, reporting achieved GB/s and tiles/s:

| arm | what it tests |
| --- | --- |
| bf4 / bf8 / bf16 weights, interleaved, the tuned 11×4 1D-mcast config | does time track bytes or tiles across all three widths |
| bf4, `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with `mlp_w1_weight_memcfg` (the path `mlp.py` takes when `mlp_1d_decode` is off) | contiguous per-bank reads instead of tile-interleaved ones |
| bf4, in0_block_w swept 1 → k_tiles/cores | reads issued as larger multi-tile transactions |
| tt-metal's DRAM read microbenchmark (`tests/tt_metal/tt_metal/perf_microbenchmark`), 576 B vs 1088 B vs 4 KiB pages | the practical DRAM ceiling per page size, independent of matmul |

If bf4 can be brought to bf8's 387 GB/s, gate+up drop from 185 to ~130 µs/layer:
**~3.5 ms/step**, lever J. If the DRAM microbenchmark shows ~390 GB/s is the practical
ceiling for *any* page size, J is closed and the matmuls are done.

**Arm 1 measured 2026-09-03** (`ttnn.linear`, M=32 × [5120, 8704], tuned 11×4 1D-mcast,
device profiler, median of 6 warm runs; the same 43,520 tiles in every row):

| weights | kernel | bytes | achieved | tiles/s |
| --- | ---: | ---: | ---: | ---: |
| bf4 | 87.7 µs | 25.1 MB | 286 GB/s | 496 M |
| bf8 | 130.0 µs | 47.3 MB | 364 GB/s | 335 M |
| bf16 | 227.8 µs | 89.1 MB | **391 GB/s** | 191 M |

Neither pure model fits — achieved GB/s *rises* with width while tiles/s *falls*. A
two-term fit, `t = fixed + bytes / rate`, gives a **marginal rate of 430–520 GB/s
(near spec) plus a fixed 20–40 µs per matmul** that does not scale with bytes. That
fixed term is 25–45% of the bf4 time and <20% of bf16's, which is the whole "bf4 reads
at 52%" effect of §1.2. So: **~390 GB/s is the practical ceiling** (bf16 reaches it;
the two bf8 model matmuls sat at 382 and 387), and **J's ceiling is ~3.0 ms/step**
(bf4 from 87.7 to ~64 µs, × 2 × 64). What decides J is whether the fixed term is
per-*tile* (page-size / read-granularity, fixable in layout) or per-*op* (pipeline
fill/drain across 44 cores, fixable only by fusing — which would lift lever I to
~1.3–2.6 ms). One more arm settles it: bf4 at N = 4352 and 17408. Arms 2–4 failed on
setup (activation must be L1-sharded for the DRAM-sharded config; the hand-built
`in0_block_w` configs tripped `matmul_device_operation.cpp:522`; the DRAM benchmark's
output was not captured) — rerun pending.

**Arms A–C measured 2026-09-03 (rerun). J as conceived is closed; a different lever
fell out.**

*A — per-op or per-tile?* bf4 across N=4352/8704/17408: `t = −6 µs + 2.24 ns/tile`,
intercept indistinguishable from zero — **no per-op fixed cost**. With the dtype sweep
(bf4 2.24, bf8 2.99, bf16 5.23 ns per tile) the per-tile cost decomposes as **~1.4 ns
fixed per transaction + ~1.5 ps/byte** (≈680 GB/s marginal streaming, near spec). For a
576-byte bf4 tile the fixed part is 62% of its time. That is the whole of §1.2's "bf4
reads at 52%".

*B — amortise it with larger reads?* No. `in0_block_w` 1→40 at bf4: the tuned 8 is the
optimum (87.6 µs); 2 is within noise; 16–40 are 10–22% *worse*; 1 is 146 µs.

*C — contiguous per-bank reads?* bf4 DRAM-width-sharded 88.2 vs 1D 87.7 — nothing.
**So the bf4 penalty is intrinsic to the tile size and no layout moves it: J is closed
at 0 ms.** Further quantisation would make it worse, as §0.5 H said.

**The lead: DRAM-sharded bf8 at K=5120 is 100.6 µs against 130.0 on the tuned 1D
path — 23% faster.** The model's bf8 in-projections (GDN qkvzab, attention QKV) are
K=5120 and run 1D (`proj_1d_decode = True`, tuned at TP=4 where per-device N is half).
If it holds at their real widths that is ~−17 µs × 64 layers ≈ **−1 ms/step, config
only** — arm D measures the four model shapes directly.

**Arm D measured 2026-09-03** — bf8, M=32, the model's real per-device widths, tuned
1D core counts vs the model's own DRAM-width-sharded config (median of 6 warm):

| projection | 1D (shipped) | DRAM-sharded | Δ | × layers |
| --- | ---: | ---: | ---: | ---: |
| GDN in-proj, K=5120 N=8256 | 120.6 µs | **97.8** | **−19%** | −1.10 ms |
| attention QKV, K=5120 N=7168 | 104.1 | **83.3** | **−20%** | −0.33 ms |
| GDN out-proj, K=3072 N=5120 | 46.2 | 43.4 | −6% | −0.14 ms |
| MLP down, K=8704 N=5120 | 122.3 | 280.0 | **+129%** | keep 1D |
| | | | | **−1.57 ms/step** |

Two mechanisms, both already in §0.5 H: `_find_grid` gives K=5120 (160 k-tiles) a 32-core
DRAM-sharded grid and K=8704 (272 k-tiles) only 16, so the down-projection collapses
on it — which is the shape upstream measured when it set `proj_1d_decode = True` at
TP=4. At TP=2 the in-projections are twice as wide per device and the balance flips.
**New lever L: DRAM-sharded decode for the K=5120 in-projections at TP=2 — ~−1.6 ms,
config only, the path already exists in `model_config` (`gdn_qkvzab_progcfg`,
`attn_qkv_fused_progcfg`, their `_weight_memcfg`s).** Needs the model-level A/B at B=1
and B=8; the microbench is one matmul on one device.

**Lever L measured 2026-09-03 — closed, negative.** `QWEN_PROJ_1D=0` on the model
(`traced_128` / `batched_128_b8`, A + in-place on in both arms, interleaved pairs):

| `exec_sync` | 1D (shipped) | DRAM-sharded | Δ |
| --- | ---: | ---: | ---: |
| B=1 | 51.30 / 51.20 | 54.35 / 54.35 | **+3.1 ms** |
| B=8 | 57.86 / 57.93 | 60.90 / 60.91 | **+3.0 ms** |

Engagement line present in every treatment arm, PCC 0.9999 unchanged. The matmul is
20% faster on the sharded path and the *layer* is 3 ms slower, because
`sharded_decode_matmul` reshards the activation to L1-width-sharded before the matmul
and converts the output back after — two relayouts per projection, 64 layers. The
1D-mcast path takes the interleaved activation directly. Upstream's choice was an
end-to-end measurement and it holds at TP=2. **Rule for this document: a matmul
microbenchmark that excludes its input/output layout conversions cannot rank paths that
differ in them** — arm D compared kernels; the model runs paths.

**3. The same per-module profile at B=8 — DONE 2026-09-03.** `test_gdn_tp[B8]` and
`test_attention_tp[B8]`, warm layer bounded by its collectives, both devices averaged.
MLP needs no re-run: `test_mlp_tp` already feeds a `T = 32` activation, i.e. one full
tile, which is exactly what B=8 pads to — **MLP kernel time is batch-invariant by
construction up to B=32**.

| module | B | ops/layer | kernel/layer | projections | CCL | small ops | µs/op | ×N per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN | 1 | 71 | 0.420 ms | 167.5 µs | 22.3 | 230.3 | 3.39 | 20.16 ms |
| GDN | **8** | 75 | **0.657 ms** | 169.0 | 21.4 | **466.1** | 6.47 | **31.51 ms** |
| attention | 1 | 44 | 0.281 ms | 168.0 | 14.2 | 99.1 | 2.42 | 4.50 ms |
| attention | **8** | 66 | 0.404 ms | 167.4 | 19.4 | **216.8** | 3.44 | **6.46 ms** |
| MLP | 1–32 | 7 | 0.346 ms | 307 | 24.8 | 14 | — | 22.13 ms |

Three things fall out.

**The projections do not move with batch** — 167.5 → 169.0 µs, 168.0 → 167.4 — because
M=1 and M=8 both pad to one 32-row tile. Everything TP=4 buys (halving weight bytes per
device) is therefore worth the same at B=8 as at B=1, and the matmul-side levers (J, I)
are batch-invariant too.

**The small-op floor doubles.** GDN's non-projection, non-CCL mass goes 11.06 →
**22.37 ms/step** and attention's 1.59 → 3.47. At eight rows the composed ops stop being
launch-floor-bound and start scaling with data: the state-sized elementwise ops on the
`[8,24,128,128]` fp32 recurrent state that cost ~9 µs at B=1 cost 25–28 µs at B=8, and
the 12 µs head-fold reshapes become 26 µs. Ops per layer also rise (71 → 75, 44 → 66).
**Lever K's ceiling at B=8 is ~22.4 ms, twice its B=1 value**, and A's −9.5 ms at B=8 is
42% of that mass against 23% at B=1 — the fused op amortises the eight rows in one
dispatch where the composed graph pays each of them.

**The budget closes.** Σkernel at B=8 = 31.51 + 6.46 + 22.13 = **60.1 ms** (+~2 for LM
head etc.) against the composed path's measured `exec_sync` of 70.6 ms → ~8.5 ms of
launch gap over ~5,100 ops, ≈1.6 µs/op, the same per-op figure as at B=1.

Practicalities from §0.5 that apply to all three: pass the exact pytest node id to
`python -m tracy` (it re-splits `-k` expressions), set `--timeout` explicitly, and
expect `-r` post-processing to abort if any markers were dropped — the per-module tests
fit the 12k-marker buffer; the full step does not.

**Cycle time (2026-09-03).** Every `--rm` container recompiles every kernel unless the
JIT cache is mounted: a `traced_128` arm is **308 s without `TT_METAL_CACHE`, 76 s
with it**, and the vLLM endpoint's readiness drops from **510–615 s to 270 s**. The
in-container work is ~74 s, 55 s of it weight loading. The tensor cache
(`TT_CACHE_PATH`, 29 GB on disk) was measured break-even after correcting for host
load and is not where the time goes — the host→device push of ~29 GB is. All nine
runners on the rig now mount both caches. An interleaved 8-arm A/B is ~12 minutes.

---

## 4. Levers, ranked

Expected gains are per step at B=1, 128 ctx, from the 54–56 ms baseline. *M* =
measured on this rig; *E* = arithmetic on §0.5's measurements.

**Re-ranked twice.** The original table put dispatch-overhead levers first; §0.5 showed
dispatch is ≤ 7 ms. This table is derived from §1.2: the two new entries, K and J, are
the two mechanisms that table exposes.

| | Lever | Expected | Effort | Risk |
| --- | --- | ---: | --- | --- |
| K | Fuse the whole GDN decode layer (71 ops → ~10) | **11.0 ms B=1 / 22.4 ms B=8 *M*** (ceilings; A has taken 2.55 / 9.5) | weeks; A is the first slice | numerics; trace-address discipline |
| A | Fused T=1 decode op — **DONE, and it already existed** | **−2.55 ms B=1 / −9.5 ms B=8 *M*** | done | PCC 0.9999, unchanged |
| B0 | Host round-trip, Python-only half: device-side untilize before readback + on-device RoPE gather | **3–5 ms at B=8 *E***; ~0.3 at B=1 | hours | none |
| J | bf4 gate/up read rate → bf8's (page size / layout) | **≤ 3.5 ms *E***; microbench decides | days | none if layout-only |
| D | `QWEN_SDPA_BF8=1` — **DONE, GSM8K gate passed (57/60 = bf16)** | **−10.8 ms ITL, −23% TTFT at 262k *M*** | done | 262k continuation byte-identical to bf16 (64 greedy tokens) |
| I | ~~Packed `[gate|up]` decode matmul~~ — **closed on §3.2's N-sweep** | **≈ +0.8 ms *M*** (worse) | — | — |
| C | bf16 GDN step + state (two flags) — **DONE, GSM8K gate passed (57/60 = fp32)** | **−1.78 ms B=1 / −3.52 ms B=8 *M*** | done | >2k-token drift unexercised |
| E | Fix `get_num_links` — **done** | **−1.35 ms *M*** | done | none observed |
| B0 | `"shard"` decode mode at TP=2 — **DONE**, one assert | **−9.7 ms at B=8 *M*** (+15%) | done | output byte-identical |
| B | Expose B0's greedy path to the **vLLM endpoint** via `sample_on_device_mode` (per-batch fallback to host for temperature > 0) | **the endpoint's share of −9.7 ms at B=8** | days | plugin contract; TopK still TP=4 |
| F | Conv shift register as a ring buffer | ≲ 0.3 ms *E* | day | trace-address safety |
| H | Matmul core count — **closed** | ~0.3 ms *M* | done | — |

Nothing else can reach more than a few ms: after §1.2 the step is 31 ms of weight
streaming at the rates the matmuls achieve, ~14 ms of small-op floors, ~5 ms of launch
and ~3 ms of CCL. K and J attack the second and first of those; everything else is
inside the noise of a session.

### K. Fuse the GDN decode layer

`forward_decode` in `gdn/tp.py` runs, per layer per token: 4 shift-register copies,
multiply + 3 `mac` + silu for the conv taps, 3 slices + reshapes to split q/k/v, 2
`repeat_interleave` for GQA, sigmoid and softplus for the gates, the 20-op recurrence,
a copy-back, `rms_norm`, silu·mul, the out-projection and an all-reduce — 71 ops, of
which two are the matmuls that move nearly all the bytes. At ~3.5 µs kernel + ~1.2 µs
launch each, the other ~68 cost ~0.32 ms per layer, **~15 ms per step**, for work that
touches a few hundred KB.

The shape of the fix is three kernels, each replacing a run of ops that share operands:

1. **conv + gates** — the K-tap causal conv over the persistent state (which also
   retires lever F's four copies), the q/k/v split, and sigmoid/softplus for beta and g.
2. **recurrence** — this is `gdn_recurrent_step` from the branch, at T=1 (lever A).
   Whether it already consumes un-normalised bf16 q/k and does the L2 norms inside
   decides whether 4–5 more ops fold into it.
3. **norm + gate** — `rms_norm` over `[B, Nv, Dv]` and the silu·mul with z.

Each is a real ttnn op with a program factory, in the same mould as the branch's
`gdn_decay/` (12 files, ~900 lines) — hence weeks, not days, and the reason A ships
first: it is the kernel that already exists, and it measures the per-op floor directly
(the A/B delta ÷ ops removed is the number the rest of K is sized from).

### A. The fused kernel already on the branch, applied to plain decode

**What exists.** `speculative-decoding/ttnn-op/gdn_decay/` is a working ttnn op,
`gdn_recurrent_step`, that runs `T` GDN recurrence steps in one dispatch and emits every
intermediate state. It is validated at `T = 1, 2, 4, 8` against the composed path, and it
is wired into the speculative *verify* path (`forward_decode_multi_fused` in
`harness/test_gdn_decode_multi.py`), where it moved verify from 1.27× to 1.53×.

**It already existed — this section was stale.** A *second*, purpose-built T=1 op,
`ttnn.transformer.decode_gated_delta_rule`, was written on 2026-08-28 (`~/opgraft-53587`:
C++ op + device kernels, plus a patched `ttnn_delta_rule_ops.py`, 578 → 672 lines) and
wired into `recurrent_gated_delta_rule_decode_ttnn` — which `gdn/tp.py:1094` calls from
`forward_decode`. So plain decode *can* take the fused path today, behind
**`QWEN_GDN_FUSED_DECODE=1`** (note: `QWEN_`, not `QWEN36_`). Every prior run of it was a
speculative-verify or `batchprobe` measurement; **the plain-decode A/B had never been
run**, which is why nothing recorded a baseline-decode number.

### A — MEASURED 2026-09-02: −2.55 ms/step, +4.7%

`traced_128`, interleaved, engagement asserted per arm:

| arm | decode | `exec_sync` |
| --- | ---: | ---: |
| no graft, flag=0 (neutrality) | 18.23 | 54.24 ms |
| graft, flag=0 (control) | 18.25 / 18.28 | 54.18 / 54.19 |
| graft, flag=1 (**fused**) | **19.14 / 19.12** | **51.63 / 51.64** |
| **mean** | **18.265 → 19.13 (+4.7%)** | **−2.55 ms** |

**At B=8 it is worth nearly 4× more** (`batched_128_b8`, same graft, two interleaved
pairs, `QWEN36_BATCHED_DECODE_MODE=host` + `QWEN_BATCHED_GROUPED=0`):

| arm | per-user tok/s | aggregate | `exec_sync` |
| --- | ---: | ---: | ---: |
| flag=0 (control) | 12.63 / 12.63 | 101.0 / 101.0 | 70.62 / 70.79 |
| flag=1 (**fused**) | **14.25 / 14.27** | **114.0 / 114.1** | **61.27 / 61.11** |
| **mean** | **12.63 → 14.26 (+12.9%)** | **101.0 → 114.05** | **−9.5 ms** |

Controls identical to three significant figures. The composed path's batch penalty is
`70.7 − 54.2 = 16.5 ms`; the fused path's is `61.2 − 51.6 = 9.6 ms` — **the fusion
scales with batch far better than the composed graph**, because the composed small ops
stop being floor-bound once each carries 8 rows while the fused op amortises them.
The B=1 control also reproduces the README (12.63 vs 12.56 per user, 101.0 vs 100.5
aggregate).

Controls reproduce to 0.16%, treatments to 0.10%. At B=1, `update` (0.28 ms) and
`readback` (0.31 ms) are unchanged, so the whole gain is device-side. Three controls
make it firm:
the **graft-neutrality** arm (rebuilt `.so`, flag off) reads the same as stock; the
**negative control** (patched `.py`, *stock* `.so`) emits
`"…decode_gated_delta_rule is not available in this ttnn build; falling back"` with zero
engagement lines; and correctness is unchanged at **PCC 0.9999**, identical to the
composed path.

**On the vLLM endpoint: +15.5% per user, +18.9% aggregate, from one env var.** The
serving image `tt-vllm:qwen38-fused-decode` already contains the op and the patched
module (`decode_gated_delta_rule present: True`), so the endpoint needs
`QWEN_GDN_FUSED_DECODE=1` and **no graft and no code change** — A is in the model layer
(`qwen36_vllm.py:363 → super().decode_forward() → ttnn_decode_forward → _forward_decode`),
unlike B0 which lives in the demo's own loop.

Measured by **inter-token latency** — 8 concurrent streams, first token of each dropped
so TTFT falls out, aggregate counted only over the window where all 8 are decoding:

| arm | median ITL | per-user | aggregate |
| --- | ---: | ---: | ---: |
| flag=0 | 81.74 / 79.68 ms | 12.23 / 12.55 | 93.4 / 93.0 |
| **flag=1** | **69.66 / 70.16 ms** | **14.36 / 14.25** | **110.2 / 111.5** |
| **mean** | **80.71 → 69.91 (−10.8 ms)** | **12.39 → 14.31 (+15.5%)** | **93.2 → 110.9 (+18.9%)** |

Controls agree to 0.4% (aggregate) and 2.5% (ITL). **The endpoint tracks the demo almost
exactly** — treated per-user 14.31 vs the demo's 14.26, control 12.39 vs 12.63 — so the
B=8 demo number transfers to production 1:1, and the endpoint costs only ~2% over the
harness.

> **Measurement note.** The first attempt at this timed whole requests, which puts TTFT
> and vLLM's per-user blocking prefill inside the number; its two control arms came out
> **41% apart** (41.3 vs 58.2 aggregate) and pair 1 showed a spurious +63%. Any endpoint
> A/B on this stack has to isolate decode via ITL, for the reason §2.4 gives.

**Realised 51% of the 5.04 ms ceiling** at B=1, and the code says why. Three costs stay
outside the op:

* the **five fp32 typecasts are still separate ops** — `beta`, `g` and `initial_state`
  are cast *before* the call, so lever C is still live and composes with A;
* **`inplace_state=False`** in the engagement line — the op returns a fresh `h` rather
  than writing into the persistent `rec_state`, so the copy-back remains. An `_inplace`
  variant exists in the op and is not what decode uses;
* the op returns `o` **ROW_MAJOR** and the graph needs TILE, so the fused path *adds* a
  `ttnn.to_layout` per layer.

Those three are the obvious next slice, and they are cheap relative to K.

**In-place — DONE 2026-09-03.** `QWEN_GDN_FUSED_INPLACE=1` threads `inplace_state=True`
through the wrapper when `init_state is rec_state` (B == Bmax); the caller skips the
copy-and-free iff `buffer_address()` matches. Two interleaved pairs each:

| | `exec_sync` off | on | Δ |
| --- | ---: | ---: | ---: |
| B=1 | 51.89 / 51.74 | 51.47 / 51.33 | **−0.42 ms** |
| B=8 | 61.31 / 61.32 | 58.19 / 57.92 | **−3.26 ms** |

PCC 0.9999 unchanged. The copy-back is 8 × 1.5 MB × 48 = 576 MB/step at B=8, so the
gain scales with batch like everything else in the GDN layer. The typecast fold and the
TILE output remain, and both are kernel changes.

> **Trap, hit 2026-09-03 while wiring `inplace_state=True`.** The op returns
> `*in.initial_state` — the caller's own buffer — but the nanobind layer hands back a
> **new Python wrapper** around it, so `new_rec is init_state` is False even when the
> write was in place. A caller that uses `is` to decide whether to skip the copy-back
> falls into the copy-and-free branch and `ttnn.deallocate(new_rec)` frees `rec_state`:
> `TT_FATAL: Input Tensor is not allocated`, then a trace-capture hang and a
> `tt-smi -r`. Compare `buffer_address()`, never object identity, when a ttnn op may
> alias an input into an output.

**Both of the open questions below are now answered.** The op does the L2-norms
internally (so inputs are *not* pre-normalised and nothing stays outside for that), and
the `h_new` two-slot CB question is moot in the current wiring because `inplace_state`
is False — it becomes live only if the copy-back is removed.

**Sizing, from §1.2.** ~20 ops × (3.5 µs kernel + 1.2 µs launch) ≈ 94 µs per layer
removed; the fused kernel itself costs perhaps 10–15 µs (it reads and writes the 1.5 MB
fp32 state once and does two small matmuls) → **~80 µs × 48 = ~4 ms**, 4–5 ms with the
L2 norms folded in. §0.5 sized this at 2–3 ms by assuming the recurrence ops total
"50–80 µs of kernel"; the per-layer average of 3.5 µs/op says ~70 µs of kernel *plus*
~24 µs of launch, and the fused op does not pay 20 launches. The per-op rows §3 asks for
settle it either way.

**Scope.** In `forward_decode`, after `q/k/v/beta/g` are formed, call
`gdn_recurrent_step` with `T = 1` in place of `recurrent_gated_delta_rule_decode_ttnn`,
behind `QWEN36_GDN_FUSED_DECODE=1` (printing a line when it engages, per the branch's
own measurement rules). Keep the in-place `ttnn.copy` into `rec_state` so decode-trace
addresses hold, and the `B < Bmax` bucketed-width slice as-is. A/B with `traced_128` at
B=1 and B=8, and the same PCC gate the branch used for the composed-vs-fused check.

~~Open questions to settle in the first hour: whether the kernel's inputs are expected
pre-L2-normalised and fp32; and whether the `h_new` two-slot CB discipline the kernel
needs is satisfied when its state input *is* the persistent `rec_state` buffer rather
than a fresh tensor.~~ **Both answered above** — norms are internal, `inplace_state` is
False so the CB question does not arise yet.

**A-lite (optional, Python-only, ~1 ms).** The three 12 µs folds in §1.2 exist because
`recurrent_gated_delta_rule_decode_ttnn` folds `v_read` and `k` to heads-in-Y only so
`delta = v − v_read` can be elementwise, and `fused_decay_and_write_ttnn` then unfolds
both again (`k_col`, `d_row`, idx 57–59) for the outer product. Doing the subtract in
Z form — unfold `v` once, `transpose` `k_row` for the outer product — removes two of the
three folds; the one before the out-projection stays (the matmul needs a `[1, 3072]`
row). ~20–25 µs/layer. Only worth doing as a measurement of the relayout cost in
isolation, since A removes all of it.

### J. The bf4 read rate

Gate and up are the only bf4 weights, 6.4 GB of the 21.5 GB per token, and they are
read at 265–277 GB/s where the bf8 down-projection with the same tile count reaches
387 GB/s. If that gap is per-transaction cost on 576-byte pages — §1.2's reading of
"time tracks tiles" — then the fix is to issue larger reads, not to change the numbers:
the DRAM-width-sharded layout `model_config.py` already builds (`mlp_w1_weight_memcfg`,
`mlp_w1_progcfg`; taken by `mlp.py` when `mlp_1d_decode` is off) reads a contiguous
per-bank shard instead of one tile per request. Upstream's own comment records the
1D-mcast path beating DRAM-sharded by 2.5% at TP=4 (42.8 vs 43.9 µs on gate), which
says the DRAM-sharded config as shipped is *also* at ~55% — so the microbenchmark in
§3 sweeps `in0_block_w` and page size rather than just flipping the flag. Ceiling
~3.5 ms/step if bf4 reaches bf8's rate; zero if the DRAM microbenchmark shows ~390 GB/s
is the practical limit for every page size. Either answer is worth having, because it
also decides whether bf8 → bf4 on any other projection could ever pay.

### I. Packed gate|up for decode

**Closed 2026-09-03 without building it.** §3.2's N-sweep ran the merged shape: bf4 at
N=17408 (one `[gate|up]` matmul) costs **188.96 µs** against **2 × 88.41 = 176.8 µs** for
the two separate matmuls — merging is **12 µs/layer slower**, because the tuned 11×4
grid's per-core N doubles and the reader amortises worse, not better. The lever's
rationale was "halves the launches and shares the in0 multicast"; the same sweep put the
per-op fixed cost at ~0 (intercept −6 µs, within noise), so there was nothing to halve.
Net ≈ **+0.8 ms/step** before the two slices a decode split would add. The packed weight
stays what it is: a prefill-AGMM artefact.

Decode runs gate and up as two matmuls over the same activation; a packed
`[gate|up]` weight already exists for the prefill AGMM path (`w.w_gate_up`). One
matmul instead of two halves the launches and shares the in0 multicast; §0.5 sized it
at 10–20 µs of the combined 182 µs, so 0.6–1.3 ms/step. Cheap, and it composes with J.

### B0 — DONE 2026-09-02: −9.7 ms at B=8 (+15%), by deleting one assert

**The demo already had the fix and it was switched off.** `text_demo.py` defaults to
`QWEN36_BATCHED_DECODE_MODE="shard"` — per-shard on-device argmax+max, reading back two
tiny `[num_devices, B]` tensors instead of the full `[B,1,vocab]` logits. The README's
serve command and `batched2.sh` both force `"host"`, documented in the file itself as
*"legacy … Baseline"*, so **every batched measurement in this repo has used the legacy
path**.

`"shard"` failed at TP=2 on `model.py:3249`:

```
AssertionError: on_device_logits=True but self.sampling is None
```

That assert conflates two callers. `"sample"` mode needs the multi-device TopK sampler;
`"shard"` mode never touches it — it only wants the vocab-sharded logits so it can run
its own tile-parallel reduce. `self.sampling` is None at TP=2 because 124,160
logits/device exceeds TopK's 65,536 ceiling, so the assert blocked `"shard"` for a
sampler it does not use. The batch padding underneath it exists only to match the
sampler's ≥32 width, and `"shard"` reads `logits.shape[2]` and slices to B anyway.

Fix: drop the assert, make the padding conditional on `self.sampling`. `"sample"` mode
self-guards in the demo (falls back to `"host"` when the sampler is absent), so nothing
else is exposed. Measured, `batched_128_b8`, fused decode on in both arms, interleaved:

| arm | per-user | aggregate | `update` | `readback` | `exec_sync` |
| --- | ---: | ---: | ---: | ---: | ---: |
| host | 14.05 / 13.67 | 112.4 / 109.3 | 3.18 / 3.27 | 6.82 / 8.43 | 61.18 / 61.46 |
| **shard** | **16.04 / 15.84** | **128.3 / 126.7** | **0.38 / 0.58** | **0.59 / 0.80** | 61.37 / 61.75 |
| **mean** | **13.86 → 15.94 (+15.0%)** | **110.9 → 127.5** | −2.7 | −6.9 | unchanged |

Host round-trip **10.85 → 1.18 ms**. `exec_sync` does not move, so the whole gain is
host-side — the `sharded_lm_head=True` path costs the same on device as the gathered
one. Correctness is the strongest gate available: the generated text is **byte-identical
across host and shard, both repetitions** (md5 `941a5f7d…`), so the per-shard argmax
picks the same tokens as a host argmax over the gathered vocab.

**This resolves a README "what we could not make work" entry** — `"shard"` decode mode
at TP=2 — and it substantially delivers lever B for greedy workloads. What remains of B
is temperature sampling, which still needs the TopK path (TP=4, or the two-halves trick).

**Combined at B=8:** baseline 12.63 per-user / 101.0 aggregate → A alone 14.26 / 114.05
→ A + shard **15.94 / 127.5**, i.e. **+26% aggregate** from a flag that already existed
and a three-line assert fix.

**What B is now — carrying this to the endpoint.** The demo's `"shard"` mode lives in
`text_demo.py`; the vLLM path (`qwen36_vllm.py` → `process_output_decode`) still reads
the full logits and samples on the host, so the served endpoint has none of the 9.7 ms.
The plugin already has the contract: `model_capabilities["supports_sample_on_device"]`
is `True`, `sample_on_device_mode: "decode_only"` is a config key, and "requests that
cannot use TT on-device sampling automatically fall back to vLLM's host-side sampling
path … selected per batch". So B is: (1) let `_validate_device_sampling_request` accept
`(1,2)` when the batch is greedy-only instead of refusing on the TopK ceiling; (2) in
`decode_forward`, for a greedy batch, run the shard argmax+max from the demo on the
`sharded_lm_head=True` output inside the trace and return token ids
(`process_output_decode(is_tokens=True)`); (3) leave temperature > 0 to the per-batch
host fallback. Temperature sampling on device stays a TP=4 / two-halves item. Measure
on the endpoint with `vllm bench serve` at concurrency 8, not the demo — and note the
demo's `update` phase also fell 3.2 → 0.4 ms under `"shard"`, so the RoPE-gather
half of the original B0 analysis is unconfirmed as a separate mechanism; check the
plugin's own input-prep time before building it.

### B0 (original analysis, kept — the readback half is now measured, the RoPE half is not)

The 8.7 ms at B=8 splits into `readback` 5.65–6.40 and `update` 2.35–3.26, and neither
needs the sampler work that makes B "days":

* **Readback.** `process_output_decode` calls `ttnn.to_torch` on a TILE-layout
  `[8, 1, 248320]` bf16 replica — ~4 MB — and the host untilizes it. The spec-decoding
  branch measured exactly this and fixed it with a device-side `ttnn.to_layout(...,
  ROW_MAJOR_LAYOUT)` before the read (`SD_DRAFT_RM_READ=1`, 13.7 → 6.7 ms in that
  path). The same call at the end of the decode trace, on the sharded LM-head output
  before the all-gather if possible, should take the 5.65–6.40 ms to roughly half.
  *E: −2.5 to −3.5 ms at B=8.*
* **Update.** `_tt_vllm_always_refresh_decode_trace_inputs = True` with the comment
  "RoPE is host-recomputed each step": eight users' cos/sin rows are built on the host
  and copied to device every token. tt_transformers' standard decode path gathers them
  on device from a persistent table via `rot_mat_idxs`; wiring that in leaves only the
  token ids and positions to copy. *E: −1.5 to −2.5 ms at B=8.*

Both are measurable with the same `QWEN36_DEBUG_DECODE_TIMING=1` phases that found them,
and both survive B (on-device greedy still needs the update path, and the readback
change is what B's fallback path uses for non-greedy requests).

### B. On-device greedy at TP=2 — demoted at B=1, **re-promoted at B=8**

*M, and the batch dependence is the whole story:*

| | `readback` | `update` | host total | share of step |
| --- | ---: | ---: | ---: | ---: |
| B=1 | 0.31 | 0.28 | **0.59 ms** | 1.1% |
| B=8 | 5.65–6.40 | 2.35–3.26 | **~8.7 ms** | **11%** |

At B=1 the ceiling really is ~1–1.8 ms and the lever is not worth days of vLLM
plumbing. **At B=8 it is ~8.7 ms — larger than A's B=1 gain and the second-biggest
lever in the table** — and B=8 is what the endpoint actually serves. §0.5's "demote it"
was measured at B=1 and should not have been generalised; the original 3–7 ms estimate
was closer to right for the batched case. Mechanism unchanged:
`_forward_decode(..., sharded_lm_head=True)` returns each device's 124,160-wide shard,
reduce it on device to `(max, argmax)` per user with the demo's tile-parallel
`ttnn.max` reshape, read back two tiny tensors, widen the two `(1,4)/(1,8)` allowlists.

### C. bf16 GDN step — a one-line A/B

**Measured 2026-09-03 — on the fused path it is two flags, not one.** The fused op
asserts a single input dtype, and `rec_state` is fp32 by default, so
`QWEN35_GDN_DECODE_BF16=1` alone would fail the assert; it needs
`QWEN35_GDN_STATE_BF16=1` too (bf16 recurrent state), which is what "halves state
traffic" in the original text actually means. A + in-place on in both arms, interleaved:

| `exec_sync` | fp32 (current) | bf16 step + state | Δ |
| --- | ---: | ---: | ---: |
| B=1 | 51.18 / 51.20 | 49.41 / 49.41 | **−1.78 ms** |
| B=8 | 57.84 / 58.00 | 54.36 / 54.45 | **−3.52 ms** |

Pairs agree to 0.01 ms at B=1. Larger than the ~1.2 ms estimated because the bf16
state halves 1.5 MB/layer of read+write, not just the five casts. **Not yet adopted:**
the demo's PCC gate is at position 0 and cannot see decay-quantisation drift; the gate
is GSM8K (2,048-token reasoning chains). **Gate passed 2026-09-03: 57/60 = 95.0%, 57/59 excluding one unparseable — the identical three items missed as fp32, with A and D also on.** No drift visible at 2k tokens. **Adopt** `QWEN35_GDN_DECODE_BF16=1 QWEN35_GDN_STATE_BF16=1`. (Longer generations than 2k remain unexercised, as for D.)

**Cumulative device step at B=1** (`exec_sync`, this rig, same session): composed 54.45
→ A 51.9 → + in-place 51.2 → + C **49.4 ms (−9.3%)**, before E's independent −1.35 ms.
At B=8: 70.6 → 61.2 → 58.0 → **54.4 ms**, aggregate 101 → 133 tok/s on host readback,
before B0's shard readback.

`QWEN35_GDN_DECODE_BF16=1` drops the five fp32 typecasts per layer — 240 ops at
~4.7 µs each is ~1.2 ms — and halves state traffic. Upstream defaulted to fp32 to avoid
decay quantisation drift over long generations, so gate it on a 2k-token generation
compared token-for-token against fp32 and on GSM8K. Subsumed by K if K lands.

### D. bf8 KV — §1.4. Config only. Gate on long-context PCC.

**Measured 2026-09-03 on the endpoint at 262k** (`QWEN_SDPA_BF8=1`, same serve command
and prompt as §2.3's acceptance run, single stream, streamed):

| | bf16 KV | **bf8 KV** | Δ |
| --- | ---: | ---: | ---: |
| TTFT | 222.3 s | **171.6 s** | **−23%** |
| median ITL | 82.6 ms | **71.8 ms** | **−10.8 ms** |
| decode | 12.11 tok/s | **13.92** | +15% |
| short request | 16.03 | 16.32 | unchanged |
| preemptions | 0 | 0 | |

§1.4 estimated 8–10 ms; the README's 64k figure was −12% TTFT, and at 262k the
quadratic layers are a larger share so TTFT moves −23%. 13.92 tok/s is within 5% of
TT's 4-die 256k target on two cards. **Not yet adopted:** this run captured timing only.
The gate the README and §1.4 require — PCC and GSM8K at long context — is still
outstanding, and it is the only thing between this number and the serve command.

**GSM8K gate, arm 1 (2026-09-03) — bf16 KV, lever A on, the README's harness verbatim
(60 items, 8 concurrent, `max_tokens=2048`, greedy): 57/60 = 95.0%; 57/59 = 96.6%
excluding one unparseable.** The README's composed-path result was 58/60 (58/59). The
misses are the README's known `13 → 12` off-by-one, the same single unparseable, and
**one new miss** (`60 → 40`). One item in sixty is inside the test's resolution, but it is
also exactly what a ~1e-6 fused-vs-composed perturbation looks like after 2,048 tokens of
reasoning (`docs/speculative-decoding.md`: "long generations diverge — typically after ~9
tokens"). It is the first downstream-accuracy number lever A has; a 200-item run would
settle whether it is noise. Ready in 165 s; 60 items in 199 s.

**Arm 2 — bf8 KV: 57/60 = 95.0%, 57/58 = 98.3% excluding two unparseables. D passes.**
Same score as bf16; bf8 *recovers* the `60 → 40` miss and instead produces one more
unparseable (`70000 → None`) — a formatting failure, not a wrong number. The off-by-one
and the trains item are common to both arms and to the README. **Adopt `QWEN_SDPA_BF8=1`.**
Caveat kept in view: GSM8K prompts are short, so this gate covers bf8-KV precision over
≤2k tokens; the "at long context" half of §1.4's gate — the 262k prompt's output compared
bf16 vs bf8 — is still unmeasured (the §2.3 acceptance run captured timing only).

**Long-context output check, 2026-09-03.** Same 261,000-token prompt, greedy, 64 tokens,
bf16-KV vs bf8-KV, both with A on: the continuations are **byte-identical** (241 chars,
md5 `62299fc53b22` both). In the same pair the bf8 arm reproduced its gains at the
longest context the endpoint serves:

| arm | TTFT (262k) | ITL at 262k | decode |
| --- | --- | --- | --- |
| bf16 KV | 221.2 s | 84.9 ms | 11.78 tok/s |
| bf8 KV | 171.8 s | 72.9 ms | 13.71 tok/s |

With GSM8K 57/60 at 4k and an identical top-1 stream at 262k, D has no remaining
measured caveat; what stays unexercised is sampled (temperature>0) output, where a
bf8 logit perturbation could flip low-margin draws.


### E. Fix the link table — **done, −1.35 ms/step**

Full derivation in §0.5. Second cable fitted, 4 links up, `p150_x2` descriptor now
validates. The gain was unreachable until `get_num_links`' static
`link_dict["P300"] = (2, 2)` was corrected: two cabled p150a are named "P300" by
device count and so inherit a real p300's on-package link count.

**The change to upstream** is one table entry, but it cannot be a literal — a genuine
p300 really does have 2 links. It needs to key off measured topology (the UMD cluster
descriptor already knows: it reported 4 `ethernet_connections` here) rather than off
`_determine_device_name`. Worth an upstream issue; the docstring already promises
control-plane querying that the implementation does not do.

**`QWEN_CCL_RS_PREFILL_LINKS` is moot for now.** `matmul_reduce_scatter_prefill` is
called from exactly one place, `gdn/tp.py:631`, behind `if self._fuse_out_mmrs_prefill:`
— which is **off by default**, so the path never runs. Setting the flag at 8k produced
no `[CCL_LINKS]` line, confirming it. If that fusion is ever enabled, raising its links
is *not* free: `_mmrs_prefill_placement` sizes RS workers as `n_rs = num_links × 2 × 9`
and reserves `ceil(n_rs / W)` rows, taking them from the matmul — on this 11×10 worker
grid, 8×6 = 48 cores down to 8×3 = 24 — and its docstring warns of "the collision that
deadlocks a full-grid fused CCL".

Relatedly, `matmul_reduce_scatter_decode` (`tp_common.py:467`) has **zero call sites** at
this version. Its `num_links=1` is dead code, not a live decode setting.

### F. Shift register

Four `ttnn.copy` per GDN layer per token (192/step) exist only to keep buffer addresses
stable for trace replay: ≲ 0.3 ms of launch plus ~0.7 ms of kernel floor. Folds into
K's first kernel; not worth doing on its own.

### G. Prefill / TTFT at 262k

At 262k the 16 attention layers' O(T²) FLOPs (~13.5 PFLOP) roughly equal the linear
layers' (~14 PFLOP), so attention efficiency is the whole superlinear term the README
documents. Known knobs: `QWEN_SDPA_BF8=1` (−12% TTFT measured at 64k), the
`_PREFILL_WARMUP_CHUNK = 2048` constant in `qwen36_vllm.py` (larger chunks amortise
per-chunk overhead; the model's `long_prefill_chunk_size` is what the demo uses), and
the SDPA q/k chunk sizes in `attention/tp.py` (comment: 128 was faster than 64/256 at
the time). Profile before touching any of it — and note again that TT's own 4-die 256k
TTFT is *worse* than yours, so this is not a hardware problem either.

---

### Closed on measurement

Kept as the record of what was tried; neither is a live lever.

#### H (closed). Decode matmul core count is already tuned — and the diagnosis was wrong

**First, a false trail worth recording.** `tp_common._find_grid` caps the grid at
`max_r, max_c = 8, 8` and picks the divisor nearest `target=32` — a Wormhole grid
assumption on a part with **11×10 = 110** worker cores. That looks like an obvious
bug. It is not reachable: `_find_grid` only feeds the **DRAM-sharded** decode matmul,
and MLP decode does not take that branch. Overriding max/target to 10×11/80 changed
the emitted config (`per_core_N` 8 → 3) and changed **nothing** measurable — core
count stayed 39, runtime flat across four arms.

**What actually runs** is the 1D mcast path, already enabled
(`model_config.py:187 self.mlp_1d_decode = True`) with Blackhole-width grids
(`decode_grid_w = compute_with_storage_grid_size().x = 11`) and per-shape tuned core
counts: gate/up `num_cores=44` → 11×4, down `num_cores=33` → 11×3. Our profile's
`CORE COUNT=32` for down matches 11×3 exactly; the DRAM-sharded path would have given
16. **We were on the tuned path the whole time.**

The `wide1d_*` comments quote ~63 µs for down where we measure 122 µs. That is not a
missing win — 63 µs would be **147% of the 512 GB/s bandwidth floor** for this
matmul. Those figures were measured at TP=4, where each device holds half the weight;
63 × 2 ≈ our 122.

**Sweeping `num_cores` at TP=2** (announced per run, both baselines bracketing):

| w1/w3/w2 | cores | gate | up | down |
| --- | --- | ---: | ---: | ---: |
| 44/44/33 (tuned) | 39/39/32 | 94.59 | 87.73 | **122.20** |
| 88/88/66 | 68/68/54 | 91.63 | 85.96 | 129.36 |
| 66/66/55 | 55/55/54 | 91.40 | 88.44 | 129.13 |
| 110/110/110 | 91/91/80 | **90.18** | 86.59 | 124.00 |
| 22/22/22 | 21/21/20 | 101.73 | 87.95 | 127.18 |
| 44/44/33 (repeat) | 39/39/32 | 95.39 | 88.99 | 121.83 |

Down is worse at every larger grid; up is flat; gate gains ~5% (≈0.3 ms/step over 64
layers). **The TP=4 tuning transfers to TP=2.**

**And quantising further would not help.** gate (bf4) and down (bf8) process *identical*
tile counts — 160×272 vs 272×160 = 43,520 each — and differ only 1.3× in time despite
bf8 moving **2× the bytes**. Time tracks tiles, not bytes: bf4 already converts only
about half its byte saving into time saving. Another quantisation step would shrink
bytes that are not the binding constraint while adding dequant work per byte.

**What is left**, unmeasured: decode runs gate and up as two matmuls over the *same*
activation, and a packed `[gate|up]` weight already exists (`w.w_gate_up`, used by the
prefill fused path). Fusing them for decode halves the dispatches and shares the
activation broadcast — worth perhaps 10–20 µs of the combined 182 µs, so ~0.6–1.3 ms
per step. That is the only remaining matmul-side idea with a plausible mechanism.

#### CPU offload: rejected, with numbers

Considered and ruled out on three independent grounds.

* **No capacity problem to solve.** 13.75 GiB/device on 32 GiB cards, ~18 GiB spare.
* **Host bandwidth is 10× below the mesh.** Threadripper PRO 9975WX, and dmidecode
  reports **8 memory slots with only 2 populated** (64 GB @ 6400 MT/s) — 2 of 8 DDR5
  channels, ~102 GB/s theoretical, 21.7 GB/s measured single-thread memcpy, against
  512 GB/s per card and 1024 GB/s for the mesh. One MLP layer unsharded (the CPU gets
  no TP) is 267 M params = 535 MB at bf16; at an optimistic 80 GB/s that is **6.7 ms**
  against **0.346 ms** on the mesh.
* **It breaks the trace, and the trace is worth 7.5×.** Decode is one captured trace
  replayed at 1.63 µs/op. A mid-stack offload forces a host sync that splits it, and
  the eager arm already measured that regime: **17.84 → 2.39 tok/s**, 81 µs/op.

Unrelated but worth acting on: **6 of 8 memory channels are empty.** Filling them
~4×s host memory bandwidth. It cannot touch the device step, but it helps the host
untilize (1.07 ms/step), the 27.5 GiB weight load, and prefill data prep.

## 5. Speculative decoding: MTP, EAGLE3, and the arithmetic

Your write-up already has the model: `speedup = mean_accepted × baseline / total`. On a
GPU, verifying K tokens costs about the same as one, so acceptance ≈ speedup. Here,
*measured*: verify at K=2 costs 102.9 ms against a 61.9 ms baseline — the second token
costs 66% of a full step, because the GDN multi-token recurrence is sequential in T and
the small-M matmuls do not amortise. That is what puts break-even at 1.94 and the result
at 1.05×.

§1.2 says why the second token is so expensive: ~23 ms of every step is per-layer fixed
cost that a second token pays again in full, and the matmuls at M=2 cost the same as
at M=1. **EAGLE3 does not change this.** A community EAGLE3 head exists for Qwen3.6-27B
(`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`, ~0.6 B params, fuses target hidden states from layers
1/31/60): it reports **acceptance length 2.2** on stock Qwen3.6-27B and 1.84× on a GPU
with chain drafting — its author notes tree drafting "neutralises gains on this hybrid
GatedDeltaNet target". At acceptance 2.2 against a break-even of 1.94, with a draft head
that is itself a 0.6 B forward per draft token plus a three-layer hidden-state tap, the
expected result on this stack is ~1.1×. There is no head for Qwen3.8-27B; training one
is a self-distillation run on tens of millions of tokens. And `vllm-tt-plugin` does not
support speculative decoding at all, so like MTP it would live in the demo harness.

**What does change it: K.** The fixed cost is mostly the GDN small-op floor, and a fused
GDN layer pays it once per layer regardless of T — the branch's own fused verify (1.27×
→ 1.53×) is that effect for the recurrence alone. If K takes the per-layer fixed cost
from ~23 ms towards ~12 ms, verify at K=2 drops from 1.66× to roughly 1.3× a plain step,
break-even falls to ~1.5, and the same acceptance of 2.04 clears it comfortably. Tracing
the MTP head (blocked today by the width-1/width-T trace hang you documented) is worth
~5 ms/step on top.

### The "EAGLE3 is 5× on decode" figure, and why it does not transfer

The EAGLE-3 paper's headline is "up to 6.5×". That number is batch 1 on a GPU, with
**tree** drafting (dozens of candidate tokens verified in one forward), against
*vanilla HuggingFace decoding* — a baseline that leaves most of the GPU idle at M=1,
so verifying 60 tree tokens costs about the same as verifying one. Production engines
with chain drafting report far less: vLLM's own EAGLE-3 write-up gives up to 1.8× on
Llama-3.1-8B and 1.6× on Llama-3.3-70B, "up to 2.5×" on RAG/maths, and notes vLLM does
not support tree decoding at all; the SGLang figure in the paper is 1.38× at batch 64.
On this model family the community head reports 1.84× at acceptance 2.2 and says tree
drafting "neutralises gains on this hybrid GatedDeltaNet target".

On these cards the ceiling is lower still, and it is arithmetic, not pessimism. The
whole gain of speculative decoding rests on the marginal token in verify being nearly
free. *Measured here*: it costs **66% of a full step** (verify at K=2 = 102.9 ms vs
61.9), because the GDN recurrence is sequential in T and the per-op floors are paid per
op regardless of M. With `verify(K) ≈ base × (0.34 + 0.66 K)`:

| accepted per step | K needed | verify(K) / base | speedup (draft ≈ 0.2 base) |
| ---: | ---: | ---: | ---: |
| 2.0 | 2 | 1.66 | 1.08× |
| 3.0 | 4 | 3.0 | 0.94× |
| 5.0 | 6 | 4.3 | **1.1×** |

Five accepted tokens per step — which itself needs a tree and a drafter far better than
2.2 — would buy 1.1×. The lever is the 0.66, not the drafter, and the 0.66 is §1.2's
small-op floor wearing a different hat: 48 layers × ~68 ops paid again for every extra
token. That is why K comes before any speculative work, and it carries a design rule:
**build K's kernels T-aware from the start** (T as a runtime argument, all T states
emitted — the shape `gdn_recurrent_step` already has). If the marginal token then costs
~0.15 of a step instead of 0.66, verify(4) ≈ 1.45 base and acceptance 3 gives ~1.9×;
that is the realistic ceiling for speculative decoding on this hardware, and the only
route to it runs through K.

Order: A, then K, then re-run `spec_generate.py` — and only then decide whether EAGLE3 is
worth training a head for.

**First re-run on the A baseline, 2026-09-03 (harness defaults, fast paths OFF).** The
harness's own baseline: 64 traced tokens at **59.7 ms/step** (16.7 tok/s; the doc's
61.9 before A — the harness carries the hidden-retention patch and its own host loop, so
it sits above the demo's 51.9). Then two findings that matter more than the speed:

* **The rollback gate fails with A on.** `SD GATE SUMMARY rollback 1/3 depths reproduce
  the reference, control separates at 1/3 → SD CHECK FAILED`. Mechanism, pending the
  control run: the T=1 fused decode op (`decode_gated_delta_rule`) and the T=3 verify op
  (`gdn_recurrent_step`) are not bit-identical, so continue-after-rollback diverges from
  the sequential reference that the gate asserts on. The doc already noted fused-vs-
  composed agreement is ~1e-6 and "long generations diverge — typically after ~9
  tokens"; with A on the *baseline* path, that divergence now lands inside the gate.
* **Acceptance fell to 1.882** (from 2.04) and spec ran at 10.9 tok/s = **0.65×** — but
  with `fast_read=False` and no fast carry, i.e. the unoptimised arm; not comparable to
  the 1.05× headline. Three runs queued to separate the effects: composed baseline
  (gate control), composed + fast paths (reproduce 1.05×), fused + fast paths (the
  re-run §5 asked for).

**Isolating matrix (same prompt, K=2, 64 tokens; each spec number against its own
baseline from the same run):**

| arm | A | fast paths | baseline ms/step | spec tok/s | ms per spec step | mean accepted | spec/baseline | rollback gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sd-offA-slow` | off | off | 61.6 | 15.08 | 137 | 2.065 | 0.93× | **3/3 pass** |
| `sd-offA-fast` | off | on | 62.8 | 16.78 | 127 | 2.129 | **1.05×** | 1/3, verify rows differ |
| `sd-onA` | on | off | 59.7 | 10.90 | 173 | 1.882 | 0.65× | 1/3 |
| `sd-onA-fast` | on | on | 61.0 | 13.55 | 145 | 1.970 | 0.83× | 1/3, verify rows differ |

Three readings:

1. **The harness is sound**: composed decode with defaults reproduces the sequential
   reference after rollback at all three depths. The README's 1.05× reproduces exactly
   under its own config (`SD_FAST_CARRY=1 SD_DRAFT_RM_READ=1`).
2. **The fast paths fail the rollback gate on their own** (A off): 1/3 depths, and the
   width-3 verify rows no longer match the sequential reference (`[279, 11, 279]` vs
   `[8831, 11, 279]`). The README did not claim gate status for that config; this is a
   new observation, recorded, not a regression from this work.
3. **A makes speculative decoding worse, not better.** With A on, each spec step costs
   18–36 ms *more* (145 vs 127 ms with fast paths, 173 vs 137 without), acceptance
   drops by 0.1–0.2, and the gate fails even with the fast paths off. Spec's T=1 steps
   evidently do not get the fused op's saving and pay something extra instead — the
   rollback path restores state from a snapshot, so `init_state is self.rec_state` is
   false and the wrapper's stable-state assumptions do not hold. Not diagnosed; a
   per-phase profile pair (`SD_DRAFT_PROFILE=1`, A on vs off) was run — below.

**Per-phase profile (fast paths on, `SD_DRAFT_PROFILE=1`; profiling adds sync points, so
absolute numbers sit above the unprofiled runs and only the A-on/A-off *difference* is
the reading):**

| ms per spec step | A off | A on | Δ |
| --- | --- | --- | --- |
| verify (T=3 op) | 127.9 | 103.3 | −24.6 (within this harness's run-to-run scatter under profiling) |
| draft (MTP) | 28.4 | 28.2 | 0 |
| commit, per step | 14.2 | 32.1 | **+17.9** |
| commit, **per rollback** | 20.6 (22/32 steps) | **49.7** (20/31 steps) | **+29.1** |
| total | 172.9 | 164.7 | |

The draft is untouched by A and the verify difference is noise-sized, but **a rollback
commit costs 2.4× more with A on**. Spread over the ~2/3 of steps that roll back, that is
the +18 ms per spec step the unprofiled arms showed. The commit restores the 48 layers'
recurrent state from the snapshot; with A the state the fused op leaves behind is not in
the same place/layout the composed path left it, so the restore stops being a plain
copy. That is fixable on the harness side (snapshot/restore in whatever config the fused
op uses) and is not a property of A in normal serving, where nothing rolls back.

Reading the commit path (`test_gdn_decode_multi.py:commit_prefix`) narrows it further: a
rollback is one state-slice copy into `rec_state` plus the conv-carry sync (two ops with
fast carry), and neither reads anything A changes. So the doubled "commit" is more
likely *attribution* — the first blocking op after A's asynchronously enqueued T=1 draft
steps landing in the commit bucket — than extra copies. The harness has a mode for
exactly this (`SD_DRAFT_PROFILE=2`, sync after enqueue); the pair below.

**Sync-after-enqueue profile (`SD_DRAFT_PROFILE=2`):**

| ms per spec step | A off | A on |
| --- | --- | --- |
| verify | 103.2 | 102.2 |
| draft (of which rope enqueue / MTP) | 18.7 (1.7 / 7.1) | **90.8 (29.7 / 43.6)** |
| commit (per rollback) | 6.6 (9.7) | **84.2 (112)** |
| hidden | 0.8 | **39.7** |
| mean total / median | 129.4 / 126.3 | **316.9 / 145.6** |

With every enqueue synced, the A-on penalty is no longer in one phase: it lands wherever
the host next touches the device, including a 30 ms *enqueue* of the two tiny rope
tensors, and the mean sits far above the median. That is the signature of intermittent
command-queue stalls, not of extra copies — the harness's spec loop runs its T=1 steps
eagerly with the fused op while traces are captured and replayed around it, and some
steps block for hundreds of ms. The steady-state penalty (median +19 ms/step) agrees
with all three earlier pairs. Root-causing this is harness work (eager fused op inside
a trace-replay loop, program-cache behaviour) and does not touch serving, where the
fused op runs inside the decode trace and nothing rolls back.

**Decision:** speculative decoding stays off in the shipped configuration. The §5
break-even argument assumed the sequential and spec paths share the per-token cost; with
A that is no longer true, and the correct fix is on the spec side (drive its T=1 steps
through the same fused op and stable-state path), not a reason to hold A back.

---

## 6. The 4× p150a upgrade

What it buys, from §1.3: ~22 tok/s with host sampling, ~26 with on-device (which becomes
legal: 62,080 logits/device), plus every TP=2-only limit in the README disappears —
`B=32`'s divisibility assert, batched prefill above B=4 (`BH = B × 48/4 = 96 ≤ 110`),
`"shard"` decode mode, and the three unmerged PRs (#53320 is still open with zero
reviews; `P300` is still not in `main`'s mesh map). Decode at 256k would be ~15 tok/s.
TTFT should not be assumed to improve.

Two things to check before the cables arrive:

* **Descriptor.** The shipped `p150_x4` descriptor is `dims [2,2]` with **4 channels**
  per edge — two cables per neighbour pair, eight cables for a 2×2. The shipped
  `p300_x2` descriptor is `dims [2,2]` with **2 channels**, i.e. exactly four cards in a
  square with one cable per edge. Since `determine_device_name` keys purely off device
  count (4 → `P150x4`), four p150a with four cables and
  `TT_MESH_GRAPH_DESC_PATH=…/p300_x2_mesh_graph_descriptor.textproto` is the same thing
  tt-metal sees on a QuietBox 2 — which is the box TT's CI targets were measured on.
  Choose from measured link count, as the gotchas doc says.
* **Logical (1,4) on physical 2×2.** The demo opens `P150x4` as a `(1,4)` mesh; on a
  2×2 the fabric maps the line onto the square. TT's own CI does it, so it works — but
  the CCL path over a square with one cable per edge is not the same as a true 1×4 line
  with 4 links per hop, so do not carry P150x4 CCL expectations across unmeasured.

---

## 7. Sequence

| Step | What | Output |
| --- | --- | --- |
| 1 | ~~§2.3 command, acceptance tests 1–3~~ **done 2026-09-03**: 262k served through vLLM, TTFT 222.3 s (README 219.9), decode 12.11 tok/s single-stream, 0 preemptions, ready in 210 s. GSM8K (test 4) not yet run | numbers in README; D (bf8 KV) queued |
| 2 | ~~§3.1~~ **done** (K 11.0 / A 5.04 ms). ~~§3.2~~ **done**: bf4 penalty is ~1.4 ns per-tile transaction cost, intrinsic — `in0_block_w` and DRAM-sharding do not move it; **J closed at 0**. tt-metal's own DRAM read peaks at **~430 GB/s (84% of spec)**; matmuls reach 391 | floor restated ~26–28 ms; lever L found and closed (+3 ms end-to-end) |
| 3 | ~~§4 A~~ **DONE end to end.** B=1 −2.55 ms (+4.7%), B=8 −9.5 ms (+12.9%, 101 → 114 aggregate), endpoint +15.5% per user / +18.9% aggregate. PCC 0.9999, output byte-identical | shipped: add `QWEN_GDN_FUSED_DECODE=1` to the README serve command |
| 3b | ~~§4 B0~~ **done: `"shard"` mode at TP=2, −9.7 ms at B=8, byte-identical output** | host round-trip 10.85 → 1.18 ms (demo only) |
| 3c | A's leftovers: ~~`inplace_state=True`~~ **done −0.42 ms B=1 / −3.26 ms B=8**. Typecast fold and TILE output remain — both kernel changes (reader / writer of `decode_gated_delta_rule`) | A's last ~1 ms, kernel work |
| 3d | §4 B: expose the shard-greedy path through `sample_on_device_mode` so the **endpoint** gets B0 too (the plugin's own guard, `qwen36_vllm.py:63`, is separate from the `model.py` assert B0 removed) **In progress 2026-09-03:** a greedy-only stand-in for `SamplingGenerator` (`wrap-3e/model.py`: no-op seed manager and trace hooks, `sample()` passes the in-trace per-shard reduce through, host picks the winner) is accepted by the Generator, the reduce runs inside the decode trace, and the first arm reached vLLM warmup before `format_sampling_params` asserted on width 8 (the real sampler always reports 32; fixed). Re-queued behind the spec profile pair. Shard-off controls from the same queue (A on, 8 streams, ITL): 71.69 ms / 13.95 per user / 102.5 aggregate and 70.83 ms / 14.12 / 109.8. Second shard arm reached vLLM warmup and failed in the host pick: the warmup decode bucket is one token wide while the Generator asks for the full width, and the pick indexed past it (the plain token path just truncates; now mirrored). Third round queued. **Works (2026-09-03 02:47, arm `s1d`):** the sampler stand-in plus in-trace reduce served the endpoint end to end — same greedy text as the controls, and at 8 streams ITL **66.89 ms / 14.95 per user / 116.8 aggregate** against the nearest shard-off control at 68.45 / 14.61 / 113.5 (controls drift down as host load falls: 71.69, 70.83, 68.45). Preliminary delta −1.6 ms (−2.3%), well under the −9.7 ms the demo showed, which is consistent with the endpoint already overlapping its logits readback with host work. Interleaved round done — **result below.** | endpoint per-user / aggregate at B=8 — measure by **ITL, not whole-request wall time** (whole-request controls scatter 41%) |
| 4 | §4 K: conv+gates kernel, then norm+gate kernel — measured at **B=8 first** | GDN layer from ~0.42 to ~0.25 ms |
| 5 | ~~§4 J~~ closed (per-tile cost intrinsic). ~~§4 I~~ closed (merged matmul is +12 µs/layer on the N-sweep). ~~§4 L~~ closed (+3 ms end-to-end despite −20% on the matmul) | three measured negatives; no matmul-side lever remains at TP=2 |
| 6 | §4 D **measured on the endpoint at 262k: −10.8 ms ITL, −23% TTFT**; GSM8K gate (bf16 vs bf8 KV, README harness) **running**. §4 C not yet run | D adopted iff GSM8K holds; C is a one-line A/B |
| 7 | §5: re-run `spec_generate.py` on the faster baseline | new break-even; decision on MTP trace / EAGLE3 |
| 8 | ~~§3.3 per-module profile at B=8~~ **done**: GDN small-op floor 11.1 → 22.4 ms/step, attention 1.6 → 3.5, projections and MLP flat. K's B=8 ceiling is 22.4 ms; J and I are batch-invariant | the batched endpoint's own lever table (§3.3) |
| — | §8 research targets R1–R4 (separate session, no hardware) | port / no-port per row; §1.4, §2.4, §5 numbers updated |
| — | §9: third card = K dev card now; fourth card → TP=4 for latency, or prefill/decode split (§9.2) for the 262k stall | 26 → ~38 tok/s with K; decode never blocked by prefill |

**Row 3d result (2026-09-03, endpoint, A on, 8 streams, ITL with first token dropped, arms interleaved):**

| arm | shard greedy | ITL | per user | aggregate | notes |
| --- | --- | --- | --- | --- | --- |
| s0a | off | 71.69 ms | 13.95 | 102.5 | |
| s0b | off | 70.83 ms | 14.12 | 109.8 | |
| s0c | off | 68.45 ms | 14.61 | 113.5 | |
| s0d | off | 71.15 ms | 14.05 | 109.8 | text md5 `d46f1ed349ef` |
| s1d | **on** | 66.89 ms | 14.95 | 116.8 | |
| s1e | **on** | 64.50 ms | 15.50 | 118.7 | |
| s1f | **on** | 67.10 ms | 14.90 | 101.0 | text md5 `d46f1ed349ef` (identical) |

Shard off: mean 70.5 ms (range 68.5–71.7). Shard on: mean 66.2 ms (range 64.5–67.1).
**−4.4 ms ITL (−6.2%), +6.6% per user**, ranges do not overlap across seven arms taken
under host load 10–40. Aggregate follows (108.9 → 112.2 mean) but one shard-on arm lost
its all-concurrent window to stream skew, so ITL is the number. Correctness: 8 × 200
greedy tokens byte-identical to the host-argmax control, and the 24-token probe matched
on every arm. The gain is under half the demo's −9.7 ms because the endpoint already
overlaps its logits readback with vLLM's host work; what remains is the readback itself
(8 × 124,160 × 2 devices of bf16 → 32 floats).

How it ships: mount `wrap-3e/model.py` (a greedy-only stand-in for `SamplingGenerator`
plus the in-trace per-shard reduce and the host winner pick), set
`QWEN36_SHARD_GREEDY=1`, and pass `"sample_on_device_mode": "decode_only"` in the
additional config. **Caveat, must be respected:** in that mode every decode is sampled
on device, and the stand-in is greedy only — a request with temperature > 0 silently
gets argmax (warned once in the log). Enable it only on endpoints that serve greedy
traffic, or first extend the plugin's request validator to reject temperature > 0.

**Composed ship configuration on the endpoint (2026-09-03 03:23–03:47, interleaved base,
ship, ship, base; A on in all four; ship adds C, D and shard-greedy; kernel cache mounted):**

| arm | config | ITL | per user | aggregate | output md5 |
| --- | --- | --- | --- | --- | --- |
| base-a | A only | 69.03 ms | 14.49 | 111.1 | `d46f1ed349ef` |
| ship-a | A + C + D + shard | **60.75 ms** | 16.46 | 128.8 | `6438f6f7e690` |
| ship-b | A + C + D + shard | **61.65 ms** | 16.22 | 108.6 | `6438f6f7e690` |
| base-b | A only | 67.62 ms | 14.79 | 115.2 | `d46f1ed349ef` |

Base mean 68.3 ms, ship mean 61.2 ms: **−7.1 ms ITL (−10.4%), +11.6% per user**; aggregate
+4.9% but one ship arm lost its all-concurrent window to stream skew, so ITL is the number.
Engagement: A and shard-greedy print their lines; C and D are read straight from the
environment by the model (`gdn/tp.py:238`, `gdn/tp.py:1103`, `attention/tp.py:138`,
`model.py:2685`) and print nothing, so their engagement is evidenced by the changed
output (shard-greedy alone is byte-identical; only C/D move numerics) and by the ITL
drop exceeding shard-greedy's own −4.4 ms. The ship output is deterministic (both ship
arms hash the same) and diverges from A-only after ~15 tokens of the probe ("located on"
→ "located at"), which is what the GSM8K gates were for. The full README table is
against the composed baseline; relative to it this endpoint stack is A's +15.5%/user
compounded with the +11.6% here.


Every step lands as a row in the README's results tables, with the same
interleaved-A/B discipline the spec-decoding doc describes. §4 E is done and §4 H is
closed; both stay in the doc as the record of what was measured.

---

## 8. Research targets for another session

Four bounded questions whose answers change a number in this document. Each has a
port / no-port criterion so the session can close it either way. GPU kernel work does
not transfer to Tensix and is out of scope; what transfers is algorithm and data.

| # | Question | Where to look | Port if | Feeds |
| --- | --- | --- | --- | --- |
| R1 | How does a single-kernel GDN recurrence step handle L2-norm, the fp32 state, and the T loop? | `flash-linear-attention` — `fused_recurrent_gated_delta_rule` (Triton), and its `chunk_gated_delta_rule` for prefill; Qwen3-Next / Qwen3.5 model code in vLLM and SGLang | the kernel does norm + decay + read + delta + write + output in one pass with per-head state resident on-chip — mirror that op boundary in K's three kernels, and take its T-loop structure for the T-aware rule in §5 | §4 K scope; §5 ceiling |
| R2 | How do vLLM and SGLang do prefix caching for hybrid GDN models? | vLLM "mamba prefix caching" / `mamba_block_size` for Qwen3-Next and Qwen3.5; SGLang's hybrid-state cache for the same models | block-aligned snapshots of the recurrent + conv state at a cache-block granularity, restorable per request — the design the TT plugin would need to drop `supports_prefix_caching: False` and make 262k a chat mode rather than one-shot (§2.4) | §2.4; §6 |
| R3 | What acceptance does the Qwen3.8-27B MTP head (and any EAGLE3 head) actually get, per K, per task? | llama.cpp (`arczhi/5060ti-qwen3.8-27b` reports 85–100% per draft token on agentic work), SGLang and vLLM MTP for Qwen3.5/3.8, `Ex0bit/Qwen3.6-27B-PRISM-EAGLE3` (2.2 accepted on stock) | a table of `mean_accepted` vs K by task family; plug into the §5 table to get the best-case speedup *before* re-running the harness | §5 |
| R4 | Does 4-bit KV hold accuracy on the Qwen3.5/3.8 attention layers, and can ttnn's paged SDPA take a `bfloat4_b` K/V at all? | llama.cpp `q4_0` KV results on this family, KIVI / TurboQuant-style papers, ttnn `sdpa_decode` / `paged_scaled_dot_product_attention_decode` dtype support | ttnn accepts bf4 K/V in the paged decode SDPA *and* the published accuracy drop at q4 KV on this family is inside the README's GSM8K margin — then it halves the 256k penalty again on top of bf8 (§1.4 → ~7 ms/step) | §1.4; §4 D |

Not worth a session: GPU quantisation formats (GGUF, NVFP4, AWQ — Tensix matmuls unpack
to bf16 in the FPU, and §0.5 H showed time tracks tiles not bytes), CUDA/Triton kernels
as code, CPU offload (§4, rejected with numbers), and general "speculative decoding
speedup" claims — §5 has the arithmetic that makes them irrelevant until K lands.

Output expected from that session: one row per target in the table above marked
port / no-port with the evidence, and the affected number in §1.4, §2.4 or §5 updated.

---

## 9. Three cards now, four soon: how to split them

There are exactly three ways to divide the model across four p150a, and they optimise
three different things. §1.2's decomposition — ~31 ms of the step scales with weights
per device, ~23 ms is per-layer fixed cost that does not — is what decides which.

| Split | What it changes | Single-stream decode | Aggregate | TTFT at 262k | Engineering |
| --- | --- | ---: | ---: | ---: | --- |
| **TP=4**, one (1,4) mesh | halves weight bytes per device | **~26 tok/s** now, **~38** after K | B=32 legal | *not* better — TT's own 4-die 256k TTFT is 286 s vs your 220 | none: the supported `P150x4` path (on-device sampling, no PR stack needed) |
| **DP=2**, two (1,2) meshes | two independent endpoints | 18 each | 2× | 220 s each, independent | none: two containers, `--device` masking, one cable per pair — you have both cables |
| **Task split**: prefill pair + decode pair | a 262k prefill no longer stalls decode | 18 (TP=2) | decode never blocks | 220 s, hidden from other users | large: adapter for tt-metal's disaggregated-prefill engine (§9.2) |
| Layer split (PP=2) for *decode* | halves weights **per pair**, not the fixed cost | 54 ms + one hop — **no gain** | 2× only with ≥2 streams interleaved | — | large; the win is memory (§9.3), not speed |
| Layer-*type* split (GDN pair + attention pair) | nothing — every token still visits every layer in order, plus 2 hops per layer | worse | — | — | do not |

A token's latency is the sum of the layers it passes through; splitting layers across
cards cannot shorten that sum, it can only run *different tokens'* layers concurrently.
So for the thing you started with — decode speed for one user — **TP=4 is the only
hardware split that helps, and K is still worth more than the two extra cards** (11 ms
vs the 15.5 ms TP=4 buys; together, 54 → 26 ms).

### 9.1 Three cards, in the meantime

TP=3 is not a shape: 4 KV heads and 24 query heads do not divide by three, and the
mesh maps have no `(1,3)`. Use the pair for serving and measurement and **the third
card as the kernel-development card for K**: the GDN unit tests and the per-module
profiles that produced §0.5 run on a single P150 (with `Qwen3.5-9B`), the fused
recurrence kernel is validated single-device on the branch, and an A/B on the pair is
never interrupted by a compile. Do not try to make the third card a prefill card: a
single p150a prefills 4k in 19.4 s and OOMs at 64k (README).

### 9.2 Prefill on one pair, decode on the other

The stall in §2.4 — one 262k prompt occupies the engine for 220 s — is a scheduling
problem, and the fix is to put prefill somewhere else. tt-metal ships the machinery:
`models/demos/common/prefill/` is a **model-agnostic disaggregated-prefill engine**
that owns "rank topology and the per-rank contiguous layer split (pipeline parallel),
the H2D input socket, the D2D inter-rank activation sockets, the request serving loop,
fabric-link lease/reclaim per chunk, and KV-chunk-table publish + WORKER_READY
handshake for cache migration" — a model plugs in through a `PrefillModelAdapter`
(`docs/ADDING_A_PREFILL_MODEL.md`) and the decode side receives migrated KV. It was
built for DeepSeek-V3 and MiniMax-M3 on Galaxies (M3's 2- and 4-galaxy pipeline
prefill in `models/demos/minimax_m3/docs/PIPELINE_PREFILL_TESTING.md`), but a rank is
just a mesh and a (1,2) pair is a mesh.

What the adapter has to provide for this model, in order of difficulty: the KV
migration (generic — paged K/V blocks, the engine already does it); the **GDN
recurrent + conv state hand-off** (model-specific — 48 layers × 1.5 MB fp32 per user,
must ride along with the KV, and M3's lightning-attention state is the precedent for a
recurrent state crossing the boundary); and the chunk-outer prefill trace running under
the engine's chunk loop instead of the demo's. Payoff: decode users never see a
prefill; a second 262k request prefills while the first decodes; and two-stage
pipeline prefill (layers 0–31 on one pair, 32–63 on the other, chunks streaming
through) is what the engine does natively — the only route in this document to a
materially lower 262k TTFT, since TP=4 does not deliver one (§1.3).

### 9.3 Layer split for memory, not speed

PP=2 for decode gives each pair half the weights (~6.9 GiB/card instead of 13.75),
which frees ~7 GiB/card for KV. At 262k that is 4 concurrent full-context users at
bf16 KV, or the room a **1M YaRN window** needs: 32 GiB/device of bf16 KV at TP=2 does
not fit anywhere; at PP=2 × TP=2 with bf8 KV it is 8 GiB/device and fits. Per-token
latency is unchanged (54 ms plus one ethernet hop), and the engine only earns its
second pair with two or more streams interleaved. Worth knowing; not worth building
before §9.2, which is the same code path with prefill in front of it.

### 9.4 Cabling for four

A (1,4) TP mesh opens on a physical 2×2: `p300_x2` descriptor with one cable per edge
(4 cables, 2 links/hop — what TT's QuietBox 2 CI runs), or `p150_x4` with two cables
per edge (8 cables, 4 links/hop, every port used). You have two cables today; DP=2
needs exactly those two. Everything in `docs/topology.md` §2.4 about sub-mesh
partitions applies: rows and row-pairs of the physical mesh are legal sub-meshes,
diagonals are not.

---

## Sources

* This repo: `README.md`, `docs/gotchas.md`, `docs/speculative-decoding.md` (branch
  `speculative-decoding`).
* tt-metal `main` @ b87f414: `models/model_targets.yaml` (`qwen3.6-27b` /
  `bh_quietbox_2`), `models/demos/blackhole/qwen36/tt/{model.py,qwen36_vllm.py,gdn/tp.py}`,
  `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py`,
  `tt_metal/fabric/mesh_graph_descriptors/{p150_x4,p300,p300_x2}_mesh_graph_descriptor.textproto`,
  `tech_reports/MetalProfiler/metal-profiler.md`, `tools/tracy/`.
* tt-metal PR [#53320](https://github.com/tenstorrent/tt-metal/pull/53320) (open, 0 reviews, 2026-09-02).
* vllm-tt-plugin @ bf77cd6: `README.md` (§ `max_model_len` and KV cache capacity),
  `src/vllm_tt_plugin/{worker.py,platform.py}`.
* [Qwen/Qwen3.8-27B config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json).
* [Ex0bit/Qwen3.6-27B-PRISM-EAGLE3](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3).
* [Blackhole p300c specifications](https://docs.tenstorrent.com/aibs/blackhole/p300.html) (2 dies, 64 GB, 1024 GB/s).
