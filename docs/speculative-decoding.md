# Speculative decoding on two Blackhole cards

Qwen3.8-27B ships a Multi-Token Prediction head (`mtp_num_hidden_layers = 1`). This is what it
took to turn that into an actual end-to-end speedup on a 2-card p150a mesh, and what it cost.

> **Status:** working. **~1.03–1.05×** at `mean_accepted = 2.04`, with break-even acceptance at
> **1.936**. Read [Caveats](#caveats) before relying on this — in particular, absolute speedups
> here carry ~±1.5% of session-to-session drift, so only within-sweep deltas are firm.

---

## Result

| | ms/step | speedup @ acc 2.04 | break-even acceptance |
| --- | ---: | ---: | ---: |
| baseline decode | 61.9 | 1.00× | — |
| speculative, K=2 | **119.8** | **1.054×** | **1.936** |

Break-even acceptance is arguably the more useful number: it is what decides whether a given
prompt distribution wins at all.

Normalised at the pooled `mean_accepted = 2.04`, three interleaved A/B pairs, host load average
3.1–6.9. Raw per-run ratios ranged 0.954–1.150 purely because acceptance varied 1.885–2.205 —
see [Measuring this honestly](#measuring-this-honestly).

The step time is stated **after normalising the verify phase across arms**. Verify is untouched by
these changes, so it acts as a control; the raw totals (135.1 → 122.3) credited ~3.2 ms of verify
drift to the result and overstated it as 1.032×.

**The speedup model:**

```
speedup = mean_accepted × baseline_ms_per_step ÷ total_ms_per_step
```

This reproduces every measured run to three decimal places. It is worth internalising, because it
says the verify path is only one of three terms — and the other two are where the time went.

---

## Why this is harder than on a GPU

48 of the 64 layers are **Gated DeltaNet** — linear attention with a recurrent state. That is what
makes long context cheap on decode (see the main README), and it is exactly what makes
speculative verify hard.

Verifying `T` draft tokens means advancing a recurrent state `T` steps *and keeping every
intermediate*, because you commit only the accepted prefix `n ≤ T` and must roll the state back to
step `n`. Full attention has no such problem: a KV cache is positional, so you truncate. A
recurrent state is not.

Upstream has no multi-token-from-state decode primitive for the tensor-parallel GDN path. So:

1. **A composed implementation** — slice/pad/copy/concat/matmul per token, all device ops, and
   therefore trace-safe.
2. **A custom ttnn op** (`ttnn-op/gdn_decay/`) computing `T` fused recurrence steps in one
   dispatch and returning all `T` intermediate states.

Both are in `speculative-decoding/harness/test_gdn_decode_multi.py`, selectable at runtime, with
the composed one kept as the reference implementation.

---

## The custom op

`ttnn._ttnn.operations.transformer.gdn_recurrent_step` — 12 files, ~900 lines, in
`speculative-decoding/ttnn-op/gdn_decay/`. Per token it computes:

```
h1     = state * exp(g)
v_read = k @ h1
delta  = (v - v_read) * beta
state' = h1 + transpose(k) @ delta      -> emitted for every t, which is what rollback needs
o      = q @ state'
```

Validated at `T = 1, 2, 4, 8` against the composed path, across 24 cores with scalar broadcast.

**Two rules this kernel is built around**, both learned the expensive way:

* **Never modify a circular buffer after `push_back`.** No exemption. An earlier version had `o`
  correct while `h_new` was wrong *despite* `o = q @ h_new` — the writer had drained the CB before
  the accumulate landed. The fix is an intermediate CB. A later "this is safe for compute-internal
  CBs" exemption was also wrong and broke the T-loop.
* **The state CB needs two slots**, so a new state can be reserved while the old one is still the
  front.

### It did not, by itself, help

| | verify | end-to-end |
| --- | ---: | ---: |
| composed | 1.27× | — |
| fused op | **1.53×** | still a **net loss** |

The kernel did what it was built to do and the system got faster at the thing that was already
fast. Drafting and rollback sat *outside* the model that motivated it. This is a documented
failure mode — Huawei's CloudMatrix384 work calls it the "pipeline break problem": MTP produces
`k+1` compute graphs per step, each dispatch costing real time.

**What actually crossed break-even was two unglamorous fixes found by instrumenting phases nobody
had measured.**

---

## Where the time actually went

Per step at K=2, instrumented (`SD_DRAFT_PROFILE=1`):

| phase | before | after | how |
| --- | ---: | ---: | --- |
| verify | 102.9 | 102.9 | traced; already fast, unchanged by any of the fixes |
| draft | 22.9 | 11.2 | device-side untilize, then tracing the MTP forward |
| rollback | 14.1 | 5.9 | persistent carry buffer |

Each fix was measured against **its own control**, and the three baselines differ — the rollback
A/B predates the untilize work, and the MTP-trace A/B had both earlier fixes on in both arms. Their
savings should not be summed.

### Tracing the MTP forward: draft 13.5 → 11.2 ms

The forward was almost pure host dispatch. Tracing it needs persistent input buffers, since
`cp_tt`, `cos`, `sin` and `emb_tt` are built fresh per call and a trace bakes addresses in.

| phase | no trace | traced |
| --- | ---: | ---: |
| `mtp` | 4.65 | **0.30** |
| `read` | 6.57 | 8.88 |
| **draft** | **13.53** | **11.16** |

Dispatch is 94% gone, but **a third of it reappears in `read`** — the replay is non-blocking, so its
device work lands in the readback wait instead of being paid up front. The honest saving is the
draft delta, −2.4 ms/step, not the −4.4 the `mtp` column alone suggests.

**Three live traces coexist.** The width-1/width-T hang documented under [Caveats](#caveats) does
*not* generalise to trace count: this trace owns its I/O (allocated once, only ever written via
`ttnn.copy`), so it does not participate in the decode-trace buffer reuse that makes the
width-1/width-T pair fatal. It is captured after the width-T trace so that verify — the expensive
one, whose loss would be silent — is not the earlier-captured party.

### Rollback: 14.1 → 5.9 ms

`commit_prefix` cost ~14 device ops per GDN layer × 48 layers ≈ 672 dispatches — for 72 MiB of
copies. Dispatch-bound, not bandwidth-bound.

Two observations collapse it: the fused path never reads `conv_states[0]`, and the `K-1` taps it
*does* read are contiguous rows, so the whole carry fits one persistent `[1, K-1, qd]` buffer.

This is correct **only** where every continuation after a rollback goes through the fused path, so
it is an explicit opt-in (`commit_prefix(..., fused_only=True)`) rather than an inferred property.
The test suite includes a deliberately failing arm that probes with the single-token path, to prove
the contract is load-bearing rather than decorative.

### Draft readback: 13.7 → 6.7 ms

The draft phase spent 60% of its time in the logits readback. Three experiments found why, none
needing a trace:

| experiment | result | conclusion |
| --- | --- | --- |
| read the same tensor twice | `read2 ≈ read` (0.93, 0.88) | **not** device wait |
| read a 4-byte tensor | 1.05 ms vs 6.6 ms for 248320-wide | ~1 ms round trip, rest scales with width |
| read one shard instead of the mesh | **zero** change | **not** DMA volume |

All three together: both readback paths were untilizing the same 248,320 elements **on the host**.
Halving the bytes on the wire never touched the real cost.

Tile → row-major is data movement, which is what the accelerator is for. `to_layout` on device,
read row-major: **13.72 → 6.68 ms/step**, spread 0.23 ms across three runs.

---

## Negative results

Recorded because they cost real time to establish, and because the reasoning that motivated them
was plausible:

| idea | result |
| --- | --- |
| **argmax on device** — return 4 bytes instead of ~1 MB | **~2× worse.** `read` 12.5 → 23.4 ms. A 248,320-wide reduction costs far more than the DMA it saves. |
| **skip the mesh composition** — the lm_head all-gathers, so one shard holds the full row | **Zero change.** Correct (verified both ways, 4/4) and simply not the cost. |
| **K=3, K=4** | No better than K=2, for structural reasons — bucketing rounds `B` to a power of two. |

---

## Measuring this honestly

This programme burned a lot of time on measurements that turned out to be vacuous. The practices
that came out of it:

* **Every flag prints a line when it engages, and the harness asserts that line is present.** An
  A/B whose arms agree *too well* is evidence the flag is inert, not that it is neutral. One
  published result had to be retracted because the runner executed `spec_generate.py` while the
  flag had been added to a similarly-named file beside it — both arms ran identical code.
* **Every fast path is checked against the slow path at runtime**, not just in a unit test. The
  first 4 drafts of every run compute the token both ways and assert equality.
* **Interleave A/B arms, never block them**, and record host load per run. On a shared host,
  contention does not corrupt results uniformly: host-dispatch-bound phases inflated 30× while
  device- and DMA-bound phases stayed flat within a few percent. That differential is a free
  diagnostic — but it means a "slightly noisy" run can be perfectly good for one column and
  worthless for the next.
* **Normalise before comparing.** Acceptance varies run to run and the speedup is linear in it.
* **Check the phases your change cannot affect.** They are controls. Interleaving arms handles
  drift *between* pairs, not drift *within* an untouched phase — and the first version of this
  page overstated the result by 1.4 points because ~3.2 ms of verify noise happened to point the
  right way.

---

## Caveats

* **Margin is small and acceptance-dependent.** Break-even sits at **1.936**; below that it drops
  under parity, and acceptance is prompt-dependent.
* **Absolute speedups carry ~±1.5% session drift.** The same configuration measured 1.018× in one
  sweep and 1.033× in another with nothing changed but the session. Only within-sweep deltas
  against a matched control are firm.
* **Output is not bit-identical to non-speculative decoding**, and this is expected. A `T`-row
  projection accumulates differently from `T` single-row ones. Agreement is ~1e-6 and the device
  is deterministic, so long generations diverge — typically after ~9 tokens. Divergence is
  *reported* by the harness, not asserted against.
* **Correctness evidence is from eager mode.** Traced *output* correctness is unverified.
* **Two live traces of different widths will hang the board.** A width-3 replay that succeeds
  immediately after its own capture hangs once width-1 replays have run in between — main thread
  blocked in `read_decode_output`, both dispatch threads spinning, recovery needs `tt-smi -r`. The
  harness works around this by capturing the wide trace last and never replaying width 1 again.
  This is why the MTP head itself is still untraced (~5 ms/step left on the table).
* Measured at K=2, `MESH_DEVICE=P300`, TP=2, host sampling.

---

## Layout

```
speculative-decoding/
  run.sh                        reference runner -- the script that produced these numbers,
                                with host-specific values lifted into env vars
  ttnn-op/gdn_decay/            custom ttnn op: T fused GDN recurrence steps, all T states out
  harness/
    test_gdn_decode_multi.py    multi-token GDN (composed + fused), rollback carry
    mtp_module.py               the MTP draft head
    spec_generate.py            end-to-end loop, phase instrumentation, correctness gates
    patch_hidden_retention.py   REQUIRED -- see below
    patch_mtp_weight_load.py    REQUIRED -- see below
docs/speculative-decoding.md    this file
```

### Two patches the harness cannot run without

Both patch files copied out of the serving image, so what gets mounted is provably image + patch:

* **`patch_hidden_retention.py`** — the target model does not retain the hidden state the MTP head
  drafts from. Without this the draft head has nothing to consume.
* **`patch_mtp_weight_load.py`** — the stock weight mapping **drops the `mtp.`-prefixed tensors**.
  Without this the head loads, runs, and produces garbage, which is a considerably worse failure
  than not loading at all.

`run.sh` applies the first one itself on every run so the mounted file can never drift.

The op drops into `ttnn/cpp/ttnn/operations/transformer/` and needs registering in
`sources.cmake`, the kernels glob in `CMakeLists.txt`, and `transformer_nanobind.cpp`.

### Flags worth knowing

| flag | effect |
| --- | --- |
| `SD_FAST_CARRY=1` | persistent rollback carry (−8.1 ms/step) |
| `SD_DRAFT_RM_READ=1` | device-side untilize before readback (−7.0 ms/step) |
| `SD_TRACE_MTP=2` | trace the MTP forward (−2.4 ms/step); `=1` probes it in isolation |
| `SD_DRAFT_PROFILE=1..4` | phase breakdown; 2 syncs after enqueue, 3 double-reads, 4 adds a 4-byte read |
| `SD_FORCE_REJECT=1` | control: degenerates to plain decoding, so `mean_accepted` must be exactly 1.000 |
| `SD_CHECK=1` | rollback and sequential-equivalence gates |
