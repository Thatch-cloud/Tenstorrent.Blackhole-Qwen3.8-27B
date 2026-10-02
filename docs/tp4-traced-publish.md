# TP4 traced publication (branch tp4/next-3-scope)

Design and CPU groundwork. Nothing here has run on a card. Every new behaviour is off, and production (`c2-packed-tp4`) is byte for byte what it was.

Labels: (m) measured, (src) read from the code, (cm) computed from measured numbers, (e) estimate, (U) unknown.

## Answer first

1. **"Traced publication" is three different things in this repo, and only one of them is a trace.**

   | Layer | Flag | What it does | State at TP4 |
   |---|---|---|---|
   | L0 | `QWEN_FAST_TRACED_PUBLISH` | Cuts the eager op count in the steady state: under `QWEN_FAST_ROUND_B1` (the image sets it) the five-op 2048-row feature-history write is skipped outright; the K/V half goes from four ops to two per bank. It captures nothing. | The history half has no two-card literal. The K/V half sliced `(1, 4, ...)`. Fixed here; it declines when the slide is on. Off in every profile. |
   | L1 | `QWEN_FAST_FUSED_COMMIT`, `_INPLACE`, `_LIVE_BANKS` | The real device trace: one captured projection trace (`T_proj`) per segment, one captured in-place slide trace per (segment, prefix). | Ported (`fused_commit_tp`), exact on one card (src, `docs/tp4-fused-commit.md`), never timed, never run with the in-trace sampler. Profiles exist, none is production. |
   | L2 | none yet | Cheaper device work inside the traces (ring-buffer history, batched K/V projection, wide grids): lever P2. | Not built. |

   The audit's P1 (-12.5 ms at 4 users, 466 eager ops, 24.5 ms host) is mostly **L1**; L0's share is the five-op history write, about 3.8 ms of the 12.0 ms of publication device time (e, section 1). The synthesis ranks the same mechanism twice, as P1 (-12.5) and as "fused commit family" (-2.4). Only a paired window can say which is nearer.

2. **De-literalisation (done, CPU-verified):** `dflash_traced_publish` no longer carries the pair's four KV heads. At TP4 its K/V fusion is installed only on the four-card sibling cache and only with `QWEN_FAST_TP_KV_SLIDE` unset. With the slide on (every four-card traffic profile) it declines, because the slide is already one op per bank and the fusion would replace it with two ops and a copy. This matters most for the fallback path: a fused-commit round that a guard refuses runs `install_publish_options`, so before this change `QWEN_FAST_TRACED_PUBLISH=1` plus the fused commit would have sliced a two-head bank with a four-head literal.
3. **What would actually move the round** is L1 on the production recipe. This branch adds one gate-only profile for that, `c2-packed-tp4-speed-strace-fcommit`, and the window templates to time it paired against `c2-packed-tp4-speed-strace`. The whole risk is the trace-capture order (below), not the arithmetic.

## 1. What the 466 ops are

Device profile of the four-user verify round (production minus the in-trace sampler and the exact tile-split all-reduce; fused commit off), 18 publication bursts, per burst (m):

- 466 ops, 12.0 ms of kernels on the chip (min 9.9, max 15.1), inside 24.5 ms of host time for the four serial `packed_commit` calls. About 116 ops a user.
- Device time by family: GenericOp (the ten slide dispatches a user, `draft_kv_slide_tp`) 3.85 ms; Matmul (the K/V projection and the feature projection) 1.78 ms; AllGatherAsync (the projection's gather) 1.12 ms; Untilize, Tilize and TilizeWithValPadding together 2.3 ms; Slice, Copy and Concat 1.6 ms.
- The 110-core and 90-core ops of 100 to 165 us in the burst listing are one chain a user: untilize, concat, tilize-with-padding, untilize, slice, tilize, copy. That is the 2048-row feature-history write (the general branch: slice, concat, trim slice, pad, copy over a 21 MB buffer), about 3.8 ms of the 12.0 (e: attributed by op family and shape, not by name; the profile CSV that would name them was pruned).
- Why it runs at TP4: the image sets `QWEN_FAST_ROUND_B1=1`, whose C7 cut skips that write where nothing can read it, but the cut lives inside the fused-steady-state branch (src, `DFlashDevice._prepare_publication_round_b1`), which `QWEN_FAST_TRACED_PUBLISH=1` selects. The TP4 profiles set the flag to 0 (for the K/V literal), so they lost C7 along with it. The fused commit marks the history stale the same way (src, `fused_commit_tp`), so the invariant is already relied on at TP4 when that lever is on; every reader of a stale history raises instead of reading.
- Commit phase of the round: 36.8 ms at 4 users, 75.1 ms at 8 (m, the production-shape logs). It is the largest phase the host exposes.

## 2. Every two-card literal in the publication path

Scan basis: the served import closure scan (`test_tp4_closure_literals`), plus a read of every module the L0 and L1 paths call.

| Where | Literal | Status |
|---|---|---|
| `dflash_traced_publish._fused_kv_history_prepare` | `(1, 4, n, 128)` slices | **Fixed.** Takes `heads`; `install_fused_kv_history` passes `tp_shapes.draft_kv_heads` for the sibling class and 4 for the pair's. The closure test's allow-list entry for it is gone, so a new literal there fails CI. |
| `dflash_traced_publish.install_fused_kv_history` | tests `isinstance(DraftKVHistory)` and reads the pair class's `prepare` as "the live one" | **Fixed.** At TP4 the live `prepare` is the sibling's. New rule: only `draft_kv_history_tp.DraftKVHistory` (module and qualname) is recognised; it declines with `tp_slide_live` when the slide is on, `bank_shape` when a bank is not `(1, kv, 2048, 128)`, `unrecognized_prepare` for any other subclass, and never installs the pair's text-patched slide scope over the sibling. Reasons are logged under `QWEN_FAST_PACKED_AUDIT=1` with the existing `fusion=declined` line. |
| `DFlashDevice.prepare_publication`, fused-steady-state branch | 5120 | No change needed. It is the hidden size of the reduced projection, the same at any width. The width that does change (the per-chip feature taps) already reads `tp_shapes.draft_taps`. |
| `DFlashDevice.project_features` | grid `(8, 10)`, `per_core_N=2` | No change needed: 80 cores x 2 tiles = 160 tiles = 5120 columns of the reduced output (src). |
| `publication_warm` | `(1, 4, 2048, 128)` bank, 2048-wide query | Already fixed by two earlier reviews (`kv_shape()`, `query_shape()`). Held by the closure test. |
| `fused_commit` (pair) | `KV_SHAPE`, `DELTA_SHAPE`, `WORKERS`, `BANKS_PER_PROGRAM`, `range(2)` chips in `slide_program` and in the retainer | Replaced by `fused_commit_tp` at TP4 (MODULE_TWINS). The twin refuses the pair names by name. Not served at TP4. |
| `fused_commit.FusedCommit.audit_round`, `install_fused_commit`, `commit_kv`, `stage_tables` | none | Loop over `get_device_tensors`, so they take the width from the mesh. Checked: no chip or head literal. |
| `draft_kv_slide` (pair) | `range(2)` program loop, four heads | Replaced by `draft_kv_slide_tp` (16 workers at the pair, 8 at four cards). |
| `draft_kv_history` (pair class) | `(1, 4, ...)` | Pair only; the sibling is built at TP4. Allow-listed with its reason. |
| `qwen_c2_profiles.json`, the TP4 traffic profile's description | still says the traced publish's K/V fusion is off "because its code carries the pair's literals" | **Left as is on purpose**: the description is part of the production profile, which must stay byte for byte. The statement is now stale: the flag is off by choice and by the slide, not by a literal. |

Not literals but width-dependent, and worth knowing before the first run: the in-place slide program holds ten banks in 80 workers at TP4 (8 workers a bank) where the pair needs two programs, and the in-place hazard argument (each worker reads tiles t and t+1 before writing t) was proved at two heads on the CPU only.

## 3. Exactness

**L0 (the eager fusion).** In the steady state (`history_rows == 2048`, permanent once reached) the general chain `slice, concat, slice, pad` always resolves to: drop the first `history_rows + prefix - 2048` rows of the old history and append every accepted row; the pad is a no-op at 2048 rows. So one slice and one concat compute the same rows. Per (layer, k/v bank) the same identity holds at any head count. Proved on the CPU, bit for bit against the sibling's own six-op chain at two heads, for every prefix 1 to 32, the ramp and the one transitional round (`test_tp4_traced_publish`). The slices are recorded in the test: torch clamps an over-long slice silently, so a surviving literal `4` would not fail on values alone (the test fails if the head count is 4). With the slide on nothing of the K/V path is installed, so what remains at TP4 is the feature-history half: under `QWEN_FAST_ROUND_B1` no history is computed at all (C7: nothing reads it, because the packed pair trace takes no history, the single-user trace copies it only without a K/V cache or with an audit, and every reader of a stale one raises). The branch has no width in it (5120 is the hidden size); `test_tp4_traced_publish` holds, at four-card tap widths, that the flag removes exactly the five history operations and nothing else, and that the pending record and the buffer swap are unchanged. Its pair twin is `test_dflash_round_b1.HistoryWriteTests`.

**L1 (the traces).** The claims, from `fused_commit.py` and `fused_commit_tp.py`:
- `T_proj` is today's op sequence at count 16 instead of count `prefix`. Every op in it is row independent (matmul, typecast, rms_norm, the CCL add, the rotary; bfp exponents are shared along a face row, one row), so rows `0..prefix-1`, the only rows a slide reads, are today's. Rows `prefix..15` hold real features where today holds zeros.
- The slides are the served kernel, byte-qualified, in place (each worker reads two tiles before writing the first). Card B showed in place equals out of place in 88 of 88 cases at four heads; two heads is the open question (CPU kernel mirrors are exact).
- Hardware proof is the fused-commit audit (`QWEN_FAST_FUSED_COMMIT_AUDIT=1`): before every fused launch today's eager publication runs, and every delta and every bank is compared bit for bit on every chip (`checked=80` a round). A mismatch is repaired to today's bytes and the gate fails the arm.

## 4. Why a per-prefix trace is not built

`project_features` and the K/V update take `prefix` rows (1 to 16), so every op shape depends on the accepted count. A captured trace bakes every op's shape and offset at capture time and only the data in fixed placeholder buffers varies. So L1 does not trace `prefix`: `T_proj` runs at 16 rows always (row independence above), and the slide trace is captured once per (segment, prefix), 16 per segment, because the kernel takes `prefix` as a runtime argument but a trace cannot change runtime arguments between replays. That is 68 traces at 4 segments (src, `docs/tp4-fused-commit.md`). The pre-built alternative, one trace per distinct prefix per bucket per user, was rejected in the L0 module's own docstring on DRAM grounds.

## 5. Trace-capture-order hazards this repo has hit

Each is a rule for any new capture, and the gate plan below tests the ones L1 touches.

1. **Anything allocated after a capture and before its replay can be overwritten.** A verify trace's replay writes its intermediates into the holes its capture freed. R2 in `fused_commit.py`: the RoPE tables and the deltas are allocated before the block's verify capture (`FusedCommit.__init__` runs right after the taps, before the warm forward), `T_proj` and the slides are captured after the GDN commit traces, and what they bake besides is pre-trace by construction. A new persistent buffer (a ring buffer for P2, say) must follow the same rule.
2. **State created by the first request-engine build after the captures.** The eight-seat hang analysis (about 0.65 confidence, the rest unresolved): the first request-engine build after both block captures creates and compiles the rows 1, 2 and 4 state in memory block B's replays then overwrite, and the next eager rows-2 warm-up hangs. At 4 seats the same mechanism is latent: post-attach allocations land in block A's freed temporaries, and production is empirically safe only by layout. **Any change to the attach allocation sequence or to the capture set can bring the hang back.** L1 adds 68 captures between the GDN commit traces and the engine builds, so it is exactly such a change. The fix under test on the 8-seat branch is a request-width warm before the captures (`QWEN_FAST_M3_REQUEST_WARM`, not on this base).
3. **Programs compiled after the warm.** Publication programs are shaped by the accepted prefix; `publication_warm` publishes every served shape at attach so no serving path compiles one. The traces remove the per-prefix shapes for the fused path, but every refused round (ramp, parity, sequential step) still takes the eager path and needs that warm.
4. **The in-trace sampler recipe depends on a trace shape.** Production serves audits off only because the pinned sampler is recorded in the verify trace (`QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1`), which keeps the shape that does not hang. The traffic profile's history: with both audits off and no sampler the platform replay and the direct smoke hung twice at the first four-user answers' tail. A new lever on top of that recipe must be proved on the hang shapes, not assumed.
5. **A collective inside a trace bakes one semaphore handle.** `T_proj` contains the projection's all-gather. Eager publications cycle the shared collectives' handles; a replayed trace reuses the captured one. One CQ and in-order execution make this safe in theory; the 8-seat analysis rates a device-level CCL state variant at about 0.20.
6. **The graph census crashes inside a capture** (`trace_census`): peak L1 inside a trace cannot be measured the usual way.
7. **Trace region size.** The profiles carry a 256 MiB trace region. 68 more traces and about 24 MB a chip of `T_proj` intermediates (src) are not measured against it. At 8 seats (two blocks of four) the count doubles.

## 6. Gate plan

Order is cheapest and most informative first; every card job starts from production live on the cards, so the first job is `agentstop` plus `unserve` (templates in `scripts/ci/references/tp4-next3-scope-jobs`).

| Step | What | Pass |
|---|---|---|
| G0 (CPU, done) | `test_tp4_traced_publish`, closure literals, profile twin rules | green |
| G1 | Production profile smoke on the image | Same lines and speed as production; no lever line anywhere; no `fusion=declined` line (L0 is off) |
| G2 | Audited fused-commit smoke (`c2-packed-tp4-gate-fcommit`) | Audit lines `mismatches=0` every round, no refusal other than ramp or parity, one-card prerequisite F1 (two-head in-place slide) already green |
| G3 | Audits-off hang shapes on `c2-packed-tp4-speed-strace-fcommit`, five consecutive completions | Five of five; one stall restarts the count and stops the window. This is the allocation-order gate: the only one that can see hazard 2 |
| G4 | Timed ABAB, `c2-packed-tp4-speed-strace` against `-fcommit`, coding prompts at 4 x 4k and 4 x 32k | Judged paired per round. Accepted-prefix sequences and texts identical (`speed_window_compare.py`). Commit phase 36.8 ms to at most about 12 ms (e); round shorter by at least 8 ms |
| G5 (not here) | Eight seats | Needs the trace region and the segment count at 8 seats, and the 8-seat hang gate; on the 8-seat branch |

L0 has no hardware step here: it is off in every profile by instruction. **It is the cheapest slice of P1 and it is one profile line away**: the arm is `c2-packed-tp4-speed-strace` plus `QWEN_FAST_TRACED_PUBLISH=1`. At TP4 it removes the five-op history write: about 3.8 ms of the 12.0 ms publication device time at four users (e), plus its host dispatch (five of about 116 ops a user), and the K/V half declines under the slide. It needs no new capture, so hazards 1 and 2 do not apply to it; its gate is the audited smoke plus the paired timing. Recommended before L1 if the window is short.

## 7. What is CPU-verified, and what needs a card

CPU-verified: the K/V fusion at two heads equals the sibling's chain bit for bit; slices carry the bank's head count; at four-card tap widths the flag removes exactly the five history operations; the decline rules and their log reason; the pair's behaviour unchanged (its own tests run beside); the flag is `0` in every profile; the new profile is the production recipe plus the three fused-commit flags and nothing else.

Needs a card: everything in the gate plan. In particular, nothing here shows the fused commit is faster, that the extra captures leave production's trace shape alone, or that the 68 traces fit the trace region.
