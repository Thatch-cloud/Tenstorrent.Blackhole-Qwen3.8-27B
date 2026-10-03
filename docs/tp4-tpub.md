# TP4 traced publication of the sequential step (branch tp4/tpub)

CPU work only. Nothing here has run on a card. Every new behaviour is off by default; production (`c2-packed-tp4`) is byte for byte what it was.

Labels: (m) measured, (src) read from the code, (e) estimate, (U) unknown.

## Answer first

1. **The traced publication of the four-user block is already built and already in the combined best.** `QWEN_FAST_FUSED_COMMIT` (with `_INPLACE` and `_LIVE_BANKS`) is the real trace (docs/tp4-traced-publish.md, layer L1): `c2-packed-tp4-best` and its twins engage it (m, runs N1c and N4b: 1310 of 1310 publications `path=fused`, audits `mismatches=0`). Production still has it off. `QWEN_FAST_TRACED_PUBLISH` (L0) captures nothing, only cuts eager ops on refused rounds, and stays `0` in every profile.
2. **What is still eager on the publication side is the lone-user (sequential) step** (m, N4b: 634 steps, median 75.2 ms). Of its parts, the one that is a pure copy to fixed addresses, and already marked "the follow-up" in the engine, is the per-step GDN carry save: 48 eager launches, 16.4 ms median host enqueue (m).
3. **This branch traces that.** `QWEN_FAST_TP4_TRACED_PUBLISH=1` makes the request engine capture the carry save and the carry restore as one trace each and replay them. The audit flag `QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT=1` compares every traced copy with the eager loop bit for bit on every chip. The flag name is the four-card sibling of `QWEN_FAST_TRACED_PUBLISH`; it is a separate flag on purpose, so the L0 cut (a different mechanism, off everywhere) and this one can be timed apart.
4. **Estimate (e):** about 15 ms of host time off each lone-user step (16.4 ms enqueue replaced by one trace launch), 75 ms to about 60 ms at rows=4. It is a host-side saving, so it shows only where the host is the step's critical path; the paired timing says.

## What is traced and why it is exact

`verifier_engine.VerifierEngine.copy_carry` runs `helper.save(slot)` or `helper.restore(slot)` for each of 48 GDN layers. At four cards the helpers are `gdn_snapshot.ActiveSnapshot(direct=True)`, so each call is `gdn_state_copy.copy_active`: one DMA program that moves slot zero of the live state to or from the carry slot, no arithmetic. The carry slots are pool storage allocated at attach (`allocate_carry`, before any request's capture), and `slot_addresses()` is checked against the recorded addresses on every call.

The trace records the same 48 calls in the same order against the same addresses, so a replay moves the same bytes. `verifier_engine_tp.VerifierEngine` overrides `copy_carry` (the pair file `verifier_engine.py` is one of the eight pinned critical files and is untouched; a test holds that).

## Capture order (the hazard this repo keeps hitting)

Anything persistent created after a trace capture can land where a replay's temporaries were and be overwritten (the confirmed eight-seat hang class). The rules followed:

- The carry slots exist before any capture (attach, `allocate_carry`). The trace allocates nothing persistent: the direct copy kernel launches over existing buffers, and `carry_trace_problem` declines (logged, eager from then on) for any helper that is not the direct copy, because the ttnn slice path allocates inside the copy.
- Both copy programs run eagerly before they are captured. The save runs as the engine build's own seed copy. The restore is an **identity warm**: the carry was seeded from slot zero a moment earlier and nothing ran between, so restoring it rewrites the same bytes. Programs are therefore compiled before any capture.
- The two captures happen once, inside the engine's own build, right after its verify and commit traces and before the build's final sync. A save made after the build never captures (`phase` must be `preparing`), so nothing is ever captured in steady state.
- A failed capture releases what it took, leaves the engine eager and re-raises (the base constructor then closes the engine); `close` releases both traces after a sync.

What this does not do, and why that is the open risk (U): the request engine is built per request, after the attach-time packed captures, so these two small traces join the engine's own verify and commit traces in the same slot of the build sequence. They add no device buffer, but they are two more traces in the 256 MiB trace region per live engine and a change to the capture set, which is exactly the kind of change that has resurfaced the hang. The gate is S2a to S2c, three consecutive audits-off completions of the hang shapes on the timed arm. A later improvement would capture the carry traces once per pool slot at attach, beside the packed captures; it needs the carry-slot to engine binding to be one to one, which this branch does not assume.

## Audit

`QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT=1` (only with the arm): on every traced copy the destination (the carry slots for a save, the live state for a restore) is read back after the trace, the eager loop runs over the same destination, it is read back again, and every tensor on every chip is compared bit for bit (so a signed zero or a denormal difference counts). The line is `[TPUB-AUDIT] op=save checked=960 mismatches=0` (960 = 48 layers x 5 tensors x 4 chips). A mismatch is logged, the eager bytes stand, and `c2_smoke_check` fails the arm.

## Flags, profiles, checks

| Item | Value |
|---|---|
| `QWEN_FAST_TP4_TRACED_PUBLISH` | `1` arms it; read once per engine at construction; default off |
| `QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT` | `1` beside the arm; audit as above |
| `c2-packed-tp4-best-strace-tpub` | gate only, timed: `c2-packed-tp4-best-strace` plus the arm flag and nothing else |
| `c2-packed-tp4-best-gate-tpub` | gate only, audited: `c2-packed-tp4-best-gate` plus the arm flag and the audit |
| Log lines | `[TPUB] carry traces engaged request=... layers=48 audit=N`; `[TPUB] carry traces declined request=...: reason`; `[TPUB-AUDIT]` |
| `c2_smoke_check.tpub_problems` | flag off: no `[TPUB` line; on: at least one engaged line, no declined line; audited: audit lines present, all `mismatches=0`, all `checked` = layers x tensors x chips |

`QWEN_FAST_CARRY_LOG=1` still works; its lines end `traced=1` when the copy was a replay.

## Not done here, and why

- **c2 (history write at 2048 rows on the sequential general branch) and c3 (a solo fused commit).** `serving_solo_lane` refuses `QWEN_FAST_FUSED_COMMIT` (the fused commit is built for the block), so the lone-user lane cannot use the existing traced publication; building a one-user fused commit is a new capture set, not a copy trace, and was not attempted without a card to prove the capture order.
- **`publish_target` (19.2 ms) and the drafter `prepare_history` (12.6 ms) of the lone-user step** are still eager: the first is the target-side GDN commit with data-dependent prefix, the second the drafter K/V write, whose proj/hist/kv pieces and its own device sync make it a larger change than a copy.
- **Promotion of the fused commit to production** needs five clean audits-off runs of the hang shapes on `c2-packed-tp4-best-strace` (templates exist on the next-3 window branch) and the rollout; this branch changes none of that.
- **Eight seats.** The carry traces are per engine, so the count scales with live engines; two blocks of four seats double the packed traces already in the region. Neither the trace-region budget nor the capture order at eight seats is measured.

## Needs a card

Everything in `scripts/ci/references/tp4-tpub-jobs`: the audited smoke (does the audit pass on every chip), the three hang-shape runs (does the extra capture leave production's trace shape alone), the paired ABAB (is the lone-user step shorter, and are texts and accepted prefixes identical). Nothing here shows the saving is real; the 15 ms is an estimate from one enqueue measurement.
