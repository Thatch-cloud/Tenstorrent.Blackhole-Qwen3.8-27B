# tp4/gluefix: GDN glue quick wins

Two flags, both off by default (production is byte-identical with them off: the twin's `_decode_packed` returns the pinned call untouched), TP4 only, strict 0/1.

## QWEN_FAST_GDN_PAIR_SLICE

The packed verify projects four 16-row users as one (1, 64, W) tile block. ttnn's Slice keeps tile layout for users 0 and 2 (begin on a tile row) but, for users 1 and 3 (begin at row 16 / 48), converts the whole input to row-major, slices, and tilizes the piece back: one whole-projection untilize (18.4 us) per odd user per GDN layer. `gdn_pair_slice_tp` serves exactly those half-tile slices with the same three ops (`to_layout` row-major, `slice`, `to_layout` tile) except that the row-major conversion is made once and shared by both odd users: about 18 us per layer, about 0.9 ms per four-user verify. Exact by construction (same ops, same bytes); the audited arm compares every such slice with ttnn's own on every chip after the replay.

It is idle when `QWEN_FAST_TP4_GDN_GLUE` (V2) is on: V2 replaces the slices with one DMA launch. It is V2's fallback, for the case V2 has to come off.

Audit: `QWEN_FAST_TP4_VGLUE_AUDIT=1` beside the flag (`c2-packed-tp4-gate-pairslice`). Markers: `[PINDIAG] tp4 pair slice engaged`, `fell back` (fails the smoke), `idle`.

## QWEN_FAST_GDN_DISPATCH_DIAG

The ~16 us gap before conv-gates users 1-3 has no host-side cause in the trace replay (no program-cache lookup, argument update or sync per op), so there is no fix flag. The diagnostic logs, for the first eight packed GDN decode calls, the ops enqueued, the `gdn_decode_conv_gates` launches, the ops between consecutive launches and the host enqueue gap (`[PINDIAG] tp4 gdn dispatch diag ...`). Host time only; it changes no op. The device side needs `TT_METAL_DEVICE_PROFILER_DISPATCH=1` on a profiled arm, which `ops_profile_plan` refuses in a profile env: a follow-up.

## Profiles and jobs

`c2-packed-tp4-speed-strace-pairslice` (timed, control `c2-packed-tp4-speed-strace`), `c2-packed-tp4-gate-pairslice` (audited), `c2-packed-tp4-speed-strace-dispatchdiag`. Templates: `scripts/ci/references/tp4-gluefix-jobs` (see `ORDER.txt`). Tests: `test_gdn_pair_slice`.
