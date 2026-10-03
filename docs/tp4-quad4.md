# tp4/quad4: the quad draft at four cards

Finding: nothing is left to fix on the four-card path for four users. The all-levers profiles (`c2-packed-tp4-best`,
`-best-strace`, `-best-gate`) already engage the quad, and the v172 D1 failure had a different cause that next-3 already fixes.
Production `c2-packed-tp4` has `QWEN_FAST_QUAD_DRAFT=0`; promoting it is a rollout decision, not a code change.

## Why v172 never formed a quad

The smoke saw the quad marker 0 times and no quad round. The quad was not at fault: users 0 and 1 (prompts of 3.4k to 3.6k
tokens) returned an end-of-text token first, and another user returned mixed-script text. The prefill programs and the GDN
scratch were first created after the attach-time packed captures, so packed replays overwrote them (the four-card prefill
defect). With slots 0 and 1 dead, the coordinator never saw groups equal to `quad_draft.PAIRS`, which `_prepare_quad` requires
(and both pairs packable), so it correctly stayed on pairs. The fix is `serving_runtime.prefill_warm_before_traces`, which warms
the eager prefill and allocates the scratch before any capture; the smoke check judges instant end-of-text, foreign script, the
warm line's place before the Metal unsafe-allocation warning, and late programs per prefill.

## Hardware evidence that it engages (all-levers recipe)

- Audited run (`best-gate`): the quad draft engaged once, 466 quad rounds, every singles audit of group [0, 1, 2, 3] equal.
- Timed run (`best-strace`): 288 quad rounds on pairs=[[0, 1, 2, 3]], no fallback or disable line.

## Tests (CPU)

`scripts/ci/test_tp4_quad_engagement.py` (allowlisted in qwen-integration-cpu.yml):

- a pair still ramping, or a dead slot in the group, never forms a quad, never disables it, never logs a fallback;
- the quad forms the round after the last ramping user packs;
- the smoke check fails the v172 signature (marker 0 times, no quad round) and passes the engaged logs; a second engage, a
  fallback or a disable fails;
- the quad is the first four slots only.

## Open: eight seats

`_prepare_quad` is tied to `quad_draft.PAIRS` (slots 0 to 3). On origin/tp4/seats8 slots 4 to 7 are live, so no quad can form
there, and the eight-seat profiles set the quad off. Extending it means a per-block quad (`QUADS=((0,1,2,3),(4,5,6,7))` in
the twin, quad state per slot tuple in the coordinator, the (4..7) masks and outputs pooled before any capture, capture
headroom per quad, a new flag, two engaged markers in the smoke check). That needs the seats8 base and a DRAM trace-region
measurement on the cards, so it is not started here.
