# tp4/next: the combined four-card S2 fast path

tp4/next is tp4/stack-fix (the speed slide and audits-off profile, per-tile exact collectives, the quad draft twin and singles
audit, and the prefill-before-capture warm) with tp4/fcommit, tp4/vglue and tp4/lanes-d0fix merged on top, in that order. Every
feature stays behind its own flag and profile.

## Combined profiles

| Profile | Base | Adds |
| --- | --- | --- |
| `c2-packed-tp4-best` (timed) | `c2-packed-tp4-speed-fcommit-quad`: audits off, K/V slide, pairs plus the quad, fused commit in place on live banks | the five verify-glue levers C1a, V4a, V2, V1, V3a |
| `c2-packed-tp4-best-gate` (audited) | `c2-packed-tp4-gate-fcommit-quad`: verify audits, fused-commit audit and singles audit on | the same five levers and `QWEN_FAST_TP4_VGLUE_AUDIT=1` |

Both are gate only and unqualified on hardware as a combination; each part has its own card evidence. The timed arm is held
against the audited arm's tokens, and for the S3a matrix against the solo engine.

## Lanes

The lone-user lanes (`QWEN_FAST_SOLO_LANE`, `QWEN_FAST_LANE`) are not on the best profiles. `serving_solo_lane.UNSUPPORTED_FLAGS` lists
`QWEN_FAST_FUSED_COMMIT`: the fused commit builds a per-block object over the M3 block's buffers and the solo block cannot take it, so the
attach refuses the combination. The five lanes profiles (`c2-packed-tp4-time-gate`, `-solo-gate`, `-solo-time-gate`, `-lanes-gate`,
`-lanes-time-gate`) stay on their own gate base, without the K/V slide they were qualified on. A lanes arm on the best config needs the
fused commit taught the solo block, or a best variant without it (speed, quad, levers, lanes); neither is built here.

## Merge notes

- The attach order is unchanged: `prefill_warm_before_traces` runs before every packed block, the D0 solo block and the fused
  commit's captures.
- CPU allowlist lines, image copy lists, the closure literals and every profile list carry the rows of all four branches.
