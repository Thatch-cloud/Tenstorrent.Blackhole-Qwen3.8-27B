# K1 verdict: launch count is not what the traced round is paying for

**Date:** 2026-09-22. **Runs:** v33 `35592099759`, v34 `35658854824`, same rig pair,
same graft, same 32768 context, four users, both `gate_passed: true` and token-exact
against the per-user single-user references.

## The measurement

`docs/trace-op-attribution.md` ranked `gdn_state_copy.copy_compact` as the cheapest
of three targets: 46.57 ms over 492 launches at a 94.6 us mean, on a 48-core DMA
grid, no arithmetic, not in the SOURCES pin. The change landed behind
`QWEN_FAST_GDN_STATE_COPY_BATCH` and collapses the packed decode's per-user compact
state moves from one launch each to one for all of them - three launches become one
on every one of the 48 GDN layers, so **96 fewer launches per round**.

| arm | flag | verify trace (mean of 33) | round (median of steady-state gaps) |
|---|---|---:|---:|
| v33 | off | 157.764 ms | 257.9 ms |
| v34 | on | 157.729 ms | 256.3 ms |

Trace: **-0.035 ms, 0.02%**. Round: **-1.6 ms, 0.6%**, against a run-to-run spread
that is wider than that in both arms. The batching bought nothing measurable.

## What that rules out

The 94.6 us per launch in the attribution table is **eager-mode host dispatch**, and
the traced round does not pay it. That report says so itself - its own round total is
1564.3 ms/device against the known 157.8 ms traced floor, roughly 10x - but the
ranked target list was still built on per-launch means. This run is the direct test
of that, and it says the launch count is not the constraint.

So the whole *collapse-the-launches* family is retired for the traced round unless a
given change also removes device work:

- **`copy_compact` batching (this change)** - measured, zero. The flag stays, default
  off; it is strictly fewer dispatches for identical bytes, so it costs nothing to
  keep and may matter on an eager path, but it is not a lever here.
- **GDN per-token slice+clone, 107.47 ms over 14,754 launches** (attribution target
  2, `gdn_multitoken_conv.py:146-168`) - the largest launch count in the round, at
  7.2-7.3 us each, i.e. dispatch-bound by the same reasoning that just failed.
  Batching it across users would change launch count and not device work. **Do not
  spend the SOURCES-pin requalification on it** without first showing it moves device
  time, which this result says it will not.

## What is left

Only changes that reduce work the device actually does:

1. **SDPA decode plus its cc=64 companion, ~595 ms combined in the eager capture**
   (333.1 ms of cc=64 paired 1:1 with 262.3 ms of SdpaDecode). Four batch-1 launches
   become one batch-4 launch - fewer launches *and* one weight/mask pass instead of
   four. This is task #39's native 64-row decode graft, already the primary.
2. **Matmul, 314.4 ms (20.1%)**, inflated by the two-tile weight re-read
   (`two_tile_decode.py:186-205`, `:263-332`) that exists to stay on the L1-resident
   one-tile decode arm. The same 64-row native path removes it.
3. **CCL, 85.3 ms over 259 launches on core count 10**, single-ethernet-link-bound.
   Fusing with the adjacent matmul removes dispatch, not fabric time, so by the same
   argument expect little - the 84 GB/s link is the floor.

Both remaining levers are the same change, and it is already the primary task. That
is the honest read: **there is no cheap dispatch win left in the traced round**, and
the 200 tok/s target rests on the native batch-4 decode kernel, not on launch
hygiene.
