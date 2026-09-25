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

---

## RETRACTION 2026-09-22: the "on" arm was not on

**This verdict's measurement is a null comparison and its conclusion is unsupported.**

v33 and v34 differ by exactly one flag, `M3NATIVE_GDN_STATE_COPY_BATCH=1`, which the
arm passes through as `QWEN_FAST_GDN_STATE_COPY_BATCH=1`. Both ran on image
`sha256:e41ef884f4c8`. A CPU tree diff of that image against a fresh build
(run 35685598363) established what its `/experiment-scripts/ci` actually holds:

    gdn_state_copy.py         4cc8f6529b94f789   built from commit 12d62af5af07
    gdn_device_loop_state.py  2c27c93ee34964cb   built from commit 4b144ce2a766

At that vintage:

- `gdn_state_copy.py` defines `page_counts`, `transfer_counts`, `copy_active`,
  `copy_compact` and `_copy_state`. There is **no `FLAG`, no `batch_enabled`, no
  `copy_compact_batch`**, and zero occurrences of the string
  `QWEN_FAST_GDN_STATE_COPY_BATCH`.
- `gdn_device_loop_state.py` line 5 is `from gdn_state_copy import copy_compact` -
  `copy_compact` alone.
- Nothing else at that vintage references the flag either. (`model_batch.py` and
  `gdn_device_loop_state.py` do contain `batch_enabled` substrings, but every one is
  `norm_batch_enabled` or `user_batch_enabled`, which are
  `gdn_batched_conv.norm_batch_enabled` and `gdn_user_batch.enabled` - unrelated.)

The batching landed in commit **7ecf980a**, which added `batch_enabled`,
`copy_compact_batch` and the import together, and added the module to neither of the
two copy lists the image build needs. So the code was never in any image, and
`e41ef884f4c8` predates it regardless.

**v34's arm therefore ran the same configuration as v33.** The table's
-0.035 ms trace and -1.6 ms round are run-to-run noise between identical arms, which
is what the original text was describing when it noted "a run-to-run spread that is
wider than that in both arms" - correctly, and for the wrong reason.

### What this does and does not invalidate

- The claim **"the batching bought nothing measurable"** is vacuously true: nothing
  was applied. It is not evidence about batching.
- The claim **"launch count is not what the traced round is paying for"** has **no
  supporting measurement**. It is not disproven; it is unsupported. The 96-fewer-
  launches figure was derived by reading the code, not observed in either run.
- The retirement of **the whole collapse-the-launches family**, including the
  14,754-launch GDN slice+clone, is withdrawn. That family is untested, not retired.
- The trace attribution in `docs/trace-op-attribution.md` and its point about
  eager-mode host dispatch (94.6 us per launch being dispatch, not device time) stand
  on their own; they did not depend on this run.

### The sting in the tail

When `7ecf980a`'s code finally does reach hardware - image v81
(`sha256:836e7abb`), which carries today's repo versions of both files - the engine
**hangs during warmup and never reaches readiness**. Two runs, two different frames,
both a ttnn op launched and never returned inside `attach_combined_runtime`:

    v41 35684239068  gdn_user_batch_conv.py:72   ttnn.transformer.gdn_decode_conv_gates
    v43 35685401900  ordered_cache.py:127        ttnn.generic_op (SOURCE_CODE kernels)

v43 had the K64 kernel graft excluded (`native_attn engaged` 0 occurrences, `runtime
binary pin overridden` 0), so the graft is not the cause - and the graft binary hash
was identical on working and hanging runs anyway (`3e40b501` over `4b7299c1`,
factory `fd8c0676`, on both `e41ef884f4c8` and v81).

So the state-copy batching is not merely unmeasured. On the evidence available it is
**broken on device**, and the only reason that has been invisible is that every
m3native run served an image built before it.

### Task status

Task #51 is reopened. The measurement it recorded did not take place.
