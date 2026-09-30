# SDPA decode K/V storage: already bf8, not bf16 - the premise is superseded

Scope: trace which K/V storage the 128 paged-SDPA-decode launches/round (four users x
16 full-attention layers x 2 bundles, `docs/sdpa-batch64-plan.md`) actually stream, and
whether converting it to `bfloat8_b` would shrink the ~2.05 ms/launch, ~40 ms visible/round
cost. Read-only, no hardware.

**Headline finding: the live four-user packed verify round already reads `bfloat8_b`
K/V, zero-copy from the vLLM serving cache.** The `ttnn.bfloat16` citation in the task
brief (`frozen_runtime_context.py:45-46`) is real, but it patches a *different*,
unrelated harness - `dspark-target-hardware.py`'s standalone `generator.allocate_kv_cache`
call, used only to produce the simulator-backend frozen numerics reference
(`frozen_target_replay.py:33`: `report.get('backend') != 'simulator' ... report.get('kv_dtype')
!= 'bfloat16'` - both conditions name the *simulator* qualification path, not live
hardware serving). `docs/sdpa-batch64-plan.md`'s section 3 cited that file for the live
launch's dtype; that citation does not support the claim it was used for. There is no
bf16 replay-owned K/V copy anywhere in the four-user packed round's call chain.

## 1. Trace: whose K/V the launch reads

`ReplayAttentionReader.__call__(self, query, keys, values, ...)` (`attention_replay.py:118`)
does not own or allocate `keys`/`values` - they are caller-supplied parameters, passed
straight through unmodified to `attention_parallel.execute` (`attention_replay.py:129`),
which passes them straight through again, unmodified, to the op
(`attention_parallel.py:17-19`: `paged_scaled_dot_product_attention_decode(stacked, keys,
values, page_table_tensor=pages, ...)`). Neither module slices, copies, or converts them.

`PackedReplayAttentionReader.__call__` (`pooled_attention_replay.py:320-334`) receives one
`keys, values` pair for the whole packed block and hands the *same* pair to every
per-user reader (`pooled_attention_replay.py:333`: `reader(selected, keys, values, ...)`)
- confirming all four users' launches address one shared physical cache, not four private
copies.

So `keys`/`values` are whatever the caller passed in. The caller is the model's own
`attention._decode_from_prep`, whose native bytecode is reused verbatim - `serial_tail`
(`attention_batch.py:116-125`) only overrides the *op* it calls
(`ttnn.transformer.paged_scaled_dot_product_attention_decode`, replaced by the reader) and
the cache-write op, via a `types.FunctionType` copy with patched globals. It changes
nothing about which tensors `_decode_from_prep` passes as `keys`/`values`. Those come from
the attention layer's own bound cache attributes, `attention.paged_k` / `attention.paged_v`
(named directly in `serving_cache_owner.py:25`).

`ServingCacheOwner` (`serving_cache_owner.py:6-36`) exists specifically to assert those
attributes, the model's own `_paged_kv_caches`, and the vLLM runner's `kv_caches` are the
**same physical tensors**, not aliases with independent lifetimes:

```
for target, serving, bound in zip(pair, serving_pair, (layer.paged_k, layer.paged_v), strict=True):
    if target is not serving or target is not bound:
        raise ValueError('Serving, model and attention caches do not share ownership')
    ...
    if (... or target.dtype != self.operations.bfloat8_b):
        raise ValueError('Qualified BF8 TP2 paged KV geometry required')
```
(`serving_cache_owner.py:26-31`, `is not` identity checks, not equality - a copy would fail
this.) `ServingCacheOwner` is constructed once at live serving attach, inside
`serving_runtime.py:167` (`owner = ServingCacheOwner(operations, runner, model)`), in the
same scope that builds the `ServingBufferPool` the packed round uses - this is the live
hardware serving path the CSV numbers come from, not an offline harness.

The write side corroborates it independently: `packed_cache_writer.py:3` states the writer
this same round uses is "the fixture's audited **BF8** read-modify-write of the paged K/V
cache" (`SegmentedOrderedCacheWriter`, wraps `ordered_cache.update`, tiled 32 rows at a
time over the packed block, `packed_cache_writer.py:70-100`).

Geometry from the same check: each of `paged_k`/`paged_v` is `(pages>=68, 2, 64, 256)`
(`serving_cache_owner.py:29`) - `2` local KV heads, `64` rows/page, `256` head_dim, matching
the task brief's signature exactly. `target_packed_pages.py` confirms the packed round's
per-row positions/page-table plumbing (`packed_rows`, `segments`) carries no K/V of its
own either - only positions and page-table indices, consistent with the reader being a
pure view over the shared cache.

**Conclusion for item 1**: the streamed storage is the vLLM paged serving cache itself,
`bfloat8_b`, referenced directly - not a recipe-owned bf16 replay copy. The 2.05 ms is
already streaming bf8 K/V.

## 2. What "changing it to bf8" requires

Nothing - it is already the qualified, running configuration. `ServingCacheOwner`'s dtype
assertion (`serving_cache_owner.py:30`) means the numerics gate that qualifies this pool
already runs against bf8 K/V; there is no bf16-K/V evidence hash to regenerate for this
launch, because bf16 K/V was never live here. The fp32-intermediates `B == 1` branch
(`dflash_combined_sim_runtime.py:17-22`) is unaffected either way - its co-conditions
(`NQH == 16 && NKH == 4 && DHt == 4`, `B == 1`) already describe a different, single-batch
call geometry than this reader's `B = len(bundle)` folded call, independent of K/V dtype.

## 3. Bytes per launch, corrected

Per-page element count is fixed by geometry (`2 * 64 * 256 = 32,768` elements); only the
per-element byte width changes with dtype. At capacity 32768 (512 pages addressed, the T16
qualified family `docs/sdpa-batch64-plan.md` section 3 uses):

| Component | bf16 (batch64 doc's assumed dtype) | bf8 (actual) |
|---|---|---|
| K | 32 MiB | 16 MiB |
| V | 32 MiB | 16 MiB |
| K+V total | 64 MiB | **32 MiB** |

The batch64 doc's own headline arithmetic (`32768 x 256 x 2 x 2 x 2 = 67,108,864 B`)
is internally correct for bf16 - it just was not describing the live launch's dtype.
Halving the byte-width term reproduces 32 MiB (`67,108,864 / 2`) - the real number.

The dense mask (`attn_mask`, `(len(bundle), 1, count*12, capacity)` bf16,
`attention_replay.py:51`) is unaffected by K/V dtype and was not counted in the batch64
doc's bandwidth line. At capacity 32768, count=4: bundle-of-3 launch = `3*48*32768*2` =
9.0 MiB; bundle-of-1 launch = 3.0 MiB.

Combined per-launch bytes (K+V bf8 + mask bf16, same "full-capacity read" simplification
the batch64 doc used): ~41.0 MiB (bundle=3) and ~35.0 MiB (bundle=1), average ~38.9 MiB.
Against the CSV's 2.053 ms average kernel duration
(`C:/Users/liamb/.claude/jobs/8376c877/tmp/profile6g/cpp_device_perf_report.csv`,
`SdpaDecodeDeviceOperation` rows, e.g. 2,129,624 ns and 1,946,050 ns kernel duration): implied
rate is **~19-20 GB/s**, not the batch64 doc's ~32 GB/s (which omitted the mask and used
the wrong K/V byte-width).

## 4. Expected saving: none available from this lever

Because bf8 K/V is already in production, **there is no further saving to claim** - the
premise "bf16 -> bf8 halves bytes, ~40 ms -> ~20 ms" cannot be executed; it already
happened, and the visible ~40 ms/round figure in `docs/trace-datamov-plan.md` is measured
*after* that halving, not before it. If the ~2.05 ms/launch is genuinely streaming-bound
(section 3's ~20 GB/s implied rate is consistent with that, though well under Blackhole's
peak DRAM bandwidth, so a real launch-overhead or compute floor cannot be ruled out from
this evidence alone), the only remaining *bytes* lever inside this call is the mask: bf16,
dense, full-capacity width, ~9-22% of a launch's streamed bytes depending on bundle width.
Shrinking the mask (dtype or a chunked/causal encoding instead of dense-provided) is a
**distinct, unscoped change** from what this task asked about, with its own numerics risk
(the mask carries additive `-inf`/large-negative bias terms into softmax; bf8's dynamic
range and the `is_causal=False` + provided-mask mode the pinned patch's own condition list
names, `dflash_combined_sim_runtime.py:19-20`, both suggest this is not a drop-in retype) -
not evaluated further here since it is outside the K/V question asked.

## 5. Alternatives at the kernel level for the same bytes

These are dtype-independent - the corrected K/V byte-width does not change the batch64
doc's core/CB conclusions, since that doc's CB-overflow numbers (section 2 there) are
**empirical**, from real runs already on the true bf8 K/V, not derived from the mistaken
bandwidth estimate this doc corrects.

- **Read K/V once per user per layer instead of twice** (merge the 2-bundle split,
  `attention_head_fold.py:52-64`): already ruled out by the batch64 doc's own CB
  measurement - widening one user's group from 4 to 8 rows on the full 110-core grid
  already overflows L1 by 256,960 B (run 35583274784, cited in that doc), a real hardware
  result, unaffected by this correction. Still infeasible without a CB-shrinking kernel
  change first.
- **GDN-style disjoint core-range concurrency** (`gdn_user_batch.py:79-96` `core_shares`,
  four users on four disjoint 24-core shares, one `MeshProgramDescriptor`): the same
  pattern applied to SDPA would need re-hosting the pinned reader/writer/compute kernels
  into a custom multi-`CoreRangeSet` wrapper (SDPA's kernel is pinned/hashed,
  `dflash_t16_native_attention_gate.py:18-25`, 14 files) rather than reusing kernels
  verbatim the way `gdn_user_batch.py:19-20` does. `gdn_user_batch.py:44-58` documents its
  own fused launch measuring **slower** than the per-user split below a 2-user threshold -
  a real precedent against assuming a win, not evidence for or against SDPA specifically.
  SDPA's 96-of-110-core usage per launch (batch64 doc, section 2) leaves less headroom for
  four disjoint shares than GDN's 24-of-110.

Neither alternative is changed in feasibility by this correction; both were already
correctly assessed as high-effort/uncertain-payoff in `docs/sdpa-batch64-plan.md`.

## 6. Effort and risk

Effort to act on the task as scoped (K/V bf16->bf8): **zero** - already shipped and
qualified, verified structurally on every serving attach by `ServingCacheOwner`. Riskiest
step in this investigation was mis-trusting a citation without checking whether the file
it named was reachable from the live call path - `frozen_runtime_context.py` never
appears in the four-user packed round's import chain (`model_batch.py`,
`pooled_attention_replay.py`, `attention_replay.py`, `attention_parallel.py`,
`serving_cache_owner.py`, `serving_runtime.py`), only in the offline
`dspark-target-hardware.py` simulator harness. If any further K/V-dtype work is wanted,
the correct next question is whether the *mask* (section 4) is worth a separate,
independently-scoped investigation - not raised or estimated here.

## Recommendation

Close this line of work: there is nothing to implement. Update
`docs/sdpa-batch64-plan.md` section 3's dtype claim and bandwidth estimate (64 MiB / ~32
GB/s) to bf8 (32 MiB / ~20 GB/s including the mask) so the number doesn't get reused
uncorrected elsewhere. If more bytes need to come out of this launch, scope the mask
(section 4) as its own task rather than folding it into a "K/V bf8" ticket that is already
done.
