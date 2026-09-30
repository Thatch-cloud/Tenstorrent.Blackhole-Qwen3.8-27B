# GDN prefill takes the slow causal-conv on every chunk

Run 35427384650. The largest single actionable finding in prefill, and it is a one-line
dispatch condition rather than a kernel.

## What was measured

`gdn/tp.py` was grafted with a log line on the causal-conv dispatch, and the same prefill
shape re-run. **576 markers, every one from the single-prefill path, none from batched:**

| valid_len | calls | what it is |
| ---: | ---: | --- |
| 2048 | **432** | a **full** chunk |
| 1288 | 48 | tail |
| 520 | 48 | tail |
| 264 | 48 | tail |

`valid_len` is **never `None`**. So this condition never fires:

```python
if self._gdn_conv1d and valid_len is None:
    conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)   # native ttnn.conv1d
else:
    conv, conv_new_state = _causal_conv1d_fir(qkv, ...)            # MAC FIR
```

`_gdn_conv1d` is `True` unconditionally, so the native `ttnn.conv1d` path **is never
taken**, including on the 432 full-chunk calls (75%) that have nothing to mask.

## What it costs

The FIR unrolls the K=4 causal convolution into three shifted windows. A shift of 1, 2 or
3 tokens is never tile-aligned, so each tap forces `untilize -> slice -> tilize` over the
full `[1, S, 5120]` activation:

```
Untilize 112 us -> Slice 108 us -> Tilize 164 us -> Ternary(mac) 122 us
```

384 us of layout to set up a 122 us multiply-accumulate, three times per GDN layer per
chunk. In the device profile that is **1,152 cycles, 473 ms, 12.8% of prefill**, of which
**357 ms is pure layout conversion**.

At the measured 75% full-chunk share, the recoverable part is roughly **355 ms, about
9.6% of prefill**. The two runs used slightly different prompt lengths, so treat the
ratio and the total as combined from adjacent measurements rather than one.

## The fix, and why it is safe

The code's own comment already states the equivalence:

> Masked buckets still pass a real valid_len (< T) so their exact masking is unchanged,
> and **for a full chunk the None slice and the valid_len==T one-hot select the identical
> rows.**

So:

```python
if self._gdn_conv1d and (valid_len is None or valid_len == T):
```

Only the **carry extraction** depends on `valid_len`, and for `valid_len == T` the two
methods are documented to select the same rows. The convolution itself never depended on
it.

## A second prize, possibly larger

The same comment warns that the `valid_len`-set path builds a one-hot via
`ttnn.from_torch`, a host write that `TT_FATAL`s inside a captured trace. And every one of
the 1,152 cycles in the profile is **untraced** (`METAL TRACE ID` empty).

Those two facts fit together: chunked prefill may be running untraced *because* a real
`valid_len` is always passed. If the dispatch fix also restores trace capture for full
chunks, the saving is larger than the conv itself, because it removes per-op dispatch
across the whole chunk. **Unverified** - worth measuring right after the dispatch fix,
not assumed.

## Why this was not visible from the source

`model.py:1017` passes `valid_len=None` for full chunks, exactly as intended:

```python
# valid_len=None: no GDN mask; trace-safe static conv capture (matches valid_len==chunk_size).
x_new = layer.forward(x, mode="prefill", chunk_size=..., valid_len=None)
```

That call site is correct and is not the one serving uses. Reading the source supported
the wrong conclusion; only logging the branch at runtime settled it. The fourth
never-firing condition found this month, after the chunk-size default, the deprecated
`num_links`, and the M1 dispatch.

## RESULT: correct, and 2.5% slower. Do not adopt

Run 35428550094. All three controls held and the gate passed on correctness:

| control | |
| --- | --- |
| lever moved | 96 markers `full=True`, 96 `full=False` - native conv1d ran for full chunks, FIR for tails |
| tokens identical | `[279, 3841, 13477, 37550, ...]` byte-for-byte across both arms |
| both arms complete | yes, two generates each |

And the measurement, now prefill-only rather than diluted by sixteen decode steps:

| arm | 2062 tokens | tok/s |
| --- | ---: | ---: |
| baseline (MAC FIR) | 0.967 s | 2133.2 |
| fixed (native conv1d) | 0.991 s | **2080.0** |

**Speedup 0.975. The fix is a 2.5% regression**, against a predicted 9.6% gain.

### Why the prediction was wrong

The estimate priced what the FIR costs and never priced what replaces it.
`_conv1d_prefill` is not layout-free:

```python
xin = ttnn.concat([conv_state, qkv], dim=1)
xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT)   # a full untilize
xin = ttnn.reshape(xin, (1, Lin, 1, C))
...                                                 # ttnn.conv1d
out = ttnn.reshape(out, (1, T, C))
out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)        # a full tilize
```

It removes two of the three layout round-trips and then spends the saving inside
the conv kernel, which costs more than three elementwise macs. Net: slightly
negative.

This is the same error family as the 9.3 GB/s: a number derived from one side of
a trade, used as if it were the trade. **Measuring the cost of what you remove
tells you nothing until you measure the cost of what you put in its place.**

### What survives

The 473 ms is still real, and still 12.8% of prefill. What is now known is that
**neither available implementation is cheap**: the FIR pays three untilize/slice/
tilize round-trips, `ttnn.conv1d` pays one round-trip plus a slower kernel, and
they land within 2.5% of each other. The cost is the layout conversion itself,
which both paths accept as unavoidable.

So the remedy is the one this project already has a pipeline for: a **tiled-layout
fused causal conv**, doing the K=4 FIR over the time axis without ever leaving
TILE layout. `GdnConvGatesDeviceOperation` is exactly that for decode and is
already in the build; prefill has no equivalent. That is a kernel build at
roughly two minutes per iteration through the graft, not a dispatch change.

Upper bound on the prize stays ~10% of prefill, and it is now bounded from below
too: anything that still untilizes will land where these two did.

### Retracted

The baseline had failed on its second prefill twice, at 6144 and 4096, with an
MMIO timeout inside 3 us of the same value, and that was flagged as possibly the
FIR path corrupting device state. **It is not.** Both arms here ran two generates
at the same length with no MMIO error at all. The earlier failures changed prompt
length between calls; the crash follows the shape change, not the conv path.

## The fusion that exists is decode-only, and that is the whole story

Checked because the obvious question after a failed dispatch swap is whether the
existing fusion work was done correctly. It was. All five custom fused ops are
engaged at runtime, confirmed from the gate run's own logs:

```
QWEN_ATTN_PREP engaged:        attn_decode_prep -> q [1,1,12,256] height-sharded
QWEN_GDN_CONV_GATES engaged:   ttnn.transformer.gdn_decode_conv_gates K=4
QWEN_GDN_NORM_GATE engaged:    norm+gate folded into decode_gated_delta_rule_packed
QWEN_GDN_PACKED_QKV engaged:   ttnn.transformer.decode_gated_delta_rule_packed
QWEN_GDN_PROJ_DIRECT engaged:  conv+gates reads qkvzab [1,1,8240] directly
```

They default to `"0"` in `gdn/tp.py` and `attention/tp.py`, and neither the
workflows nor `worker.py` set them, so the image environment does. Worth knowing
that the defaults are off: a run launched without the image environment silently
measures the unfused model.

**Every one of them is decode.** `attn_decode_prep`, `gdn_decode_conv_gates`,
`decode_gated_delta_rule_packed`. Prefill has custom ops for the scan -
`ChunkGdnScan` and `ChunkGdnPrep`, 517 ms of real work between them - but nothing
for the conv and gates.

That single fact accounts for the whole prefill/decode asymmetry without needing
another hypothesis:

| | decode | prefill |
| --- | --- | --- |
| fused conv/gates op | yes, engaged | **none** |
| residue | 3.44 ms over 15 op types, diffuse | 637 ms layout, 80% of op calls do no arithmetic |
| conv implementation | one fused op | FIR with 3 untilize/slice/tilize, or conv1d with 1 |

So the dispatch swap was trying to reach a fused-op win by choosing between two
unfused stock ops, and could not: both pay the layout conversion, which is why
they landed within 2.5% of each other.

**The fix is the prefill sibling of `gdn_decode_conv_gates`**, built on the same
pattern through the same graft pipeline that produced the decode five, operating
in TILE layout throughout. It is the only approach measured to be capable of
taking the 473 ms, because it is the only one that does not untilize.

## VERDICT: shift-by-matmul is 4.5x slower. The kernel is not built

Run 35429279459, before any kernel work, exactly to test the premise the build
rested on.

| arm, [1, 2048, 5120] bf16, 4 taps | ms |
| --- | ---: |
| slice FIR, what the model runs today | 2.377 |
| shift by batched 32x32 matmul, never leaving TILE | **10.706** |

**PCC 1.0, max absolute difference 0.0.** The arms agree exactly, so the
composition is right and the timing means what it says: the approach is
**4.5x slower**, not faster.

### It could never have won, and arithmetic says so without the run

Per 32x32 output tile, for one tap:

| | MACs |
| --- | ---: |
| the elementwise multiply-accumulate a FIR tap actually needs | 1,024 |
| a 32x32 matmul used to shift the rows for it | **32,768** |

The matmul exists only to *move* rows and costs **32x the arithmetic of the
multiply it enables**; across four taps and both the self and previous tile it is
256x. No amount of kernel polish recovers a factor of 256 in issued work - the
measured 4.5x is the composition being partly amortised, not the cost being
avoidable.

**This should have been computed before the probe was written, and certainly
before a kernel was proposed.** The proposal was made on the strength of the
pattern existing elsewhere in the codebase - chunk_gdn_prep's eye/tril/ones - and
"someone else uses this primitive" was allowed to stand in for "this primitive is
cheap for this purpose". chunk_gdn_prep matmuls because it genuinely needs a row
mixing; a conv does not.

### What this closes, and what it leaves

The prefill conv line of attack is closed at the ttnn level, and the three
approaches now measured all land in the same place:

| approach | result |
| --- | --- |
| slice FIR (current) | baseline, 3 untilize/slice/tilize per layer-chunk |
| native `ttnn.conv1d` | 0.975x, run 35428550094 |
| shift-by-matmul | **0.222x**, this run |

The 473 ms is real, and nothing reachable through composed ttnn ops takes it. A
kernel would need a genuine sub-tile row-shift primitive - addressing rows
*within* a tile in the compute kernel - and no evidence has been gathered that
Tensix exposes one. Until someone establishes that it does, **the prefill conv
should be left alone**, and the 12.8% treated as the cost of a K=4 causal
convolution on tiled hardware rather than as available headroom.

The cost of finding this out was one probe and roughly ten minutes of hardware,
against a kernel build estimated in days. That trade is the only part of this
worth repeating.
