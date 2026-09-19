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
