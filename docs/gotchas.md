# Gotchas

Things that cost us hours. Most present as something other than what they are.

---

## Device numbers are not stable across a board reset

`tt-smi -r` re-enumerates. Three different mappings observed on one host in one
afternoon:

| board | run 1 | after reset A | after reset B |
| --- | --- | --- | --- |
| A | `/dev/…/0` | `/dev/…/1` | `/dev/…/0` |
| B | `/dev/…/1` | `/dev/…/0` | `/dev/…/1` |
| C | `/dev/…/2` | `/dev/…/2` | `/dev/…/2` |

Use `/dev/tenstorrent/by-id/blackhole-<serial>`, which is keyed on the board.

**But by-id alone is not enough.** `readlink -f` returns a *device number*, valid
only until the next re-enumeration:

```bash
D=$(readlink -f /dev/tenstorrent/by-id/blackhole-XXXX)   # -> /dev/tenstorrent/2
tt-smi -r 0 1 2                                          # RENUMBERS
docker run --device $D ...                               # $D is now stale
```

**Resolve after the reset, never before.** We did this and lost a run to it.

---

## Not every pair of cards can form a mesh

Cabling is a graph. On a 3-card line:

```
A ══2 links══ B ══2 links══ C
              ← hub
```

`{A,B}` and `{B,C}` are valid `(1,2)` meshes. **`{A,C}` is not** — two hops,
no direct link. An allocator picking "any 2 of 3" gets it wrong one time in three.

**The failure is silent.** No error. The mesh opens, weights load, and the only
symptom is one line of fabric telemetry:

```
Logical  ... intra-mesh degree histograms mesh0 {1:2}    <- requested
Physical ... intra-mesh degree histograms mesh0 {0:2}    <- reality
```

`{0:2}` is two nodes of degree **zero**. Assert that Logical and Physical agree
after opening the mesh and **before loading weights**. It is cheap and it catches
both stale device numbers and real cabling faults.

Discover adjacency with `test_system_health`, which reports per-channel state and
names the peer:

```
eth channel 9 core 0-9 link UP (QSFP), connected to Chip 1 Physical Slot: 1 core 0-4
```

It needs `TT_MESH_GRAPH_DESC_PATH` at v0.77, or all its tests fail and it prints
nothing — which reads exactly like a dead fabric.

---

## Links train at board init only

After re-cabling you read **zero** links until `tt-smi -r`. We spent hours
suspecting cables, seating, and the cable spec. The cabling had been correct the
whole time; the boards simply had not been reset since.

---

## The mesh descriptor's channel count is asserted, not advisory

tt-metal ships both of these, and they disagree:

| Descriptor | dims | `channels` |
| --- | --- | ---: |
| `p150_x2_mesh_graph_descriptor.textproto` | `[1,2]` | **4** |
| `p300_mesh_graph_descriptor.textproto` | `[1,2]` | **2** |

One QSFP-DD cable gives **2 links per hop**. So the shipped `p150_x2` fails with
`TT_FATAL: Expected 4 eth links` on correctly-cabled hardware, and `p300` is the
right file. **Choose from measured link count, not from card model.**

And always pass `TT_MESH_GRAPH_DESC_PATH`. Without it:

```
warning | Op | Failed to discover available ethernet links; falling back to 1 link
```

Every collective silently runs on one link, in a path whose all-gather expects
two. Wrong, not fatal — the worst combination.

---

## Two Blackhole devices are a "P300"

`models/tt_transformers/tt/model_config.py` keys purely off device **count**:

```python
{1: "P100"/"P150", 2: "P300", 4: "P150x4", 8: "P150x8", 32: "BHGLX"}
```

Two boards on a cable are a P300 as far as the model stack is concerned. Note the
C++ `ClusterType` is *board*-derived and will say `P150_X2` — the two naming
systems disagree, and the model path uses the device-count one.

---

## "Out of Memory … but bank size is 0 B" is not an OOM

```
Out of Memory: Not enough space to allocate 1024 B L1_SMALL buffer across 64 banks,
where each bank needs to store 16 B, but bank size is 0 B
```

The region **does not exist**. `ttnn.conv1d` in the GDN prefill path needs an
L1_SMALL arena; the demo opens its device with `l1_small_size=24576` and vLLM
opens with none unless told:

```
--additional-config '{"tt": {"l1_small_size": 24576}}'
```

---

## vLLM will not start any Qwen3.5/3.6/3.8 model without `--no-enable-prefix-caching`

```
ValidationError: --mamba-block-size can only be set with --enable-prefix-caching
```

vLLM classifies Gated DeltaNet as Mamba-style and branches on prefix caching to
size `mamba_block_size`:

```python
if <prefix caching enabled>:  mamba_block_size = cache_config.block_size      # 16
else:                         mamba_block_size = model_config.max_model_len   # validates
```

Core runs that **before** the TT platform disables prefix caching, so
`mamba_block_size` is left at 16 and the validator then rejects it. The platform
knows the answer — the model declares `supports_prefix_caching: False` — it just
applies it too late.

Disable prefix caching **explicitly up front** so core takes the else-branch.

---

## Batch limits come from head count, not just memory

`chunk_gdn_phased_program_factory.cpp:137`:

```
BH <= ncores
  info: num_heads 192 exceeds compute cores 110
```

`BH = B × Nv_tp`, where `Nv_tp = linear_num_value_heads / TP`. With 48 value heads:

| | `Nv_tp` | B=2 | B=4 | B=8 |
| --- | ---: | ---: | ---: | ---: |
| TP=2 | 24 | 48 ✓ | 96 ✓ | **192 ✗** |
| TP=4 | 12 | 24 ✓ | 48 ✓ | 96 ✓ |

This binds **batched prefill only**, and only for prompts ≤256 tokens — longer
prompts prefill per-user and sidestep it. Hence `QWEN_BATCHED_GROUPED=0` to make
short-prompt B=8 work at TP=2.

Separately, **B=32 fails at TP=2** with
`per_core_M % out_subblock_h == 0` — a divisibility assert; the shape factors at
TP=4 and does not at TP=2.

---

## The demo's defaults assume four cards

Two of them will stop a 2-card run dead:

```python
_MESH_SHAPE = {...}.get(os.environ.get("MESH_DEVICE"), (1, 4))   # default FOUR
_mode = os.environ.get("QWEN36_BATCHED_DECODE_MODE", "shard")     # needs on-device sampling
```

The second is worth understanding: `"shard"` requires the on-device sampler, which
requires ≤65,536 logits/device, which requires TP=4 (vocab 248,320 gives 124,160
at TP=2). There is a `"host"` mode that works — and the auto-fallback covers
`"sample"` but **not** `"shard"`, so the default asserts rather than degrading.

---

## `tt-smi` is not on the non-interactive PATH

```
$ ssh host 'tt-smi -r 0 1 2'
bash: tt-smi: command not found
```

It lives at `~/.local/bin/tt-smi`. A login shell finds it; `bash -s` and systemd
units do not. A reset that silently did not happen presents as a dead fabric, not
as a missing binary.

---

## `QWEN_SDPA_BF8` means opposite things at different batch sizes

| Config | Verdict |
| --- | --- |
| B=1, 64k | **don't** — buys 12% TTFT for precision upstream marks "validate PCC at long ctx" |
| B=8, 64k | **required** — without it the allocator OOMs |

It halves paged KV, so it is an optimisation in one place and an enabling
condition in another. Upstream gates it with a correctness caveat, so if you need
it at long context, validate accuracy for that configuration specifically.

---

## Hangs need an external watchdog

Some failures wedge the device rather than erroring. `pytest --timeout` uses
`method: signal`, which a wedged device op may not honour. Any CI lane running
these tests needs a wall-clock kill plus `tt-smi -r`, or one bad commit parks the
hardware indefinitely.

Distinguish a hang from slow progress by telemetry: a wedged die sits at **full
aiclk drawing full power with the log not advancing**.
