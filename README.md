# Qwen3.5 / 3.6 / 3.8-27B on two Tenstorrent Blackhole cards

Running Tenstorrent's `qwen36` model on a **2-card** Blackhole p150a mesh, with a
working OpenAI-compatible vLLM endpoint — including the fixes and configuration it
took to get there.

Upstream documents this model family at **4 and 8 cards** (`P150x4`, `P150x8`).
Every test docstring says `MESH_DEVICE=P150x4`. There is no published path for two
cards, and several defaults assume four. This is that path, measured.

> **Status:** working, with caveats stated inline. The tt-metal fix stack this
> depends on is **unmerged and largely unreviewed** — see [Dependencies](#dependencies).
> Read that section before putting this in front of anyone.

---

## Results

Hardware: 2 × Blackhole **p150a** (32 GiB each), linked by one QSFP-DD cable
(2 ethernet links), opened as a `(1,2)` mesh. Model `Qwen/Qwen3.8-27B`, weights
`bfloat8_b` with `bfloat4_b` on MLP gate/up — upstream's default, no quantisation
work required.

### Single stream, full context ladder

| Context | TTFT | Decode |
| ---: | ---: | ---: |
| 128 | 0.19 s | 18.19 tok/s |
| 4,096 | 1.04 s | 18.01 tok/s |
| 65,536 | 27.3 s | 16.19 tok/s |
| ~104k¹ | 51.8 s | 15.80 tok/s |
| **262,144** | 219.9 s | **12.76 tok/s** |

¹ the 128k case is capped ~104k by its source text; only the 256k run fills its window.

**Decode falls 30% across a 2048× context range.** The architecture is hybrid —
48 of 64 layers are Gated DeltaNet (linear attention, context-independent per
token), 16 are full attention. Long context is nearly free on decode; the cost is
all in prefill.

**TTFT turns superlinear past ~100k**, which is the 16 quadratic layers taking
over:

| step | context ratio | time ratio |
| --- | ---: | ---: |
| 64k → ~104k | 1.63× | 1.90× |
| ~104k → 262k | 2.52× | **4.24×** |

Size long-context capacity assuming **quadratic TTFT above 100k**, not linear.

### Versus a single card

The 27B fits on one 32 GiB card (27.49 GiB at bf8/bf4). Two cards buy more than
the obvious 2×:

| Context | 1 card TTFT | 2 cards TTFT | 1 card decode | 2 cards decode |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.338 s | **0.19 s** | 7.8 | **18.19** |
| 4k | 19.4 s | **1.04 s** | 7.7 | **18.01** |
| 64k | **OOM** | **27.3 s** | — | **16.19** |

64k TTFT improves **6.4×** (175 s → 27 s), well beyond a card-count argument: the
single card needed `QWEN_SDPA_BF8=1` *and* was thrashing its memory ceiling, so
removing both compounds.

### Batched serving (B=8)

| Context | TTFT | Per-user decode | **Aggregate** |
| ---: | ---: | ---: | ---: |
| 128 | 4.2 s | 12.56 tok/s | **100.5 tok/s** |
| 4,096 | 57.7 s | 12.23 tok/s | **97.9 tok/s** |
| 8,192 | 90.1 s | 11.71 tok/s | **93.6 tok/s** |
| 65,536² | 275.1 s | 9.81 tok/s | **78.5 tok/s** |

² requires `QWEN_SDPA_BF8=1`; without it the allocator OOMs.

**~5.5× aggregate throughput at B=8**, for a 31% per-user decode cost.

### Correctness

**GSM8K: 58/60 = 96.7%** (canonical `openai/grade-school-math` test set, greedy,
via the live endpoint at B=8, `max_tokens=2048`). Excluding the one item that
returned no parseable answer: 58/59 = 98.3%.

Use **`max_tokens=2048`**: an earlier pass at 640 scored 55/60, but two of the
three "failures" were truncation mid-reasoning rather than wrong arithmetic.
Reasoning models need the budget.

Only 2 genuine misses remain, and one is `expected 13, got 12` — an off-by-one
after correct reasoning. That is a model error, not a kernel one.

This validates the stack **as a composition** — quantisation, tensor-parallel
sharding, the patched kernels, and vLLM's scheduler and paged KV all preserving
multi-step arithmetic. It is *not* a side-by-side against a CUDA reference, so it
rules out gross regression rather than subtle drift.

---

## Dependencies

This does **not** work on stock tt-metal. It needs three unmerged PRs plus a
one-character fix of our own.

| PR | What it fixes |
| --- | --- |
| [#53314](https://github.com/tenstorrent/tt-metal/pull/53314) | conv2d channel-chunking — the depthwise GDN conv does not fit L1 at TP=2 |
| [#53319](https://github.com/tenstorrent/tt-metal/pull/53319) | `ttnn.slice` tile-window carve — static CB clash in the GDN qkv-carry slice |
| [#53320](https://github.com/tenstorrent/tt-metal/pull/53320) | qwen36 demo/model layer, incl. adding `"P300": (1, 2)` to the mesh map |

Root cause for the conv wall is documented in
[#53303](https://github.com/tenstorrent/tt-metal/issues/53303): the conv L1
estimator is **group-blind**, computing an 838 MB weight tensor for a depthwise
conv whose real weight is ~80 KB.

> **Read this before depending on it.** These PRs are authored by an agent account
> (`ctxbot`) from a fork and carry essentially no human review — two have zero
> reviews, one has only a bot review. Their silicon receipts are detailed and
> matched our hardware exactly (we reproduced #53319's reported failure
> byte-for-byte: same program number, same two L1 addresses, same file and line).
> That is strong corroboration. It is not maintainer endorsement.

### Our fix on top: batch truncation in the GDN FIR conv

`patches/53320-fix-fir-batch-truncation.patch` — **one character.**

#53320 replaced a Python slice with an explicit `ttnn.slice` to keep a
row-major hop in DRAM (a sound fix for an L1 OOM). But writing the bounds out by
hand pinned the **batch** end to a literal `1`:

```diff
-        x_slice = x_padded[:, k : k + T]                                # keeps all B rows
+        x_slice = ttnn.slice(x_padded, (0, k, 0), (1, k + T, D), ...)   # batch end LITERAL 1
```

Identical for B=1; for B≥2 every user past the first is silently discarded. It
surfaces far downstream as `Ends 2 must be less than or equal to the shape of the
tensor 1`. The change was intended as a no-op — its own comment says *"Pure memory
placement, numerics unchanged"* — and the module's own tests only ever run
`batch_size=1`, so nothing caught it.

Fix is `(1, k + T, D)` → `(B, k + T, D)`; `B` is already in scope. Verified on
silicon: batched GDN prefill at B=2 and B=4 pass at PCC 0.99998–1.00000.

---

## Build

```bash
scripts/tt-build-images.sh --prstack --vllm
```

Three layers, each taking `BASE` as a build arg so they compose:

| Image | Contains | Size |
| --- | --- | ---: |
| `tt-bringup` | tt-metal + ttnn + `test_system_health` | ~15 GB |
| `tt-serving` | + torch / transformers / pytest | ~22 GB |
| `tt-vllm` | + stock vLLM and the TT plugin | + ~2 GB |

Set `REGISTRY` (default `localhost:5000`) and `TAG` (default `v0.77.0-rc1`).

`--from serving` / `--from vllm` skip earlier stages. Worth knowing: the
`-prstack` layer builds **FROM the built bring-up image**, so applying the patches
triggers an *incremental* tt-metal rebuild — about a minute, versus hours from
scratch.

### The vLLM layer

Built from [`tenstorrent/vllm-tt-plugin`](https://github.com/tenstorrent/vllm-tt-plugin)
— **stock vLLM plus a plugin-only repo**, which is where Tenstorrent are moving.
(An earlier version of this repo used their vLLM *fork*; the maintainer's guidance
on [tenstorrent/vllm#473](https://github.com/tenstorrent/vllm/issues/473) is to
migrate, and we have.)

Their install script is worth reading before you replace it with something
simpler. vLLM's PyPI metadata is generated on a CUDA machine, so a plain install
resolves `requirements/cuda.txt` — torch, `flashinfer`, `tilelang`, `nvidia-*` —
**regardless of `VLLM_TARGET_DEVICE`**. That fights tt-metal over `torch` and adds
several GB. They fetch vLLM's `requirements/common.txt` at the pinned tag, install
that explicitly, then install vLLM itself `--no-deps --no-binary`. torch stays the
tt-metal one by construction.

---

## Serve

```bash
docker run -d --name ttserve -p 127.0.0.1:8000:8000 -w /opt/vllm-tt-plugin \
  --device /dev/tenstorrent/<a> --device /dev/tenstorrent/<b> \
  -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
  -v $HOME/tt-hf-cache:/root/.cache/huggingface \
  -e HF_MODEL=Qwen/Qwen3.8-27B -e MESH_DEVICE=P300 \
  -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
  -e VLLM_PLUGINS=tt,tt_model_registry -e VLLM_RPC_TIMEOUT=100000 \
  -e QWEN36_BATCHED_DECODE_MODE=host \
  $REGISTRY/tt-vllm:v0.77.0-rc1-prstack-plugin \
  python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.8-27B --served-model-name qwen3.8-27b \
    --max_model_len 4096 --max-num-seqs 8 --no-enable-prefix-caching \
    --block-size 64 --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --port 8000 --host 0.0.0.0 \
    --additional-config '{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D", "trace_region_size": 1073741824}}'
```

**Two deliberate choices in that command, both security-relevant.**

`-p 127.0.0.1:8000:8000`, not `-p 8000:8000`. Docker's default publishes on
*every* host interface and [punches through the host firewall to do
it](https://docs.docker.com/engine/network/port-publishing/) — this endpoint has
no authentication, so the bare form puts an open inference server on your
network. Bind it to loopback and put a reverse proxy in front if you need it
reachable. Note that vLLM's own `--api-key`
[does not protect every endpoint](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/),
so it is not a substitute. `--host 0.0.0.0` stays as-is: that is the *container's*
interface, and the published port is what actually controls exposure.

`$HOME/tt-hf-cache`, not `$HOME/.cache/huggingface`. Hugging Face
[stores your access token in that directory](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables),
and this image is built from unmerged, largely unreviewed upstream code. A
dedicated cache directory keeps the token out of the container while still
letting weights download and persist. Stricter still: mount only the model
snapshot, read-only.

**Readiness takes ~510 s** — weights, warmup, and prefill trace capture. Budget
at least 600 s. A default 30 s initial delay will restart-loop forever, and the
symptom reads as a broken image rather than an impatient probe.

(For reference, the older fork-based build was 330–375 s. The plugin build is
~40% slower to start; cause not established, first-run JIT cache population being
the leading suspect.)

**Every flag is load-bearing.** See [docs/gotchas.md](docs/gotchas.md) for what
each one prevents. The short version:

| Flag | Without it |
| --- | --- |
| `--no-enable-prefix-caching` | engine won't construct — affects **every** Qwen3.5/3.6/3.8 GDN model |
| `l1_small_size: 24576` | `ttnn.conv1d` cannot allocate; error misleadingly reads as OOM |
| `--max-num-seqs 8` | B=32 hits a matmul divisibility assert at TP=2 |
| `--reasoning-parser qwen3` | users receive raw chain-of-thought in `content` |
| `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` | tool calls arrive as raw `<tool_call><function=...>` markup in `content`, `tool_calls` stays null, and `tool_choice: "auto"` is rejected |
| `MESH_DEVICE=P300` | the demo path defaults to `(1,4)` and asks for four cards |

Two Blackhole devices are a **`P300`** as far as tt-metal is concerned —
`determine_device_name` keys purely off device count, regardless of whether
they're two dies on one board or two boards on a cable.

---

## What we could not make work

Recorded because absence of evidence is useful too.

**B=32 at TP=2.** Fails with `per_core_M % out_subblock_h == 0`
(`matmul_program_config.cpp:1069`) — one bug with four manifestations, in unit
tests and end-to-end alike. A *divisibility* assert: the shape factors at TP=4 and
does not at TP=2, because halving device count doubles per-device M. B=8 works.

**Batched prefill above B=4.** `BH ≤ ncores` in
`chunk_gdn_phased_program_factory.cpp:137`, where `BH = B × Nv_tp` and
`Nv_tp = linear_num_value_heads / TP`. At TP=2 with 48 value heads that's
`8 × 24 = 192 > 110` compute cores. Prompts over 256 tokens sidestep it by
prefilling per-user, which is why long-context B=8 works and *short*-context B=8
needs `QWEN_BATCHED_GROUPED=0`.

**On-device sampling.** Hard-refused below TP=4: vocab 248,320 gives 124,160
logits/device at TP=2, over the 65,536 ceiling. Host sampling works
(`QWEN36_BATCHED_DECODE_MODE=host`).

---

## Licence and provenance

The original work here — the Dockerfiles, the build script, and the documentation
— is MIT, per [LICENSE](LICENSE).

**The patch in `patches/` is not.** It is a one-line change to tt-metal, which is
Apache-2.0, so it is a derivative work of Apache-2.0 code and is offered under
**Apache-2.0**, not MIT. The same applies to the equivalent in-line fix applied by
`docker/tenstorrent-bringup-prstack.Dockerfile`. It is offered upstream.

tt-metal, the vLLM plugin, and the model weights are the property of their
respective owners under their own licences and are neither vendored nor
redistributed here — the images fetch them at build time. See [NOTICE](NOTICE) for
the component-by-component breakdown.

Measurements were taken on 2 × p150a on 2026-08-24 at tt-metal `v0.77.0-rc1`. They
are single runs unless stated; where we repeated a configuration it reproduced to
within 0.5%.
