# tt-metal custom ops for Qwen3.8-27B on two Blackhole p150a cards

This directory holds the sources of the five custom ttnn operations that the Qwen3.8-27B fast-serving stack needs,
and the changes that register them, as a patch set against tt-metal `v0.77.0-rc1`. They are not in upstream tt-metal.
Without them the serving image cannot be rebuilt from this repository (issue #33).

| Op (Python name on `ttnn.transformer`) | Directory under `ops/` |
|---|---|
| `attn_decode_prep` | `attn_prep` |
| `gdn_decode_conv_gates` | `gdn_conv_gates` |
| `gdn_decode_norm_gate` | `gdn_norm_gate` |
| `gdn_decay` | `gdn_decay` |
| `decode_gated_delta_rule_packed` | `decode_gated_delta_rule` |

Base: tt-metal `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9` (tag `v0.77.0-rc1`).

## What is here

| Path | What |
|---|---|
| `prstack/` | Three upstream pull-request diffs (53314, 53319, 53320), our one-line FIR fix (`0004-...patch`), `ORDER`. The diffs are the exact bytes we built with. |
| `tt-metal.diff` | One patch on top of the prstack: op registration (`transformer/sources.cmake`, `transformer/CMakeLists.txt`, `transformer_nanobind.cpp` binding all five ops), the `nlp_concat_heads_decode` batch-64 changes, the SDPA fp32-draft and decode tree-scratch edits, and the two model files `attention/tp.py` and `gdn/tp.py` plus `ttnn_delta_rule_ops.py`. |
| `ops/` | The five op directories, as they sit in the tree (host code, device kernels, nanobind). |
| `k64j/` | The SDPA prefill-chain and decode kernels and the two program factories of the "K64j" tier (below). |
| `optional/lazy-ccl-links.patch` | Two small edits to `all_gather_async.cpp` and `reduce_scatter_minimal_async.cpp` that the first serving image carried. Not part of the production source set; see "Open points". |
| `apply-to-tt-metal.sh` | Applies the above to a checkout, with a `--check` mode that touches nothing. |
| `verify.sh`, `markers.txt` | Post-build checks: exact set of `QWEN_` strings in the binary, and the Python bindings. |
| `MANIFEST.txt`, `regenerate-manifest.sh` | sha256 of every file in this directory; `apply-to-tt-metal.sh` checks it first. The script rewrites it after a deliberate edit. |
| `model-sources.sha256` | The pinned bytes of five model files (`attention/tp.py`, `gdn/tp.py`, `layer.py`, `mlp.py`, `model_config.py`) that the serving image's model graft refuses to install without. `--source-sha256` checks them. |
| `LICENSE` | Apache-2.0. See "Licence". |

## Tiers

- **`ops`**: prstack + `tt-metal.diff` + `ops/`. Enough for the five ops and the decode path they serve.
- **`k64j`** (default): `ops` + `k64j/`. The binary family the two-card serving numbers were measured on: it adds the
  SDPA decode kernels with tail/share/slice modes, the runtime-extent flag (0x20) and the SDPA prefill chain. The
  `exact`, `c2` and `c2-packed` profiles need it; `RECIPES.md` says which profile needs which tier.

The K64j tier is plain source: the two factories (`sdpa_decode_program_factory.cpp`, `sdpa_program_factory.cpp`) replace
the versions that `tt-metal.diff` produces, and five qwen kernels are added next to the stock ones. Nothing else changes,
and no stock kernel is edited.

## Apply

```
git clone --branch v0.77.0-rc1 https://github.com/tenstorrent/tt-metal.git --recurse-submodules --shallow-submodules
./apply-to-tt-metal.sh --root ./tt-metal --check         # touches nothing
./apply-to-tt-metal.sh --root ./tt-metal --tier k64j \
    --source-sha256 model-sources.sha256                    # applies, then checks the pinned model sources
```

`--source-sha256` is optional. The serving image's model graft refuses to install unless `attention/tp.py`, `gdn/tp.py`,
`layer.py`, `mlp.py` and `model_config.py` carry pinned bytes; this flag runs the same check right after the patch.

The script refuses to run unless `HEAD` is the base commit and the tracked files are unmodified. It copies the op
directories with a plain `cp -a` into directories that must not exist yet, and `touch`es every file it writes.

## Build

Inside the tt-metal tree (the Docker stages in `docker/` do exactly this):

```
CMAKE_BUILD_PARALLEL_LEVEL=<jobs> ./build_metal.sh --build-tests --enable-ccache
```

For a tree that is already configured and built (an incremental rebuild after `apply-to-tt-metal.sh`):

```
cmake --build build_Release --target install   # rebuilds and installs both libraries and the Python extension
cp -f build_Release/lib/_ttnncpp.so build_Release/ttnn/_ttnncpp.so
```

Rebuild **both** libraries. The C++ operations are in `_ttnncpp.so`, but the Python bindings for the five ops are in
`_ttnn.so`. A build that stops after `_ttnncpp.so`, or `ninja ttnncpp` alone, passes the C++ link and then fails the
`ttnn.transformer.*` check, because the bindings were never compiled. `build_Release/lib/_ttnncpp.so` and
`build_Release/ttnn/_ttnncpp.so` are two copies; the serving image runs one binary at both paths, so make them the
same file (`cp -f build_Release/lib/_ttnncpp.so build_Release/ttnn/_ttnncpp.so`) before you hash or ship it.
`ttnn/ttnn/_ttnn.so` is an installed copy, not a link: a bare `ninja ttnn/_ttnn.so` leaves it stale and `import ttnn` then
loads the old extension, so use the `install` target above (or copy the new file over it).

## Verify

```
./verify.sh --root ./tt-metal --tier k64j
```

It checks that the set of `QWEN_*` strings in both `_ttnncpp.so` copies is exactly the set in `markers.txt`
(5 strings for `ops`, 6 for `k64j`), that the five names are callable on `ttnn.transformer` after `import ttnn`, and
that the two copies are the same build. An exact match, not a superset, is deliberate: a binary that replaced the
whole library with another build silently loses patches, and a missing marker is the only visible sign.

Your binary will not have the same sha256 as the maintainers' binary. Anything that pins a binary hash (the
`QWEN_FAST_RUNTIME_BINARY_SHA256` variable of the serving image, the c2-packed admission record) must take the hash of
your build; `BUILD.md` shows where.

## Runtime needs the op directories

Device kernels are compiled from source when an op first runs. A serving image therefore needs the op directories,
with their `device/kernels`, at the same place in the tree (`/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/...`)
as well as the two libraries. A tree built and kept in place, as the Docker stages do, satisfies this. Copying only the
`.so` files into another image does not: the first call fails with
`Kernel file ... doesn't exist in any of the searched paths`.

## Traps

- **Unity build.** The transformer ops are compiled as unity builds: several `.cpp` files share one translation unit. A
  file-scope `namespace cb { constexpr ... }` of circular-buffer indices in two ops collides. Each op in `ops/` uses its
  own namespace (`cbap`, `cbcg`, `cbng`, `cbd`); keep that when you edit them.
- **Stale unity TUs.** A unity TU is rebuilt only when one of its member files is newer than the object. After copying
  files over an existing build tree, `touch` them; `apply-to-tt-metal.sh` does.
- **Runtime arguments.** `emplace_runtime_args` takes an initializer list or a `std::vector<std::variant<uint32_t,
  Buffer*>>`. Pass `Buffer*`, never a raw address: a program-cache hit re-resolves the buffer, and the decode trace
  depends on it.
- **DRAM access.** Every DRAM access in a kernel must be a whole tile page. Sub-page reads and writes silently do
  nothing on this stack.
- **Compute kernels.** A binary op whose two operands come from one circular buffer (different tile indices), or span two
  separately pushed blocks, stalled the unpack-math-pack handshake on Blackhole three times. Read index 0 of two distinct
  rings.
- **Copying onto a directory nests.** `cp -a dir existing/` and `docker cp dir container:/existing` both create
  `existing/dir`. The script copies only into absent directories for this reason.
- **A hung kernel needs a card reset** (`tt-smi -r`), and `timeout` on `docker run` does not stop the container.
- **CRLF.** Files written on Windows with CRLF make the bash scripts fail with "unexpected end of file". This directory
  ships `.gitattributes` with `* -text`, so git leaves every byte alone, including the pinned patch files.
- **Replacing a whole library.** Copying another build's `_ttnncpp.so` over a build drops every source edit the copied
  build lacks. Compare `QWEN_` strings (`verify.sh`) before serving.

## Provenance

- `prstack/53314.diff`, `53319.diff`, `53320.diff` are tenstorrent/tt-metal pull requests #53314 (conv2d channel
  chunking), #53319 (slice tile window) and #53320 (qwen36 demo and model layer), saved on 2026-08-23. The PRs are
  unmerged. On 2026-10-03 the live `.diff` of #53319 and #53320 was byte-identical to ours; the live #53314 differed. The
  Dockerfile this repository used to carry downloaded the live diff, so it built a different tree than the one the
  numbers were measured on; the vendored copies are the ones to use.
- `0004-fir-batch-truncation.patch` is the batch-truncation fix for the GDN FIR convolution described in
  `patches/53320-fix-fir-batch-truncation.patch`.
- `tt-metal.diff` and `ops/` are the working tree of the maintainers' op build container minus the prstack, with the
  three model files `attention/tp.py`, `gdn/tp.py` and `ttnn_delta_rule_ops.py` taken from the first serving image
  (the model graft pins those bytes). Files in `ops/` and the diff keep their SPDX headers.
- `k64j/` kernels are the qwen kernels of the K64j graft source; the two factories were compiled into the K64j binary.

## Open points

- **`decode_gated_delta_rule` version.** Two versions of this op's kernel and factory exist. This bundle ships the
  build container's (the "state-fast" variant: `cb_gexp` in the output format, no state conversion). The first serving
  image carried an older kernel next to it. The maintainers have not yet established which pair the production binary
  ran. If you need the older kernel it is a matter of replacing `ops/decode_gated_delta_rule` and the matching
  `MANIFEST.txt` lines; results are expected to be numerically close but the kernel-cache key differs.
- **Lazy-CCL edits.** `optional/lazy-ccl-links.patch` is not part of the production source set and is not applied by
  default. If a rebuilt tree shows a pair-fabric regression, apply it with `--with-lazy-ccl`.
- **Clean-room rebuild.** The sources here were cross-checked byte for byte against the maintainers' trees; a build from
  this bundle on a fresh base has not been compared with the production binary (it cannot be: binaries are not
  reproducible across builds). Acceptance is `verify.sh`, the unit tests that run inside the image, and a smoke run.

## Licence

The five op directories carry their own SPDX headers: files marked `Thatch Cloud` are Apache-2.0, and so are the files
that begin with a Tenstorrent header (`gdn_decay`, `decode_gated_delta_rule`, and the patches against Tenstorrent
files). Everything under this directory is offered under the Apache License 2.0 (`LICENSE`), not under the MIT licence
of the rest of the repository. The patches modify Tenstorrent source files; this is the Apache-2.0 section 4(b) notice
that the files in `tt-metal.diff`, `prstack/` and `k64j/` are modified versions of tt-metal sources
(`sdpa`, `sdpa_decode`, `nlp_concat_heads_decode`, `conv2d`, `slice`, `transformer_nanobind.cpp` and others).
