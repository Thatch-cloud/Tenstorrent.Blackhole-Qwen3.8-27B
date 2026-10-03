# Building the two-card images from source

This page builds, from this repository and public upstreams only, the image chain that the [two-card recipes](RECIPES.md)
run on. It replaces the internal registry and the prebuilt binaries the earlier Dockerfiles assumed (issue #33).

> **Status: stages 1-6 are reproducible from this repository; stages 7-8 (the serving image) are not yet, so none of
> the recipes in [RECIPES.md](RECIPES.md) can be run from a clean clone today.** What is missing is listed exactly in
> "What is not here yet".

**What is reproducible today, and what is not.** Stages 1-6 below (tt-metal, the upstream PR stack, the vLLM plugin and
the five custom ops, in two tiers) build from public sources plus [`tt-metal-custom-ops/`](tt-metal-custom-ops/), and
each ends in a check. Stages 7-8 (the serving runtime tree and the C2 serving layer) have portable Dockerfiles, but
their build context still comes from the maintainers' CI and part of it is missing from this repository. The gap is
listed exactly in "What is not here yet". Nothing in this page has been built end to end by the maintainers from a clean
checkout of this branch; stages 1-6 were checked as far as `apply-to-tt-metal.sh --check` against upstream tt-metal
(every patch applies cleanly in order), not by a Docker build.

## 1. Host

- Linux host with Docker (BuildKit, the default `docker build`; not a buildx container driver, which would try to pull
  the local base images). Containers are Ubuntu 22.04; the host kernel driver and firmware are Tenstorrent's. Firmware
  19.12.0 is what the recipes were developed on; the kernel-module version is not recorded here.
- Two p150a cards, two free 1 GiB hugepages (`/dev/hugepages-1G`), `SYS_NICE` for the container.
- Disk: about 15 GB per base layer, plus the model (roughly 54 GB: 27 billion parameters in bf16) and the weight and kernel caches.
  Exact totals have not been measured.
- Stage 1 is a multi-hour compile of tt-metal; later stages are incremental (ccache). Build times have not been
  recorded either.
- Links: build stage 1, then count the card-to-card links with the health check it contains (list your boards with
  `ls -l /dev/tenstorrent/by-id/` and use two of them):

  ```bash
  docker run --rm --device "$(readlink -f /dev/tenstorrent/by-id/<board-0>)" \
    --device "$(readlink -f /dev/tenstorrent/by-id/<board-1>)" \
    -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
    localhost:5000/tt-bringup:v0.77.0-rc1 build/test/tt_metal/tt_fabric/test_system_health
  ```

  Four links: every recipe. Two links: `general-2link` only.

## 2. Check the clone

```bash
git clone --branch recipes/two-card https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B
cd Tenstorrent.Blackhole-Qwen3.8-27B
python3 scripts/ci/sanitise_scan.py --dir tt-metal-custom-ops --block-only   # must end "0 BLOCK" (shapes only, see its --help)
(cd tt-metal-custom-ops && grep -v '^#' MANIFEST.txt | sha256sum -c --quiet)  # the bundle is intact
```

## 3. Build the chain

```bash
REGISTRY=localhost:5000 JOBS=<cores> scripts/two-card-build.sh --tier k64j     # stages 1-4 and 6
REGISTRY=localhost:5000 JOBS=<cores> scripts/two-card-build.sh --from ops --tier ops   # optional: stage 5 as well
```

`REGISTRY` is only a name prefix. Nothing has to listen on it: `FROM` resolves a local image first. The script prints,
for each custom-ops image, `RUNTIME_BINARY_SHA256=<hash of your _ttnncpp.so>`; keep it for stage 8.

| Stage | Dockerfile | Output tag | What it does |
|---|---|---|---|
| 1 | `docker/tenstorrent-bringup.Dockerfile` | `tt-bringup:v0.77.0-rc1` | clones tt-metal at the tag (`9f9cd4fd`), builds it with its test targets |
| 2 | `docker/two-card/10-tt-prstack.Dockerfile` | `tt-bringup:v0.77.0-rc1-prstack` | applies the **vendored** upstream PR diffs 53314, 53319, 53320 and our FIR fix, rebuilds |
| 3 | `docker/tenstorrent-serving.Dockerfile` | `tt-serving:v0.77.0-rc1-prstack` | serving Python requirements |
| 4 | `docker/tenstorrent-vllm-plugin.Dockerfile` | `tt-vllm:v0.77.0-rc1-prstack-plugin` | vllm-tt-plugin at `bf77cd6`, vLLM |
| 5 | `docker/two-card/40-qwen-ops.Dockerfile` | `tt-vllm:qwen38-k` | tier `ops`: the five ops, rebuild of both libraries, `verify.sh` |
| 6 | `docker/two-card/50-qwen-k64j.Dockerfile` | `tt-vllm:qwen38-k64j` | tier `k64j`: `ops` plus the SDPA kernels and factories, rebuild, `verify.sh` |

Stages 5 and 6 are siblings: both start from stage 4. **K64j is the default.** It is the binary family the recipes'
measurements were made on. The `ops` tier is a stepping stone and a fallback for the `general*` profiles with
`QWEN_FAST_SDPA_PF=0`, a combination the maintainers have not measured.

Stage 2 deliberately does not download pull-request diffs. The pull requests are unmerged and move: the live diff of
#53314 already differs from the one we built with, so a fresh build from the old Dockerfile gets a different tree.

## 4. Verify

Each custom-ops stage ends with `tt-metal-custom-ops/verify.sh`, whose output is kept in the image at
`/opt/qwen-custom-ops.verify.txt`:

- both `_ttnncpp.so` copies carry exactly the expected set of `QWEN_` strings (5 for `ops`, 6 for `k64j`), no more, no
  fewer (a whole-library replacement silently drops source edits; this is the only visible sign);
- `import ttnn` works and `attn_decode_prep`, `gdn_decode_norm_gate`, `gdn_decode_conv_gates`,
  `decode_gated_delta_rule_packed` and `gdn_decay` are callable on `ttnn.transformer`;
- the two library copies are the same build.

The five model files the serving layer overlays are also checked against `docker/qwen-c2-graft/source.sha256` during
stages 5 and 6, the same pins the C2 image enforces.

To run the checks by hand on any tt-metal checkout: `tt-metal-custom-ops/apply-to-tt-metal.sh --check`, then build,
then `verify.sh`. The bundle's README lists the traps (unity-build namespaces, stale translation units, the
`_ttnn.so` that `ninja ttnncpp` does not rebuild).

## 5. The model

```bash
huggingface-cli download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --cache-dir /path/to/hf-hub-cache
```

Mount that directory at `/models` when you run (RECIPES.md). The profiles refer to the snapshot by that revision.

## 6. Serving image (stages 7 and 8)

Both Dockerfiles are portable. Stage 7's runtime tree is published in `bundle/` (see `bundle/README.md`); stage 8 needs
the DFlash2 fixtures for the fast tier and a stage 6 binary hash. Neither stage has been rebuilt end to end from this
repository by the maintainers: the P8 image was built from the same recipe earlier, from CI artifacts. Read "What is not
here yet" and "Pins that `exact`, `c2` and `c2-packed` check" before relying on the speculative-decoding profiles.

- Stage 7, `docker/two-card/qwen-fast-serving.Dockerfile`: the fast-path runtime tree on top of the **k64j-tier** image
  (`tt-vllm:qwen38-k64j`, the default `BASE`). `BASE` and `REGISTRY` are build arguments; no registry or digest is
  hard-coded. The copy no longer replaces your libraries with a prebuilt one. Tag the result
  `localhost:5000/qwen-fast-serving:p8` (the default `BASE` of stage 8). The build context is the repository root: the
  `COPY` lines read this repository's `scripts/ci/` (the overlay, which wins over same-named bundle files) and `bundle/`
  (`experiment-scripts/`, `experiment-optimisation/`, `speculative-decoding/`, `serving-bundle.json`, all published):

  ```bash
  docker build -f docker/two-card/qwen-fast-serving.Dockerfile -t localhost:5000/qwen-fast-serving:p8 .
  ```

  `native-cache-manifest.json` and a prebuilt `_ttnncpp.so` are **not** inputs of this stage (this Dockerfile copies
  neither; the one consumer of the manifest, `serving_native_install.py`, is removed from this stage, item 2 below). Two image-build steps run checks: the `unittest` step (14 test modules, CPU only) and
  `serving_image_preflight.py`, which pins the sha256 of three files under `/opt/tt-metal`
  (`p150_x2_mesh_graph_descriptor.textproto`, `models/common/modules/tt_ccl.py`, `models/common/sampling/tt_sampling.py`).
  No patch in the build chain touches them, so a v0.77.0-rc1 tree should match; `sha256sum` them in your k64j image
  before building (expected `e5d25de8...`, `ec24c3ab...`, `d3d32ea9...`; full values in
  `scripts/ci/dram-mlp-down-hardware.json`, lines 273-275). The preflight is fatal and has no override.
  The step also clones `vllm-tt-plugin` at `bf77cd63` from GitHub, so it needs network access.
- Stage 8, `docker/two-card/qwen-c2-serving.Dockerfile`: the model graft, the overlay, the prefix stage and the
  serving environment. Build arguments:
  - `BASE`: the stage-7 image (default `${REGISTRY}/qwen-fast-serving:p8`);
  - `RUNTIME_BINARY_SHA256`: **required**, the hash your stage 6 build printed. The image refuses to build when the
    installed `_ttnncpp.so` differs, and sets `QWEN_FAST_RUNTIME_BINARY_SHA256` to it;
  - `SOURCE_REVISION`: **required**, the commit the context is staged from;
  - `GRAFT_INSTALL`: `0` by default, because the base is a source build; `1` installs a prebuilt set from `prebuilt-graft/`
    (none is published);
  - `KERNEL_CACHE`: defaults to `/experiment-cache/kernels-qwen`, a link to the writable `/models/.qwen-c2`.

  Stage the context and build (after stage 7 exists):

  ```bash
  python3 scripts/ci/c2_overlay.py stage --repo . --out ctx8        # needs the full history of this clone
  cp docker/two-card/qwen-c2-serving.Dockerfile ctx8/Dockerfile
  mkdir -p ctx8/prebuilt-graft                                         # empty: GRAFT_INSTALL=0 installs nothing from it
  echo "two-card build" > ctx8/build-stamp
  scripts/two-card-fixtures.sh ctx8                                  # fast tier only; see below. Without it: mkdir ctx8/fixture ctx8/draft-config
  docker build -t localhost:5000/qwen-c2-serving:local \
    --build-arg RUNTIME_BINARY_SHA256=<hash stage 6 printed> --build-arg SOURCE_REVISION="$(cat ctx8/source-revision)" ctx8
  ```

  Run `scripts/ci/c2_image_provenance.py` only against the unmodified internal Dockerfile: it parses that file's
  `ARG BASE` line and binary pins. `localhost:5000/qwen-c2-serving:local` is the image name RECIPES.md runs.

### Fast tier (the `exact`, `c2` and `c2-packed` profiles)

These need the DFlash2 draft fixtures in the image. `scripts/two-card-fixtures.sh OUT_DIR` fetches them from the public
Hugging Face repository `incoai/Qwen3.8-27B-DFlash2` at revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4` (the
repository's head has moved since; keep the pinned revision) and writes `OUT_DIR/fixture/` and
`OUT_DIR/draft-config/` in the layout stage 8 copies. `FIXTURE_CACHE` names the directory it caches downloads in. It
needs torch and transformers and about 10 GB; it has not been run end to end from a clean checkout by the maintainers.

## Pins that `exact`, `c2` and `c2-packed` check

Found by reading the code against the k64j source set, not by running a build. Report what you see.

**`general`, `general-2link` and `general-prefix` avoid every pin below.** They do not set `qwen_fast_t16`, so the
combined-runtime attach never runs, and none of these modules executes for them: `runtime_binary_override`,
`dflash_t16_native_scope`, `sdpa_tree_scratch`, the frozen evidence, the evidence gates, `packed_any_admission`. They need
none of the evidence directories.

**`exact` and `c2` are refused on a source-built k64j tree, and no environment variable or build argument unblocks them.**
The k64j build patches the SDPA prefill factory, so the on-disk file is not the one the pins name:

| What | Pinned value | Your k64j tree |
|---|---|---|
| `sdpa/device/sdpa_program_factory.cpp` (combined factory) | `fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783` | `bfab8558d889ad215f0e9ee732c75a4142be1a1e7f37810f4ca8c5e3c73bdf65` (`tt-metal-custom-ops/MANIFEST.txt`) |
| `sdpa_decode/device/sdpa_decode_program_factory.cpp` | `05708e6d...` (original) or `3e0a69af...` (patched; `scripts/ci/probe_sdpa_decode_sources.py`) | `bb4dc6a759d40054d31089792f617a1d66f5837da2065fa12f61438d09a55080` |
| the `_ttnncpp.so` binary | `4b7299c1c9233b25aad310bc9a9d751a0631c6af0602cb1a151f934b4bfa07ea` (maintainers' build) | whatever stage 6 printed |

- **Binary override (`runtime_binary_override.py`).** Stage 8 sets `QWEN_FAST_RUNTIME_BINARY_SHA256` to the hash you pass as
  `RUNTIME_BINARY_SHA256`. With it set, `install` admits your binary only if the on-disk `sdpa_program_factory.cpp`
  still hashes to `fd8c0676...`; yours hashes to `bfab8558...`, so it refuses. Without the variable the binary pin
  (`4b7299c1...`) refuses instead. `build_k64j.sh` derives `bfab8558` from `fd8c0676` plus `apply_factory_pf.py`.
- **T16 evidence.** `dflash-t16-31.json` and `dflash-t16-2048.json` (in `bundle/.../dflash-t16-native-evidence/`) record
  15 native SDPA sources with the factory at `fd8c0676...`; `dflash_t16_native_scope.admit` refuses on any mismatch.
  Fixing the override alone does not help.
- **Evidence gates** pin further native sources: `mlp_down_grid_gate` pins `qwen36/tt/tp_common.py`;
  `gdn_direct_window` and `shared_qk_norm_scatter` pin their own lists. Whether those match your tree is **unknown**.
  `dflash_combined_sim_runtime.native_hashes('/opt/tt-metal')` computes the live hashes to compare with the reports.
- **`sdpa_tree_scratch.audit`** pins the *decode* factory and two decode kernels (not the prefill factory), so the k64j
  decode factory (`bb4dc6a7...`) refuses it. At serve time it runs only in `c2-packed`'s `check_runtime`; `exact` and `c2`
  use only `history_limit` and `validate_request` from `dspark_8k_admission`.
- **`c2-packed`** also pins the K64j binary, four kernels and `packed_any_evidence.json`.

Why the maintainers' images pass: they keep the audited baseline on disk (prefill factory `fd8c0676`, decode factory
`3e0a69af`) under a separately built binary. The factory `.cpp` files are compiled into the binary, so the on-disk copies
do not change what runs; the pins then vouch for the sources the binary was built from only by convention.

**Two ways forward. Neither is done here, and neither is a measurement.**

1. *No code change, a bypass.* After stage 6, write the combined factory over the on-disk file, the way the maintainers'
   images are laid out: take the **original** factory (the file at the tt-metal base, with its sha256 equal to
   `ORIGINAL_FACTORY = a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc`; stop if it is not), and write
   `dflash_combined_sim_runtime.factory_bytes(original)` to
   `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`. The pins for the factory then pass. It
   makes the audits describe sources your binary was not built from, so treat a pass as "the rest of the checks
   pass", not as qualification. The other 14 T16 pins and the gate pins must still match, and the decode pins for
   `c2-packed` need the same treatment for the decode factory. Do this only in a throwaway image.
2. *Code change, by the maintainers.* Accept `bfab8558` (and `bb4dc6a7` for the tree-scratch audit; both already in
   `MANIFEST.txt`) as descendants of the pinned factories. That needs a judgement that the prefill-chain and k64j
   changes do not touch the qualified draft path, plus fresh evidence runs. It has not been made.

## What is not here yet

Honest list, so that nobody loses a day on it:

1. **Stage 7 has not been built from this repository.** `bundle/` is published (flattened, hash-manifested, cut from the
   qualified image, with operator-specific material withheld: `bundle/README.md` lists what and why). The Dockerfile
   was checked by reading it: every `COPY` source exists, and every module-level import of the image's unit-test step
   resolves inside the image tree. It was not built. One build failure was found and fixed by reading: the
   `test_serving_runtime` step imports `serving_prefill_admission`, `serving_request_quarantine` and
   `packed_any_admission` (with `packed_any_evidence.json`) inside test bodies, and they are in neither the bundle nor
   the earlier copy list, so the Dockerfile now copies them. Expect to find the next one the same way. Report it.
2. **`serving_native_install.py`** is removed from the stage 7 copy: it refuses any binary but the maintainers'
   own and re-applies a patch the source build already contains. Its other output, `/opt/qwen-serving/native-install.json`, is
   not written; whether the fast path's start-up check needs it has not been tested on a rebuilt image.
3. **The frozen-evidence tree** is now in `bundle/experiment-scripts/ci/frozen-evidence/` (byte-exact, only the files
   the reports pin). The `general*` profiles do not read it; `exact`, `c2` and `c2-packed` do. Having the files does not
   mean the checks pass on your build: see the next section.
4. **c2-packed evidence.** The packed-admission check pins the maintainers' binary, four kernels and an evidence record.
   On your own binary it needs fresh evidence runs on your cards, and the harnesses are not published here. **When the
   check refuses, the engine does not start** (by reading the code: the refusal, `AdmissionRefused`, is not caught at
   attach). Earlier versions of these docs said it falls back to serving as `c2`; that is wrong.
5. **Hardware validation of a rebuilt image.** The maintainers can only re-check the two-link path (`general-2link`) on
   their current cabling; the four-link profiles, rebuilt from source, are unvalidated.
6. **`scripts/ci/build-c2-serving-image.sh`** and several CI helpers still carry the maintainers' paths and registry. They
   are the internal build driver; use `scripts/two-card-build.sh`, `scripts/two-card-fixtures.sh` and the Dockerfiles in
   `docker/two-card/` instead.

## Troubleshooting

- **Link refusal at start (`STRICT_INIT`):** the pair trained fewer links than the descriptor declares. Use
  `general-2link`, or fix the cabling.
- **`[QWEN-SDPA-PF]` refusal:** the image sets `QWEN_FAST_SDPA_PF=1`; the `ops` tier lacks the prefill-chain factory.
  Run with `-e QWEN_FAST_SDPA_PF=0` (unmeasured) or use the `k64j` tier.
- **`ttnn.transformer` lacks an op after `ninja`:** `ttnn/ttnn/_ttnn.so` is an installed copy that a bare `ninja` does not
  refresh. Run `cmake --build build_Release --target install` (see `tt-metal-custom-ops/README.md`).
- **`Kernel file ... doesn't exist in any of the searched paths`:** the op directories are missing from the tree the
  container runs. The libraries alone are not enough.
- **Hugepage NUMA warning:** add `--cap-add SYS_NICE`.
- **Stale objects after copying sources over a build tree:** `touch` them; `apply-to-tt-metal.sh` does.
- **Apply refuses (`HEAD is ...`):** the checkout is not tt-metal `v0.77.0-rc1`, or has local changes.
