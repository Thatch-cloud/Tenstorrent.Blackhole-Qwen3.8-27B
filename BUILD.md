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

Not runnable from this repository alone yet (see "What is not here yet"). The Dockerfiles are portable and are
published so that the shape of the build is visible and the missing inputs are exactly the ones listed.

- Stage 7, `docker/two-card/qwen-fast-serving.Dockerfile`: the fast-path runtime tree on top of the **k64j-tier** image
  (`tt-vllm:qwen38-k64j`, the default `BASE`). `BASE` and `REGISTRY` are build arguments; no registry or digest is
  hard-coded. The copy no longer replaces your libraries with a prebuilt one. Tag the result
  `localhost:5000/qwen-fast-serving:p8` (the default `BASE` of stage 8). Its build context is this repository's
  `scripts/ci/` plus a `bundle/` directory (`experiment-scripts/`, `experiment-optimisation/`, `speculative-decoding/`,
  `serving-bundle.json`) that this repository does not publish.
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

## What is not here yet

Honest list, so that nobody loses a day on it:

1. **The serving runtime tree for stage 7.** The earlier build read it from CI artifacts that no longer exist
   (`bundle/experiment-scripts`, `bundle/experiment-optimisation`, `bundle/speculative-decoding`,
   `bundle/serving-bundle.json`, `native-cache-manifest.json`). A flattened, hash-manifested replacement, cut from the
   qualified image, is planned and is not in this branch. Until it is, stage 7 cannot be built from this repository
   alone, and neither can stage 8 (which builds on it).
2. **`serving_native_install.py`** is removed from the stage 7 copy: it refuses any binary but the maintainers'
   own and re-applies a patch the source build already contains. Its other output, `/opt/qwen-serving/native-install.json`, is
   not written; whether the fast path's start-up check needs it has not been tested on a rebuilt image.
3. **The frozen-evidence tree** that the fast path's startup check reads (`<runtime>/frozen-evidence/`) exists only in
   the maintainers' images. The `general*` profiles do not need it; `exact`, `c2` and `c2-packed` do.
4. **c2-packed evidence.** The packed-admission check pins the maintainers' binary and an evidence record. On your own
   binary it needs fresh evidence runs on your cards. The harnesses are not published here. Until then c2-packed serves
   as `c2`.
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
