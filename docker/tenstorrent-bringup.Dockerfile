# syntax=docker/dockerfile:1.7
#
# Tenstorrent Blackhole bring-up image.
#
# WHY WE BUILD THIS RATHER THAN PULL ONE
#
# The published release image
# (ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64) ships the
# ttnn Python wheel and shared libraries only. The fabric diagnostics that
# multi-card bring-up needs — `test_system_health` and
# `tools/scaleout/run_cluster_validation` — are C++ *test* targets that exist only
# in a source build. There is no published image variant carrying them: probed
# `-dev-amd64`, plain `-amd64` and `upstream-tests-*` on ghcr, none exist.
#
# WHAT THIS IS ACTUALLY NEEDED FOR (measured 23 Aug 2026)
#
# The release image runs a single p150a correctly — 1x1 mesh opens, a real ttnn
# op executes on silicon. But tt-metal's topology discovery reports a physical
# intra-mesh degree histogram of {0:N} for EVERY subset of cards (3 cards {0:3},
# any pair {0:2}, a single card {0:1}). Every card has degree zero: no
# inter-card ethernet link is discovered, despite the QSFP-DD fabric being
# cabled. Consequences:
#
#   - 3 cards is classed a CUSTOM cluster and refuses to initialise without
#     TT_MESH_GRAPH_DESC_PATH pointing at a mesh graph descriptor.
#   - Supplying one gets further but then fails topology mapping, because a 1x3
#     line needs degrees {1:2, 2:1} and a 1x2 pair needs {1:2}, and the hardware
#     presents neither.
#   - Only 1x1 works today, which is a standard shape needing no descriptor.
#
# A lead worth chasing with these binaries: discovery also warns
# "Unknown motherboard '<board>' ... falling back to bus_id as tray_id. Add
# this motherboard and its bus IDs to mobo_to_bus_ids in
# physical_system_discovery.cpp". tt-metal does not know this board, and physical
# discovery uses motherboard→bus-id maps to infer card layout. Whether that is
# the cause of the missing links or merely coincident is exactly what
# test_system_health is for.
#
# BUILD
#   docker buildx build -f docker/tenstorrent-bringup.Dockerfile \
#     --build-arg TT_METAL_REF=v0.60.0-rc1 --build-arg JOBS=16 \
#     -t localhost:5000/tt-bringup:v0.60.0-rc1 .
#
# RUN (single card — the only shape that works today)
#   docker run --rm --device /dev/tenstorrent/2 \
#     -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
#     localhost:5000/tt-bringup:v0.60.0-rc1 \
#     build/test/tt_metal/tt_fabric/test_system_health
#
# RUN (all cards, for fabric diagnosis — needs the descriptor)
#   docker run --rm --device /dev/tenstorrent \
#     -v /dev/hugepages-1G:/dev/hugepages-1G \
#     -v /opt/tenstorrent/mesh:/opt/tenstorrent/mesh:ro --cap-add SYS_NICE \
#     -e TT_MESH_GRAPH_DESC_PATH=/opt/tenstorrent/mesh/p150_x3.textproto \
#     localhost:5000/tt-bringup:v0.60.0-rc1 \
#     build/test/tt_metal/tt_fabric/test_system_health
#
# --cap-add SYS_NICE is not optional for performance: without it UMD cannot
# hwloc-bind hugepages to the NUMA node of the device and logs
# "Hugepage allocation is not on NumaNode matching TT Device ... decreased
# Device->Host perf (Issue #893)" for every card.

# v0.77 line: the Adartras Qwen3.6 bundle requires ttnn >= 0.77, and v0.60.0-rc1
# — which this defaulted to first — CANNOT open a 2- or 3-card mesh on this rig
# (`Unknown cluster type`, reproduced against the OFFICIAL v0.60.0-rc1 image, so
# it is tt-metal and not this Dockerfile). tt-metal's git tags run to
# v0.78.0-dev*; v0.60 is old, not new.
#
# NOTE: ghcr *image* tags are not tt-metal *git* tags. `v0.59.0-rc60` exists as a
# published image and does not exist as a git ref. Check with
# `git ls-remote --tags https://github.com/tenstorrent/tt-metal.git` before pinning.
ARG TT_METAL_REF=v0.77.0-rc1

# Ubuntu 22.04 deliberately: it is Tenstorrent's documented baseline for
# Blackhole. A host running a much newer Ubuntu and Python is far outside that
# baseline — getting the supported userspace regardless of host OS is the whole
# reason this work is containerised rather than installed on the host.
FROM ubuntu:22.04 AS build

ARG TT_METAL_REF
ARG JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    TT_METAL_HOME=/opt/tt-metal \
    PYTHONDONTWRITEBYTECODE=1

# python3-pip and python3-venv are NOT pulled in by install_dependencies.sh
# --mode build, and create_venv.sh below needs both. Without them the ttnn python
# install fails, `import ttnn` silently resolves to the bare source directory as a
# namespace package, and every C++-backed attribute is missing at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git git-lfs sudo curl python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pin the source. A moving ref would make two builds of "the same" image produce
# different binaries, which is precisely what the gate records must not have.
RUN git clone --depth 1 --branch "${TT_METAL_REF}" --recurse-submodules --shallow-submodules \
        https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME}

WORKDIR ${TT_METAL_HOME}

# Vendor script rather than a hand-maintained apt list: the dependency set moves
# with the source, and duplicating it here is how the image silently drifts from
# what the pinned ref actually needs.
#
# No flags. Both of the obvious ones are wrong here, for opposite reasons.
#
# NOT `--hugepages`: it runs configure_hugepages(), which installs tt-system-tools
# and calls `systemctl enable --now`. There is no systemd in a docker build, so it
# exits 1 and fails the layer. That is also right on the merits — hugepages, udev
# rules and the systemd units are HOST concerns. Allocate the 1GB pool and enable
# tenstorrent-hugepages.service on the host before running any of these images.
#
# NOT `--docker` either, counter-intuitive as that reads. The flag deliberately
# OMITS the build toolchain, because Tenstorrent's own container builds inherit it
# from a base tool image:
#
#     # Add cmake only if not in Docker
#     if [ "$docker" -ne 1 ]; then PACKAGES+=("cmake"); fi
#     ...
#     else echo "[INFO] Skipping CMake install in Docker (cmake provided via tool image)"
#
# We build FROM ubuntu:22.04, so nothing provides it and build_metal.sh dies with
# `cmake: command not found`. `--docker` also skips install_sfpi and
# install_mpi_ulfm. The bare invocation is the documented way to get build deps:
# cmake 4.0.2 from Kitware, ninja, clang-20, SFPI and OpenMPI-ULFM.
#
# The flag set MOVES between versions — this is the third correction. At v0.60,
# hugepages were tied to a default `baremetal` mode that had to be escaped with
# `--mode build`; at v0.77 `--mode` does not exist and hugepages are opt-in. If you
# change TT_METAL_REF, read this script's own `--help` and package lists for that
# ref before assuming any flag still means what it did.
RUN ./install_dependencies.sh && rm -rf /var/lib/apt/lists/*

# --build-tests is the entire point: it produces
# build/test/tt_metal/tt_fabric/test_system_health and the scaleout validation
# tooling that the release image lacks.
#
# Parallelism is CMAKE_BUILD_PARALLEL_LEVEL, NOT a script flag. build_metal.sh
# has no --jobs/-j option and rejects one outright ("invalid option -- 'j'");
# it shells out to `cmake --build` without a job count and lets CMake read the
# environment. -c enables ccache, which is what makes the cache mount above
# worth having on a rebuild.
RUN --mount=type=cache,target=/root/.ccache \
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    ./build_metal.sh --build-tests --enable-ccache

# A venv with ttnn installed from the tree we just built.
#
# NOT tt-metal's create_venv.sh, deliberately. That script builds a *developer*
# environment: it force-downgrades pip to 21.2.4 and installs
# requirements-dev.txt, which is black, mypy, pre-commit, twine, clang-format —
# and, at line 8, `git+https://github.com/tenstorrent/tt-smi.git@v3.0.20`, whose
# `pyluwen` pin does not resolve on PyPI:
#
#     ERROR: Could not find a version that satisfies the requirement
#            pyluwen (unavailable) (from tt-smi)
#
# That cost 535 seconds of installing tooling this image has no use for, in order
# to fail on one of them. tt-smi belongs on the HOST (it drives board resets and
# host-side monitoring); nothing in the container calls it.
#
# ttnn's real runtime surface is four packages, declared in pyproject.toml:
# numpy<2, loguru, networkx, graphviz. `pip install -e .` resolves exactly those
# against the already-built tree.
#
# The step this replaces was `pip install ... || pip install ... || true`, where
# pip was not installed at all — so it could never have worked, and the `|| true`
# made the failure silent. The image shipped with working C++ binaries and a ttnn
# that imported as an empty namespace package: test_system_health ran fine while
# every ttnn call raised AttributeError. No `||` fallbacks here. If the python env
# cannot be built, the image must fail to build.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}" \
 && "${VIRTUAL_ENV}/bin/python3" -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && "${VIRTUAL_ENV}/bin/python3" -m pip install --no-cache-dir -e .

# The venv first on PATH, so `python3` is the one with ttnn in it. Keeping the
# python bindings and the C++ test binaries on ONE tt-metal commit is the point
# of building both here: mixing our image's binaries with the release image's
# wheel already produced an API skew (GetNumAvailableDevices exists in one and
# not the other), which is exactly the kind of thing that invalidates a gate record.
ENV PATH="/opt/venv/bin:${TT_METAL_HOME}/build/test/tt_metal/tt_fabric:${PATH}" \
    ARCH_NAME=blackhole \
    TT_METAL_HOME=/opt/tt-metal

# Fail the build rather than ship another image whose python is quietly broken.
RUN python3 -c "import ttnn; assert ttnn.__file__, 'ttnn imported as a namespace package'; \
    print('ttnn OK:', ttnn.__file__)"

WORKDIR ${TT_METAL_HOME}
CMD ["/bin/bash"]
