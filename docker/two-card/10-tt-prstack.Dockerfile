# syntax=docker/dockerfile:1.7
#
# Stage 2 of the two-card chain: tt-bringup + the upstream PR stack, from VENDORED diffs.
#
# docker/tenstorrent-bringup-prstack.Dockerfile downloads the live diff of each pull request. Pull requests move:
# the live diff of #53314 already differs from the one the two-card numbers were measured with. This stage applies
# the exact bytes kept in tt-metal-custom-ops/prstack/ (53314, 53319, 53320, then our one-line FIR fix), in the order
# of prstack/ORDER, and needs no network for it.
#
# BUILD CONTEXT: a directory holding tt-metal-custom-ops/ (scripts/two-card-build.sh creates it):
#   docker build -f docker/two-card/10-tt-prstack.Dockerfile --build-arg BASE=localhost:5000/tt-bringup:v0.77.0-rc1 \
#       -t localhost:5000/tt-bringup:v0.77.0-rc1-prstack <context>
ARG REGISTRY=localhost:5000
ARG BASE=${REGISTRY}/tt-bringup:v0.77.0-rc1
FROM ${BASE}

ARG JOBS=48
ENV TT_METAL_HOME=/opt/tt-metal
WORKDIR /opt/tt-metal

COPY tt-metal-custom-ops/ /opt/qwen-custom-ops/

# apply-to-tt-metal.sh refuses unless HEAD is 9f9cd4fd (v0.77.0-rc1) and the tracked files are untouched.
RUN /opt/qwen-custom-ops/apply-to-tt-metal.sh --root /opt/tt-metal --tier prstack \
 && mkdir -p /opt/tt-prstack \
 && cp /opt/qwen-custom-ops/prstack/ORDER /opt/tt-prstack/APPLIED.txt \
 && cp /opt/qwen-custom-ops/prstack/*.diff /opt/qwen-custom-ops/prstack/*.patch /opt/tt-prstack/

# Incremental rebuild, same flags as the base image: only the conv2d and slice sources changed.
RUN --mount=type=cache,target=/root/.ccache \
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    ./build_metal.sh --build-tests --enable-ccache

RUN python3 -c "import ttnn; print('ttnn OK:', ttnn.__file__)" \
 && test -x build/test/tt_metal/tt_fabric/test_system_health \
 && grep -q '(B, k + T, D)' models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py \
 && echo 'PR stack applied, FIR batch fix present' && cat /opt/tt-prstack/APPLIED.txt

WORKDIR /opt/tt-metal
CMD ["/bin/bash"]
