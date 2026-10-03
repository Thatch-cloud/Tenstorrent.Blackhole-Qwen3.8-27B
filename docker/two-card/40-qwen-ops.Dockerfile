# syntax=docker/dockerfile:1.7
#
# Stage 5 of the two-card chain: the five custom ttnn ops, compiled in. Tier "ops" of tt-metal-custom-ops/.
#
# FROM the stage-4 image (tt-vllm:...-prstack-plugin). Applies tt-metal.diff and the op directories over the PR stack,
# rebuilds BOTH _ttnncpp.so and _ttnn.so (the Python bindings of the new ops live in _ttnn.so), and runs verify.sh:
# exact QWEN_ marker set, five bindings callable, same binary at both paths.
#
# The op directories stay in the tree on purpose: device kernels are compiled from source at first use.
#
# BUILD CONTEXT: a directory holding tt-metal-custom-ops/.
#   docker build -f docker/two-card/40-qwen-ops.Dockerfile -t localhost:5000/tt-vllm:qwen38-k <context>
# This is the ops tier. docker/two-card/qwen-fast-serving.Dockerfile builds FROM the k64j tier (stage 6) by default.
ARG REGISTRY=localhost:5000
ARG BASE=${REGISTRY}/tt-vllm:v0.77.0-rc1-prstack-plugin
FROM ${BASE}

ARG JOBS=48
ENV TT_METAL_HOME=/opt/tt-metal
WORKDIR /opt/tt-metal

COPY tt-metal-custom-ops/ /opt/qwen-custom-ops/

RUN /opt/qwen-custom-ops/apply-to-tt-metal.sh --root /opt/tt-metal --tier ops \
      --prstack-applied /opt/tt-prstack/APPLIED.txt \
      --source-sha256 /opt/qwen-custom-ops/model-sources.sha256 \
      --record /opt/qwen-custom-ops.applied.json

# Incremental: the new op sources, the registration and the SDPA factories are the only changes since stage 2.
RUN --mount=type=cache,target=/root/.ccache \
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    ./build_metal.sh --build-tests --enable-ccache \
 && ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so \
 && cp -f build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so

# A pipe to tee would hide a failing verify.sh behind tee's exit status (sh has no pipefail), so keep the status.
RUN VLLM_PLUGINS='' /opt/qwen-custom-ops/verify.sh --root /opt/tt-metal --tier ops \
 > /opt/qwen-custom-ops.verify.txt 2>&1; rc=$?; cat /opt/qwen-custom-ops.verify.txt; exit $rc

WORKDIR /opt/tt-metal
CMD ["/bin/bash"]
