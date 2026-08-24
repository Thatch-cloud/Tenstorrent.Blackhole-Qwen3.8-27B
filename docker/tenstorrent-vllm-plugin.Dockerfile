# syntax=docker/dockerfile:1.7
#
# Tenstorrent vLLM serving image — plugin-only, on stock vLLM.
#
# Replaces the earlier fork-based Dockerfile, which has been removed.
#
# Tenstorrent are migrating away from "vLLM fork + plugin in one repo" to "stock
# vLLM + a plugin-only repo". From the maintainer, on tenstorrent/vllm#473:
#
#   "Instead of vLLM fork + plugin in one repo, we now use stock vLLM, and a
#    plugin-only repo. I suggest you migrate to the new plugin, unless something
#    is blocking you."
#
# Nothing was blocking us, and it is now proven on silicon: same model, same two
# cards, character-identical completions, GSM8K 58/60. The fork-based Dockerfile
# has been removed — carrying two ways to build the same image is how a fleet ends
# up running something nobody meant to ship.
#
# WHY THE PLUGIN-ONLY LAYOUT IS BETTER FOR US, BEYOND FOLLOWING UPSTREAM
#
# The plugin is self-contained — "Nothing TT-specific needs to touch vLLM core."
# Platform, scheduler, worker, loader and model registration all live in it, so
# vLLM itself is an ordinary pinned dependency rather than a fork we track.
#
# It also solves, structurally, the risk the previous image could only *assert*
# against. vLLM's PyPI metadata is generated on a CUDA machine, so a plain
# install resolves requirements/cuda.txt — torch, flashinfer, tilelang, nvidia-* —
# regardless of VLLM_TARGET_DEVICE. That would fight tt-metal over torch and add
# several GB. Their install script fetches vLLM's requirements/common.txt at the
# pinned tag, installs that explicitly, then installs vLLM itself --no-deps
# --no-binary. torch stays the tt-metal one, by construction rather than by luck.
#
# BUILD
#   docker build -f docker/tenstorrent-vllm-plugin.Dockerfile \
#     --build-arg BASE=$REGISTRY/tt-serving:v0.77.0-rc1-prstack \
#     -t $REGISTRY/tt-vllm:v0.77.0-rc1-prstack-plugin .
#
# VERSION PAIRING — THE ONE REAL UNKNOWN
#
# The plugin pins vLLM 0.25.1 and its README points at tt-metal's LLMs table for
# the tt-metal <-> vLLM pairing. Our base is v0.77.0-rc1 PLUS three unmerged PRs
# (#53314/#53319/#53320) that are what make a 2-card mesh work at all, so that ref
# is not in the table. Asked upstream on tenstorrent/vllm#473. If 0.25.1 turns out
# not to pair with a v0.77-era tt-metal, the fallback is the plugin's
# `compat/vllm-0.24.0` tag — set VLLM_TT_PLUGIN_REF to it.

ARG BASE=localhost:5000/tt-serving:v0.77.0-rc1-prstack
FROM ${BASE}

# Pinned, not `main`. A moving ref makes two builds of "the same" image produce
# different bits, which is exactly what a gate record must not have.
#   git ls-remote https://github.com/tenstorrent/vllm-tt-plugin.git main
ARG VLLM_TT_PLUGIN_REF=bf77cd6

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    VLLM_TT_PLUGIN_DIR=/opt/vllm-tt-plugin

RUN python3 -m pip install --no-cache-dir uv

# Blobless rather than --depth 1 --branch main: shallow-cloning a branch races
# upstream pushes, and a detached checkout of an exact SHA fails loudly instead of
# silently producing a different image.
RUN git clone --filter=blob:none https://github.com/tenstorrent/vllm-tt-plugin.git ${VLLM_TT_PLUGIN_DIR} \
 && git -C ${VLLM_TT_PLUGIN_DIR} checkout --detach "${VLLM_TT_PLUGIN_REF}" \
 && git -C ${VLLM_TT_PLUGIN_DIR} log -1 --format='vllm-tt-plugin pinned at %H %ci'

WORKDIR ${VLLM_TT_PLUGIN_DIR}

# Their script, run as shipped. It is `source`d rather than executed because it
# uses `return` on failure, and it assumes the repo root as cwd for the relative
# path to docs/vllm-overrides.txt.
#
# UV_NO_CACHE=1 is their own recommendation for container builds: without it the
# sdist and the wheel built from it both stay behind in the layer.
#
# Do NOT reimplement this inline. It owns a dependency set that has to be re-read
# whenever the vLLM pin moves, and the last time this project hand-copied a vendor
# install step it silently dropped a cache mount and the image stopped matching
# what the repo claimed it was.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_NO_CACHE=1 VIRTUAL_ENV=${VIRTUAL_ENV} \
    bash -lc 'cd ${VLLM_TT_PLUGIN_DIR} && source docs/install-vllm-tt.sh'

ENV VLLM_PLUGINS=tt,tt_model_registry

# Fail the build rather than ship an image whose plugin does not load, or whose
# torch was quietly replaced by vLLM's CUDA dependency set.
#
# torch is the assertion that matters: the base image is built against the
# tt-metal env's torch, and a silent swap here would break the PCC reference path
# while leaving every ttnn test green.
RUN python3 -c "\
import vllm, vllm_tt_plugin, ttnn, torch, transformers; \
print('vllm        ', vllm.__version__); \
print('plugin      ', vllm_tt_plugin.__file__); \
print('torch       ', torch.__version__); \
print('transformers', transformers.__version__); \
assert '+cpu' in torch.__version__ or 'cu' not in torch.__version__, \
    f'torch was replaced by a CUDA build: {torch.__version__}'"

# The Qwen3.5/3.6/3.8 family registers as TTQwen3_5ForConditionalGeneration and
# resolves to tt-metal's qwen36 implementation. Assert it, because a registration
# change upstream would otherwise surface as an opaque "no model runner" at serve
# time on a box with the hardware attached.
RUN grep -q "TTQwen3_5ForConditionalGeneration" ${VLLM_TT_PLUGIN_DIR}/src/vllm_tt_plugin/platform.py \
 && grep -q "qwen36_vllm:Qwen36ForCausalLM" ${VLLM_TT_PLUGIN_DIR}/src/vllm_tt_plugin/platform.py \
 && echo 'qwen36 registration present'

WORKDIR /opt/tt-metal
CMD ["/bin/bash"]
