# syntax=docker/dockerfile:1.7
#
# Tenstorrent model/serving image — the bring-up image plus what it takes to run
# a model.
#
# Layered rather than merged on purpose. `tenstorrent-bringup.Dockerfile` answers
# "is this hardware right?" and must stay lean enough to run on a host that has
# other work to do. This image answers "does a model run?", and drags in
# torch, transformers and the rest of tt-metal's dev requirements — several GB
# that have no business in a diagnostic tool.
#
#   tt-bringup       tt-metal + ttnn + test_system_health        ~15 GB
#   tt-serving       + model/test deps (this file)               + torch et al
#
# BUILD (base must exist locally or in the registry first)
#   docker build -f docker/tenstorrent-serving.Dockerfile \
#     --build-arg BASE=localhost:5000/tt-bringup:v0.77.0-rc1 \
#     -t localhost:5000/tt-serving:v0.77.0-rc1 .
#
# RUN — single die, which is the validated upstream path
#   docker run --rm --device /dev/tenstorrent/0 \
#     -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -e HF_MODEL=Qwen/Qwen3.5-9B -e MESH_DEVICE=P150 \
#     localhost:5000/tt-serving:v0.77.0-rc1 \
#     pytest models/demos/blackhole/qwen36/tests/unit/ -v
#
# Device masking is done with `--device /dev/tenstorrent/<id>`, NOT with
# TT_VISIBLE_DEVICES / TT_METAL_VISIBLE_DEVICES. Measured on this rig: mapping one
# device node gives the process exactly one chip, with no CUSTOM cluster and no
# mesh descriptor needed. Which of the two env vars a given build honours is an
# open question upstream; the container boundary sidesteps it entirely.

ARG BASE=localhost:5000/tt-bringup:v0.77.0-rc1
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/tt-metal/build/test/tt_metal/tt_fabric:/usr/local/bin:/usr/bin:/bin \
    TT_METAL_HOME=/opt/tt-metal \
    ARCH_NAME=blackhole

WORKDIR /opt/tt-metal

# tt-metal's dev requirements, installed as shipped.
#
# HISTORY, because it will look like it needs a workaround and does not: at
# v0.60 this file's line 8 was
#   git+https://github.com/tenstorrent/tt-smi.git@v3.0.20
# whose pyluwen pin does not resolve on PyPI —
#   ERROR: Could not find a version that satisfies the requirement
#          pyluwen (unavailable) (from tt-smi)
# — and it took the whole install down with it. **Upstream removed it by v0.77**
# (verified: no tt-smi/pyluwen entry in the 113-line file at this ref), so there
# is nothing to strip here. If you pin TT_METAL_REF back below v0.77, add
# `sed -i '/tt-smi/d'` and drop it again. tt-smi is a host tool regardless —
# it drives board resets and host-side monitoring; nothing here calls it.
#
# The CPU torch index is what create_venv.sh uses. This image never needs a CUDA
# torch: torch is here for the reference implementations the PCC tests compare
# TTNN output against, and that comparison runs on the host CPU.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r tt_metal/python_env/requirements-dev.txt

# Fail the build rather than ship an image that cannot run the demos.
RUN python3 -c "import ttnn, torch, transformers, pytest; \
    print('serving deps OK:', 'torch', torch.__version__, '| transformers', transformers.__version__)"

CMD ["/bin/bash"]
