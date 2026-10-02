# The drafter fine-tune's TRAINING image: the pinned tt-metal tree built WITH tt-train (the Python package ttml), plus the upstream
# fixes of patches/tt-train/backports.json. Nothing in the serving image changes: the only contracts between training and serving are the
# Hugging Face safetensors checkpoint and the feature-file format (scripts/ci/ft_dataset.py).
#
# NOT BUILT, NOT RUN. It is written against the bring-up image's own recipe (docker/tenstorrent-bringup.Dockerfile builds the tree with
# `build_metal.sh --build-tests` and WITHOUT `--build-tt-train`); the first build happens in a window the owner grants, on the rig,
# whose root is under disk pressure: re-measure free space first and never push this image to the registry.
#
#   docker build --build-arg BASE=<the bring-up image the operator already holds, built from the v0.77.0-rc1 tree> \
#       -f docker/tt-train/Dockerfile -t tt-train-probe .
#
# The base is a build ARG on purpose: its name belongs in the operator's private plan, not in this public repository.
ARG BASE
FROM ${BASE}

ARG JOBS=8
WORKDIR /opt/tt-metal

# The upstream fixes, in the manifest's order. An entry whose status is `ready` must have its <pr>.patch beside the manifest: the build
# stops at a missing file rather than training on an unfixed backward.
COPY patches/tt-train/ /opt/patches/tt-train/
RUN python3 - <<'PY'
import json, os, subprocess, sys
manifest = json.load(open('/opt/patches/tt-train/backports.json'))
for entry in manifest['backports']:
    path = '/opt/patches/tt-train/%s.patch' % entry['pr']
    if entry['status'] == 'ready':
        if not os.path.isfile(path):
            sys.exit('backport %s is ready but its patch is missing' % entry['pr'])
        subprocess.check_call(['git', 'apply', '--check', path], cwd='/opt/tt-metal')
        subprocess.check_call(['git', 'apply', path], cwd='/opt/tt-metal')
    else:
        print('backport %s: %s, not applied' % (entry['pr'], entry['status']))
PY

# tt-train is a separate CMake target of the same tree: an incremental ccache build on the tt-metal that already drives the four-card
# fabric. No `||` fallback: if ttml cannot be built the image must fail to build.
RUN --mount=type=cache,target=/root/.ccache \
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" ./build_metal.sh --build-tt-train --enable-ccache

# The Python package from the tree just built, into the base image's own virtual environment.
RUN "${VIRTUAL_ENV:-/opt/venv}/bin/python3" -m pip install --no-cache-dir -e tt-train

# The probe and its F2 references (CPU code the device results are compared with).
COPY scripts/ci/ttml_probe.py scripts/ci/dflash2_torch.py scripts/ci/tau_lab_report.py scripts/ci/acceptance_report.py \
     scripts/ci/tp4_mesh.py scripts/ci/qwen_p150x4_ring_mesh_graph_descriptor.textproto /opt/probe/
ENV PYTHONPATH=/opt/probe TT_MESH_GRAPH_DESC_PATH=/opt/probe/qwen_p150x4_ring_mesh_graph_descriptor.textproto
ENTRYPOINT ["python3", "/opt/probe/ttml_probe.py"]
