# syntax=docker/dockerfile:1.7
#
# Tenstorrent bring-up image + the upstream TP=2 fix stack.
#
# WHY THIS EXISTS
#
# Our rig reproduces upstream PR #53319's reported failure byte-for-byte —
# same program number, same two L1 addresses, same file and line:
#
#   ours:   program 74 clash ... L1 buffer at 1051648, CB region ends 1422208
#           at models/demos/blackhole/qwen36/tt/gdn/tp.py:298
#   #53319: "traced_128 fatals ... in program 74 ... 1051648 vs 1422208, thrown
#            from Untilize reached via the GDN qkv-carry slice (gdn/tp.py:298)"
#
# So this is not a speculative upgrade. We are hitting the exact bug the stack
# fixes, on the hardware shape it was validated against (2x p150a as a 1x2 mesh,
# which upstream calls "P300").
#
#   #53314  conv2d channel-chunking     ttnn/cpp/.../conv2d.cpp (+423)   C++
#   #53319  ttnn.slice tile-window      ttnn/cpp/.../slice.cpp  (+39)    C++
#   #53320  qwen36 demo/model layer     models/demos/.../qwen36 (+51)    Python
#
# ALL THREE APPLY CLEAN to v0.77.0-rc1 (verified with `git apply --check` before
# this file was written), despite GitHub reporting mergeable_state=behind. That
# is consistent with the finding that current main is byte-identical to
# v0.77.0-rc1 at every site these PRs touch.
#
# HEALTH WARNING ON PROVENANCE
#
# These PRs are authored by an agent account (`ctxbot`) from a fork and carry
# essentially no human review: #53319 and #53320 have zero reviews, #53314 has
# only a Copilot bot review. Their silicon receipts are detailed and
# self-consistent, and our byte-exact failure match is strong corroboration —
# but this is an UNMERGED, UNREVIEWED stack. Do not promote an image built from
# it to anything user-facing without a PCC/eval gate of your own.
#
# WHY IT LAYERS ON THE BUILT IMAGE RATHER THAN REBUILDING FROM SOURCE
#
# FROM tt-bringup:v0.77.0-rc1 keeps the already-built tree at /opt/tt-metal, so
# build_metal.sh does an INCREMENTAL rebuild. Only four C++ files change, so with
# the ccache mount below most objects are hits and this costs minutes rather than
# the hours a from-scratch build costs. The tag in the image name still refers to
# the tt-metal ref; `-prstack` marks the delta.
#
# BUILD
#   docker build -f docker/tenstorrent-bringup-prstack.Dockerfile \
#     -t localhost:5000/tt-bringup:v0.77.0-rc1-prstack .
#
# THEN the serving layer on top of it, reusing the existing file unchanged:
#   docker build -f docker/tenstorrent-serving.Dockerfile \
#     --build-arg BASE=localhost:5000/tt-bringup:v0.77.0-rc1-prstack \
#     -t localhost:5000/tt-serving:v0.77.0-rc1-prstack .
#
# TEST — the run that failed at gdn/tp.py:298 before this stack
#   docker run --rm --device /dev/tenstorrent/0 --device /dev/tenstorrent/1 \
#     -v /dev/hugepages-1G:/dev/hugepages-1G --cap-add SYS_NICE \
#     -v $HOME/hf-cache:/root/.cache/huggingface \
#     -e HF_MODEL=Qwen/Qwen3.6-27B -e MESH_DEVICE=P300 \
#     -e TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
#     localhost:5000/tt-serving:v0.77.0-rc1-prstack \
#     pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not 128k"
#
# No mesh-descriptor bind-mount and no ~/tt-patch-p300 mount are needed here:
# #53320 adds "P300": (1, 2) to the demo's own map, and the P300 descriptor
# already ships inside the image.

ARG BASE=localhost:5000/tt-bringup:v0.77.0-rc1
FROM ${BASE}

ARG PR_STACK="53314 53319 53320"
ARG JOBS=48

ENV TT_METAL_HOME=/opt/tt-metal
WORKDIR /opt/tt-metal

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fetch and apply each PR as a diff rather than cherry-picking its commits. The
# clone in the base image is --depth 1, so it has no history to cherry-pick
# against; a diff needs only the working tree. `git apply` without --3way on
# purpose: a 3-way merge here could silently resolve against blobs we do not
# have, and a clean-apply requirement is what makes this image reproducible.
#
# Each diff is kept at /opt/tt-prstack so the image can say exactly what it is.
RUN --network=default mkdir -p /opt/tt-prstack && cd /opt/tt-prstack \
 && for pr in ${PR_STACK}; do \
      echo "=== fetching #${pr} ===" && \
      curl -fsSL "https://github.com/tenstorrent/tt-metal/pull/${pr}.diff" -o "${pr}.diff" && \
      wc -l "${pr}.diff"; \
    done \
 && cd /opt/tt-metal \
 && for pr in ${PR_STACK}; do \
      echo "=== applying #${pr} ===" && \
      git apply --verbose "/opt/tt-prstack/${pr}.diff"; \
    done \
 && git diff --stat | tail -12

# OUR OWN FIX ON TOP OF #53320 — batch truncation in the GDN FIR conv.
#
# #53320 replaced a Python slice with an explicit ttnn.slice to keep the rm_only
# hop in DRAM (a sound fix for an L1 OOM at bank_manager.cpp:451). But writing the
# bounds by hand pinned the BATCH end to a literal 1, so every user past the first
# is silently discarded whenever B >= 2:
#
#   -  x_slice = x_padded[:, k : k + T]                                # keeps all B rows
#   +  x_slice = ttnn.slice(x_padded, (0, k, 0), (1, k + T, D), ...)   # batch end LITERAL 1
#
# The failure surfaces far downstream, at gdn/tp.py:934, as
#   "Ends 2 must be less than or equal to the shape of the tensor 1"
# because conv arrives [1,T,D] where [B,T,D] was required.
#
# One character. Verified on silicon: batched GDN prefill B=2 and B=4 pass at
# PCC 0.99998–1.00000 with per-row lens [128, 96, 64, 128].
#
# Applied with python rather than sed so the count is ASSERTED — if upstream
# changes this line, the build fails loudly instead of silently not patching.
# Remove once the fix lands upstream; see patches/53320-fix-fir-batch-truncation.patch.
RUN python3 - <<'PYFIX'
import pathlib
p = pathlib.Path("/opt/tt-metal/models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py")
s = p.read_text(encoding="utf-8")
old = "x_slice = ttnn.slice(x_padded, (0, k, 0), (1, k + T, D), memory_config=_dram)"
new = "x_slice = ttnn.slice(x_padded, (0, k, 0), (B, k + T, D), memory_config=_dram)"
n = s.count(old)
assert n == 1, f"FIR batch-truncation fix: expected exactly 1 occurrence, found {n}"
p.write_text(s.replace(old, new), encoding="utf-8")
print("applied FIR batch-truncation fix: batch end 1 -> B")
PYFIX

# Incremental rebuild. Same flags as the base image's build, so the only
# difference between the two images is the patch set.
RUN --mount=type=cache,target=/root/.ccache \
    CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    ./build_metal.sh --build-tests --enable-ccache

# ttnn is installed editable (-e) in the base, so the Python side picks the
# patched sources up with no reinstall. The C++ side is the .so we just rebuilt.
RUN python3 -c "import ttnn; print('ttnn OK:', ttnn.__file__)" \
 && test -x build/test/tt_metal/tt_fabric/test_system_health \
 && echo 'test_system_health present'

# Record the patch set in the image so a running container can prove what it is.
RUN printf '%s\n' ${PR_STACK} > /opt/tt-prstack/APPLIED.txt \
 && echo "applied PRs:" && cat /opt/tt-prstack/APPLIED.txt

WORKDIR /opt/tt-metal
CMD ["/bin/bash"]
