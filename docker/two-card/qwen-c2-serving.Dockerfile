# Stage 8 of the two-card chain: the serving layer. It is stage 7 (the fast-path runtime tree, tagged
# qwen-fast-serving:p8 by default) plus the model-tree graft, the overlay and the contract that makes a platform-launched
# vLLM match the qualification runs'. NOT RUNNABLE FROM THIS REPOSITORY ALONE yet: stage 7 needs a runtime tree that is not
# published (BUILD.md, "What is not here yet").
#
# BUILD CONTEXT: a directory staged by `python3 scripts/ci/c2_overlay.py stage --repo . --out <dir>`, to which BUILD.md
# section 6 says to add this file as Dockerfile, fixture/, draft-config/, build-stamp and an (empty) prebuilt-graft/.
#
# Portable build arguments (see BUILD.md). REGISTRY is only a name prefix; a local image is used first.
#   BASE                    the stage-7 image (docker/two-card/qwen-fast-serving.Dockerfile output), tag qwen-fast-serving:p8
#   RUNTIME_BINARY_SHA256   sha256 of the _ttnncpp.so this image runs. REQUIRED, no default: it is the hash of
#                           YOUR build (stage 5/6 prints it), and the image refuses to build on a mismatch.
#   GRAFT_INSTALL           0 (default): the base image already holds a source build of the K64j tier
#                           (docker/two-card/50-qwen-k64j.Dockerfile); install nothing and only check the binary hash.
#                           1: copy prebuilt libraries and op directories from prebuilt-graft/ in the build context over
#                           the base image's. You need your own prebuilt set for this; none is published.
#                           prebuilt-graft/ must exist in the context either way (an empty directory is enough).
#   KERNEL_CACHE            directory (inside the container) for the compiled-kernel cache. The default sits under
#                           /experiment-cache, which is a link to /models/.qwen-c2 (the writable mount of RECIPES.md),
#                           so a --read-only container can still compile its kernels.
#   SOURCE_REVISION         REQUIRED: the commit the context was staged from (the staged file source-revision).
ARG REGISTRY=localhost:5000
ARG BASE=${REGISTRY}/qwen-fast-serving:p8
FROM ${BASE}
ARG KERNEL_CACHE=/experiment-cache/kernels-qwen
ARG RUNTIME_BINARY_SHA256
ARG GRAFT_INSTALL=0
LABEL qwen.c2-serving="1" qwen.serving-qualified="false"

# K64j: the decode factory with the runtime extent (flag 0x20) and its kernels, built into _ttnncpp.so (the SDPA
# changes of tt-metal-custom-ops/k64j/). With GRAFT_INSTALL=1 each prebuilt binary is copied over its path and each op
# directory over the image's (the image's copy is removed first); with 0 the base already holds the source build.
COPY prebuilt-graft/ /opt/qwen-c2/prebuilt-graft/
RUN set -eu; test -n "${RUNTIME_BINARY_SHA256}"; g=/opt/qwen-c2/prebuilt-graft; \
    if [ "${GRAFT_INSTALL}" = 1 ]; then \
    for pair in _ttnn.so:/opt/tt-metal/ttnn/ttnn/_ttnn.so \
                _ttnncpp.so:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so \
                _ttnncpp.so:/opt/tt-metal/build_Release/lib/_ttnncpp.so; do \
      target=$(readlink -f "${pair#*:}"); cp -f "$g/${pair%%:*}" "$target"; \
    done; \
    for pair in attn_prep:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/attn_prep \
                nlp_concat_heads_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode \
                sdpa_decode:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode \
                sdpa:/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa; do \
      target=$(readlink -f "${pair#*:}"); rm -rf "$target"; cp -a "$g/${pair%%:*}" "$target"; \
    done; \
    test "$(sha256sum < "$(readlink -f /opt/tt-metal/ttnn/ttnn/_ttnn.so)" | cut -c1-64)" = "$(sha256sum < $g/_ttnn.so | cut -c1-64)"; \
    fi; \
    test "$(sha256sum < /opt/tt-metal/build_Release/lib/_ttnncpp.so | cut -c1-64)" = "${RUNTIME_BINARY_SHA256}"; \
    test "$(sha256sum < /opt/tt-metal/build_Release/ttnn/_ttnncpp.so | cut -c1-64)" = "${RUNTIME_BINARY_SHA256}"

# The model-tree graft (staged by c2_overlay.py): the five wired model files, the GDN prefill conv op and the
# gate/up packing, one file each. source.sha256 pins the originals the graft was cut from; graft.sha256 the grafted bytes.
COPY graft/ /opt/qwen-c2/graft/
COPY graft.sha256 source.sha256 /opt/qwen-c2/
RUN set -eu; root=/opt/tt-metal/models/demos/blackhole/qwen36/tt; cd /opt/qwen-c2; \
    sha256sum -c --quiet graft.sha256; \
    sed "s#  graft/\(.*\)\.orig\$#  $root/\1#" source.sha256 | sha256sum -c --quiet; \
    cd graft; \
    for file in model_config.py attention/tp.py gdn/tp.py mlp.py layer.py \
                gdn/gdn_prefill_conv_exact.py gdn/gdn_prefill_conv_exact_compute.cpp \
                gdn/gdn_prefill_conv_exact_reader.cpp gdn/gdn_prefill_conv_exact_writer.cpp \
                mlp_c1e_pack.cpp mlp_c1e_pack.py packed_weight_check.cpp; do \
      install -D -m 0644 "$file" "$root/$file"; \
    done; \
    sed "s#  graft/#  $root/#" /opt/qwen-c2/graft.sha256 | sha256sum -c --quiet

# Every file docker/qwen-c2-overlay.txt names, laid over the base image's fast-path trees (/experiment-scripts/ci,
# /speculative-decoding/harness) and the plugin's copy of the policy. That manifest is the one list:
# c2_overlay.py stages the context from it, installs from it here (recording each destination's sha256
# before and after). The internal build driver checked the built image against it too; it is not part of this branch.
# Then the contract's boot hook, and every overlaid scripts/ci test module, run inside the image.
COPY qwen-c2-overlay.txt c2_overlay.py /opt/qwen-c2/
COPY overlay/ /opt/qwen-c2/overlay/
# /opt/qwen-c2/mesh holds the checked-in mesh graph descriptors the TP4 and 2-link profiles name; the installer
# refuses a destination whose directory the image lacks, so the directory is made first.
RUN set -eu; \
    install -d -m 0755 /opt/qwen-c2/mesh; \
    python3 -B /opt/qwen-c2/c2_overlay.py install --manifest /opt/qwen-c2/qwen-c2-overlay.txt \
      --root /opt/qwen-c2/overlay --record /opt/qwen-c2/overlay-install.json; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    echo 'import qwen_c2_boot' > "$site/qwen_c2_serving.pth"; \
    tests=$(python3 -B /opt/qwen-c2/c2_overlay.py tests --manifest /opt/qwen-c2/qwen-c2-overlay.txt); \
    cd /experiment-scripts/ci && VLLM_PLUGINS='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B -m unittest $tests

# Conversation prefix reuse (the TT prefix-reuse design): the AST stages that graft it into the TT
# plugin (scheduler.py, model_runner.py, worker.py) and the model tree (model.py, qwen36_vllm.py), none
# of them an overlay destination. apply runs each stage from the overlaid tree, compiles the result and
# records every target's sha256 before and after, and the sha256 of every overlaid module the stages
# imported, which c2_image_provenance (f) holds the image to. Once the stage table patches anything it
# refuses unless vllm_tt_plugin and models import from the trees it patches and every target holds its
# pinned original bytes (the source.sha256 pattern above, checked before anything is written); with the
# table empty it only records what it found. check then imports every patched module with
# QWEN_PREFIX_REUSE unset and =1 (each must load the patched file) and re-runs the base image's installed-plugin
# tests and the overlay's in-image tests against the patched files with the switch unset - the files
# every profile serves. Every grafted branch stays off unless QWEN_PREFIX_REUSE=1, which only the
# general-prefix profiles set.
COPY qwen_prefix_stage.py /opt/qwen-c2/
RUN set -eu; cd /experiment-scripts/ci && VLLM_PLUGINS='' python3 -B /opt/qwen-c2/qwen_prefix_stage.py apply \
      --modules /experiment-scripts/ci --record /opt/qwen-c2/prefix-stage.json; \
    tests=$(python3 -B /opt/qwen-c2/c2_overlay.py tests --manifest /opt/qwen-c2/qwen-c2-overlay.txt); \
    python3 -B /opt/qwen-c2/qwen_prefix_stage.py check --record /opt/qwen-c2/prefix-stage.json --tests $tests

# The DFlash2 draft: its config (the speculative model path) and the fixture weights.
COPY draft-config/ /draft-config/
COPY fixture/ /experiment-dflash-fixture/
# Caches live on one persistent, writable mount (/models, the Hugging Face cache you mount at run time); the
# paths below stay under /experiment-cache, so TT_CACHE_PATH and TT_METAL_CACHE read the same in every profile.
RUN rm -rf /experiment-cache && ln -s /models/.qwen-c2 /experiment-cache

# The qualification environment (68 variables) less the three the profile sets (QWEN_FAST_MAX_POSITION,
# QWEN_DSPARK_REQUEST_CONTEXT, QWEN_FAST_OUTPUT_BUDGET), plus the non-QWEN settings.
ENV QWEN_ATTN_PREP=1 QWEN_CARDS_ALLOCATED=1 QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1 QWEN_FABRIC_LINK_PROBE=1 \
    QWEN_FAST_C1_EXACT=1 QWEN_FAST_CARRY_LOG=1 QWEN_FAST_DRAFT_BF8=1 QWEN_FAST_EARLY_DRAFT=1 \
    QWEN_FAST_FAST_COMMIT=1 QWEN_FAST_FAULTHANDLER=1 QWEN_FAST_FOUR_AS_TWO=0 QWEN_FAST_FUSED_COMMIT=1 \
    QWEN_FAST_FUSED_COMMIT_INPLACE=1 QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1 QWEN_FAST_GDN_AFTER_PAIRS=1 \
    QWEN_FAST_GDN_PREFILL_CONV=1 QWEN_FAST_GDN_SEQ_BLOCK=1 QWEN_FAST_GDN_USER_BATCH=1 QWEN_FAST_MEMORY_LEDGER=1 \
    QWEN_FAST_NATIVE_ATTN=1 QWEN_FAST_PACKED_AUDIT=1 QWEN_FAST_PACKED_PROPOSAL=1 QWEN_FAST_PACKED_STEP=1 \
    QWEN_FAST_PADDED_BLOCK=1 QWEN_FAST_PAIR_MASK_REFRESH=1 QWEN_FAST_PAIR_ROW_EXACT=1 QWEN_FAST_PHASE_LOG=1 \
    QWEN_FAST_PHASE_TIMING=1 QWEN_FAST_PIPELINED_COMMITS=1 QWEN_FAST_PIPELINED_PROPOSALS=1 \
    QWEN_FAST_PIPELINED_PUBLISH=1 QWEN_FAST_PRESTAGE=1 QWEN_FAST_PUBLISH_PREWARM=1 QWEN_FAST_QUAD_DRAFT=1 \
    QWEN_FAST_REPLAY_GROUP_ROWS=8 QWEN_FAST_ROUND_B1=1 QWEN_FAST_ROUND_FENCES=1 \
    QWEN_FAST_RUNTIME_BINARY_SHA256=${RUNTIME_BINARY_SHA256} \
    QWEN_FAST_SDPA_MODES=tail,share,slice QWEN_FAST_SDPA_PF=1 QWEN_FAST_SDPA_PF_FLAGS=0x3 \
    QWEN_FAST_SEQ_PUBLISH_LOG=1 QWEN_FAST_SHARD_CHECK=0 QWEN_FAST_SHARED_CCL=1 QWEN_FAST_SINGLE_GATEUP=1 \
    QWEN_FAST_SKIP_BLOCK_STREAM=1 QWEN_FAST_TRACED_PUBLISH=1 QWEN_FAST_VERIFY_T1=1 QWEN_FAST_VERIFY_T2=1 \
    QWEN_FROZEN_COMBINED_RUNTIME=1 QWEN_GDN_CONV_GATES=1 QWEN_GDN_DIRECT_WINDOW=1 QWEN_GDN_FUSED_DECODE=1 \
    QWEN_GDN_FUSED_INPLACE=1 QWEN_GDN_GATE_EXP_ABBA=0 QWEN_GDN_GROUPED_GATHER_ABBA=0 QWEN_GDN_NORM_GATE=1 \
    QWEN_GDN_PACKED_QKV=1 QWEN_GDN_PROJ_DIRECT=1 QWEN_GDN_SHARED_QK_EXPERIMENT=1 QWEN_HARDWARE_TESTS=1 \
    QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1 QWEN_PROJECTION_LINKS=4 QWEN_SDPA_BF8=1 QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 \
    QWEN_SKIP_UNUSED_SINGLETON_POSITIONS=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 TT_METAL_HOME=/opt/tt-metal \
    MESH_DEVICE=P300 OMP_NUM_THREADS=8 TT_CACHE_PATH=/experiment-cache/weights TT_METAL_CACHE=${KERNEL_CACHE} \
    VLLM_CACHE_ROOT=/tmp/vllm-cache QWEN_C2_SERVING=1

# Provenance, last so a new commit or stamp never invalidates the cached layers above (an ARG
# busts the cache of every RUN after it). The base image's revision label names the base's commit; this one names
# the commit the context was staged from (c2_overlay.py stage writes source-revision). build-stamp is any file you
# create (BUILD.md writes a short note into it).
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
COPY source-revision build-stamp /opt/qwen-c2/
RUN test -n "${SOURCE_REVISION}" && test "$(cat /opt/qwen-c2/source-revision)" = "${SOURCE_REVISION}"
