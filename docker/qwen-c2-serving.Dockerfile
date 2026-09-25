# C2 as a self-contained serving base: P8 (qwen-fast-serving:ci-be9e184e) with everything the
# m3native gate arm bind-mounted for tags v235/v238 baked in, plus the serving contract
# (scripts/ci/serving_c2_contract.py) that makes a platform-launched vLLM match the gate's.
# Thatch.Server's tt-serving-image.yml layers its runtime on top (docker/tenstorrent-serving.Dockerfile).
# Built on the rig by scripts/ci/build-c2-serving-image.sh, which stages this context.
ARG BASE=qwen-fast-serving:ci-be9e184e672756c8eee040f03e48dbff58e68fda
FROM ${BASE}
ARG KERNEL_CACHE
LABEL thatch.qwen.c2-serving="1" thatch.qwen.serving-qualified="false"

# K64i: the arm mounts each binary over its path and each op directory over the image's
# (a directory mount REPLACES the directory, so the image's copy is removed first).
COPY opgraft-K64i/ /opt/qwen-c2/opgraft-K64i/
RUN set -eu; g=/opt/qwen-c2/opgraft-K64i; \
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
    test "$(sha256sum < /opt/tt-metal/build_Release/lib/_ttnncpp.so | cut -c1-64)" = cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4; \
    test "$(sha256sum < "$(readlink -f /opt/tt-metal/ttnn/ttnn/_ttnn.so)" | cut -c1-64)" = "$(sha256sum < $g/_ttnn.so | cut -c1-64)"

# The v235 model-tree graft (artifact m3native-graft-sha-36087022223): the five wired model
# files, the GDN prefill conv op (lever #2) and the C1e gate/up packing, one file each.
# source.sha256 pins the originals the graft was cut from; graft.sha256 the grafted bytes.
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

# The fast path's evidence tree changes for serving, and the plugin's copy of the policy.
COPY ci/ /opt/qwen-c2/ci/
RUN set -eu; cd /opt/qwen-c2/ci; \
    cp serving_fast_policy.py packed_verifier.py serving_request_factory.py pooled_attention_replay.py \
       serving_c2_contract.py test_serving_c2_contract.py test_serving_fast_policy.py qwen_c2_profiles.json \
       /experiment-scripts/ci/; \
    cp serving_fast_policy.py /opt/qwen-fast-plugin/src/vllm_tt_plugin/qwen_fast_policy.py; \
    cp qwen_c2_profiles.json /opt/qwen-c2/profiles.json; \
    site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    cp qwen_c2_boot.py "$site/qwen_c2_boot.py"; \
    echo 'import qwen_c2_boot' > "$site/qwen_c2_serving.pth"; \
    cd /experiment-scripts/ci && VLLM_PLUGINS='' python3 -B -m unittest test_serving_c2_contract test_serving_fast_policy

# The DFlash2 draft: its config (the speculative model path) and the fixture weights.
COPY draft-config/ /draft-config/
COPY fixture/ /experiment-dflash-fixture/
# Caches live on the platform's one persistent mount (/models = the host's hf-cache/hub); the
# paths stay the gate's, so TT_CACHE_PATH and TT_METAL_CACHE read exactly as they did there.
RUN rm -rf /experiment-cache && ln -s /models/.qwen-c2 /experiment-cache

# The v235 environment (m3native-gate.json qwen_configuration, 68 variables) less the three
# the profile sets (QWEN_FAST_MAX_POSITION, QWEN_DSPARK_REQUEST_CONTEXT, QWEN_FAST_OUTPUT_BUDGET),
# plus the arm's non-QWEN settings.
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
    QWEN_FAST_RUNTIME_BINARY_SHA256=cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4 \
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
