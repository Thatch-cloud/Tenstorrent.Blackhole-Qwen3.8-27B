ARG BASE=zot.thatch.local:5000/tt-vllm@sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465
FROM ${BASE}
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
LABEL thatch.qwen.serving-qualified="false"
COPY bundle/experiment-scripts /experiment-scripts
COPY bundle/experiment-optimisation /experiment-optimisation
COPY bundle/speculative-decoding /speculative-decoding
COPY bundle/serving-bundle.json /opt/qwen-serving/serving-bundle.json
COPY bundle/_ttnncpp.so /tmp/qwen-runtime.so
COPY native-cache-manifest.json /tmp/native-cache-manifest.json
COPY scripts/ci/serving_native_install.py /experiment-scripts/ci/serving_native_install.py
COPY scripts/ci/serving_image_preflight.py /experiment-scripts/ci/serving_image_preflight.py
COPY scripts/ci/serving_fast_policy.py scripts/ci/serving_lifecycle.py scripts/ci/serving_request_factory.py /experiment-scripts/ci/
COPY scripts/ci/serving_plugin_patch.py scripts/ci/serving_dflash_registry.py /experiment-scripts/ci/
COPY scripts/ci/serving_canary_runner.py /experiment-scripts/ci/
COPY scripts/ci/serving_fast_request.py /experiment-scripts/ci/
COPY scripts/ci/serving_runtime.py scripts/ci/serving_gather_experiment.py scripts/ci/gdn_grouped_gather.py scripts/ci/gdn_grouped_gather_gate.py scripts/ci/gdn_grouped_gather_scope.py /experiment-scripts/ci/
ENV PYTHONPATH=/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal
ENV PYTHONDONTWRITEBYTECODE=1
RUN if [ ! -e /optimisation ]; then ln -s /experiment-optimisation /optimisation; fi \
    && test "$(readlink -f /optimisation)" = /experiment-optimisation
RUN python3 -B /experiment-scripts/ci/serving_native_install.py \
    --root /opt/tt-metal --scripts /experiment-scripts/ci --optimisation /experiment-optimisation \
    --binary /tmp/qwen-runtime.so --manifest /tmp/native-cache-manifest.json \
    --output /opt/qwen-serving/native-install.json
RUN git clone --filter=blob:none https://github.com/tenstorrent/vllm-tt-plugin.git /opt/qwen-fast-plugin \
    && git -C /opt/qwen-fast-plugin checkout --detach bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c \
    && python3 -B /experiment-scripts/ci/serving_plugin_patch.py /opt/qwen-fast-plugin \
    && python3 -m pip install --no-deps -e /opt/qwen-fast-plugin
COPY scripts/ci/test_serving_*.py /experiment-scripts/ci/
RUN OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VLLM_PLUGINS='' python3 -B -m unittest \
    test_serving_fast_request test_serving_vllm_contract test_serving_vllm_state \
    test_serving_page_binding test_serving_runner_bridge test_serving_request_factory \
    test_serving_worker_hook test_serving_lifecycle test_serving_cache_owner test_serving_runtime test_serving_gather_experiment
RUN VLLM_PLUGINS='' python3 -c 'import ttnn; from importlib.metadata import version; assert version("vllm").split("+")[0] == "0.25.1"; assert all(callable(getattr(ttnn.transformer, name)) for name in ("attn_decode_prep", "gdn_decode_norm_gate", "gdn_decode_conv_gates", "decode_gated_delta_rule_packed"))'
WORKDIR /opt/tt-metal
RUN VLLM_PLUGINS='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B /experiment-scripts/ci/serving_image_preflight.py --root /opt/tt-metal --output /opt/qwen-serving/startup-preflight.json
RUN VLLM_PLUGINS='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B -m unittest test_serving_scheduler test_serving_vllm_installed test_serving_dflash_registry_installed
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
