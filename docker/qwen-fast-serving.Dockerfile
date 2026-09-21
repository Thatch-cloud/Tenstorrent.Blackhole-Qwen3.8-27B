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
COPY scripts/ci/serving_worker_hook.py /experiment-scripts/ci/
COPY scripts/ci/serving_one_in_flight.py /experiment-scripts/ci/
COPY scripts/ci/runtime_binary_override.py /experiment-scripts/ci/
COPY scripts/ci/serving_vllm_packed.py /experiment-scripts/ci/
COPY scripts/ci/serving_vllm_contract.py /experiment-scripts/ci/
COPY scripts/ci/model_batch.py /experiment-scripts/ci/
COPY scripts/ci/serving_packed_bridge.py /experiment-scripts/ci/
COPY scripts/ci/serving_sequential_step.py /experiment-scripts/ci/
COPY scripts/ci/serving_vllm_state.py /experiment-scripts/ci/
COPY scripts/ci/serving_buffer_pool.py /experiment-scripts/ci/
COPY scripts/ci/dflash_device.py scripts/ci/draft_attention_branch.py scripts/ci/draft_mlp_branch.py scripts/ci/draft_convolution.py scripts/ci/draft_convolution_fused.py scripts/ci/draft_convolution_fused_io.cpp scripts/ci/verifier_engine.py scripts/ci/verifier_pack.py scripts/ci/draft_kv_history.py /experiment-scripts/ci/
COPY scripts/ci/serving_runtime.py scripts/ci/serving_gather_experiment.py scripts/ci/gdn_grouped_gather.py scripts/ci/gdn_grouped_gather_gate.py scripts/ci/gdn_grouped_gather_scope.py /experiment-scripts/ci/
COPY scripts/ci/packed_verifier.py scripts/ci/serving_packed_step.py scripts/ci/gdn_device_loop_state.py scripts/ci/gdn_records.py scripts/ci/dflash_request_runtime.py /experiment-scripts/ci/
COPY scripts/ci/gdn_snapshot.py scripts/ci/dflash_prefill_window.py /experiment-scripts/ci/
COPY scripts/ci/pooled_attention_replay.py /experiment-scripts/ci/
COPY scripts/ci/target_packed_pages.py /experiment-scripts/ci/
COPY scripts/ci/dflash_batched_mask.py /experiment-scripts/ci/
COPY scripts/ci/packed_shapes.py scripts/ci/packed_cache_writer.py scripts/ci/force_argmax.py scripts/ci/gdn_prefix.py /experiment-scripts/ci/
COPY scripts/ci/two_tile_norm.py scripts/ci/two_tile_decode.py /experiment-scripts/ci/
COPY scripts/ci/serving_runner_bridge.py /experiment-scripts/ci/
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
    test_serving_worker_hook test_serving_lifecycle test_serving_cache_owner test_serving_runtime test_serving_gather_experiment \
    test_serving_runtime_binary_override
RUN VLLM_PLUGINS='' python3 -c 'import ttnn; from importlib.metadata import version; assert version("vllm").split("+")[0] == "0.25.1"; assert all(callable(getattr(ttnn.transformer, name)) for name in ("attn_decode_prep", "gdn_decode_norm_gate", "gdn_decode_conv_gates", "decode_gated_delta_rule_packed"))'
WORKDIR /opt/tt-metal
RUN VLLM_PLUGINS='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B /experiment-scripts/ci/serving_image_preflight.py --root /opt/tt-metal --output /opt/qwen-serving/startup-preflight.json
RUN VLLM_PLUGINS='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B -m unittest test_serving_scheduler test_serving_vllm_installed test_serving_dflash_registry_installed
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
