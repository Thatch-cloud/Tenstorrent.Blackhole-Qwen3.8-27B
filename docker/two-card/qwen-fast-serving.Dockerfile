# Stage 7 of the two-card chain: the fast-path runtime tree on top of the custom-ops image. NOT RUNNABLE FROM THIS
# REPOSITORY ALONE yet: the COPY sources named bundle/... below are a runtime tree that the earlier build read from CI
# artifacts which no longer exist (BUILD.md, "What is not here yet").
#
# Portable build arguments. REGISTRY is only a name prefix: FROM resolves a local image first, so no registry has to
# run. BASE is the K64j-tier image (tt-vllm:qwen38-k64j, docker/two-card/50-qwen-k64j.Dockerfile; the `ops` tier lacks
# the prefill-chain factory the serving environment turns on). It is a source build, so this copy does NOT replace its
# libraries: the earlier step that installed a prebuilt _ttnncpp.so over the image's is removed.
#
# BUILD CONTEXT: this repository's scripts/ci/ (the COPY lines) plus a directory bundle/ holding experiment-scripts/,
# experiment-optimisation/, speculative-decoding/ and serving-bundle.json: the serving runtime tree, which this
# repository does not publish.
ARG REGISTRY=localhost:5000
ARG BASE=${REGISTRY}/tt-vllm:qwen38-k64j
FROM ${BASE}
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
LABEL qwen.serving-qualified="false"
COPY bundle/experiment-scripts /experiment-scripts
COPY bundle/experiment-optimisation /experiment-optimisation
COPY bundle/speculative-decoding /speculative-decoding
COPY bundle/serving-bundle.json /opt/qwen-serving/serving-bundle.json
COPY scripts/ci/serving_image_preflight.py /experiment-scripts/ci/serving_image_preflight.py
COPY scripts/ci/serving_fast_policy.py scripts/ci/serving_lifecycle.py scripts/ci/serving_request_factory.py /experiment-scripts/ci/
COPY scripts/ci/serving_plugin_patch.py scripts/ci/serving_dflash_registry.py /experiment-scripts/ci/
COPY scripts/ci/serving_canary_runner.py /experiment-scripts/ci/
COPY scripts/ci/serving_fast_request.py /experiment-scripts/ci/
COPY scripts/ci/serving_worker_hook.py /experiment-scripts/ci/
COPY scripts/ci/serving_one_in_flight.py /experiment-scripts/ci/
COPY scripts/ci/runtime_binary_override.py /experiment-scripts/ci/
COPY scripts/ci/dflash_proposal_trace.py /experiment-scripts/ci/
COPY scripts/ci/gdn_user_batch.py scripts/ci/gdn_user_batch_conv.py scripts/ci/gdn_seq_block.py /experiment-scripts/ci/
COPY scripts/ci/dflash_packed_proposal.py scripts/ci/dflash_packed_proposal_coordinator.py /experiment-scripts/ci/
COPY scripts/ci/dflash_pipelined_publish.py /experiment-scripts/ci/
COPY scripts/ci/dflash_traced_publish.py /experiment-scripts/ci/
COPY scripts/ci/profiled_block_stream_override.py /experiment-scripts/ci/
# LLK profiling (llk_zone_override, imported by serving_runtime only under QWEN_LLK_ZONES, and its two helpers):
# the C2 overlay carries them; named here too so a rebuild from HEAD has what serving_runtime imports.
COPY scripts/ci/llk_zone_override.py scripts/ci/llk_kernels.py scripts/ci/llk_zones.py /experiment-scripts/ci/
COPY scripts/ci/serving_vllm_packed.py /experiment-scripts/ci/
COPY scripts/ci/serving_vllm_contract.py /experiment-scripts/ci/
COPY scripts/ci/model_batch.py /experiment-scripts/ci/
COPY scripts/ci/serving_packed_bridge.py /experiment-scripts/ci/
COPY scripts/ci/serving_sequential_step.py /experiment-scripts/ci/
COPY scripts/ci/serving_vllm_state.py /experiment-scripts/ci/
COPY scripts/ci/serving_buffer_pool.py /experiment-scripts/ci/
COPY scripts/ci/dflash_device.py scripts/ci/draft_attention_branch.py scripts/ci/draft_mlp_branch.py scripts/ci/draft_convolution.py scripts/ci/draft_convolution_fused.py scripts/ci/draft_convolution_fused_io.cpp scripts/ci/verifier_engine.py scripts/ci/verifier_pack.py scripts/ci/draft_kv_history.py scripts/ci/serving_parked_engines.py /experiment-scripts/ci/
COPY scripts/ci/serving_runtime.py scripts/ci/serving_gather_experiment.py scripts/ci/gdn_grouped_gather.py scripts/ci/gdn_grouped_gather_gate.py scripts/ci/gdn_grouped_gather_scope.py /experiment-scripts/ci/
COPY scripts/ci/packed_verifier.py scripts/ci/serving_packed_step.py scripts/ci/gdn_device_loop_state.py scripts/ci/gdn_records.py scripts/ci/dflash_request_runtime.py /experiment-scripts/ci/
# gdn_state_copy travels with gdn_device_loop_state, which imports batch_enabled
# and copy_compact_batch from it. Both were added by commit 7ecf980a (K1: one
# launch for the packed decode state moves) and neither exists in the bundle,
# whose tree is commit 77d6995a. an earlier run died at engine init with
# "cannot import name batch_enabled from gdn_state_copy"; it went unnoticed
# because the m3native lane serves a different image. test_serving_image_copy_closure
# now fails on CPU for any COPYed file needing a symbol the bundle cannot supply.
# The .cpp travels with it: gdn_state_copy builds its KernelDescriptor with
# kernel_source=Path(__file__).with_suffix(".cpp"), so the SIBLING file is the
# kernel. Commit 7ecf980a changed both together - the new python emits runtime
# args as group-count + worker-index + 15-value blocks and the new kernel reads
# that layout. Copying only the .py left the new python driving the bundle's OLD
# kernel, which read a count where an address belongs, issued a DMA that never
# completed, and wedged the core - so the NEXT generic_op hung, which is why earlier runs
# hung at two different ops in the same warmup pass.
# K5-A (QWEN_FAST_GDN_SEQ_BLOCK, default off) is the same shape: gdn_seq_block.py (above, beside
# gdn_user_batch_conv, which imports it, as model_batch and packed_verifier do) generates its kernels
# from the SIBLING gdn_seq_block_{compute,reader,writer}.cpp (gdn_seq_block.SOURCES), so they travel here.
COPY scripts/ci/gdn_state_copy.py scripts/ci/gdn_state_copy.cpp scripts/ci/gdn_seq_block_compute.cpp scripts/ci/gdn_seq_block_reader.cpp scripts/ci/gdn_seq_block_writer.cpp /experiment-scripts/ci/
COPY scripts/ci/gdn_snapshot.py scripts/ci/dflash_prefill_window.py /experiment-scripts/ci/
COPY scripts/ci/pooled_attention_replay.py /experiment-scripts/ci/
COPY scripts/ci/target_packed_pages.py /experiment-scripts/ci/
COPY scripts/ci/dflash_batched_mask.py /experiment-scripts/ci/
COPY scripts/ci/packed_shapes.py scripts/ci/packed_cache_writer.py scripts/ci/force_argmax.py scripts/ci/gdn_prefix.py /experiment-scripts/ci/
COPY scripts/ci/two_tile_norm.py scripts/ci/two_tile_decode.py /experiment-scripts/ci/
COPY scripts/ci/serving_runner_bridge.py /experiment-scripts/ci/
COPY scripts/ci/ordered_cache.py /experiment-scripts/ci/
# Image A: serving_startup (the block-stream skip, QWEN_FAST_SKIP_BLOCK_STREAM /
# QWEN_FAST_SINGLE_GATEUP) and dflash_combined_request (the FusedT16Arm skip,
# QWEN_FAST_SINGLE_GATEUP) were in neither list and shipped at their bundle version;
# both were byte-identical to 77d6995a when overlaid. memory_ledger is new
# (QWEN_FAST_MEMORY_LEDGER); serving_startup, serving_runtime and serving_packed_step
# import it. Every flag defaults off.
COPY scripts/ci/serving_startup.py scripts/ci/dflash_combined_request.py scripts/ci/memory_ledger.py /experiment-scripts/ci/
# Verify-trace T1 (QWEN_FAST_VERIFY_T1, default off): packed_verifier, gdn_device_loop_state
# and gdn_user_batch import it, so it must reach the image beside them.
COPY scripts/ci/verify_trace_t1.py /experiment-scripts/ci/
# Verify-trace T2 (QWEN_FAST_VERIFY_T2, default off): verify_trace_t2.RUNTIME_FILES, the one table.
# gdn_user_batch_conv, model_batch, packed_verifier and serving_packed_step import it; the
# packed windows kernel is the SIBLING .cpp of its driver, so the pair travels together.
COPY scripts/ci/verify_trace_t2.py scripts/ci/gdn_conv_windows_packed.py scripts/ci/gdn_conv_windows_packed.cpp scripts/ci/packed_ordered_cache.py /experiment-scripts/ci/
# Variable-user packed rounds M1 (QWEN_FAST_PADDED_PROBE, default off): packed_verifier imports
# padded_probe when the flag is set. M0 changes only modules already listed above.
COPY scripts/ci/padded_probe.py /experiment-scripts/ci/
# Round-fence plan H1a (QWEN_FAST_PRESTAGE, _PRESTAGE_AUDIT, QWEN_FAST_ROUND_FENCES; every flag default
# off): packed_verifier imports verify_prestage at module level, serving_packed_step, serving_worker_hook
# and serving_lifecycle reach it too, so it must reach the image beside them.
COPY scripts/ci/verify_prestage.py /experiment-scripts/ci/
# Round-fence plan H1b (QWEN_FAST_FUSED_COMMIT, _INPLACE, _LIVE_BANKS, _AUDIT; every flag default off):
# packed_verifier, serving_packed_step and dflash_proposal_trace import fused_commit when a flag is set,
# so it must reach the image beside them. It drives the bundle's own slide kernel (the served driver's .cpp).
COPY scripts/ci/fused_commit.py /experiment-scripts/ci/
# Round-fence plan H2 (QWEN_FAST_EARLY_DRAFT, QWEN_FAST_GDN_AFTER_PAIRS; both default off): serving_worker_hook
# and packed_verifier import early_draft when a flag is set, so it must reach the image beside them.
COPY scripts/ci/early_draft.py /experiment-scripts/ci/
# The pair drafter's row-1 fix (QWEN_FAST_PAIR_ROW_EXACT, default off): dflash_proposal_trace, draft_attention_branch
# and dflash_packed_proposal import pair_row_exact when the flag is set, so it must reach the image beside them.
COPY scripts/ci/pair_row_exact.py /experiment-scripts/ci/
# Q4, the four-user 64-row draft pass (QWEN_FAST_QUAD_DRAFT, default off): the packed proposal coordinator,
# dflash_device and the draft branches import quad_draft when the flag is set, and it drives the SIBLING
# quad_conv_io.cpp (the probed 64-row conv I/O kernel, sha256-pinned in quad_draft.CONV_KERNEL_SHA256),
# so both travel here.
COPY scripts/ci/quad_draft.py scripts/ci/quad_conv_io.cpp /experiment-scripts/ci/
# Publish prewarm (QWEN_FAST_PUBLISH_PREWARM, default off): serving_request_factory imports publish_prewarm when the
# flag is set, so it must reach the image beside it.
COPY scripts/ci/publish_prewarm.py /experiment-scripts/ci/
# S2 B6 (publication_warm): packed_verifier imports it at an extent block's attach (QWEN_FAST_EXTENT_REPLAY=1, never
# set in this image), so it travels beside it.
COPY scripts/ci/publication_warm.py /experiment-scripts/ci/
ENV PYTHONPATH=/experiment-scripts/ci:/speculative-decoding/harness:/opt/tt-metal/ttnn:/opt/tt-metal
ENV PYTHONDONTWRITEBYTECODE=1
RUN if [ ! -e /optimisation ]; then ln -s /experiment-optimisation /optimisation; fi \
    && test "$(readlink -f /optimisation)" = /experiment-optimisation
RUN git clone --filter=blob:none https://github.com/tenstorrent/vllm-tt-plugin.git /opt/qwen-fast-plugin \
    && git -C /opt/qwen-fast-plugin checkout --detach bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c \
    && python3 -B /experiment-scripts/ci/serving_plugin_patch.py /opt/qwen-fast-plugin \
    && python3 -m pip install --no-deps -e /opt/qwen-fast-plugin
COPY scripts/ci/test_serving_*.py /experiment-scripts/ci/
COPY scripts/ci/test_packed_verifier.py /experiment-scripts/ci/
RUN OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VLLM_PLUGINS='' python3 -B -m unittest \
    test_serving_fast_request test_serving_vllm_contract test_serving_vllm_state \
    test_serving_page_binding test_serving_runner_bridge test_serving_request_factory \
    test_serving_worker_hook test_serving_lifecycle test_serving_cache_owner test_serving_runtime test_serving_gather_experiment \
    test_serving_runtime_binary_override test_serving_profiled_block_stream_override test_serving_startup
RUN VLLM_PLUGINS='' python3 -c 'import ttnn; from importlib.metadata import version; assert version("vllm").split("+")[0] == "0.25.1"; assert all(callable(getattr(ttnn.transformer, name)) for name in ("attn_decode_prep", "gdn_decode_norm_gate", "gdn_decode_conv_gates", "decode_gated_delta_rule_packed"))'
WORKDIR /opt/tt-metal
RUN VLLM_PLUGINS='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B /experiment-scripts/ci/serving_image_preflight.py --root /opt/tt-metal --output /opt/qwen-serving/startup-preflight.json
RUN VLLM_PLUGINS='' VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -B -m unittest test_serving_scheduler test_serving_vllm_installed test_serving_dflash_registry_installed
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
