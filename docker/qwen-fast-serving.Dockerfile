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
COPY scripts/ci/serving_solo_lane.py /experiment-scripts/ci/
COPY scripts/ci/request_width_warm.py /experiment-scripts/ci/
COPY scripts/ci/runtime_binary_override.py /experiment-scripts/ci/
COPY scripts/ci/dflash_proposal_trace.py /experiment-scripts/ci/
COPY scripts/ci/gdn_user_batch.py scripts/ci/gdn_user_batch_conv.py scripts/ci/gdn_seq_block.py /experiment-scripts/ci/
COPY scripts/ci/dflash_packed_proposal.py scripts/ci/dflash_packed_proposal_coordinator.py /experiment-scripts/ci/
COPY scripts/ci/dflash_pipelined_publish.py /experiment-scripts/ci/
COPY scripts/ci/dflash_traced_publish.py /experiment-scripts/ci/
COPY scripts/ci/profiled_block_stream_override.py /experiment-scripts/ci/
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
# gdn_state_copy travels with gdn_device_loop_state, which imports batch_enabled
# and copy_compact_batch from it. Both were added by commit 7ecf980a (K1: one
# launch for the packed decode state moves) and neither exists in the bundle,
# whose tree is commit 77d6995a. Run 35683127469 died at engine init with
# "cannot import name batch_enabled from gdn_state_copy"; it went unnoticed
# because the m3native lane serves a different image. test_serving_image_copy_closure
# now fails on CPU for any COPYed file needing a symbol the bundle cannot supply.
# The .cpp travels with it: gdn_state_copy builds its KernelDescriptor with
# kernel_source=Path(__file__).with_suffix(".cpp"), so the SIBLING file is the
# kernel. Commit 7ecf980a changed both together - the new python emits runtime
# args as group-count + worker-index + 15-value blocks and the new kernel reads
# that layout. Copying only the .py left the new python driving the bundle's OLD
# kernel, which read a count where an address belongs, issued a DMA that never
# completed, and wedged the core - so the NEXT generic_op hung, which is why runs
# 35684239068 and 35685401900 hung at two different ops in the same warmup pass.
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
# Tensor-parallel width (S2 TP4 port; TP2 unless QWEN_FAST_TP names four cards): serving_startup and, through
# mesh_link_policy, feature_collective / quad_draft / dflash_device import these, so they travel beside them.
COPY scripts/ci/tp_shapes.py scripts/ci/tp_addresses.py scripts/ci/tile_collective_tp.py scripts/ci/tp4_mesh.py scripts/ci/mesh_link_policy.py scripts/ci/gdn_user_batch_tp.py scripts/ci/tp_kernels.py scripts/ci/gdn_multitoken_conv_tp.py scripts/ci/gdn_records_tp.py scripts/ci/gdn_commit_dma_tp.py scripts/ci/gdn_commit_dma_tp.cpp scripts/ci/gdn_multitoken_tp.py scripts/ci/gdn_conv_prefix_copy_tp.cpp scripts/ci/gdn_conv_windows_tp.cpp scripts/ci/attention_replay_tp.py scripts/ci/attention_mask_replay_tp.py scripts/ci/attention_parallel_tp.py scripts/ci/attention_fold_dma_tp.py scripts/ci/attention_fold_dma_tp.cpp scripts/ci/attention_mask_replay_tp.cpp scripts/ci/serving_cache_owner.py /experiment-scripts/ci/
# Verify-trace T1 (QWEN_FAST_VERIFY_T1, default off): packed_verifier, gdn_device_loop_state
# and gdn_user_batch import it, so it must reach the image beside them.
COPY scripts/ci/verify_trace_t1.py /experiment-scripts/ci/
# Verify-trace T2 (QWEN_FAST_VERIFY_T2, default off): verify_trace_t2.RUNTIME_FILES, the one table.
# gdn_user_batch_conv, model_batch, packed_verifier and serving_packed_step import it; the
# packed windows kernel is the SIBLING .cpp of its driver, so the pair travels together.
COPY scripts/ci/verify_trace_t2.py scripts/ci/gdn_conv_windows_packed.py scripts/ci/gdn_conv_windows_packed.cpp scripts/ci/packed_ordered_cache.py scripts/ci/page_width_tp4.py /experiment-scripts/ci/
# TP4 verify-glue levers (tp4/vglue; QWEN_FAST_TP4_*, default off): tp4_vglue.RUNTIME_FILES, the one table.
COPY scripts/ci/tp4_vglue.py scripts/ci/gdn_commit_lanes_tp.cpp scripts/ci/gdn_rows_dma_tp.py scripts/ci/gdn_rows_dma_tp.cpp scripts/ci/gdn_device_loop_state_tp.py scripts/ci/gdn_block_conv_tp.py scripts/ci/attention_block_fold_tp.py scripts/ci/attention_block_fold_tp.cpp scripts/ci/extent_attention_fold_tp.py scripts/ci/gdn_pair_slice_tp.py /experiment-scripts/ci/
# The long-context SDPA configuration (tp4/sdpa-long, QWEN_FAST_TP4_SDPA, default off): sdpa_long_tp.RUNTIME_FILES, imported by extent_attention_fold_tp.
COPY scripts/ci/sdpa_long_tp.py /experiment-scripts/ci/
# The drafter's hidden-width RMS norms on a wide grid (tp4/next-3-cheap; QWEN_FAST_TP4_DRAFT_WIDE, default off): dflash_device,
# draft_attention_branch and draft_mlp_branch import draft_wide_tp, so it must reach the image beside them.
COPY scripts/ci/draft_wide_tp.py /experiment-scripts/ci/
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
# quad_conv_io.cpp (the card-B probed 64-row conv I/O kernel, sha256-pinned in quad_draft.CONV_KERNEL_SHA256),
# so both travel here.
COPY scripts/ci/quad_draft.py scripts/ci/quad_conv_io.cpp /experiment-scripts/ci/
# The four-card twin of the quad (quad_draft_tp: tp_addresses.install() puts it in quad_draft's place at QWEN_FAST_TP=4 only,
# MODULE_TWINS) and the singles audit (QWEN_FAST_DRAFT_SINGLES_AUDIT, imported by the coordinator when the flag is set).
COPY scripts/ci/quad_draft_tp.py scripts/ci/draft_singles_audit.py /experiment-scripts/ci/
# The four-card twin of the fused commit (fused_commit_tp: tp_addresses.install() puts it in fused_commit's place at QWEN_FAST_TP=4
# only, MODULE_TWINS; packed_verifier, serving_packed_step, verify_prestage and dflash_proposal_trace import fused_commit lazily, so
# they reach it). It borrows the pair's fused_commit (listed above) for every name it does not redefine and drives the bundle's
# slide kernel through draft_kv_slide_tp.
COPY scripts/ci/fused_commit_tp.py /experiment-scripts/ci/
# Publish prewarm (QWEN_FAST_PUBLISH_PREWARM, default off): serving_request_factory imports publish_prewarm when the
# flag is set, so it must reach the image beside it.
COPY scripts/ci/publish_prewarm.py /experiment-scripts/ci/
# S2 B6 (publication_warm): packed_verifier imports it at an extent block's attach (QWEN_FAST_EXTENT_REPLAY=1, never
# set in this image), so it travels beside it.
COPY scripts/ci/publication_warm.py /experiment-scripts/ci/
# S2T-07 (the drafter at four cards): these carry the per-chip counts from tp_shapes (unchanged at the pair) and the four-card
# twins dflash_device / serving_request_factory select at QWEN_FAST_TP=4, so they travel beside their importers.
COPY scripts/ci/serving_page_binding.py scripts/ci/verifier_inputs.py scripts/ci/feature_projection_tp.py scripts/ci/draft_kv_history_tp.py scripts/ci/draft_kv_slide_tp.py scripts/ci/verifier_engine_tp.py /experiment-scripts/ci/
# Four-card twins of the pinned pair helpers (S2 TP4 port): tp_addresses.install() rebinds the pair's functions to these at QWEN_FAST_TP=4 only;
# the pair's own modules stay the bundle's bytes (test_tp2_pins).
COPY scripts/ci/draft_convolution_tp.py scripts/ci/draft_convolution_fused_tp.py scripts/ci/draft_head_layout_tp.py scripts/ci/draft_kv_projection_tp.py scripts/ci/draft_mlp_tp.py scripts/ci/draft_shared_head_tp.py scripts/ci/feature_collective_tp.py scripts/ci/pair_row_exact_tp.py scripts/ci/attention_head_fold_tp.py scripts/ci/gdn_conv_prefix_copy_tp.py scripts/ci/gdn_working_state_tp.py scripts/ci/mtp_hidden_rows_tp.py scripts/ci/gdn_batched_conv_tp.py scripts/ci/gdn_conv_windows_tp.py scripts/ci/attach_scopes_tp.py scripts/ci/draft_attention_tp.py scripts/ci/ordered_cache_tp.py /experiment-scripts/ci/
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
