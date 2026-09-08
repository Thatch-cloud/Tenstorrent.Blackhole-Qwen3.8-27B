"""Full-model serial rollback oracle across attention page boundaries, not batched verification speed."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from gdn_snapshot import ActiveSnapshot
from attention_batch import capture_operation
from verifier_trace_profile import PROFILE_BASE_FLAGS, PROFILE_BEST_FLAGS


def active_serial_logits(logits, vocab_size):
    if logits.shape[-1] != vocab_size or logits.numel() // vocab_size not in (1, 8):
        raise ValueError("Expected native B1 logits with optional Bmax8 API padding")
    return logits.reshape(-1, vocab_size)[:1]


def cache_geometry(shape):
    dimensions = tuple(shape)
    if len(dimensions) != 4 or dimensions[0] < 2 or dimensions[2] != 64:
        raise ValueError("Expected at least two 64-token KV pages")
    return dimensions[1:]


def logical_kv_prefix(host, valid_tokens):
    if type(valid_tokens) is not int or not 0 < valid_tokens <= 128:
        raise ValueError("Expected a valid prefix of the first two KV pages")
    if len(host.shape) != 4 or host.shape[0] != 2 or host.shape[2] != 64:
        raise ValueError("Expected two 64-token KV pages")
    heads, width = host.shape[1], host.shape[3]
    return host.permute(1, 0, 2, 3).reshape(heads, 128, width)[:, :valid_tokens]


def logical_kv_chunk(host, first_page, valid_tokens):
    if type(first_page) is not int or first_page < 0 or type(valid_tokens) is not int:
        raise ValueError("Expected integer page offset and valid-token count")
    if len(host.shape) != 4 or host.shape[0] < 1 or host.shape[2] != 64:
        raise ValueError("Expected a nonempty chunk of 64-token pages")
    count = min(host.shape[0] * 64, valid_tokens - first_page * 64)
    if count <= 0:
        raise ValueError("Chunk contains no valid tokens")
    return host.permute(1, 0, 2, 3).reshape(host.shape[1], -1, host.shape[3])[:, :count]


def verification_widths(max_rows, *, packed_checkpoints, ordered_cache, deferred_commit, attribution):
    if type(max_rows) is not int or max_rows not in (16, 32):
        raise ValueError('Maximum verification width must be 16 or 32')
    if max_rows == 32 and (not packed_checkpoints or not ordered_cache or attribution):
        raise ValueError('T32 requires the packed-history ordered-cache gate without attribution')
    return (1, 2, 4, 8, 16, 32) if max_rows == 32 else (1, 2, 4, 8, 16)


def main():
    if os.environ.get("QWEN_HARDWARE_TESTS") != "1" or os.environ.get("QWEN_CARDS_ALLOCATED") != "1":
        raise RuntimeError("Explicit hardware allocation required")
    if os.environ.get("TT_METAL_SIMULATOR") or os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        raise RuntimeError("Fast-dispatch hardware required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--coding-cost", action="store_true")
    parser.add_argument("--serial-sdpa", action="store_true")
    parser.add_argument("--attribution", action="store_true")
    parser.add_argument('--device-profile', action='store_true')
    parser.add_argument('--profile-context', type=int, choices=(4095, 16383))
    parser.add_argument('--prefix-zero-reuse', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument("--compact-gdn", action="store_true")
    parser.add_argument("--reuse-gdn-input", action="store_true")
    parser.add_argument("--skip-row-clones", action="store_true")
    parser.add_argument("--hoist-row-layout", action="store_true")
    parser.add_argument("--device-loop-gdn", action="store_true")
    parser.add_argument('--compact-prologue', action='store_true')
    parser.add_argument('--batch-conv', action='store_true')
    parser.add_argument('--packed-checkpoints', action='store_true')
    parser.add_argument('--deferred-commit', action='store_true')
    parser.add_argument('--ordered-cache', action='store_true')
    parser.add_argument('--commit-dma', action='store_true')
    parser.add_argument('--captured-commit', action='store_true')
    parser.add_argument('--max-rows', type=int, choices=(16, 32), default=16)
    parser.add_argument('--replay-inputs', action='store_true')
    parser.add_argument('--device-selection', action='store_true')
    parser.add_argument('--request-pilot', action='store_true')
    parser.add_argument('--norm-batch', action='store_true')
    parser.add_argument('--grouped-attention', action='store_true')
    parser.add_argument('--attention-dma', action='store_true')
    parser.add_argument('--attention-parallel', action='store_true')
    parser.add_argument('--attention-tree', action='store_true')
    parser.add_argument('--attention-replay', action='store_true')
    parser.add_argument('--attention-mask-once', action='store_true')
    parser.add_argument('--replay-group-rows', type=int, choices=(4, 8), default=4)
    parser.add_argument('--attention-engine', action='store_true')
    parser.add_argument('--attention-engine-wide', action='store_true')
    parser.add_argument('--target-features', action='store_true')
    parser.add_argument('--target-feature-prefill', action='store_true')
    parser.add_argument('--target-feature-batch', action='store_true')
    parser.add_argument('--target-feature-replay', action='store_true')
    parser.add_argument('--target-feature-prefix', action='store_true')
    options = parser.parse_args()
    coding_request = os.environ.get('QWEN_CODING_REQUEST', '0')
    mtp_drafts = os.environ.get('QWEN_MTP_DRAFTS', '0')
    dflash_drafts = os.environ.get('QWEN_DFLASH_DRAFTS', '0')
    dflash_capture = os.environ.get('QWEN_DFLASH_CAPTURE', '0')
    dflash_commit_abba = os.environ.get('QWEN_DFLASH_COMMIT_ABBA', '0')
    if dflash_commit_abba not in ('0', '1') or (dflash_commit_abba == '1' and
            (dflash_capture != '1' or dflash_drafts != '7')):
        parser.error('Commit-only GDN ABBA requires the captured T8 DFlash2 request')
    if dflash_capture not in ('0', '1') or (dflash_capture == '1' and dflash_drafts == '0'):
        parser.error('Captured drafting requires an explicit DFlash2 request')
    if dflash_drafts not in ('0', '7', '31') or (dflash_drafts != '0' and
            (mtp_drafts != '0' or coding_request != '1' or not options.norm_batch or not options.request_pilot
             or options.attention_engine or options.attention_engine_wide or options.replay_inputs
             or os.environ.get('QWEN_LOOKUP_CAP_ABBA', '0') != '0'
             or os.environ.get('QWEN_FABRIC_LINK_PROBE', '0') != '1'
             or os.environ.get('QWEN_PROJECTION_LINKS', '0') != '4')):
        parser.error('DFlash2 requires the isolated coding norm-engine request and explicit four-link pair')
    if mtp_drafts not in ('0', '7') or (mtp_drafts != '0' and
            (coding_request != '1' or not options.norm_batch or not options.request_pilot
             or options.attention_engine or options.attention_engine_wide
             or os.environ.get('QWEN_LOOKUP_CAP_ABBA', '0') != '0'
             or os.environ.get('QWEN_FABRIC_LINK_PROBE', '0') != '1')):
        parser.error('MTP requires the explicit short coding norm-engine, audited pair and no ABBA options')
    fabric_link_probe = os.environ.get('QWEN_FABRIC_LINK_PROBE', '0')
    if fabric_link_probe not in ('0', '1') or (fabric_link_probe == '1' and
            (coding_request != '1' or not options.norm_batch or not options.request_pilot
             or options.attention_engine or options.attention_engine_wide
             or os.environ.get('QWEN_LOOKUP_CAP_ABBA', '0') != '0')):
        parser.error('Fabric request comparison requires coding norm-engine without other ABBA options')
    if fabric_link_probe == '1':
        from sampling_link_policy import audit, sampler_links
        fabric_sources = audit('/opt/tt-metal', os.environ)
    if coding_request not in ('0', '1') or (coding_request == '1' and not options.request_pilot):
        raise ValueError('Coding workload requires explicit request-pilot mode')
    if coding_request == '1' and (options.attention_engine or options.attention_engine_wide):
        raise ValueError('Short coding workload is not qualified for long-context attention replay')
    lookup_cap_abba = os.environ.get('QWEN_LOOKUP_CAP_ABBA', '0')
    if lookup_cap_abba not in ('0', '1') or (lookup_cap_abba == '1' and
            (coding_request != '1' or not options.norm_batch)):
        raise ValueError('Lookup cap comparison requires the explicit norm-batched coding request')
    if options.prefix_zero_reuse and (not options.batch or not options.coding_cost or not options.packed_checkpoints
            or any((options.request_pilot, options.replay_inputs, options.deferred_commit,
                options.attribution, options.device_selection, options.target_features, options.target_feature_batch))):
        raise ValueError('Prefix zero reuse requires a standalone static packed-checkpoint coding-cost experiment')
    if options.profile_context is not None and not options.device_profile:
        raise ValueError('Profile context requires device profiling')
    if options.correctness_only and (not options.batch or not options.coding_cost or any((
            options.device_profile, options.attribution, options.deferred_commit, options.replay_inputs,
            options.device_selection, options.request_pilot))):
        raise ValueError('Correctness-only requires a standalone static coding-cost matrix')
    if options.device_profile:
        required = PROFILE_BASE_FLAGS
        if any(getattr(options, name) for name in PROFILE_BEST_FLAGS):
            required = (*required, *PROFILE_BEST_FLAGS)
        allowed = (*required, 'device_profile', 'profile_context', 'max_rows', 'replay_group_rows')
        if not all(getattr(options, name) for name in required) or any(
                value for name, value in vars(options).items() if name not in allowed):
            raise ValueError('Device profiling requires the matched static packed-GDN ordered-cache configuration')
        if not all(os.environ.get(name) == '1' for name in (
                'TTNN_OP_PROFILER', 'TT_METAL_DEVICE_PROFILER', 'TT_METAL_PROFILER_TRACE_TRACKING')):
            raise ValueError('Device profiling requires operation, device and trace tracking profilers')
    if options.target_feature_prefill and not options.target_features:
        raise ValueError('Prefill features require the standalone eager target feature gate')
    if options.target_feature_prefix and not options.target_feature_replay:
        raise ValueError('Feature prefix publication requires feature replay validation')
    if options.target_feature_replay and (not options.replay_inputs or not options.norm_batch or
            options.target_features or options.target_feature_batch):
        raise ValueError('Target feature replay requires standalone retained norm-batch replay')
    if options.target_feature_batch and (not options.batch or not options.norm_batch or options.max_rows != 32 or any(
            (options.target_features, options.request_pilot, options.replay_inputs, options.deferred_commit,
             options.grouped_attention, options.attention_replay, options.attribution, options.device_selection))):
        raise ValueError('Batched target features require standalone T32 static norm-batch verification')
    if options.target_features and (options.max_rows != 16 or options.replay_group_rows != 4 or any(
            value for name, value in vars(options).items() if name not in ('target_features', 'target_feature_prefill', 'max_rows', 'replay_group_rows'))):
        raise ValueError('Target feature gate requires standalone eager native settings')
    if options.attention_engine_wide and (not options.attention_engine or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1'):
        raise ValueError('Wide request comparison requires attention engine and process-fixed compact scratch')
    if options.replay_group_rows == 8 and (not options.attention_replay or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1'):
        raise ValueError('Eight-row replay requires replay attention and process-fixed compact scratch')
    if options.attention_mask_once and not options.attention_replay:
        raise ValueError('Shared attention masks require retained replay certification')
    if options.attention_tree and not options.attention_parallel:
        raise ValueError('Eight-row attention requires parallel attention')
    if options.attention_tree and os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':
        raise ValueError('Eight-row attention requires process-fixed compact native scratch')
    if options.attention_engine and (not options.request_pilot or not options.norm_batch):
        raise ValueError('Attention engine requires the matched norm-batch request pilot')
    if options.attention_replay and (not options.norm_batch or not options.replay_inputs or options.grouped_attention
            or options.device_selection or options.request_pilot or options.attribution):
        raise ValueError('Replay attention is limited to retained norm-batch replay certification')
    if options.attention_parallel and not options.attention_dma:
        raise ValueError('Parallel attention requires DMA layout')
    if options.attention_dma and not options.grouped_attention:
        raise ValueError('Attention DMA requires grouped attention')
    if options.grouped_attention and (not options.norm_batch or options.deferred_commit or options.replay_inputs
            or options.device_selection or options.request_pilot or options.attribution):
        raise ValueError('Grouped attention is limited to static norm-batch correctness and timing')
    if options.norm_batch and (not options.packed_checkpoints or not options.coding_cost or not options.ordered_cache
            or options.attribution
            or (options.deferred_commit and not options.replay_inputs)):
        raise ValueError('Norm batching requires packed static selection or retained-replay verification')
    if options.request_pilot and (not options.device_selection or options.max_rows != 32):
        raise ValueError('Request pilot requires the T32 device-selection configuration')
    if options.device_selection and (not options.coding_cost or not options.packed_checkpoints or not options.ordered_cache
                                    or options.deferred_commit or options.attribution or options.replay_inputs):
        raise ValueError('Device selection requires the standalone static ordered-cache coding-cost gate')
    if options.replay_inputs and (not options.deferred_commit or not options.ordered_cache):
        raise ValueError('Replay gate requires retained histories and ordered cache')
    if options.max_rows == 32 and options.deferred_commit and not options.commit_dma:
        raise ValueError('T32 decisions require simulator-certified DMA publication')
    widths = verification_widths(options.max_rows, packed_checkpoints=options.packed_checkpoints,
        ordered_cache=options.ordered_cache, deferred_commit=options.deferred_commit, attribution=options.attribution)
    if options.coding_cost and not options.batch:
        raise ValueError("Coding cost requires the batched candidate")
    if options.serial_sdpa and not options.batch:
        raise ValueError("Serial SDPA requires the batched candidate")
    if options.attribution and (not options.batch or not options.serial_sdpa or options.coding_cost):
        raise ValueError("Attribution requires batch/B1-SDPA and is separate from the cost matrix")
    if options.compact_gdn and not (options.coding_cost and options.serial_sdpa):
        raise ValueError("Compact GDN requires the coding-context B1-SDPA gate")
    if options.reuse_gdn_input and not options.compact_gdn:
        raise ValueError("GDN input reuse requires compact GDN")
    if options.skip_row_clones and not options.reuse_gdn_input:
        raise ValueError("Clone removal requires reused GDN input")
    if options.hoist_row_layout and not options.skip_row_clones:
        raise ValueError("Layout hoisting requires selective clone removal")
    if options.device_loop_gdn and not options.hoist_row_layout:
        raise ValueError('Device loop requires the previous full row-layout control')
    if options.compact_prologue and not options.device_loop_gdn:
        raise ValueError('Compact prologue requires device-loop GDN')
    if options.batch_conv and not options.compact_prologue:
        raise ValueError('Batched convolution requires compact-prologue control')
    if options.packed_checkpoints and not options.batch_conv:
        raise ValueError('Packed checkpoints require batched convolution')
    if options.deferred_commit and not options.packed_checkpoints:
        raise ValueError('Deferred commit requires packed checkpoints')
    if options.commit_dma and not options.deferred_commit:
        raise ValueError('Fused commit requires post-verification retained records')
    if options.captured_commit and not options.commit_dma:
        raise ValueError('Captured commit requires simulator-certified fused publication')
    if options.ordered_cache and not (options.packed_checkpoints and options.serial_sdpa):
        raise ValueError('Ordered cache requires the packed-history B1 SDPA control')
    if options.ordered_cache:
        from ordered_cache import HASHES as CACHE_HASHES, load_kernels as load_cache_kernels
        cache_kernels = load_cache_kernels('/opt/tt-metal')
    if options.deferred_commit or options.request_pilot:
        sys.path.insert(0, '/experiment-speculative')
        from greedy_verify import select_prefix
    lengths = (4095, 16383) if options.coding_cost or options.attribution else (63, 64, 65)
    if options.profile_context is not None:
        lengths = (options.profile_context,)
    if options.target_feature_batch:
        lengths = (63, 64, 65)
    if options.target_feature_prefill:
        lengths = (63, 64, 65, 127, 128, 129)
    if options.attention_replay:
        lengths = (4096, 16384)
    prefixes = (0, 1, options.max_rows // 2, options.max_rows) if options.coding_cost else tuple(range(options.max_rows + 1))
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    root = Path("/experiment/results")
    report = dict(passed=False, checks=[], negative_controls=[], rows=options.max_rows,
                  scope="Native sequential 64-layer target; active GDN restore and logical KV rollback, no drafter or speed claim")
    report["batched_candidate"] = options.batch
    report['instrumented_timing'] = options.device_profile
    report['correctness_only'] = options.correctness_only
    report['profile_configuration'] = {name: getattr(options, name) for name in (*PROFILE_BASE_FLAGS, *PROFILE_BEST_FLAGS)}
    report['prefix_zero_reuse'] = options.prefix_zero_reuse
    report["serial_sdpa"] = options.serial_sdpa
    report['grouped_attention'] = options.grouped_attention
    report['attention_dma'] = options.attention_dma
    report['attention_parallel'] = options.attention_parallel
    report['attention_tree'] = options.attention_tree
    if options.attention_tree:
        report['attention_tree_layer_prerequisite'] = 34074774493
        report['attention_tree_scope'] = 'Static T4 versus T8 parallel groups; identical native scratch, DMA and GDN'
    report['attention_replay'] = options.attention_replay
    report['attention_mask_once'] = options.attention_mask_once
    report['replay_group_rows'] = options.replay_group_rows
    if options.attention_mask_once:
        report['attention_mask_once_simulator_prerequisite'] = '20260907T030808Z-298'
        report['attention_mask_once_scope'] = f'One shared mask refresh per sixteen-layer forward; {options.replay_group_rows}-row reader, native math unchanged'
    report['attention_engine'] = options.attention_engine
    if options.target_feature_replay:
        report['feature_taps'] = [5, 19, 33, 47, 61]
        report['feature_replay_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('feature_rows.py', 'target_features.py', 'full_replay.py', 'feature_prefix.py')}
    report['attention_engine_wide'] = options.attention_engine_wide
    if options.attention_engine:
        from sdpa_tree_scratch import audit
        report['attention_engine_native_sources'] = audit('/opt/tt-metal', patched=options.attention_engine_wide)
        report['attention_engine_replay_prerequisite'] = 34081751556 if options.attention_engine_wide else 34070163839
        report['attention_engine_scope'] = 'Both request arms use norm batching and identical mask-family proposal limits'
        if options.attention_engine_wide:
            report['attention_engine_scope'] = 'Four versus eight-row replay; both arms share masks and identical compact native scratch'
        report['attention_engine_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('attention_request_plan.py', 'attention_replay.py', 'attention_mask_replay.py',
                         'attention_mask_replay.cpp', 'model_batch.py', 'verifier_engine.py', 'verifier_inputs.py')}
    if options.attention_replay:
        from sdpa_tree_scratch import audit
        report['attention_replay_native_sources'] = audit('/opt/tt-metal', patched=options.replay_group_rows == 8)
        report['attention_replay_prerequisite'] = 34069798251
        report['attention_replay_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('attention_replay.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp', 'attention_parallel.py')}
    if options.attention_parallel:
        from sdpa_tree_scratch import audit
        report['attention_parallel_layer_prerequisite'] = 34067099712
        report['attention_parallel_native_sources'] = audit('/opt/tt-metal', patched=options.attention_tree)
        report['attention_parallel_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('attention_parallel.py', 'attention_grouped.py', 'attention_head_fold.py')}
    if options.attention_dma:
        report.update(attention_dma_layer_prerequisite=34064053339,
            attention_dma_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('attention_fold_dma.py', 'attention_fold_dma.cpp')})
    if options.grouped_attention:
        report.update(grouped_attention_layer_prerequisite=34061952468,
            grouped_attention_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('attention_grouped.py', 'attention_head_fold.py', 'model_batch.py', 'full_batch_timing.py')})
    report["compact_gdn"] = options.compact_gdn
    if options.ordered_cache:
        report.update(ordered_cache=True, cache_native_hashes=CACHE_HASHES,
                      cache_generated_hashes={role: hashlib.sha256(source.encode()).hexdigest() for role, source in cache_kernels.items()},
                      cache_adapter_sha256=hashlib.sha256(Path(__file__).with_name('ordered_cache.py').read_bytes()).hexdigest())
    if options.commit_dma:
        report.update(commit_dma=True, commit_workers_per_chip=96,
            commit_dma_hashes={suffix: hashlib.sha256(Path(__file__).with_name(f'gdn_commit_dma.{suffix}').read_bytes()).hexdigest() for suffix in ('py', 'cpp')})
    report['captured_commit'] = options.captured_commit
    if options.replay_inputs:
        report.update(replay_inputs=True, replay_scope='Forced two-block trace reuse; no drafter or committed throughput',
            replay_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('full_replay.py', 'verifier_inputs.py', 'gdn_records.py')})
    output_path = root / ("full-batch.json" if options.batch else "full-prefix.json")
    if options.coding_cost:
        output_path = root / "full-coding-cost.json"
    if options.compact_gdn:
        output_path = root / "full-compact-gdn.json"
        report["compact_gdn_prerequisite"] = 34005970668
        report["native_t1_state_path"] = True
    if options.reuse_gdn_input:
        output_path = root / "full-gdn-input-reuse.json"
        report["input_reuse_prerequisite"] = 34006233354
        report["reuse_gdn_input"] = True
    if options.skip_row_clones:
        output_path = root / "full-gdn-row-clones.json"
        report.update(skip_row_clones=True, ownership_audit=34009341359)
    if options.hoist_row_layout:
        output_path = root / "full-gdn-row-layout.json"
        report.update(hoist_row_layout=True, layout_prerequisite=34009858516)
    if options.device_loop_gdn:
        output_path = root / 'full-gdn-device-loop.json'
        if options.norm_batch:
            from gdn_vsplit_norm_batch import audit as norm_batch_audit
            report['norm_batch'] = norm_batch_audit(Path('/opt/tt-metal'))
            report['norm_batch_min_rows'] = 8
            report['norm_batch_layer_prerequisite'] = 34051597269
        report.update(device_loop_gdn=True, native_t1_retained=True,
                      device_loop_min_rows=8 if options.compact_prologue and not options.packed_checkpoints else 2,
                      legacy_gdn_flags_describe_paired_control=True,
                      checkpoint_materialization='all internal prefixes; selected external checkpoint')
        if options.compact_prologue:
            report.update(compact_prologue=True, prior_device_loop_run=34023117059,
                checkpoint_materialization='all recurrent prefixes; selected/final convolution prefixes; selected external checkpoint')
        if options.batch_conv:
            report.update(batched_convolution=True, dma_windows=True,
                          prior_full_model_run=34024642720, batched_convolution_prerequisite=34027345128)
        if options.packed_checkpoints:
            report.update(packed_convolution_checkpoints=True, prior_full_model_run=34027510486,
                          packed_checkpoint_prerequisite=34028407207,
                          checkpoint_materialization='all recurrent prefixes; all convolution prefixes in packed windows; selected external checkpoint')
        if options.deferred_commit:
            report.update(deferred_gdn_commit=True, dynamic_commits=[],
                          commit_scope='Post-readback greedy decision or explicit abort, then retained GDN history restore; forced proposal fixtures, not an end-to-end drafter benchmark',
                          component_timing_scope='Readback, greedy selection and GDN commit only; excludes verification, construction and capture',
                          records_sha256=hashlib.sha256(Path(__file__).with_name('gdn_records.py').read_bytes()).hexdigest(),
                          selector_sha256=hashlib.sha256(Path('/experiment-speculative/greedy_verify.py').read_bytes()).hexdigest())
    if options.attribution:
        output_path = root / "full-batch-attribution.json"
    report.update(context_lengths=lengths, rollback_prefixes=prefixes, eligible_for_serving_gate=False)
    if options.batch:
        report["scope"] = "64-layer batched target with static positions, per-layer GDN prefix snapshots and serial shared-page KV writes; no drafter or speed claim"
    if options.coding_cost:
        report["scope"] = "64-layer static coding-context correctness and full-logit block costs; no committed-token throughput"
    if options.deferred_commit:
        report['scope'] = '64-layer forced-draft post-verification acceptance and commit correctness; no actual drafter or committed-throughput measurement'
    if options.attribution:
        report["scope"] = "In-situ fenced eager stage attribution at coding contexts; not critical-path device timing or a throughput gain"
        report["attribution_prerequisite"] = 34002876975
    if options.device_selection:
        report.update(device_selection=True, scope='Paired verifier plus selection/readback; no drafting or dynamic commit',
            selection_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('full_device_selection.py', 'force_argmax.py', 'model_batch.py')})
    mesh = None
    generator = None
    sampler = None
    feature_prefix_pool = {}
    try:
        source = Path("/opt/tt-metal")
        if options.device_loop_gdn:
            from gdn_multitoken import HANDOFF_HASHES, load_kernels, validate_handoff_runtime
            validate_handoff_runtime(source)
            kernels = load_kernels(source, True)
            report.update(full_layer_prerequisite=34022668338, handoff_runtime_hashes=HANDOFF_HASHES,
                generated_hashes={name: hashlib.sha256(value.encode()).hexdigest() for name, value in kernels.items()},
                adapter_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('gdn_device_loop_state.py', 'gdn_multitoken_conv.py')})
            if options.batch_conv:
                report['adapter_hashes'].update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('gdn_batched_conv.py', 'gdn_conv_windows.py', 'gdn_conv_windows.cpp')})
            if options.packed_checkpoints:
                report['adapter_hashes'].update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('gdn_conv_prefix_copy.py', 'gdn_conv_prefix_copy.cpp')})
        if options.skip_row_clones:
            ownership_hashes = {
                "ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp": "817b571dc619eef7af7988ad90e3eda4a89632af3477ca935048b05dc52aea6f",
                "ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp": "6ec5f59e394c9497c9ef87282b67e934be62c5b74c51bc1b88c9789efae38023",
            }
            for name, expected in ownership_hashes.items():
                if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                    raise ValueError("Pinned slice ownership implementation changed")
            report["ownership_source"] = ownership_hashes
        expected_source = {
            "models/demos/blackhole/qwen36/tt/gdn/tp.py": "f767d0648ae01b0b1c0bb7bf601f5490661707b845c11ce6dbebb86ba0f84dc9",
            "models/demos/blackhole/qwen36/tt/qwen36_vllm.py": "cda38c3121b7a61417885469c224c0c69189fda899fbf8361565f4d93125c2fe",
            "models/tt_transformers/tt/generator.py": "4c2633ba8e5e6b0430550ef99409e9a6f0e0a901b4c6627540c579eb9b7d5a3e",
        }
        if options.batch or options.target_features:
            expected_source.update({
                "models/demos/blackhole/qwen36/tt/attention/tp.py": "e0c685a43796f6f8a0ba42fd70a9533b502461b50fdda15e51c8753340f3dc3a",
                "models/demos/blackhole/qwen36/tt/model.py": "c977f3808c39c9dacde5a62a1e30c09dbb55b27d272fecaa9ffea09991270391",
            })
        report["source"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in expected_source}
        if report["source"] != expected_source:
            raise ValueError("Unreviewed native GDN/Generator source")
        report["model_source_sha256"] = hashlib.sha256((source / "models/demos/blackhole/qwen36/tt/model.py").read_bytes()).hexdigest()
        report["copy_kernel_sha256"] = hashlib.sha256(Path(__file__).with_name("gdn_state_copy.cpp").read_bytes()).hexdigest()
        if report["copy_kernel_sha256"] != "4f921500b63817a5f26f288725a9da1fee9223841a68d058d96fac0f67a23428":
            raise ValueError("Direct copy kernel must pass the layer gate first")
        report["prerequisite_run"] = 33999114362
        spec = importlib.util.spec_from_file_location("baseline", Path(__file__).with_name("baseline-client.py"))
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dflash_fixtures = None
        if dflash_drafts != '0':
            from full_dflash_request import load_dflash_fixtures
            print(json.dumps(dict(dflash_stage='verify-complete-checkpoint-before-device-open')), flush=True)
            dflash_fixtures = load_dflash_fixtures('/experiment-dflash-fixture')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1073741824)
        mesh.enable_program_cache()
        if options.target_feature_prefix:
            from feature_prefix import allocate_prefix_pool
            feature_prefix_pool = allocate_prefix_pool(ttnn, lambda prefix: ttnn.from_torch(
                torch.zeros((1, 1, prefix, 5120), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)))
            report['feature_prefix_allocation'] = 'Mesh-open pool before model initialization or any trace capture'
            report['feature_prefix_pool_tensors'] = sum(len(group) for group in feature_prefix_pool.values())
        weights = os.environ["MODEL_WEIGHTS_DIR"]
        config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
        generator = Qwen36ForCausalLM.initialize_vllm_model(config, mesh, max_batch_size=8, max_seq_len=65536)
        model = generator.model[0]
        if len(model.layers) != 64 or model.args.vocab_size != 248320:
            raise ValueError("Expected frozen 64-layer target")
        if options.device_selection:
            import inspect
            from models.common.sampling.generator import SamplingGenerator
            from models.tt_transformers.tt.ccl import TT_CCL
            sampler = SamplingGenerator(args=model.args, mesh_device=mesh, tt_ccl=TT_CCL(mesh))
            sampler.set_trace_bucket(1)
            report['sampling_generator_sha256'] = hashlib.sha256(Path(inspect.getsourcefile(SamplingGenerator)).read_bytes()).hexdigest()
        from dflash_memory import request_kv_allocation
        allocation = request_kv_allocation(dflash_drafts)
        report['kv_allocation'] = allocation
        kv_cache = generator.allocate_kv_cache((allocation['physical_pages'], model.args.n_local_kv_heads, 64, model.args.head_dim),
                                               ttnn.bfloat16, len(model.layers))
        page_table = torch.arange(1024, dtype=torch.int32).reshape(1, 1024)
        layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        caches = [tensor for pair in model._paged_kv_caches for tensor in pair]
        if len(layers) != 48 or len(caches) != 32:
            raise ValueError("Expected 48 GDN layers and 16 attention K/V pairs")
        helpers = [ActiveSnapshot(layer, ttnn, direct=True) for layer in layers]
        saved = [helper.allocate() for helper in helpers]
        scratch = [helper.allocate() for helper in helpers]
        candidate_saved = [helper.allocate() for helper in helpers] if options.batch else []
        replay_initial = [helper.allocate() for helper in helpers] if options.batch else []
        report["snapshot_bytes_per_chip"] = sum(math.prod(tensor.padded_shape) * 2
                                                 for state in saved + scratch + candidate_saved + replay_initial for tensor in state)
        report["kv_dtypes"] = [str(tensor.dtype) for tensor in caches]

        def addresses():
            return [tensor.buffer_address() for layer in layers for tensor in [layer.rec_state, *layer.conv_states]]

        original_addresses = addresses()

        def save(destination):
            for helper, state in zip(helpers, destination, strict=True):
                helper.save(state)

        def restore(source_state=saved):
            if addresses() != original_addresses:
                raise AssertionError("Native state addresses changed")
            for helper, state in zip(helpers, source_state, strict=True):
                helper.restore(state)

        def digest(tensor):
            return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def local_host(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError("Both chips required")
            return [ttnn.to_torch(shard) for shard in shards]

        def state_digest(state):
            return [digest(host) for layer in state for tensor in layer for host in local_host(tensor)]

        def live_digest():
            save(scratch)
            return state_digest(scratch)

        def kv_digest(valid_tokens):
            result = []
            if not 0 < valid_tokens <= 65536:
                raise ValueError("KV prefix exceeds the identity-mapped request allocation")
            for tensor in caches:
                heads, block_size, width = cache_geometry(tensor.shape)
                if block_size != 64:
                    raise ValueError("Expected 64-token KV pages")
                total_pages = math.ceil(valid_tokens / 64)
                for first_page in range(0, total_pages, 64):
                    end_page = min(first_page + 64, total_pages)
                    chunk = ttnn.slice(tensor, (first_page, 0, 0, 0), (end_page, heads, 64, width))
                    for host in local_host(chunk):
                        result.append(digest(logical_kv_chunk(host, first_page, valid_tokens)))
                    ttnn.deallocate(chunk)
            return result

        def inactive_digest():
            return [digest(value[1:] if slot == 0 else value[:, 1:])
                    for helper in helpers for slot, tensor in enumerate(helper.live) for value in local_host(tensor)]

        def decode(token, position, trace, pages=None):
            result = generator.decode_forward(tokens=torch.tensor([[token]], dtype=torch.int32),
                start_pos=torch.tensor([position], dtype=torch.int32), page_table=page_table if pages is None else pages,
                kv_cache=kv_cache, enable_trace=trace, read_from_device=True)
            logits = result[0] if isinstance(result, tuple) else result
            return logits.clone()

        def argmax(logits):
            return int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax())

        def prefill(prompt, *, return_logits=False):
            generator.prev_page_table = None
            logits, _ = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), page_table,
                kv_cache, [len(prompt)], empty_slots=[0],
                enable_trace=not (options.target_features or options.target_feature_batch or mtp_drafts != '0' or dflash_drafts != '0'))
            if addresses() != original_addresses:
                raise AssertionError("Prefill replaced persistent decode state")
            return logits.clone() if return_logits else argmax(logits)

        def batched(tokens, length, prefix, trace, *, deferred=False, abort=False, known_seed=None):
            from model_batch import ModelBatch

            fixture = ModelBatch(model, tokens, length, page_table, helpers, candidate_saved, len(tokens) if deferred else prefix,
                                 serial_sdpa=options.serial_sdpa, compact_gdn=options.compact_gdn,
                                 reuse_gdn_input=options.reuse_gdn_input, skip_row_clones=options.skip_row_clones,
                                 hoist_row_layout=options.hoist_row_layout, device_loop_gdn=options.device_loop_gdn,
                                 compact_prologue=options.compact_prologue, batch_conv=options.batch_conv,
                                 packed_checkpoints=options.packed_checkpoints, retain_records=deferred,
                                 norm_batch=options.norm_batch, grouped_attention=options.grouped_attention,
                                 attention_dma=options.attention_dma, attention_parallel=options.attention_parallel,
                                 attention_tree=options.attention_tree,
                                 prefix_zero_reuse=options.prefix_zero_reuse,
                                 attention_replay=options.attention_replay,
                                 attention_mask_once=options.attention_mask_once,
                                 replay_group_rows=options.replay_group_rows,
                                 ordered_cache=options.ordered_cache)
            captured = None
            output = None
            commit_traces = {}
            setup_ms = None
            try:
                if trace:
                    save(replay_initial)
                    captured, output = capture_operation(ttnn, mesh, fixture.run)
                else:
                    output = fixture.run()
                if captured is not None:
                    restore(replay_initial)
                    ttnn.execute_trace(mesh, captured, cq_id=0, blocking=True)
                if deferred:
                    if options.captured_commit:
                        from gdn_commit_dma import prepare
                        setup_started = time.perf_counter()
                        layers = [[*state.entry, result['states'], *result['packed_conv_states'],
                                   state.gdn.rec_state, *state.gdn.conv_states, *checkpoint]
                                  for state, result, checkpoint in fixture.retained.records]
                        publications = {index: prepare(mesh, layers, index) for index in range(len(tokens) + 1)}
                        for publication in publications.values():
                            publication()
                        ttnn.synchronize_device(mesh)
                        for index, publication in publications.items():
                            commit_traces[index], unused = capture_operation(ttnn, mesh, publication)
                        ttnn.synchronize_device(mesh)
                        setup_ms = (time.perf_counter() - setup_started) * 1000
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                actual = [value.reshape(len(tokens), model.args.vocab_size).clone() for value in local_host(output)]
                if deferred:
                    readback_finished = time.perf_counter()
                    if len(actual) != 2 or not torch.equal(actual[0], actual[1]):
                        raise AssertionError('Both chips must agree before committing')
                    if not abort and tokens[0] != known_seed:
                        raise ValueError('Greedy verification must consume the already-emitted seed')
                    decision = None if abort else select_prefix(tokens[1:], actual[0].float().argmax(dim=-1).tolist(),
                                                                 vocab_size=model.args.vocab_size, max_proposals=options.max_rows - 1)
                    selected = 0 if abort else decision.state_rows
                    selection_finished = time.perf_counter()
                    fixture.retained.commit(selected, dma=options.commit_dma,
                        publication=(lambda index: ttnn.execute_trace(mesh, commit_traces[index], cq_id=0, blocking=True)) if commit_traces else None)
                    ttnn.synchronize_device(mesh)
                    commit_finished = time.perf_counter()
                    report['dynamic_commits'].append(dict(length=length, trace=trace, selected_state_rows=selected,
                        accepted_proposals=0 if abort else decision.accepted, abort=abort,
                        emitted=[] if abort else list(decision.emitted), next_input=known_seed if abort else decision.next_input,
                        readback_select_commit_ms=(commit_finished - started) * 1000,
                        readback_ms=(readback_finished - started) * 1000,
                        selection_ms=(selection_finished - readback_finished) * 1000,
                        commit_ms=(commit_finished - selection_finished) * 1000,
                        commit_setup_ms=setup_ms,
                        retained_layers=len(fixture.retained.records)))
                return actual
            finally:
                for commit_trace in commit_traces.values():
                    ttnn.release_trace(mesh, commit_trace)
                if captured is not None:
                    ttnn.release_trace(mesh, captured)
                if output is not None:
                    ttnn.deallocate(output)
                fixture.close()

        generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=False, max_batch_size=1,
                                      num_blocks=1024, can_sample_on_device=False)
        save(saved)
        save(scratch)
        restore()
        kv_digest(128)
        warm_lengths, warm_widths = lengths, widths
        if mtp_drafts != '0' or dflash_drafts != '0':
            from coding_request import make_prompt
            mtp_prompt = make_prompt(tokenizer)
            warm_lengths = (len(mtp_prompt),)
            warm_widths = tuple(rows for rows in widths if rows <= (int(dflash_drafts) + 1 if dflash_drafts != '0' else 8))
        if options.batch:
            from model_batch import ModelBatch
            save(replay_initial)
            restore(replay_initial)
            for length in warm_lengths:
                kv_digest(length + options.max_rows + 2)
                for rows in warm_widths:
                    fixture = ModelBatch(model, [1] * rows, length, page_table, helpers, candidate_saved, rows,
                                         serial_sdpa=options.serial_sdpa, compact_gdn=options.compact_gdn,
                                         reuse_gdn_input=options.reuse_gdn_input, skip_row_clones=options.skip_row_clones,
                                         hoist_row_layout=options.hoist_row_layout, device_loop_gdn=options.device_loop_gdn,
                                         compact_prologue=options.compact_prologue, batch_conv=options.batch_conv,
                                         packed_checkpoints=options.packed_checkpoints, ordered_cache=options.ordered_cache,
                                         norm_batch=options.norm_batch)
                    output = fixture.run()
                    ttnn.deallocate(output)
                    fixture.close()
        if dflash_drafts != '0':
            from full_dflash_request import warm_dflash_prefill
            report['dflash_prefill_warmup'] = warm_dflash_prefill(ttnn, model, mtp_prompt, prefill)
        elif mtp_drafts == '0':
            generator.warmup_model_prefill(kv_cache=kv_cache, enable_trace=not (options.target_features or options.target_feature_batch))
        else:
            from full_mtp_request import prefill_with_hidden
            print(json.dumps(dict(mtp_stage='warm-prefill-before-native-trace', length=len(mtp_prompt))), flush=True)
            warm_seed = prefill(mtp_prompt)
            captured_seed, warm_hidden = prefill_with_hidden(ttnn, model, mtp_prompt, prefill)
            report['mtp_prefill_warmup'] = dict(native_seed=warm_seed, captured_seed=captured_seed,
                hidden_shape=list(warm_hidden.shape), before_native_trace=True)
            del warm_hidden
            if warm_seed != captured_seed:
                raise AssertionError(f'MTP prefill capture changed the seed before any trace: {warm_seed} != {captured_seed}')
        generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=not (options.target_features or options.target_feature_batch), max_batch_size=1,
                                      num_blocks=1024, can_sample_on_device=False, skip_trace_precompile=True)
        if options.target_feature_batch:
            from full_batched_features import verify_batched_features
            from full_target_features import feature_prompts
            from gdn_multitoken_conv import addresses as tensor_addresses
            output_path = root / 'target-feature-batch.json'
            report.update(scope='Real target serial versus batched feature rows; no drafter or throughput claim',
                feature_checks=[], feature_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('full_batched_features.py', 'full_target_features.py', 'target_features.py', 'model_batch.py')})
            for prompt in feature_prompts(tokenizer, baseline.make_prompt, lengths):
                for rows in (8, 16, 32):
                    result = verify_batched_features(model, prompt, (5, 19, 33, 47, 61), rows,
                        prefill=prefill, decode=decode,
                        batch_decode=lambda tokens, position: batched(tokens, position, len(tokens), False),
                        live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest,
                        snapshot=lambda value: ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                        release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(tensor_addresses(ttnn, value))),
                        local_host=local_host)
                    report['feature_checks'].append(result)
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(result), flush=True)
            report['passed'] = True
            return
        if options.target_features:
            from full_target_features import feature_prompts, verify_features
            from full_prefill_features import verify_prefill_features
            from gdn_multitoken_conv import addresses as tensor_addresses
            output_path = root / 'target-features.json'
            if options.target_feature_prefill:
                output_path = root / 'target-feature-prefill.json'
            report.update(scope='Real target eager B1 feature boundaries; no neural drafter or throughput claim',
                feature_checks=[], feature_sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('full_target_features.py', 'target_features.py', 'full_prefill_features.py')})
            verify = verify_features
            prefill_callback = dict(prefill=prefill)
            if options.target_feature_prefill:
                verify = verify_prefill_features
                prefill_callback = dict(prefill_logits=lambda prompt: prefill(prompt, return_logits=True))
                report['scope'] = 'Real target eager single-chunk prefill features; no neural drafter or throughput claim'
            for prompt in feature_prompts(tokenizer, baseline.make_prompt, lengths):
                result = verify(model, prompt, (5, 19, 33, 47, 61), **prefill_callback, decode=decode,
                    live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest,
                    snapshot=lambda value: ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(tensor_addresses(ttnn, value))),
                    local_host=local_host)
                report['feature_checks'].append(result)
                output_path.write_text(json.dumps(report, indent=2))
                print(json.dumps(result), flush=True)
            report['passed'] = True
            return
        base_prompt = baseline.make_prompt(tokenizer, max(lengths) + 128, 0) if options.coding_cost or options.attribution else baseline.make_prompt(tokenizer, 128, 0)
        timing_fixtures = []
        def measure_fixture(prompt, oracle, rows):
            from full_batch_timing import measure
            return measure(model, oracle[:rows], len(prompt), page_table, helpers, candidate_saved,
                prefill=lambda: prefill(prompt), save_initial=lambda: save(saved), restore_initial=restore,
                state_digest=live_digest, kv_digest=kv_digest, local_host=local_host, serial_sdpa=options.serial_sdpa,
                compact_gdn=options.compact_gdn, checkpoint_digest=lambda: state_digest(candidate_saved),
                reuse_gdn_input=options.reuse_gdn_input, skip_row_clones=options.skip_row_clones,
                hoist_row_layout=options.hoist_row_layout, device_loop_gdn=options.device_loop_gdn,
                compact_prologue=options.compact_prologue, batch_conv=options.batch_conv,
                packed_checkpoints=options.packed_checkpoints, ordered_cache=options.ordered_cache,
                norm_batch=options.norm_batch, grouped_attention=options.grouped_attention,
                attention_dma=options.attention_dma, attention_parallel=options.attention_parallel,
                attention_tree=options.attention_tree, device_profile=options.device_profile,
                prefix_zero_reuse=options.prefix_zero_reuse)

        if coding_request == '1':
            from coding_request import make_prompt
            coding_prompt = make_prompt(tokenizer)
            lengths = (len(coding_prompt),)
            report['coding_task'] = 'merge_intervals_v1'
            report['thinking_enabled'] = False
        for length in lengths:
            prompt = baseline.make_prompt(tokenizer, length, 0) if options.request_pilot else base_prompt[:length]
            if coding_request == '1':
                prompt = coding_prompt
            if not options.request_pilot and len(prompt) != length:
                raise AssertionError("Insufficient fixed prompt tokens")
            if options.request_pilot:
                from full_request import measure_request, terminal_ids
                eos_ids = terminal_ids(weights, model.args.vocab_size)
                report['terminal_ids'] = eos_ids
                report['generation_config_sha256'] = hashlib.sha256((Path(weights) / 'generation_config.json').read_bytes()).hexdigest()
                report.update(scope='Actual lookup request pilot on synthetic repeated code; not a coding-quality benchmark')
                if coding_request == '1':
                    report['scope'] = 'Single non-repeated coding request with lookup drafting; exact native token/state comparison, not held-out coding-quality certification'
                report['request_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                             for name in ('full_request.py', 'verifier_engine.py', 'full_request_pair.py')}

                if dflash_drafts != '0':
                    from full_dflash_request import measure_dflash_request, summarize_dflash_requests, summarize_dflash_commit_requests
                    output_path = root / 'full-dflash-request.json'
                    report.update(scope='Complete five-layer DFlash2 coding request; exact target verification, not held-out quality certification',
                        context_lengths=[len(prompt)], request_checks=[])
                    with sampler_links(sampler.tt_sampling, 4):
                        axis = sampler.tt_sampling._get_sampling_cluster_axis()
                        actual, topology = sampler.tt_sampling._get_force_argmax_all_gather_config(axis)
                        if actual != 4 or topology != ttnn.Topology.Linear:
                            raise AssertionError('DFlash2 target sampler must use four physical-pair links')
                        arms = ((False, True), (False, False), (False, False))
                        if dflash_commit_abba == '1':
                            arms = ((False, True), (True, True), (False, False), (True, False), (True, False), (False, False))
                        for commit_only, feature_audit in arms:
                            print(json.dumps(dict(dflash_stage='complete-request', audit_features=feature_audit,
                                commit_only_gdn=commit_only, repetition=len(report['request_checks']))), flush=True)
                            result = measure_dflash_request(ttnn, model, sampler, prompt, page_table, helpers,
                                fixtures=dflash_fixtures, prefill=prefill, decode=decode, live_digest=live_digest,
                                kv_digest=kv_digest, inactive_digest=inactive_digest, eos_ids=eos_ids,
                                audit_features=feature_audit, block_rows=int(dflash_drafts) + 1,
                                proposal_capture=dflash_capture == '1', commit_only_gdn=commit_only)
                            result.update(kind=report['scope'], coding_task=report['coding_task'],
                                output_text=tokenizer.decode(result['emitted'], skip_special_tokens=False),
                                ended_with_eos=result['emitted'][-1] in eos_ids, sampler_num_links=4,
                                fabric_sources=fabric_sources)
                            report['request_checks'].append(result)
                            output_path.write_text(json.dumps(report, indent=2))
                    summarize = summarize_dflash_commit_requests if dflash_commit_abba == '1' else summarize_dflash_requests
                    report['request_summary'] = summarize(report['request_checks'])
                    report['passed'] = True
                    print(json.dumps(report['request_summary']), flush=True)
                    return

                if mtp_drafts != '0':
                    from full_mtp_request import measure_mtp_request
                    from full_request_pair import summarize_requests
                    output_path = root / 'full-mtp-request.json'
                    report['scope'] = 'MTP device-chained drafting ABBA; both arms retain exact KV-only repair'
                    report['instrumented_timing'] = False
                    report['context_lengths'] = [len(prompt)]
                    report['mtp_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('full_mtp_request.py', 'mtp_device_step.py', 'mtp_prefill.py',
                                     'mtp_request_runtime.py', 'mtp_hidden_capture.py', 'mtp_hidden_rows.py', 'mtp_cache_only.py', 'mtp_device_chain.py',
                                     'force_argmax.py', 'full_request_pair.py', 'attention_request_plan.py')}
                    report['mtp_module_sha256'] = hashlib.sha256(Path('/experiment-speculative/mtp_module.py').read_bytes()).hexdigest()
                    report['request_checks'] = []
                    with sampler_links(sampler.tt_sampling, 4):
                        axis = sampler.tt_sampling._get_sampling_cluster_axis()
                        actual, topology = sampler.tt_sampling._get_force_argmax_all_gather_config(axis)
                        if actual != 4 or topology != ttnn.Topology.Linear:
                            raise AssertionError('MTP request sampler must use four physical-pair links')
                        for device_chain in (False, True, True, False):
                            print(json.dumps(dict(mtp_device_chain=device_chain,
                                repetition=len(report['request_checks']))), flush=True)
                            result = measure_mtp_request(ttnn, model, sampler, prompt, page_table, helpers,
                                weights=weights, prefill=prefill, decode=decode, live_digest=live_digest,
                                kv_digest=kv_digest, inactive_digest=inactive_digest, eos_ids=eos_ids,
                                max_drafts=int(mtp_drafts), native_sampling_rows=True,
                                short_context=True, kv_only_repair=True, device_chain=device_chain)
                            result.update(kind=report['scope'], coding_task=report['coding_task'],
                                output_text=tokenizer.decode(result['emitted'], skip_special_tokens=False),
                                ended_with_eos=result['emitted'][-1] in eos_ids, sampler_num_links=4,
                                fabric_sources=fabric_sources, mtp_device_chain=device_chain)
                            report['request_checks'].append(result)
                            output_path.write_text(json.dumps(report, indent=2))
                            if result['committed_decode_tokens'] == 0:
                                raise AssertionError('MTP request must exercise decode')
                    report['request_summary'] = summarize_requests(report['request_checks'], arm_key='mtp_device_chain')
                    report['passed'] = True
                    print(json.dumps(report['request_summary']), flush=True)
                    return

                def request_measure(*, norm_batch=options.norm_batch, attention_replay=False, attention_wide=False, lookup_cap=False, sampling_links=False):
                    from contextlib import nullcontext
                    links = 4 if sampling_links else 1
                    scope = sampler_links(sampler.tt_sampling, links) if fabric_link_probe == '1' else nullcontext()
                    with scope:
                        engine_factory = None
                        if fabric_link_probe == '1' and sampling_links:
                            from candidate_runtime import CandidateRuntime
                            engine_factory = CandidateRuntime
                        if fabric_link_probe == '1':
                            axis = sampler.tt_sampling._get_sampling_cluster_axis()
                            actual, topology = sampler.tt_sampling._get_force_argmax_all_gather_config(axis)
                            if actual != links or topology != ttnn.Topology.Linear:
                                raise AssertionError('Request sampling link configuration was clamped')
                        result = measure_request(model, sampler, prompt, page_table, helpers, prefill=prefill, decode=decode,
                            live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest, eos_ids=eos_ids,
                            norm_batch=norm_batch, attention_replay=attention_replay or options.attention_engine_wide,
                            max_new_tokens=513 if coding_request == '1' else 129,
                            lookup_max_rows=8 if lookup_cap or fabric_link_probe == '1' else 32,
                            family_routing=options.attention_engine, attention_mask_once=options.attention_engine_wide,
                            replay_group_rows=8 if attention_wide else 4, engine_factory=engine_factory)
                    if fabric_link_probe == '1':
                        result.update(sampling_links=sampling_links, sampler_num_links=links, fabric_sources=fabric_sources)
                        if sampling_links:
                            from candidate_runtime import PROFILE, EVIDENCE_RUN
                            result.update(candidate_profile=PROFILE, candidate_basis_run=EVIDENCE_RUN,
                                candidate_runtime_sha256=hashlib.sha256(Path(__file__).with_name('candidate_runtime.py').read_bytes()).hexdigest())
                    if options.attention_engine_wide:
                        result['attention_wide'] = attention_wide
                    if lookup_cap_abba == '1':
                        result['lookup_cap'] = lookup_cap
                    if coding_request == '1':
                        result['kind'] = report['scope']
                        result['coding_task'] = report['coding_task']
                        result['output_text'] = tokenizer.decode(result['emitted'], skip_special_tokens=False)
                        result['ended_with_eos'] = result['emitted'][-1] in eos_ids
                    report.setdefault('request_checks', []).append(result)
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(result), flush=True)
                    if result['committed_decode_tokens'] == 0:
                        raise AssertionError('Request pilot must exercise decode, not only terminal prefill')
                    return result

                if options.norm_batch:
                    from full_request_pair import measure_requests
                    unused_requests, comparison = measure_requests(request_measure,
                        arm_key='sampling_links' if fabric_link_probe == '1' else 'lookup_cap' if lookup_cap_abba == '1' else 'attention_wide' if options.attention_engine_wide else 'attention_replay' if options.attention_engine else 'norm_batch')
                    comparison['length'] = len(prompt)
                    report.setdefault('request_comparisons', []).append(comparison)
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(comparison), flush=True)
                else:
                    request_measure()
                continue
            oracle = [prefill(prompt)]
            oracle_logits = []
            for position in range(8 if options.device_profile else options.max_rows * (2 if options.replay_inputs else 1) + 2):
                logits = decode(oracle[-1], length + position, False)
                oracle_logits.append(logits)
                oracle.append(argmax(logits))
            timing_fixtures.append((prompt, oracle))
            report.setdefault("prompts", []).append(dict(length=length,
                tokens_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                kind="Truncated deterministic repeated-code fixture, not a coding-quality benchmark"))
            if options.device_profile:
                measurement = measure_fixture(prompt, oracle, 8)
                report.setdefault('timings', []).append(measurement)
                output_path.write_text(json.dumps(report, indent=2))
                print(json.dumps(measurement), flush=True)
                continue
            if options.device_selection:
                from full_device_selection import measure_selection
                for rows in widths:
                    result = measure_selection(model, sampler, prompt, oracle, page_table, helpers, candidate_saved, replay_initial,
                        rows=rows, prefill=prefill, decode=decode, save=save, restore=restore,
                        live_digest=live_digest, kv_digest=kv_digest, local_host=local_host, norm_batch=options.norm_batch)
                    report.setdefault('selection_checks', []).append(result)
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(result), flush=True)
                continue
            if options.attribution:
                from full_batch_attribution import measure
                for rows in (1, 16):
                    result = measure(model, oracle[:rows], length, page_table, helpers, candidate_saved,
                        prefill=lambda: prefill(prompt), save_initial=lambda: save(saved), restore_initial=restore,
                        state_digest=live_digest, kv_digest=kv_digest, local_host=local_host)
                    report.setdefault("attribution", []).append(result)
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(dict(length=length, rows=rows, exact=result["exact"],
                                          trace_median_ms=result["trace_median_ms"],
                                          totals=result["passes"][-1]["totals"])), flush=True)
                continue
            for trace in (False, True):
                prefill(prompt)
                for index, expected in enumerate(oracle_logits):
                    if not torch.equal(decode(oracle[index], length + index, trace), expected):
                        raise AssertionError("Native eager/trace baseline differs")
                report.setdefault("mode_checks", []).append(dict(length=length, trace=trace, logits_exact=True))
                if options.batch:
                    for rows in widths:
                        prefill(prompt)
                        reference_logits = [decode(oracle[index], length + index, trace) for index in range(rows)]
                        expected = torch.cat([active_serial_logits(value, model.args.vocab_size) for value in reference_logits], dim=0)
                        expected_state = live_digest()
                        expected_kv = kv_digest(length + rows)
                        prefill(prompt)
                        actual = batched(oracle[:rows], length, rows, trace)
                        if any(not torch.equal(value, expected) for value in actual):
                            differences = [dict(chip=chip, unequal=int((value != expected).sum()),
                                                nonfinite_actual=int((~torch.isfinite(value)).sum()),
                                                max_abs=float((value.float() - expected.float()).abs().max())
                                                if torch.isfinite(value).all() and torch.isfinite(expected).all() else None)
                                           for chip, value in enumerate(actual)]
                            report["logit_difference"] = dict(length=length, rows=rows, trace=trace, differences=differences)
                            raise AssertionError("Full-model batched logits differ from native B1")
                        if live_digest() != expected_state or kv_digest(length + rows) != expected_kv:
                            raise AssertionError("Full-model batched final state/KV differs")
                        if state_digest(candidate_saved) != expected_state:
                            raise AssertionError("Per-layer end checkpoint differs from global end state")
                        report.setdefault("batch_checks", []).append(dict(length=length, rows=rows, trace=trace,
                            logits_exact=True, all_gdn_states_exact=True, valid_kv_exact=True))
                        output_path.write_text(json.dumps(report, indent=2))
                        print(json.dumps(dict(length=length, rows=rows, trace=trace, batched_exact=True)), flush=True)
                for prefix in prefixes:
                    if prefill(prompt) != oracle[0]:
                        raise AssertionError("Reference prefill seed changed")
                    for index in range(prefix):
                        decode(oracle[index], length + index, trace)
                    save(saved)
                    expected_state = state_digest(saved)
                    expected_kv = kv_digest(length + prefix)
                    expected_logits = [decode(oracle[index], length + index, trace) for index in range(prefix, prefix + 2)]
                    expected_final_state = live_digest()
                    expected_final_kv = kv_digest(length + prefix + 2)

                    if prefill(prompt) != oracle[0]:
                        raise AssertionError("Candidate prefill seed changed")
                    expected_inactive = inactive_digest() if options.commit_dma else None
                    if options.batch:
                        proposals = oracle[:prefix] + [(oracle[index] + 137) % model.args.vocab_size for index in range(prefix, options.max_rows)]
                        actual = batched(proposals, length, prefix, trace, deferred=options.deferred_commit,
                                         abort=prefix == 0, known_seed=oracle[0])
                        if prefix:
                            expected = torch.cat([active_serial_logits(value, model.args.vocab_size) for value in oracle_logits[:prefix]], dim=0)
                            if any(not torch.equal(value[:prefix], expected) for value in actual):
                                raise AssertionError("Rejected future rows changed accepted-prefix logits")
                        if options.deferred_commit:
                            decision = report['dynamic_commits'][-1]
                            if decision['selected_state_rows'] != prefix or decision['next_input'] != oracle[prefix]:
                                raise AssertionError('Post-verification decision differs from forced-rejection oracle')
                            if decision['emitted'] != oracle[1:prefix + 1]:
                                raise AssertionError('Post-verification emission accounting differs from oracle')
                            if state_digest(candidate_saved) != expected_state:
                                raise AssertionError('Deferred external checkpoint differs from selected state')
                        else:
                            restore(candidate_saved)
                    else:
                        for index in range(prefix):
                            decode(oracle[index], length + index, trace)
                        for index in range(prefix, options.max_rows):
                            decode((oracle[index] + 137) % model.args.vocab_size, length + index, trace)
                        restore()
                    if live_digest() != expected_state or kv_digest(length + prefix) != expected_kv:
                        raise AssertionError(f"Rollback state/KV prefix mismatch: {length=} {prefix=} {trace=}")
                    if options.commit_dma:
                        if inactive_digest() != expected_inactive:
                            raise AssertionError('Fused commit changed an inactive native GDN slot')
                        report['dynamic_commits'][-1]['all_inactive_slots_exact'] = True
                    for index, expected in zip(range(prefix, prefix + 2), expected_logits, strict=True):
                        actual = decode(oracle[index], length + index, trace)
                        if not torch.equal(actual, expected):
                            raise AssertionError(f"Full-logit continuation mismatch: {length=} {prefix=} {trace=}")
                    if live_digest() != expected_final_state or kv_digest(length + prefix + 2) != expected_final_kv:
                        raise AssertionError("Corrected continuation state/KV mismatch")
                    report["checks"].append(dict(length=length, prefix=prefix, trace=trace,
                        logits_exact=True, all_gdn_states_exact=True, valid_kv_exact=True, correction_steps=2))
                    if prefix == 0:
                        prefill(prompt)
                        for index in range(options.max_rows):
                            decode((oracle[index] + 137) % model.args.vocab_size, length + index, trace)
                        stale = decode(oracle[0], length, trace)
                        stale_detected = not torch.equal(stale, expected_logits[0])
                        prefill(prompt)
                        wrong_pages = page_table.clone()
                        if options.coding_cost:
                            wrong_pages.fill_(8199)
                        else:
                            wrong_pages[0, 0] = 8199
                        wrong = decode(oracle[0], length, trace, wrong_pages)
                        page_detected = not torch.equal(wrong, expected_logits[0])
                        report["negative_controls"].append(dict(length=length, trace=trace,
                            stale_gdn_detected=stale_detected, wrong_page_detected=page_detected))
                        if not stale_detected or not page_detected:
                            raise AssertionError("Full-model negative control was not detected")
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(dict(length=length, prefix=prefix, trace=trace, exact=True)), flush=True)
        if options.device_profile:
            if addresses() != original_addresses or len(report.get('timings', [])) != len(lengths):
                raise AssertionError('Incomplete stable-state T8 profile contexts')
            report.update(passed=True, timing_scope='Isolated instrumented T8 attribution; broad correctness matrix runs separately')
            return
        if options.request_pilot:
            expected_requests = len(lengths) * (4 if options.norm_batch else 1)
            if addresses() != original_addresses or len(report.get('request_checks', [])) != expected_requests:
                raise AssertionError('Incomplete request pilot matrix')
            if options.norm_batch and len(report.get('request_comparisons', [])) != len(lengths):
                raise AssertionError('Incomplete matched request comparison matrix')
            report['passed'] = True
            return
        if options.device_selection:
            if addresses() != original_addresses or len(report.get('selection_checks', [])) != len(lengths) * len(widths):
                raise AssertionError('Incomplete stable-state device-selection matrix')
            report['passed'] = True
            return
        if options.replay_inputs:
            from full_replay import verify_replay
            replay_widths = (2, 16, 32) if options.max_rows == 32 else (2, 16)
            for prompt, oracle in timing_fixtures:
                for rows in replay_widths:
                    for first_prefix in (0, 1, rows):
                        for second_prefix in (1, rows):
                            result = verify_replay(model, prompt, oracle, page_table, helpers, candidate_saved, replay_initial,
                                rows=rows, first_prefix=first_prefix, second_prefix=second_prefix,
                                prefill=prefill, decode=decode, save=save, restore=restore, state_digest=state_digest,
                                live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest, local_host=local_host,
                                norm_batch=options.norm_batch, attention_replay=options.attention_replay,
                                attention_mask_once=options.attention_mask_once, replay_group_rows=options.replay_group_rows,
                                feature_taps=(5, 19, 33, 47, 61) if options.target_feature_replay else (),
                                feature_publication=options.target_feature_prefix,
                                feature_prefix_buffers=(feature_prefix_pool[0, first_prefix],
                                    feature_prefix_pool[1, second_prefix]) if options.target_feature_prefix else None)
                            report.setdefault('replay_checks', []).append(result)
                            output_path.write_text(json.dumps(report, indent=2))
                            print(json.dumps(result), flush=True)
            if len(report.get('replay_checks', [])) != 6 * len(replay_widths) * len(lengths):
                raise AssertionError('Missing changed-metadata replay cases')
        if addresses() != original_addresses:
            raise AssertionError("Persistent state addresses changed")
        if options.batch and not options.attribution and len(report.get("batch_checks", [])) != len(lengths) * 2 * len(widths):
            raise AssertionError("Missing batched width/mode checks")
        if not options.attribution and len(report["checks"]) != len(lengths) * 2 * len(prefixes):
            raise AssertionError("Missing rollback cases")
        if options.deferred_commit and len(report['dynamic_commits']) != len(lengths) * 2 * len(prefixes):
            raise AssertionError('Missing post-verification commits')
        if options.coding_cost and not options.deferred_commit and not options.correctness_only:
            report["timing_scope"] = "Captured full-logit blocks with one preselected end checkpoint; no drafter, dynamic selection or complete speculative commit pipeline"
            for prompt, oracle in timing_fixtures:
                for rows in widths:
                    measurement = measure_fixture(prompt, oracle, rows)
                    report.setdefault("timings", []).append(measurement)
                    output_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(measurement), flush=True)
            if len(report.get("timings", [])) != len(lengths) * len(widths):
                raise AssertionError("Missing full-model timing fixtures")
            if options.batch:
                from full_matrix import validate_static_matrix
                validate_static_matrix(report, options.max_rows)
        if options.attribution and (len(report.get("attribution", [])) != 4 or
                                   not all(value["exact"] for value in report["attribution"])):
            raise AssertionError("Missing exact attribution fixtures")
        if options.correctness_only:
            report['timing_scope'] = 'Broad correctness matrix only; no timings collected'
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if hasattr(error, 'evidence'):
            report['failure_evidence'] = error.evidence
        raise
    finally:
        output_path.write_text(json.dumps(report, indent=2))
        if sampler is not None:
            sampler.reset_trace()
        if mesh is not None:
            if generator is not None:
                for store in getattr(generator, "_bucket_trace_store", {}).values():
                    for per_device in store[0].values():
                        if per_device:
                            for trace in per_device.values():
                                ttnn.release_trace(mesh, trace)
            for group in feature_prefix_pool.values():
                for value in group:
                    ttnn.deallocate(value)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
