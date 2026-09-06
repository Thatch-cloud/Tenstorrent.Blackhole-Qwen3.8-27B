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
    if max_rows == 32 and (not packed_checkpoints or not ordered_cache or deferred_commit or attribution):
        raise ValueError('T32 requires the static packed-history ordered-cache gate without deferred decisions or attribution')
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
    options = parser.parse_args()
    if options.device_selection and (not options.coding_cost or not options.packed_checkpoints or not options.ordered_cache
                                    or options.deferred_commit or options.attribution or options.replay_inputs):
        raise ValueError('Device selection requires the standalone static ordered-cache coding-cost gate')
    if options.replay_inputs and (options.max_rows != 16 or not options.deferred_commit or not options.ordered_cache):
        raise ValueError('Replay gate requires T16 retained histories and ordered cache')
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
    if options.deferred_commit:
        sys.path.insert(0, '/experiment-speculative')
        from greedy_verify import select_prefix
    lengths = (4095, 16383) if options.coding_cost or options.attribution else (63, 64, 65)
    prefixes = (0, 1, options.max_rows // 2, options.max_rows) if options.coding_cost else tuple(range(options.max_rows + 1))
    import torch
    import ttnn
    from transformers import AutoConfig, AutoTokenizer
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    root = Path("/experiment/results")
    report = dict(passed=False, checks=[], negative_controls=[], rows=options.max_rows,
                  scope="Native sequential 64-layer target; active GDN restore and logical KV rollback, no drafter or speed claim")
    report["batched_candidate"] = options.batch
    report["serial_sdpa"] = options.serial_sdpa
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
        if options.batch:
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
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1073741824)
        mesh.enable_program_cache()
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
        kv_cache = generator.allocate_kv_cache((8200, model.args.n_local_kv_heads, 64, model.args.head_dim),
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

        def prefill(prompt):
            generator.prev_page_table = None
            logits, _ = generator.prefill_forward(torch.tensor([prompt], dtype=torch.int32), page_table,
                kv_cache, [len(prompt)], empty_slots=[0], enable_trace=True)
            if addresses() != original_addresses:
                raise AssertionError("Prefill replaced persistent decode state")
            return argmax(logits)

        def batched(tokens, length, prefix, trace, *, deferred=False, abort=False, known_seed=None):
            from model_batch import ModelBatch

            fixture = ModelBatch(model, tokens, length, page_table, helpers, candidate_saved, len(tokens) if deferred else prefix,
                                 serial_sdpa=options.serial_sdpa, compact_gdn=options.compact_gdn,
                                 reuse_gdn_input=options.reuse_gdn_input, skip_row_clones=options.skip_row_clones,
                                 hoist_row_layout=options.hoist_row_layout, device_loop_gdn=options.device_loop_gdn,
                                 compact_prologue=options.compact_prologue, batch_conv=options.batch_conv,
                                 packed_checkpoints=options.packed_checkpoints, retain_records=deferred,
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
                                                                 vocab_size=model.args.vocab_size)
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
        if options.batch:
            from model_batch import ModelBatch
            save(replay_initial)
            restore(replay_initial)
            for length in lengths:
                kv_digest(length + options.max_rows + 2)
                for rows in widths:
                    fixture = ModelBatch(model, [1] * rows, length, page_table, helpers, candidate_saved, rows,
                                         serial_sdpa=options.serial_sdpa, compact_gdn=options.compact_gdn,
                                         reuse_gdn_input=options.reuse_gdn_input, skip_row_clones=options.skip_row_clones,
                                         hoist_row_layout=options.hoist_row_layout, device_loop_gdn=options.device_loop_gdn,
                                         compact_prologue=options.compact_prologue, batch_conv=options.batch_conv,
                                         packed_checkpoints=options.packed_checkpoints, ordered_cache=options.ordered_cache)
                    output = fixture.run()
                    ttnn.deallocate(output)
                    fixture.close()
        generator.warmup_model_prefill(kv_cache=kv_cache, enable_trace=True)
        generator.warmup_model_decode(kv_cache=kv_cache, enable_trace=True, max_batch_size=1,
                                      num_blocks=1024, can_sample_on_device=False, skip_trace_precompile=True)
        base_prompt = baseline.make_prompt(tokenizer, max(lengths) + 128, 0) if options.coding_cost or options.attribution else baseline.make_prompt(tokenizer, 128, 0)
        timing_fixtures = []
        for length in lengths:
            prompt = base_prompt[:length]
            if len(prompt) != length:
                raise AssertionError("Insufficient fixed prompt tokens")
            oracle = [prefill(prompt)]
            oracle_logits = []
            for position in range(options.max_rows * (2 if options.replay_inputs else 1) + 2):
                logits = decode(oracle[-1], length + position, False)
                oracle_logits.append(logits)
                oracle.append(argmax(logits))
            timing_fixtures.append((prompt, oracle))
            report.setdefault("prompts", []).append(dict(length=length,
                tokens_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                kind="Truncated deterministic repeated-code fixture, not a coding-quality benchmark"))
            if options.device_selection:
                from full_device_selection import measure_selection
                for rows in widths:
                    result = measure_selection(model, sampler, prompt, oracle, page_table, helpers, candidate_saved, replay_initial,
                        rows=rows, prefill=prefill, decode=decode, save=save, restore=restore,
                        live_digest=live_digest, kv_digest=kv_digest, local_host=local_host)
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
        if options.device_selection:
            if addresses() != original_addresses or len(report.get('selection_checks', [])) != len(lengths) * len(widths):
                raise AssertionError('Incomplete stable-state device-selection matrix')
            report['passed'] = True
            return
        if options.replay_inputs:
            from full_replay import verify_replay
            for prompt, oracle in timing_fixtures:
                for rows in (2, 16):
                    for first_prefix in (0, 1, rows):
                        for second_prefix in (1, rows):
                            result = verify_replay(model, prompt, oracle, page_table, helpers, candidate_saved, replay_initial,
                                rows=rows, first_prefix=first_prefix, second_prefix=second_prefix,
                                prefill=prefill, decode=decode, save=save, restore=restore, state_digest=state_digest,
                                live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest, local_host=local_host)
                            report.setdefault('replay_checks', []).append(result)
                            output_path.write_text(json.dumps(report, indent=2))
                            print(json.dumps(result), flush=True)
            if len(report.get('replay_checks', [])) != 12 * len(lengths):
                raise AssertionError('Missing changed-metadata replay cases')
        if addresses() != original_addresses:
            raise AssertionError("Persistent state addresses changed")
        if options.batch and not options.attribution and len(report.get("batch_checks", [])) != len(lengths) * 2 * len(widths):
            raise AssertionError("Missing batched width/mode checks")
        if not options.attribution and len(report["checks"]) != len(lengths) * 2 * len(prefixes):
            raise AssertionError("Missing rollback cases")
        if options.deferred_commit and len(report['dynamic_commits']) != len(lengths) * 2 * len(prefixes):
            raise AssertionError('Missing post-verification commits')
        if options.coding_cost and not options.deferred_commit:
            from full_batch_timing import measure
            report["timing_scope"] = "Captured full-logit blocks with one preselected end checkpoint; no drafter, dynamic selection or complete speculative commit pipeline"
            for prompt, oracle in timing_fixtures:
                for rows in widths:
                    measurement = measure(model, oracle[:rows], len(prompt), page_table, helpers, candidate_saved,
                        prefill=lambda: prefill(prompt), save_initial=lambda: save(saved), restore_initial=restore,
                        state_digest=live_digest, kv_digest=kv_digest, local_host=local_host, serial_sdpa=options.serial_sdpa,
                        compact_gdn=options.compact_gdn, checkpoint_digest=lambda: state_digest(candidate_saved),
                        reuse_gdn_input=options.reuse_gdn_input, skip_row_clones=options.skip_row_clones,
                        hoist_row_layout=options.hoist_row_layout, device_loop_gdn=options.device_loop_gdn,
                        compact_prologue=options.compact_prologue, batch_conv=options.batch_conv,
                        packed_checkpoints=options.packed_checkpoints, ordered_cache=options.ordered_cache)
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
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
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
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
