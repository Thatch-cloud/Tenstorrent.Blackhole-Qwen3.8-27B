"""Real-weight MTP coding request with aligned prompt KV and exact target publication."""

import hashlib
import json
from pathlib import Path
import time

from full_request import measure_request
from gdn_multitoken_conv import addresses, release_owned
from mtp_device_step import MTPDeviceStep
from mtp_prefill import AlignedMTPStep, initialize_prompt
from mtp_request_runtime import MTPRequestRuntime
from target_features import LayerOutputCapture


def validate_prompt_hidden(parts, length):
    import torch

    if (len(parts) != 2 or type(length) is not int or not 0 < length <= 256
            or any(part.device.type != 'cpu' or part.dtype != torch.bfloat16 or part.ndim != 4
                or tuple(part.shape[:2]) != (1, 1) or not length <= part.shape[2] <= 256
                or part.shape[3] != 5120 or not torch.isfinite(part[:, :, :length]).all() for part in parts)
            or parts[0].shape != parts[1].shape
            or not torch.equal(parts[0][:, :, :length], parts[1][:, :, :length])):
        raise ValueError('Complete replicated finite BF16 target prefill rows required on both chips')
    return parts[0].clone()


def load_embedding(weights):
    import torch
    from safetensors import safe_open

    root = Path(weights)
    key = 'model.language_model.embed_tokens.weight'
    index = root / 'model.safetensors.index.json'
    shard = (root / json.loads(index.read_text())['weight_map'][key]).resolve()
    if not shard.is_relative_to(root.resolve()) or shard.suffix != '.safetensors':
        raise ValueError('Embedding shard must belong to the frozen checkpoint')
    with safe_open(str(shard), framework='pt') as source:
        embedding = source.get_tensor(key).to(torch.bfloat16)
    if tuple(embedding.shape) != (248320, 5120):
        raise ValueError('Pinned full embedding geometry required')
    return embedding, dict(embedding_key=key, index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())


def prefill_with_hidden(operations, model, prompt, prefill):
    from models.tt_transformers.tt.common import Mode

    capture = LayerOutputCapture(model, (63,), snapshot=operations.clone,
        release=operations.deallocate,
        storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))
    normalized = None
    try:
        with capture.capture():
            seed = prefill(prompt)
        normalized = model.norm(capture.outputs()[0], mode=Mode.PREFILL)
        parts = [operations.to_torch(part) for part in operations.get_device_tensors(normalized)]
        return seed, validate_prompt_hidden(parts, len(prompt))
    finally:
        if normalized is not None:
            operations.deallocate(normalized)
        capture.close()


def measure_mtp_request(operations, model, sampler, prompt, pages, helpers, *, weights,
                        prefill, decode, live_digest, kv_digest, inactive_digest, eos_ids, max_drafts=7):
    import torch
    from models.tt_transformers.tt.ccl import TT_CCL
    from mtp_module import NAMES, Qwen36MTP, load_mtp_weights

    if (type(max_drafts) is not int or max_drafts not in (1, 3, 7, 15, 31)
            or not 0 < len(prompt) <= 256 or len(model.layers) != 64 or model.num_devices != 2
            or tuple(pages.shape) != (1, 1024) or not torch.equal(pages, torch.arange(1024).reshape(1, 1024))):
        raise ValueError('Pinned short coding request, identity pages and explicit MTP draft width required')
    mesh = model.mesh_device
    owned, prompt_rows = [], []
    step = None
    prefill_calls = 0
    metadata = dict(max_drafts=max_drafts, head='native-full-vocabulary-force-argmax',
                    mtp_weight_names=list(NAMES), kernel_math='reused native MTP, no new kernel math')

    def status(stage, **values):
        print(json.dumps(dict(mtp_stage=stage, **values)), flush=True)

    def captured_prefill(tokens):
        nonlocal prefill_calls
        prefill_calls += 1
        status('native-prefill' if prefill_calls == 1 else 'candidate-prefill')
        if prefill_calls == 1:
            seed = prefill(tokens)
            status('native-prefill-complete', seed=seed)
            return seed
        if prefill_calls != 2:
            raise AssertionError('MTP gate must initialize exactly one fresh candidate request')
        seed, hidden = prefill_with_hidden(operations, model, tokens, prefill)
        prompt_rows.append(hidden)
        metadata['prefill_hidden_shape'] = list(hidden.shape)
        status('candidate-prefill-complete', seed=seed)
        return seed

    def factory():
        nonlocal step
        if prefill_calls != 2 or len(prompt_rows) != 1:
            raise AssertionError('Real MTP preparation requires the fresh candidate prompt features')
        status('load-native-mtp')
        embedding, manifest = load_embedding(weights)
        metadata.update(manifest)
        mtp = Qwen36MTP(mesh, model.args, load_mtp_weights(weights), TT_CCL(mesh))
        owned.extend(mtp.allocate_kv((1024, model.args.n_local_kv_heads, 64, model.args.head_dim), operations.bfloat16))
        mapper = operations.ReplicateTensorToMesh(mesh)
        draft_pages = operations.from_torch(pages, device=mesh, dtype=operations.int32,
            layout=operations.ROW_MAJOR_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        owned.append(draft_pages)
        for _ in range(2):
            owned.append(operations.from_torch(torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16),
                device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        anchor, staged = owned[-2:]
        step = MTPDeviceStep(operations, model, mtp, embedding, draft_pages, sampler)
        status('prepare-native-mtp-traces')
        started = time.perf_counter()
        step.prepare()
        metadata['mtp_trace_setup_ms'] = 1000 * (time.perf_counter() - started)

        def stage_row(row):
            source = operations.from_torch(row.contiguous(), dtype=operations.bfloat16,
                layout=operations.TILE_LAYOUT, mesh_mapper=mapper)
            operations.copy_host_to_device_tensor(source, staged)
            return staged

        status('initialize-aligned-mtp-prompt', rows=len(prompt) - 1)
        started = time.perf_counter()
        metadata['prompt_alignment'] = initialize_prompt(step, prompt, prompt_rows.pop(), anchor,
            stage_row=stage_row, copy_hidden=operations.copy)
        operations.synchronize_device(mesh)
        metadata['mtp_prompt_ms'] = 1000 * (time.perf_counter() - started)
        status('prepare-target-verifier', max_rows=max_drafts + 1)
        return MTPRequestRuntime(AlignedMTPStep(step), anchor, copy_hidden=operations.copy, max_drafts=max_drafts)

    try:
        result = measure_request(model, sampler, prompt, pages, helpers,
            prefill=captured_prefill, decode=decode, live_digest=live_digest, kv_digest=kv_digest,
            inactive_digest=inactive_digest, eos_ids=eos_ids, max_new_tokens=513, norm_batch=True,
            lookup_max_rows=max_drafts + 1, mtp_factory=factory,
            progress=lambda block: status('committed-block', **{key: block[key] for key in (
                'position', 'rows', 'accepted', 'committed', 'draft_ms', 'select_commit_ms', 'cycle_ms')}))
        result['mtp'] = metadata
        result['native_committed_tokens_per_second'] = (
            1000 * result['committed_decode_tokens'] / result['native_decode_ms'] if result['native_decode_ms'] else None)
        result['target_reached'] = bool(result['committed_tokens_per_second'] and result['committed_tokens_per_second'] >= 200)
        result['qualification'] = 'One exact native coding request; not held-out coding quality or sustained service throughput'
        return result
    finally:
        if step is not None:
            step.close()
        release_owned(operations, owned)
