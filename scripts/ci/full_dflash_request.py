"""Complete DFlash2 request pilot; exact target checks, not held-out coding quality."""

import hashlib
import json
import math
from pathlib import Path

from dflash_device import DFlashDevice
from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS
from draft_attention_fixture import load_attention
from draft_convolution_fixture import load_convolution
from draft_mlp_fixture import load_mlp
from draft_projection_full_fixture import load_projection
from draft_remaining_layers_fixture import load_layer
from draft_selector_fixture import load_selector
from full_request import measure_request
from gdn_multitoken_conv import addresses
from target_features import LayerOutputCapture


def warm_dflash_prefill(operations, model, prompt, prefill):
    native_seed = prefill(prompt)
    captured = LayerOutputCapture(model, TARGET_TAPS,
        snapshot=lambda value: operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG),
        release=operations.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))
    try:
        with captured.capture():
            captured_seed = prefill(prompt)
        outputs = captured.outputs()
        if native_seed != captured_seed or any(value.shape[2] < len(prompt) for value in outputs):
            raise AssertionError('DFlash2 prefill capture must preserve seed and all prompt rows')
        return dict(native_seed=native_seed, captured_seed=captured_seed, taps=list(TARGET_TAPS),
            shapes=[list(value.shape) for value in outputs], before_native_trace=True)
    finally:
        captured.close()


def summarize_dflash_requests(requests):
    if len(requests) != 3 or [entry.get('instrumented_timing') for entry in requests] != [True, False, False]:
        raise ValueError('One complete feature audit followed by two uninstrumented requests required')
    reference = requests[0]
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size', 'committed_decode_tokens')
    for entry in requests:
        if (any(entry.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact', 'ended_with_eos'))
                or any(entry[key] != reference[key] for key in identity)
                or entry.get('selected_drafter') != 'dflash2' or entry.get('sampler_num_links') != 4
                or not entry.get('fabric_sources') or entry['fabric_sources'] != reference.get('fabric_sources')):
            raise ValueError('Complete identical outputs, exact target state and the audited four-link pair required')
        count = entry['committed_decode_tokens']
        if (type(count) is not int or count <= 0 or count != len(entry['emitted']) - 1
                or sum(block['committed'] for block in entry['blocks']) != count
                or entry['dflash']['committed_feature_rows'] != count
                or entry['dflash']['proposal_calls'] <= 0):
            raise ValueError('Measured decode and complete committed feature accounting required')
        for key in ('decode_ms', 'prefill_ms', 'engine_setup_ms', 'feature_setup_ms'):
            value = entry[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite request and setup measurements required')
        if entry.get('setup_amortized') is not False or entry.get('cross_request_trace_reuse') is not False:
            raise ValueError('Do not hide eager request setup or imply cross-request trace reuse')
    checks = reference['dflash']['feature_checks']
    if (reference.get('committed_tokens_per_second') is not None or not checks
            or sum(check['rows'] for check in checks) != 2 * len(TARGET_TAPS) * reference['committed_decode_tokens']
            or any(check['exact'] is not True for check in checks)):
        raise ValueError('Feature audit is correctness evidence, not a throughput sample')
    measured = requests[1:]
    tokens = sum(entry['committed_decode_tokens'] for entry in measured)
    decode_ms = sum(entry['decode_ms'] for entry in measured)
    throughput = 1000 * tokens / decode_ms
    return dict(committed_tokens_per_second=throughput,
        per_request_tokens_per_second=[1000 * entry['committed_decode_tokens'] / entry['decode_ms'] for entry in measured],
        committed_tokens=tokens, measured_requests=2, feature_audit_requests=1,
        context=len(reference['prompt_tokens']), streams=1, target_reached=throughput >= 200,
        prefill_setup_decode_ms=[entry['prefill_setup_decode_ms'] for entry in measured],
        feature_setup_ms=[entry['feature_setup_ms'] for entry in measured],
        scope='Complete coding-request pilot; not a component rate, MTP comparison or held-out quality certification')


def load_dflash_fixtures(root):
    root = Path(root)
    attention_manifest, attention = load_attention(root / 'attention')
    convolution_manifest, convolution = load_convolution(root / 'convolution')
    mlp_manifest, mlp = load_mlp(root / 'mlp')
    projection_manifest, projection, norm = load_projection(root / 'projection')
    selector_manifest, selector = load_selector(root / 'selector')
    manifests = dict(attention=attention_manifest, convolution=convolution_manifest, mlp=mlp_manifest,
        projection=projection_manifest, selector=selector_manifest, layers=[])
    layers = [(attention, convolution, mlp)]
    for layer in range(1, 5):
        manifest, weights = load_layer(root / f'layer-{layer}', layer)
        normalized = {name.replace(f'layers.{layer}.', 'layers.0.', 1): value for name, value in weights.items()}
        layers.append((normalized, normalized, normalized))
        manifests['layers'].append(manifest)
    return manifests, layers, {'fc.weight': projection, 'hidden_norm.weight': norm}, selector


def measure_dflash_request(operations, model, sampler, prompt, pages, helpers, *, fixtures,
                          prefill, decode, live_digest, kv_digest, inactive_digest, eos_ids,
                          audit_features=False, max_new_tokens=513):
    import torch
    from models.tt_transformers.tt.ccl import TT_CCL

    if type(audit_features) is not bool or len(prompt) > 2048:
        raise ValueError('Explicit feature-audit policy and bounded prompt required')
    manifests, layers, projection, selector = fixtures
    capture = device = runtime = None
    prefill_count = 0
    golden_features = {}
    feature_checks = []

    def status(stage, **values):
        print(json.dumps(dict(dflash_stage=stage, **values)), flush=True)

    def new_capture():
        return LayerOutputCapture(model, TARGET_TAPS,
            snapshot=lambda value: operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG),
            release=operations.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))

    def captured_prefill(tokens):
        nonlocal capture, prefill_count
        if capture is not None:
            capture.close()
        capture = new_capture()
        with capture.capture():
            seed = prefill(tokens)
        prefill_count += 1
        return seed

    def gold_decode(token, position, traced):
        if not audit_features:
            return decode(token, position, traced)
        observed = new_capture()
        try:
            with observed.capture():
                output = decode(token, position, False)
            golden_features[position] = tuple(tuple(operations.to_torch(shard).clone()
                for shard in operations.get_device_tensors(value)) for value in observed.outputs())
            return output
        finally:
            observed.close()

    def validate_features(features, prefix, position):
        for offset, (tap, feature) in enumerate(zip(TARGET_TAPS, features, strict=True)):
            shards = operations.get_device_tensors(feature)
            if len(shards) != 2:
                raise AssertionError('Both target-feature shards required')
            for chip, shard in enumerate(shards):
                expected = torch.cat([golden_features[position + row][offset][chip] for row in range(prefix)], dim=2)
                actual = operations.to_torch(shard)[..., :prefix, :]
                if not torch.equal(actual, expected):
                    raise AssertionError(f'Committed target-feature mismatch: tap={tap}, chip={chip}, position={position}')
                feature_checks.append(dict(tap=tap, chip=chip, position=position, rows=prefix, exact=True))

    def factory():
        nonlocal device, runtime
        if prefill_count != 2 or capture is None:
            raise ValueError('DFlash2 setup requires fresh candidate prefill features')
        status('prepare-five-layer-device-drafter')
        device = DFlashDevice(operations, model, TT_CCL(model.mesh_device), layers, projection, selector,
            capture.outputs(), position=len(prompt))
        capture.close()
        runtime = DFlashRequestRuntime(device, position=len(prompt),
            validate_features=validate_features if audit_features else None)
        return runtime

    try:
        result = measure_request(model, sampler, prompt, pages, helpers,
            prefill=captured_prefill, decode=gold_decode, live_digest=live_digest, kv_digest=kv_digest,
            inactive_digest=inactive_digest, eos_ids=eos_ids, max_new_tokens=max_new_tokens,
            norm_batch=True, lookup_max_rows=8, native_sampling_rows=True,
            feature_factory=factory, progress=lambda block: status('committed-block', **block))
        result['dflash'] = dict(checkpoints=manifests, target_taps=list(TARGET_TAPS),
            block_rows=8, max_drafts=7, mask_token_id=248070,
            policy='Five learned BF16 layers, shared target head top16 and CPU FP64 learned greedy selector',
            attention='Composed precise control; unqualified native SDPA is not enabled',
            feature_history='Two preallocated 2048-row buffers; only committed prefixes are projected and published',
            execution='Eager request integration; setup, dispatch and compilation costs are not amortized',
            proposal_calls=device.proposal_calls if device is not None else 0,
            committed_feature_rows=runtime.committed_feature_rows if runtime is not None else 0,
            feature_checks=feature_checks, audit_features=audit_features)
        result['instrumented_timing'] = audit_features
        if audit_features:
            result['committed_tokens_per_second'] = None
        result['target_reached'] = bool(result['committed_tokens_per_second'] and result['committed_tokens_per_second'] >= 200)
        result['qualification'] = __doc__
        result['sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('full_dflash_request.py', 'dflash_device.py', 'dflash_request_runtime.py', 'prepared_target_features.py')}
        return result
    finally:
        if device is not None:
            device.close()
        if capture is not None:
            capture.close()
