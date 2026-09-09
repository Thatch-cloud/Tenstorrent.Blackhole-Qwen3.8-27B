"""Complete DFlash2 request pilot; exact target checks, not held-out coding quality."""

import hashlib
import faulthandler
import json
import math
from pathlib import Path

from dflash_device import DFlashDevice
from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS
from dflash_prefill_window import PrefillWindowCapture, prefill_window, validate_prefill_chunks
from draft_attention_fixture import load_attention
from draft_convolution_fixture import load_convolution
from draft_mlp_fixture import load_mlp
from draft_projection_full_fixture import load_projection
from draft_remaining_layers_fixture import load_layer
from draft_selector_fixture import load_selector
from gdn_multitoken_conv import addresses
from target_features import LayerOutputCapture


def warm_dflash_prefill(operations, model, prompt, prefill):
    window = prefill_window(len(prompt))
    native_seed = prefill(prompt)
    captured = PrefillWindowCapture(operations, model, len(prompt), TARGET_TAPS)
    try:
        with captured.capture():
            captured_seed = prefill(prompt)
        outputs = captured.outputs()
        if native_seed != captured_seed or any(value.shape[2] != window['rows'] for value in outputs):
            raise AssertionError('DFlash2 prefill capture must preserve the seed and exact valid tail window')
        return dict(native_seed=native_seed, captured_seed=captured_seed, taps=list(TARGET_TAPS),
            shapes=[list(value.shape) for value in outputs], feature_window=window,
            chunks=captured.chunks, before_native_trace=True)
    finally:
        captured.close()


def audit_prefill_assembly(operations, captured):
    import torch

    pieces = [child.outputs() for child in captured.children]
    checks = []
    for index, (tap, feature) in enumerate(zip(TARGET_TAPS, captured.outputs(), strict=True)):
        actual = operations.get_device_tensors(feature)
        parts = [operations.get_device_tensors(piece[index]) for piece in pieces]
        if len(actual) != 2 or any(len(part) != 2 for part in parts):
            raise AssertionError('Both chips required for assembled prefill features')
        for chip, shard in enumerate(actual):
            expected = torch.cat([operations.to_torch(part[chip]) for part in parts], dim=2).contiguous()
            output = operations.to_torch(shard).contiguous()
            if not torch.equal(output.view(torch.int16), expected.view(torch.int16)):
                raise AssertionError('Assembled prefill features differ from their exact ordered chunk snapshots')
            checks.append(dict(tap=tap, chip=chip, **captured.window, exact=True))
    return checks


def summarize_dflash_requests(requests, *, audit_only=False):
    if (type(audit_only) is not bool or len(requests) != (1 if audit_only else 3)
            or [entry.get('instrumented_timing') for entry in requests] != ([True] if audit_only else [True, False, False])):
        raise ValueError('One complete feature audit followed by two uninstrumented requests required')
    reference = requests[0]
    commit_only = reference.get('commit_only_gdn', False)
    fused_convolution = reference.get('fused_convolution', False)
    cache_history = reference.get('cache_history', False)
    cache_projection_capture = reference.get('cache_projection_capture', False)
    live_query_qk = reference.get('live_query_qk', False)
    native_proposal_attention = reference.get('native_proposal_attention', False)
    if (type(commit_only) is not bool or type(fused_convolution) is not bool or type(cache_history) is not bool
            or type(cache_projection_capture) is not bool or (cache_projection_capture and not cache_history)
            or type(live_query_qk) is not bool or (live_query_qk and not cache_history)
            or type(native_proposal_attention) is not bool or (native_proposal_attention and
                (not cache_history or live_query_qk or cache_projection_capture))):
        raise ValueError('Explicit Boolean GDN and convolution arms required')
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size', 'committed_decode_tokens')
    for entry in requests:
        if (any(entry.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact', 'ended_with_eos'))
                or any(entry[key] != reference[key] for key in identity)
                or entry['dflash'].get('block_rows') != reference['dflash'].get('block_rows')
                or entry['dflash'].get('proposal_capture', False) != reference['dflash'].get('proposal_capture', False)
                or entry.get('commit_only_gdn', False) is not commit_only
                or entry.get('fused_convolution', False) is not fused_convolution
                or entry.get('cache_history', False) is not cache_history
                or entry.get('cache_projection_capture', False) is not cache_projection_capture
                or entry.get('live_query_qk', False) is not live_query_qk
                or entry.get('native_proposal_attention', False) is not native_proposal_attention
                or entry.get('selected_drafter') != 'dflash2' or entry.get('sampler_num_links') != 4
                or not entry.get('fabric_sources') or entry['fabric_sources'] != reference.get('fabric_sources')):
            raise ValueError('Complete identical outputs, exact target state and the audited four-link pair required')
        if entry['dflash'].get('cache_projection_replays', 0) != (len(entry['blocks']) if cache_projection_capture else 0):
            raise ValueError('Every committed cached block requires exactly one selected projection replay')
        if live_query_qk and (entry['dflash'].get('live_query_qk') is not True
                or entry['dflash'].get('proposal_capture') is not True or entry['dflash'].get('block_rows') != 8
                or not entry['dflash'].get('proposal_contexts') or type(entry['dflash'].get('validated_live_masks')) is not int
                or entry['dflash'].get('validated_live_masks') != len(entry['dflash']['proposal_contexts'])):
            raise ValueError('Every captured live-query bucket requires a validated mask')
        if native_proposal_attention and (entry['dflash'].get('native_proposal_attention') is not True
                or entry['dflash'].get('proposal_capture') is not True or entry['dflash'].get('block_rows') != 8
                or not entry['dflash'].get('proposal_contexts')
                or type(entry['dflash'].get('validated_native_proposal_masks')) is not int
                or entry['dflash']['validated_native_proposal_masks'] != len(entry['dflash']['proposal_contexts'])
                or entry['dflash'].get('attention') != 'Native BF16 proposal-only attention; not an exact replacement'):
            raise ValueError('Every native proposal bucket requires its own validated mask and explicit approximate policy')
        count = entry['committed_decode_tokens']
        if (type(count) is not int or count <= 0 or count != len(entry['emitted']) - 1
                or sum(block['committed'] for block in entry['blocks']) != count
                or entry['dflash']['committed_feature_rows'] != count
                or entry['dflash']['proposal_calls'] <= 0):
            raise ValueError('Measured decode and complete committed feature accounting required')
        for key in ('decode_ms', 'prefill_ms', 'engine_setup_ms', 'feature_setup_ms', 'prefill_setup_decode_ms'):
            value = entry[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite request and setup measurements required')
        if entry.get('setup_amortized') is not False or entry.get('cross_request_trace_reuse') is not False:
            raise ValueError('Do not hide eager request setup or imply cross-request trace reuse')
        if not entry['instrumented_timing'] and (entry.get('gdn_verify_checks') or entry['dflash'].get('convolution_checks')
                or entry['dflash'].get('cache_checks') or entry['dflash'].get('history_checks')
                or not math.isclose(entry['committed_tokens_per_second'], 1000 * count / entry['decode_ms'], rel_tol=1e-12)):
            raise ValueError('Uninstrumented throughput must describe the measured complete decode')
    checks = reference['dflash']['feature_checks']
    if len(reference['prompt_tokens']) > 2048 or 'prefill_window' in reference['dflash']:
        window = prefill_window(len(reference['prompt_tokens']))
        prefills = reference['dflash'].get('prefill_chunks', [])
        if len(prefills) != 2:
            raise ValueError('Both complete native and candidate prefill chunk records required')
        expected = [dict(chip=chip, **piece, exact=True)
            for chunks in prefills for piece in validate_prefill_chunks(len(reference['prompt_tokens']), chunks)
            for tap in TARGET_TAPS for chip in range(2)]
        assembled = [dict(tap=tap, chip=chip, **window, exact=True)
            for prefill in range(2) for tap in TARGET_TAPS for chip in range(2)]
        if (reference['dflash'].get('prefill_checks') != expected
                or reference['dflash'].get('prefill_assembly_checks') != assembled
                or any(entry['dflash'].get('prefill_window') != window for entry in requests)
                or any(entry['dflash'].get('prefill_chunks') != prefills for entry in requests)
                or any(entry['dflash'].get('prefill_checks') or entry['dflash'].get('prefill_assembly_checks') for entry in requests[1:])):
            raise ValueError('Both audited prefills require exact tail-window snapshots; measured requests must not audit')
    if fused_convolution:
        convolution_checks = reference['dflash'].get('convolution_checks', [])
        expected_convolutions = [(block['position'], layer, chip) for block in reference['blocks']
            if block['source'] == 'dflash2' and block['rows'] > 1
            for layer in range(5) for convolution in range(4) for chip in range(2)]
        if (len(convolution_checks) != 40 * reference['dflash']['proposal_calls']
                or [(check.get('position'), check.get('layer'), check.get('chip')) for check in convolution_checks] != expected_convolutions
                or any(check.get('exact') is not True or check.get('rows') != 32 for check in convolution_checks)):
            raise ValueError('All twenty learned convolutions per proposal require exact composed-path audits on both chips')
    if reference['dflash'].get('proposal_capture'):
        trace_checks = reference['dflash'].get('proposal_trace_checks', [])
        if (len(trace_checks) != reference['dflash']['proposal_calls']
                or any(check.get('exact') is not True for check in trace_checks)
                or any(entry['dflash'].get('proposal_contexts') != reference['dflash'].get('proposal_contexts') for entry in requests)
                or not reference['dflash'].get('proposal_contexts')):
            raise ValueError('Complete same-context eager-versus-trace proposal audit required')
    if cache_history:
        positions = [len(reference['prompt_tokens']), *(block['position'] + block['committed'] for block in reference['blocks'])]
        expected = [dict(position=position, rows=min(position, 2048), layer=layer, head=head, chip=chip, exact=True)
            for position in positions for layer in range(5) for head in ('k', 'v') for chip in range(2)]
        if (reference['dflash'].get('proposal_capture') is not True
                or reference['dflash'].get('cache_checks') != reference['dflash'].get('proposal_trace_checks')
                or reference['dflash'].get('history_checks') != expected):
            raise ValueError('Every cached proposal and committed historical frontier require exact full-projection audits')
    elif any(entry['dflash'].get('cache_checks') or entry['dflash'].get('history_checks') for entry in requests):
        raise ValueError('Uncached control cannot claim cache audits')
    if (reference.get('committed_tokens_per_second') is not None or not checks
            or sum(check['rows'] for check in checks) != 2 * len(TARGET_TAPS) * reference['committed_decode_tokens']
            or any(check['exact'] is not True for check in checks)):
        raise ValueError('Feature audit is correctness evidence, not a throughput sample')
    if commit_only:
        expected = [dict(position=block['position'], rows=block['rows'], unchanged=True)
            for block in reference['blocks'] if block['rows'] > 1]
        if not expected or reference.get('gdn_verify_checks') != expected:
            raise ValueError('Every multirow verification must leave native GDN unchanged before the decision')
    if audit_only:
        return dict(context=len(reference['prompt_tokens']), streams=1, committed_tokens=reference['committed_decode_tokens'],
            scope='Single complete instrumented request audit; not PP or TG', audit_only=True)
    measured = requests[1:]
    tokens = sum(entry['committed_decode_tokens'] for entry in measured)
    decode_ms = sum(entry['decode_ms'] for entry in measured)
    throughput = 1000 * tokens / decode_ms
    context = len(reference['prompt_tokens'])
    prefill_ms = sum(entry['prefill_ms'] for entry in measured)
    benchmark = dict(pp_tokens_per_second=1000 * context * len(measured) / prefill_ms,
        ctx_tokens=context, tg_tokens_per_second=throughput, streams=1,
        mode='Offline single-stream complete coding request',
        verify_rows=reference['dflash'].get('block_rows', 8),
        measured_requests=len(measured),
        committed_decode_tokens_per_request=[entry['committed_decode_tokens'] for entry in measured],
        per_request_pp_tokens_per_second=[1000 * context / entry['prefill_ms'] for entry in measured],
        per_request_tg_tokens_per_second=[1000 * entry['committed_decode_tokens'] / entry['decode_ms'] for entry in measured],
        mean_prefill_ms=prefill_ms / len(measured),
        mean_prefill_setup_decode_ms=sum(entry['prefill_setup_decode_ms'] for entry in measured) / len(measured),
        pp_scope='Target prefill helper including feature capture and first-token selection; excludes draft setup',
        ctx_scope='Actual prompt tokens including chat template, before generation; not maximum cache capacity',
        tg_scope='Committed decode after the prefill seed, including draft, verify/readback and publication; excludes prefill/setup',
        aggregation='Sum of tokens divided by sum of measured time; audit requests excluded')
    return dict(committed_tokens_per_second=throughput,
        per_request_tokens_per_second=[1000 * entry['committed_decode_tokens'] / entry['decode_ms'] for entry in measured],
        committed_tokens=tokens, measured_requests=2, feature_audit_requests=1,
        context=context, streams=1, target_reached=throughput >= 200, benchmark=benchmark,
        block_rows=reference['dflash'].get('block_rows', 8),
        proposal_capture=reference['dflash'].get('proposal_capture', False),
        commit_only_gdn=commit_only,
        fused_convolution=fused_convolution,
        cache_history=cache_history,
        cache_projection_capture=cache_projection_capture,
        live_query_qk=live_query_qk,
        native_proposal_attention=native_proposal_attention,
        prefill_setup_decode_ms=[entry['prefill_setup_decode_ms'] for entry in measured],
        feature_setup_ms=[entry['feature_setup_ms'] for entry in measured],
        scope='Complete coding-request pilot; not a component rate, MTP comparison or held-out quality certification')


def summarize_dflash_commit_requests(requests):
    return summarize_dflash_abba_requests(requests, arm_key='commit_only_gdn')


def summarize_dflash_convolution_requests(requests):
    if any(entry.get('commit_only_gdn') is not True for entry in requests):
        raise ValueError('Convolution ABBA must retain the qualified commit-only verifier in both arms')
    return summarize_dflash_abba_requests(requests, arm_key='fused_convolution')


def summarize_dflash_cache_requests(requests):
    if any(entry.get('commit_only_gdn') is not True or entry.get('fused_convolution') is not True for entry in requests):
        raise ValueError('K/V-cache ABBA must retain commit-only GDN and fused convolution in both arms')
    return summarize_dflash_abba_requests(requests, arm_key='cache_history')


def summarize_dflash_abba_requests(requests, *, arm_key):
    if any(entry.get('native_proposal_attention', False) is not False for entry in requests):
        raise ValueError('Different proposal arithmetic requires its separate policy comparison gate')
    if arm_key not in ('commit_only_gdn', 'fused_convolution', 'cache_history', 'cache_projection_capture', 'live_query_qk', 'target_four_links'):
        raise ValueError('Explicit isolated DFlash2 experiment arm required')
    if (len(requests) != 6
            or [entry.get(arm_key) for entry in requests] != [False, True, False, True, True, False]
            or [entry.get('instrumented_timing') for entry in requests] != [True, True, False, False, False, False]):
        raise ValueError('Two arm audits followed by measured control/candidate/candidate/control requests required')
    reference = requests[0]
    identity = ('prompt_tokens', 'emitted', 'max_new_tokens', 'eos_ids', 'vocab_size', 'committed_decode_tokens',
        'fabric_sources', 'sources')
    block_identity = ('rows', 'source', 'accepted', 'match_length', 'position', 'input_tokens', 'committed')
    draft_identity = ('checkpoints', 'block_rows', 'proposal_capture', 'proposal_contexts', 'proposal_calls',
        'committed_feature_rows', 'target_taps', 'policy')
    if reference['dflash'].get('proposal_capture') is not True or reference['dflash'].get('block_rows') != 8:
        raise ValueError('Commit-only comparison requires the qualified captured T8 proposer')
    expected_blocks = [tuple(block[key] for key in block_identity) for block in reference['blocks']]
    for entry in requests:
        if (any(entry[key] != reference[key] for key in identity)
                or any(entry.get(key, False) is not reference.get(key, False)
                    for key in ('commit_only_gdn', 'fused_convolution', 'cache_history', 'cache_projection_capture', 'live_query_qk', 'target_four_links') if key != arm_key)
                or any(entry['dflash'][key] != reference['dflash'][key] for key in draft_identity)
                or [tuple(block[key] for key in block_identity) for block in entry['blocks']] != expected_blocks):
            raise ValueError('Matched ABBA requires identical inputs, proposals, acceptance, sources and four-link configuration')
    control = summarize_dflash_requests([requests[index] for index in (0, 2, 5)])
    candidate = summarize_dflash_requests([requests[index] for index in (1, 3, 4)])
    return dict(control=control, candidate=candidate, measured_order=['control', 'candidate', 'candidate', 'control'],
        committed_tokens_per_second=candidate['committed_tokens_per_second'],
        candidate_over_control=candidate['committed_tokens_per_second'] / control['committed_tokens_per_second'],
        target_reached=candidate['target_reached'],
        scope='Matched complete coding-request ABBA; audit requests excluded; not held-out coding-quality certification')


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


def summarize_dflash_projection_requests(requests):
    if any(any(entry.get(key) is not True for key in ('commit_only_gdn', 'fused_convolution', 'cache_history'))
            for entry in requests):
        raise ValueError('Projection ABBA must retain cached K/V, commit-only GDN and fused convolution in both arms')
    return summarize_dflash_abba_requests(requests, arm_key='cache_projection_capture')


def summarize_dflash_live_query_requests(requests):
    if any(any(entry.get(key) is not True for key in ('commit_only_gdn', 'fused_convolution', 'cache_history'))
            or entry.get('cache_projection_capture', False) is not False for entry in requests):
        raise ValueError('Live-query ABBA must retain the cached T8 lead without projection-capture changes')
    return summarize_dflash_abba_requests(requests, arm_key='live_query_qk')


def measure_dflash_request(operations, model, sampler, prompt, pages, helpers, *, fixtures,
                          prefill, decode, live_digest, kv_digest, inactive_digest, eos_ids,
                          audit_features=False, max_new_tokens=513, block_rows=8, proposal_capture=False,
                          commit_only_gdn=False, fused_convolution=False, cache_history=False, cache_projection_capture=False,
                          profile_verifier=False, live_query_qk=False, native_proposal_attention=False):
    import torch
    from full_request import measure_request
    from models.tt_transformers.tt.ccl import TT_CCL

    if (type(audit_features) is not bool or type(proposal_capture) is not bool or type(commit_only_gdn) is not bool
            or type(fused_convolution) is not bool or (fused_convolution and not proposal_capture)
            or type(cache_history) is not bool or (cache_history and (not proposal_capture or block_rows != 8))
            or type(cache_projection_capture) is not bool or (cache_projection_capture and not cache_history)
            or type(live_query_qk) is not bool or (live_query_qk and (not cache_history or cache_projection_capture))
            or type(native_proposal_attention) is not bool or (native_proposal_attention and
                (not cache_history or not commit_only_gdn or not fused_convolution or live_query_qk
                 or cache_projection_capture or profile_verifier or len(prompt) != 4096))
            or type(profile_verifier) is not bool or (profile_verifier and (not audit_features or not commit_only_gdn
                or not proposal_capture or not fused_convolution or not cache_history or cache_projection_capture
                or block_rows != 8 or len(prompt) != 4096 or live_query_qk))
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 65536 - len(prompt) - 32
            or type(block_rows) is not int or block_rows not in (8, 32)):
        raise ValueError('Explicit feature-audit policy and bounded prompt required')
    window = prefill_window(len(prompt))
    manifests, layers, projection, selector = fixtures
    capture = device = runtime = None
    prefill_count = 0
    golden_features = {}
    feature_checks = []
    prefill_checks = []
    prefill_chunks = []
    prefill_assembly_checks = []

    def status(stage, **values):
        if stage == 'committed-block':
            faulthandler.dump_traceback_later(180, exit=True)
        print(json.dumps(dict(dflash_stage=stage, **values)), flush=True)

    def new_capture():
        return LayerOutputCapture(model, TARGET_TAPS,
            snapshot=lambda value: operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG),
            release=operations.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))

    def captured_prefill(tokens):
        nonlocal capture, prefill_count
        if capture is not None:
            capture.close()
        capture = PrefillWindowCapture(operations, model, len(prompt), TARGET_TAPS,
            checks=prefill_checks if audit_features else None)
        with capture.capture():
            seed = prefill(tokens)
        capture.outputs()
        if audit_features:
            prefill_assembly_checks.extend(audit_prefill_assembly(operations, capture))
        prefill_chunks.append(list(capture.chunks))
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
        faulthandler.dump_traceback_later(180, exit=True)
        device = DFlashDevice(operations, model, TT_CCL(model.mesh_device), layers, projection, selector,
            capture.outputs(), position=len(prompt), progress=status if audit_features else None, block_rows=block_rows,
            proposal_capture=proposal_capture, max_new_tokens=max_new_tokens, fused_convolution=fused_convolution,
            feature_start=window['start'], cache_history=cache_history, cache_projection_capture=cache_projection_capture,
            live_query_qk=live_query_qk, native_proposal_attention=native_proposal_attention)
        capture.close()
        runtime = DFlashRequestRuntime(device, position=len(prompt),
            validate_features=validate_features if audit_features else None)
        faulthandler.dump_traceback_later(180, exit=True)
        return runtime

    observer = None
    try:
        if profile_verifier:
            from request_verifier_profile import RequestVerifierProfile

            observer = RequestVerifierProfile(operations, model.mesh_device)
        result = measure_request(model, sampler, prompt, pages, helpers,
            prefill=captured_prefill, decode=gold_decode, live_digest=live_digest, kv_digest=kv_digest,
            inactive_digest=inactive_digest, eos_ids=eos_ids, max_new_tokens=max_new_tokens,
            norm_batch=True, lookup_max_rows=block_rows, native_sampling_rows=True,
            commit_only_gdn=commit_only_gdn, audit_commit_only_gdn=commit_only_gdn and audit_features,
            feature_factory=factory, progress=lambda block: status('committed-block', **block),
            **(dict(verifier_observer=observer) if observer is not None else {}))
        if observer is not None:
            result['verifier_profile'] = observer.summary()
        result['fused_convolution'] = fused_convolution
        result['cache_history'] = cache_history
        result['cache_projection_capture'] = cache_projection_capture
        result['live_query_qk'] = live_query_qk
        result['native_proposal_attention'] = native_proposal_attention
        result['dflash'] = dict(checkpoints=manifests, target_taps=list(TARGET_TAPS),
            prefill_window=window, prefill_checks=prefill_checks, prefill_chunks=prefill_chunks,
            prefill_assembly_checks=prefill_assembly_checks,
            convolution_checks=device.convolution_checks,
            cache_checks=device.proposal_capture.cache_checks if proposal_capture else [],
            history_checks=device.kv_history.checks if device.kv_history is not None else [],
            cache_projection_replays=device.kv_history.projection.calls if cache_projection_capture else 0,
            live_query_qk=live_query_qk, validated_live_masks=len(device.validated_live_masks),
            native_proposal_attention=native_proposal_attention,
            validated_native_proposal_masks=len(device.validated_native_proposal_masks),
            block_rows=block_rows, max_drafts=block_rows - 1, mask_token_id=248070,
            checkpoint_trained_block_rows=8, block_width_extrapolation=block_rows != 8,
            policy='Five learned BF16 layers, shared target head top16 and CPU FP64 learned greedy selector',
            attention='Native BF16 proposal-only attention; not an exact replacement' if native_proposal_attention
                else 'Composed precise control; unqualified native SDPA is not enabled',
            head_layout='Native QKV head split and concatenation; no generic sub-tile reshape',
            feature_history='Two preallocated 2048-row buffers; only committed prefixes are projected and published',
            execution='Eager request integration; setup, dispatch and compilation costs are not amortized',
            proposal_capture=proposal_capture,
            proposal_contexts=list(device.proposal_capture.buckets) if proposal_capture else [],
            proposal_trace_checks=device.proposal_capture.checks if proposal_capture else [],
            proposal_calls=device.proposal_calls if device is not None else 0,
            committed_feature_rows=runtime.committed_feature_rows if runtime is not None else 0,
            feature_checks=feature_checks, audit_features=audit_features)
        result['dflash']['synchronized_stage_diagnostics'] = audit_features
        if proposal_capture:
            result['dflash']['execution'] = 'Fixed-context captured five-layer proposals; per-request setup is not amortized'
            if audit_features and len(device.proposal_capture.checks) != device.proposal_calls:
                raise AssertionError('Every audited proposal requires exact eager-versus-trace checks')
        result['instrumented_timing'] = audit_features
        if audit_features:
            result['committed_tokens_per_second'] = None
        result['target_reached'] = bool(result['committed_tokens_per_second'] and result['committed_tokens_per_second'] >= 200)
        result['qualification'] = __doc__
        result['sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('full_dflash_request.py', 'dflash_device.py', 'dflash_request_runtime.py', 'prepared_target_features.py',
                         'draft_head_layout.py', 'draft_attention_branch.py', 'draft_mlp_branch.py', 'draft_shared_head.py',
                         'draft_selector.py', 'dflash_proposal_inputs.py', 'dflash_proposal_trace.py',
                         'draft_kv_projection.py', 'draft_kv_projection_trace.py', 'draft_kv_history.py', 'request_verifier_profile.py',
                         'dflash_prefill_window.py', 'coding_context_request.py',
                         'gdn_device_loop_state.py', 'model_batch.py', 'verifier_engine.py', 'full_request.py',
                         'draft_convolution.py', 'draft_convolution_fused.py',
                         'draft_convolution_fused_compute.cpp', 'draft_convolution_fused_io.cpp')}
        from live_attention_gate import source_hashes

        result['sources'].update(source_hashes())
        from proposal_native_request import source_hashes as proposal_source_hashes

        result['sources'].update(proposal_source_hashes())
        return result
    finally:
        try:
            if device is not None:
                device.close()
            if capture is not None:
                capture.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
