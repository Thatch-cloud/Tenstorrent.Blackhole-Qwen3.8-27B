"""Complete full-history DSpark coding-request experiment; target exactness, not held-out quality certification."""

import json

from dspark_device import DSparkDevice
from dspark_intake import TAPS
from dspark_prefill import FullHistoryCapture
from dspark_projection import tensor_digest
from dspark_request_runtime import DSparkRequestRuntime
from gdn_multitoken_conv import addresses
from target_features import LayerOutputCapture


def measure_dspark_request(operations, model, sampler, prompt, pages, helpers, *, collectives,
        parameters, layer_weights, predecessor, successor, rotary, prefill, decode,
        live_digest, kv_digest, inactive_digest, eos_ids, audit_features=False, max_new_tokens=257):
    import torch
    from full_request import measure_request

    if (type(audit_features) is not bool or type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 513
            or not 1 <= len(prompt) <= 8192 - max_new_tokens):
        raise ValueError('Explicit audit policy and full-history capacity for the complete request required')
    capture = drafter = runtime = None
    golden_features, prefill_hashes = {}, None
    prefill_records, feature_checks = [], []
    seed = None

    def status(stage, **values):
        print(json.dumps(dict(dspark_stage=stage, **values)), flush=True)

    def captured_prefill(tokens):
        nonlocal capture, seed, prefill_hashes
        if capture is not None:
            capture.close()
        capture = FullHistoryCapture(operations, model, len(prompt))
        status('prefill', ordinal=len(prefill_records), context=len(prompt), audit=audit_features)
        with capture.capture():
            seed = prefill(tokens)
        chunks = capture.outputs()
        records = [dict(start=chunk.start, rows=chunk.rows, bucket=chunk.features[0].shape[2]) for chunk in chunks]
        if audit_features:
            observed = []
            for chunk in chunks:
                for tap, value in zip(TAPS, chunk.features, strict=True):
                    shards = operations.get_device_tensors(value)
                    if len(shards) != 2:
                        raise AssertionError('Both actual prefill feature shards required')
                    for chip, shard in enumerate(shards):
                        observed.append(dict(start=chunk.start, rows=chunk.rows, tap=tap, chip=chip,
                            sha256=tensor_digest(operations.to_torch(shard)[..., :chunk.rows, :])))
            if prefill_hashes is not None and observed != prefill_hashes:
                raise AssertionError('Fresh candidate prefill features differ from native controls')
            prefill_hashes = observed
        prefill_records.append(records)
        return seed

    def gold_decode(token, position, traced):
        if not audit_features:
            return decode(token, position, traced)
        observed = LayerOutputCapture(model, TAPS,
            snapshot=lambda value: operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG),
            release=operations.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))
        try:
            with observed.capture():
                output = decode(token, position, False)
            features = []
            for value in observed.outputs():
                shards = operations.get_device_tensors(value)
                if len(shards) != 2:
                    raise AssertionError('Both native decode feature shards required')
                features.append(tuple(operations.to_torch(shard).clone() for shard in shards))
            golden_features[position] = tuple(features)
            return output
        finally:
            observed.close()

    def validate_features(features, prefix, position):
        for index, (tap, value) in enumerate(zip(TAPS, features, strict=True)):
            shards = operations.get_device_tensors(value)
            if len(shards) != 2:
                raise AssertionError('Both verifier feature shards required')
            for chip, shard in enumerate(shards):
                expected = torch.cat([golden_features[position + row][index][chip] for row in range(prefix)], dim=2)
                actual = operations.to_torch(shard)[..., :prefix, :]
                if not torch.equal(actual, expected):
                    raise AssertionError(f'DSpark committed feature mismatch: tap={tap}, chip={chip}, position={position}')
                feature_checks.append(dict(tap=tap, chip=chip, position=position, rows=prefix, exact=True))

    def factory():
        nonlocal drafter, runtime
        status('project_full_prefill_history', context=len(prompt))
        drafter = DSparkDevice(operations, model, collectives, parameters, layer_weights, predecessor, successor,
            capture.outputs(), rotary, position=len(prompt), proposals=15)
        capture.close()
        status('warm_fifteen_query_proposal_before_verifier_capture')
        drafter.propose(seed, 15)
        runtime = DSparkRequestRuntime(drafter, position=len(prompt), validate_features=validate_features if audit_features else None)
        return runtime

    try:
        result = measure_request(model, sampler, prompt, pages, helpers, prefill=captured_prefill, decode=gold_decode,
            live_digest=live_digest, kv_digest=kv_digest, inactive_digest=inactive_digest, eos_ids=eos_ids,
            max_new_tokens=max_new_tokens, norm_batch=True, native_sampling_rows=True,
            lookup_max_rows=16, feature_factory=factory, feature_drafter_name='dspark',
            progress=lambda block: status('committed-block', **block))
        expected_checks = len(result['blocks']) * len(TAPS) * 2 if audit_features else 0
        if (runtime is None or len(prefill_records) != 2 or len(feature_checks) != expected_checks
                or runtime.committed_feature_rows != result['committed_decode_tokens']
                or drafter.position != len(prompt) + result['committed_decode_tokens']):
            raise AssertionError('Complete target-feature publication and exact full-history frontier required')
        result['dspark'] = dict(proposals=15, verifier_rows=16, full_history=True, prefill_chunks=prefill_records,
            prefill_hashes=prefill_hashes, feature_checks=feature_checks, audit_features=audit_features,
            committed_feature_rows=runtime.committed_feature_rows, final_position=drafter.position,
            execution='Eager full-history proposal; batched captured target verifier; all request-loop costs retained',
            proposal_trace=False, packed_token_readbacks_per_proposal=2, checkpoint_trained_block_rows=16,
            published_serving_proposals=7, wider_proposal_acceptance_qualified=False)
        result['instrumented_timing'] = audit_features
        if audit_features:
            result['committed_tokens_per_second'] = None
        result['kind'] = 'Full-history DSpark coding-request pilot; not held-out coding quality'
        result['qualification'] = __doc__
        return result
    finally:
        try:
            if drafter is not None:
                drafter.close()
        finally:
            if capture is not None:
                capture.close()
