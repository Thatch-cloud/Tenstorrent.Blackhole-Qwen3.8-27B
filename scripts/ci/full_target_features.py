"""Eager target-feature boundary validation; no draft model or acceptance claim."""

from model_batch import instance_overrides
from target_features import LayerOutputCapture


def verify_features(model, prompt, tap_ids, *, prefill, decode, live_digest, kv_digest, inactive_digest,
                    snapshot, release, storage_ids, local_host):
    import torch

    taps = tuple(tap_ids)
    if not taps or len(set(taps)) != len(taps) or any(
            type(index) is not int or not 0 <= index < len(model.layers) - 1 for index in taps):
        raise ValueError('Unique target taps with following decoder layers required')
    position = len(prompt)
    seed = prefill(prompt)
    inactive = inactive_digest()
    gold = decode(seed, position, False)
    gold_state, gold_kv = live_digest(), kv_digest(position + 1)
    if inactive_digest() != inactive:
        raise AssertionError('Native decode changed inactive slots')
    references = {}
    capture = None

    def input_hook(index, original):
        def forward(hidden, *args, **kwargs):
            if index in references:
                raise AssertionError('Reference input executed more than once')
            references[index] = snapshot(hidden)
            return original(hidden, *args, **kwargs)
        return forward

    def check_output(actual):
        if not torch.equal(actual, gold) or live_digest() != gold_state or kv_digest(position + 1) != gold_kv:
            raise AssertionError('Feature instrumentation changed native logits, GDN or valid KV')
        if inactive_digest() != inactive:
            raise AssertionError('Feature instrumentation changed inactive slots')

    try:
        if prefill(prompt) != seed:
            raise AssertionError('Reference prefill seed changed')
        bindings = [(model.layers[index + 1], 'forward', input_hook(index, model.layers[index + 1].forward))
                    for index in taps]
        with instance_overrides(bindings):
            reference_logits = decode(seed, position, False)
        check_output(reference_logits)
        if set(references) != set(taps):
            raise AssertionError('Missing next-layer input reference')
        reference_host = {index: local_host(value) for index, value in references.items()}
        for index in tuple(references):
            value = references.pop(index)
            release(value)
        if prefill(prompt) != seed:
            raise AssertionError('Captured prefill seed changed')
        capture = LayerOutputCapture(model, taps, snapshot=snapshot, release=release, storage_ids=storage_ids)
        with capture.capture():
            actual = decode(seed, position, False)
        check_output(actual)
        checks = []
        for index, feature in zip(taps, capture.outputs(), strict=True):
            parts = local_host(feature)
            if len(parts) != 2 or len(reference_host[index]) != 2:
                raise AssertionError('Both target feature shards required')
            for chip, (part, expected) in enumerate(zip(parts, reference_host[index], strict=True)):
                if not torch.equal(part, expected):
                    raise AssertionError(f'Post-layer feature differs from next-layer input at layer {index}, chip {chip}')
                checks.append(dict(layer=index, chip=chip, shape=list(part.shape), exact=True))
        return dict(length=position, token=seed, tap_ids=list(taps), checks=checks,
                    logits_exact=True, state_exact=True, valid_kv_exact=True, inactive_exact=True,
                    scope='Eager B1 post-layer versus next-layer input; no trace, token alignment or drafter certification')
    finally:
        try:
            if capture is not None:
                capture.close()
        finally:
            for value in references.values():
                release(value)
