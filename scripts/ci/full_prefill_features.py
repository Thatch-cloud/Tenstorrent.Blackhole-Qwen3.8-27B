"""Single-chunk eager prefill feature boundary oracle; no trained-drafter parity claim."""

from model_batch import instance_overrides
from target_features import LayerOutputCapture


def verify_prefill_features(model, prompt, tap_ids, *, prefill_logits, decode, live_digest, kv_digest,
                            inactive_digest, snapshot, release, storage_ids, local_host):
    import torch

    taps = tuple(tap_ids)
    length = len(prompt)
    if (not 0 < length <= 256 or not taps or len(set(taps)) != len(taps)
            or any(type(index) is not int or not 0 <= index < len(model.layers) - 1 for index in taps)):
        raise ValueError('Bounded single-chunk prompt and unique taps with following layers required')
    gold = prefill_logits(prompt).clone()
    gold_state, gold_kv, gold_inactive = live_digest(), kv_digest(length), inactive_digest()
    seed = int(gold.reshape(-1, model.args.vocab_size)[-1].float().argmax())
    gold_continuation = decode(seed, length, False).clone()
    gold_final_state, gold_final_kv = live_digest(), kv_digest(length + 1)
    references = {}
    capture = None

    def input_hook(index, original):
        def forward(hidden, *args, **kwargs):
            if index in references:
                raise AssertionError('Single-chunk prefill reference executed more than once')
            references[index] = snapshot(hidden)
            return original(hidden, *args, **kwargs)
        return forward

    def check_prefill(actual):
        if not torch.equal(actual, gold) or live_digest() != gold_state or kv_digest(length) != gold_kv:
            raise AssertionError('Prefill feature instrumentation changed logits, GDN or valid KV')
        if inactive_digest() != gold_inactive:
            raise AssertionError('Prefill feature instrumentation changed inactive slots')

    try:
        bindings = [(model.layers[index + 1], 'forward', input_hook(index, model.layers[index + 1].forward))
                    for index in taps]
        with instance_overrides(bindings):
            reference_logits = prefill_logits(prompt)
        check_prefill(reference_logits)
        if set(references) != set(taps):
            raise AssertionError('Missing prefill next-layer input references')
        reference_host = {index: [part.clone() for part in local_host(value)] for index, value in references.items()}
        for index in tuple(references):
            release(references.pop(index))
        capture = LayerOutputCapture(model, taps, snapshot=snapshot, release=release, storage_ids=storage_ids)
        with capture.capture():
            actual = prefill_logits(prompt)
        check_prefill(actual)
        checks = []
        for index, feature in zip(taps, capture.outputs(), strict=True):
            parts = local_host(feature)
            if len(parts) != 2 or len(reference_host[index]) != 2:
                raise AssertionError('Exactly two prefill feature shards required')
            for chip, (part, expected) in enumerate(zip(parts, reference_host[index], strict=True)):
                shape = tuple(part.shape)
                if (len(shape) != 4 or shape[:2] != (1, 1) or not length <= shape[2] <= 256
                        or shape[3] != 2560 or tuple(expected.shape) != shape):
                    raise AssertionError('Unexpected TP2 prefill feature geometry')
                if not torch.equal(part[:, :, :length], expected[:, :, :length]):
                    raise AssertionError(f'Prefill feature row mismatch at layer {index}, chip {chip}')
                checks.append(dict(layer=index, chip=chip, shape=list(shape), valid_rows=length,
                    padding_rows=shape[2] - length, exact=True))
        capture.close()
        capture = None
        if (not torch.equal(decode(seed, length, False), gold_continuation)
                or live_digest() != gold_final_state or kv_digest(length + 1) != gold_final_kv):
            raise AssertionError('Prefill feature capture changed native decode continuation')
        if inactive_digest() != gold_inactive:
            raise AssertionError('Decode continuation changed inactive slots')
        return dict(length=length, tap_ids=list(taps), checks=checks, logits_exact=True, state_exact=True,
            valid_kv_exact=True, inactive_exact=True, correction_steps=1,
            scope='Eager single-chunk valid prefill rows versus next-layer inputs; padding excluded, no trace or drafter parity')
    finally:
        try:
            if capture is not None:
                capture.close()
        finally:
            for value in references.values():
                release(value)
