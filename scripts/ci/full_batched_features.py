"""Serial versus batched target features; eager token-row alignment only."""

from target_features import LayerOutputCapture


def verify_batched_features(model, prompt, tap_ids, rows, *, prefill, decode, batch_decode,
                            live_digest, kv_digest, inactive_digest, snapshot, release, storage_ids, local_host):
    import torch

    if type(rows) is not int or rows not in (8, 16, 32):
        raise ValueError('Eight, sixteen or thirty-two verification rows required')
    taps = tuple(tap_ids)
    if not taps or len(set(taps)) != len(taps) or any(
            type(index) is not int or not 0 <= index < len(model.layers) for index in taps):
        raise ValueError('Unique valid decoder taps required')
    position, vocab = len(prompt), model.args.vocab_size
    capture_options = dict(snapshot=snapshot, release=release, storage_ids=storage_ids)
    token = seed = prefill(prompt)
    inactive = inactive_digest()
    tokens, gold_logits, serial_features = [], [], []
    for offset in range(rows):
        tokens.append(token)
        capture = LayerOutputCapture(model, taps, **capture_options)
        try:
            with capture.capture():
                logits = decode(token, position + offset, False)
            if logits.shape[-1] != vocab or logits.numel() // vocab not in (1, 8):
                raise AssertionError('Native B1 logits geometry changed')
            active = logits.reshape(-1, vocab)[:1].clone()
            gold_logits.append(active)
            serial_features.append([local_host(feature) for feature in capture.outputs()])
            token = int(active.float().argmax(dim=-1)[0])
        finally:
            capture.close()
    gold_state, gold_kv = live_digest(), kv_digest(position + rows)
    if inactive_digest() != inactive:
        raise AssertionError('Serial feature capture changed inactive slots')
    if prefill(prompt) != seed:
        raise AssertionError('Batched feature prefill seed differs')
    capture = LayerOutputCapture(model, taps, **capture_options)
    try:
        with capture.capture():
            actual_logits = batch_decode(tokens, position)
        expected_logits = torch.cat(gold_logits, dim=0)
        if len(actual_logits) != 2 or any(not torch.equal(part.reshape(rows, vocab), expected_logits) for part in actual_logits):
            raise AssertionError('Batched logits differ from serial decode')
        if live_digest() != gold_state or kv_digest(position + rows) != gold_kv or inactive_digest() != inactive:
            raise AssertionError('Batched feature capture changed native state, KV or inactive slots')
        checks = []
        for tap_offset, (layer, feature) in enumerate(zip(taps, capture.outputs(), strict=True)):
            parts = local_host(feature)
            if len(parts) != 2 or any(len(features[tap_offset]) != 2 for features in serial_features):
                raise AssertionError('Both feature shards required')
            for chip, part in enumerate(parts):
                serial = [features[tap_offset][chip] for features in serial_features]
                if any(list(value.shape) != [1, 1, 1, 2560] for value in serial):
                    raise AssertionError('Certified eager B1 feature geometry changed')
                if list(part.shape) != [1, 1, rows, 2560]:
                    raise AssertionError('Batched feature token axis or shard width changed')
                expected = torch.cat(serial, dim=2)
                if not torch.equal(part, expected):
                    raise AssertionError(f'Batched feature token alignment differs at layer {layer}, chip {chip}')
                checks.append(dict(layer=layer, chip=chip, rows=rows, shape=list(part.shape), exact=True))
        return dict(length=position, rows=rows, input_tokens=tokens, tap_ids=list(taps), checks=checks,
                    logits_exact=True, state_exact=True, valid_kv_exact=True, inactive_exact=True,
                    scope='Eager serial B1 versus verifier feature rows; no traced reuse or drafter certification')
    finally:
        capture.close()
