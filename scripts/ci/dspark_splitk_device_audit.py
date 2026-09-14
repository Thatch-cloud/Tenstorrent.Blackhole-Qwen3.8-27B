"""Exact first-eager device layout readback for simulator diagnostics only."""

import json

import torch

from dspark_splitk_layout import fold_mask, fold_query


def audit_layout(operations, original, folded, *, stripe_keys=False):
    original_devices = [operations.get_device_tensors(tensor) for tensor in original]
    folded_devices = [operations.get_device_tensors(tensor) for tensor in folded]
    counts = [len(devices) for devices in original_devices + folded_devices]
    if not counts or len(set(counts)) != 1 or counts[0] != 2:
        raise ValueError('Two matching simulator chip replicas required for layout audit')
    for chip in range(counts[0]):
        query, key, value, mask = [operations.to_torch(devices[chip]) for devices in original_devices]
        expected = (fold_query(query).reshape(1, 4, 128, 128),
            key.reshape(4, 1, key.shape[2], 128), value.reshape(4, 1, value.shape[2], 128),
            fold_mask(mask).reshape(4, 1, 128, mask.shape[3]))
        if stripe_keys:
            if key.shape[2] != 512:
                raise ValueError('Exact 512-key redistribution audit required')
            indices = torch.arange(512).reshape(32, 16).transpose(0, 1).flatten()
            expected = (expected[0], expected[1].index_select(2, indices),
                expected[2].index_select(2, indices), expected[3].index_select(3, indices))
        for name, devices, reference in zip(('query', 'key', 'value', 'mask'), folded_devices, expected):
            actual = operations.to_torch(devices[chip])
            exact = actual.shape == reference.shape and torch.equal(actual, reference)
            print(json.dumps(dict(stage='splitk-device-layout', chip=chip, tensor=name,
                passed=exact, stripe_keys=stripe_keys,
                actual_shape=list(actual.shape), expected_shape=list(reference.shape))), flush=True)
            if not exact:
                raise AssertionError('Split-K device layout differs from exact host oracle: ' + name)
