"""Exact host-side checks for captured TP2 target feature rows."""


def compare_feature_rows(serial, captured, tap_ids):
    import torch

    taps = tuple(tap_ids)
    rows = len(serial)
    if not rows or not taps or len(captured) != len(taps) or any(len(step) != len(taps) for step in serial):
        raise ValueError('Complete serial and captured feature taps required')
    checks = []
    for index, layer in enumerate(taps):
        parts = captured[index]
        if len(parts) != 2 or any(len(step[index]) != 2 for step in serial):
            raise ValueError('Exactly two feature shards required')
        for chip, actual in enumerate(parts):
            expected_parts = [step[index][chip] for step in serial]
            if any(list(part.shape) != [1, 1, 1, 2560] for part in expected_parts):
                raise ValueError('Native serial feature geometry changed')
            if list(actual.shape) != [1, 1, rows, 2560]:
                raise ValueError('Captured feature geometry changed')
            if not torch.equal(actual, torch.cat(expected_parts, dim=2)):
                raise AssertionError(f'Stale or misaligned captured features at layer {layer}, chip {chip}')
            checks.append(dict(layer=layer, chip=chip, rows=rows, exact=True))
    return checks
