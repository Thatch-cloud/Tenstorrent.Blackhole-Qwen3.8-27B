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
            expected = torch.cat(expected_parts, dim=2)
            if not torch.equal(actual, expected):
                mismatch = actual != expected
                first = tuple(mismatch.nonzero()[0].tolist())
                raise AssertionError(f'Stale or misaligned captured features at layer {layer}, chip {chip}: '
                    f'{int(mismatch.sum())}/{actual.numel()} differ, first={first}, '
                    f'actual={actual[first].item()}, expected={expected[first].item()}')
            checks.append(dict(layer=layer, chip=chip, rows=rows, exact=True))
    return checks


def check_published_features(serial, published, tap_ids, local_host, *, stage):
    checks = []
    for phase, offset, prefix, copies in published:
        try:
            if len(copies) != (len(tap_ids) if prefix else 0):
                raise AssertionError('Abort or committed feature tap count differs')
            compared = compare_feature_rows(serial[offset:offset + prefix],
                [local_host(value) for value in copies], tap_ids) if prefix else []
        except (AssertionError, ValueError) as error:
            raise AssertionError(f'Published features failed: {stage=}, {phase=}, {offset=}, {prefix=}: {error}') from error
        checks.append(dict(phase=phase, offset=offset, prefix=prefix, stage=stage, checks=compared, exact=True))
    return checks
