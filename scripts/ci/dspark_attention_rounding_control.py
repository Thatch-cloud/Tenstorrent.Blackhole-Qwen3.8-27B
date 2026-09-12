"""CPU-only rounding hypotheses; not a model of Tensix arithmetic or a qualification gate."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch

import dspark_full_attention


def bf16_storage(value, truncate=False):
    if truncate:
        return ((value.contiguous().view(torch.int32) >> 16) << 16).view(torch.float32)
    return value.bfloat16().float()


def online_attention(query, key, value, mask, *, round_output=False, round_probability=False,
        round_statistics=False, truncate_scale=False, truncate_storage=False):
    scale = torch.tensor(128 ** -.5)
    if truncate_scale:
        scale = ((scale.view(torch.int32) >> 16) << 16).view(torch.float32)
    maximum = torch.full((*query.shape[:-1], 1), float('-inf'))
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(query)
    for start in range(0, key.shape[2], 64):
        scores = query @ key[:, :, start:start + 64].transpose(-1, -2)
        scores = scores + mask[:, :, :, start:start + 64]
        updated = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        if round_statistics:
            updated = bf16_storage(updated, truncate_storage)
        correction = ((maximum - updated) * scale).exp()
        if round_statistics:
            correction = bf16_storage(correction, truncate_storage)
        probability = ((scores - updated) * scale).exp()
        if round_probability:
            probability = bf16_storage(probability, truncate_storage)
        partial = probability @ value[:, :, start:start + 64]
        if round_output:
            partial = bf16_storage(partial, truncate_storage)
        numerator = numerator * correction + partial
        if round_output:
            numerator = bf16_storage(numerator, truncate_storage)
        denominator = denominator * correction + probability.sum(-1, keepdim=True)
        maximum = updated
    return (numerator / denominator).bfloat16().float()


def main():
    torch.set_num_threads(1)
    source = Path(__file__).with_name('dspark-native-8k-attention-probe.py')
    spec = importlib.util.spec_from_file_location('attention_fixture', source)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    records = []
    with patch.object(dspark_full_attention, 'MAX_CONTEXT', probe.CAPACITY):
        for case, fixture in enumerate(probe.fixtures()):
            for chip in range(2):
                query = fixture['query'][chip:chip + 1, :, :probe.PROPOSALS].float()
                key = probe.joined(fixture, 'key')[chip:chip + 1].float().repeat_interleave(4, dim=1)
                value = probe.joined(fixture, 'value')[chip:chip + 1].float().repeat_interleave(4, dim=1)
                mask = fixture['mask'][:, :, :probe.PROPOSALS].float()
                expected = probe.reference(fixture, chip)[:, :, :probe.PROPOSALS]
                variants = ((False, False, False, False), (True, False, False, False),
                    (False, True, False, False), (True, True, False, False),
                    (False, False, True, False), (True, False, True, False),
                    (False, False, False, True), (True, False, True, True))
                variants = tuple((*variant, False) for variant in variants) + (
                    (True, False, False, True, True), (True, False, True, True, True))
                for round_output, round_probability, round_statistics, truncate_scale, truncate_storage in variants:
                    actual = online_attention(query, key, value, mask,
                        round_output=round_output, round_probability=round_probability,
                        round_statistics=round_statistics, truncate_scale=truncate_scale,
                        truncate_storage=truncate_storage)
                    failed = ~torch.isclose(actual, expected, rtol=.01, atol=.01)
                    records.append(dict(case=case, chip=chip, round_output=round_output,
                        round_probability=round_probability, round_statistics=round_statistics,
                        truncate_scale=truncate_scale, truncate_storage=truncate_storage,
                        failed=int(failed.sum()),
                        max_abs=float((actual - expected).abs().max())))
                    if not any((round_output, round_probability, round_statistics, truncate_scale)) and failed.any():
                        raise AssertionError('FP32 online-softmax CPU control failed')
    print(json.dumps(dict(qualification=False, device_access=False, records=records), indent=2))


if __name__ == '__main__':
    main()
