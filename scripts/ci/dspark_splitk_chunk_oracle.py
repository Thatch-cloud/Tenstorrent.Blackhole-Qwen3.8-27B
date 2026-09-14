"""CPU-only chunk-statistics ablation; not a model of every device rounding step."""

import json

import torch

from dspark_ladder_fixtures import fixture_probe


def compare_chunk_statistics(*, round_maxima=False, truncate_correction_scale=False):
    reports = []
    with fixture_probe(128, 256) as probe:
        fixture = probe.fixtures()[0]
        query = fixture['query'][0:1].float()
        key = probe.joined(fixture, 'key')[0:1].float().repeat_interleave(4, 1)
        value = probe.joined(fixture, 'value')[0:1].float().repeat_interleave(4, 1)
        scale = 128 ** -.5
        correction_scale = float(torch.tensor(scale).view(torch.int32).bitwise_and(-65536).view(torch.float32)) if truncate_correction_scale else scale
        scores = query @ key.transpose(-1, -2) + fixture['mask'].float()
        expected = probe.reference(fixture, 0)
        for chunk_size in (512, 256):
            for bf16_statistics in (False, True):
                def rounded(tensor):
                    return tensor.bfloat16().float() if bf16_statistics else tensor

                maximum = total = numerator = None
                for start in range(0, scores.shape[-1], chunk_size):
                    chunk = scores[..., start:start + chunk_size]
                    next_maximum = chunk.amax(-1, keepdim=True)
                    if maximum is not None:
                        next_maximum = torch.maximum(maximum, next_maximum)
                    if round_maxima:
                        next_maximum = next_maximum.bfloat16().float()
                    probabilities = torch.where(torch.isneginf(next_maximum), 0.,
                        torch.exp((chunk - next_maximum) * scale))
                    partial_sum = rounded(probabilities.sum(-1, keepdim=True))
                    partial_output = probabilities @ value[..., start:start + chunk_size, :]
                    if maximum is None:
                        total, numerator = partial_sum, partial_output
                    else:
                        correction = rounded(torch.where(torch.isneginf(maximum), 0.,
                            torch.exp((maximum - next_maximum) * correction_scale)))
                        total = rounded(rounded(total * correction) + partial_sum)
                        numerator = numerator * correction + partial_output
                    maximum = next_maximum
                actual = (numerator / total).bfloat16().float()
                close = torch.isclose(actual, expected, rtol=.01, atol=.01)
                reports.append(dict(chunk_size=chunk_size, bf16_statistics=bf16_statistics,
                    round_maxima=round_maxima, truncate_correction_scale=truncate_correction_scale,
                    failed_elements=int((~close).sum()),
                    max_abs=float((actual - expected).abs().max()), device_qualified=False))
    return reports


if __name__ == '__main__':
    print(json.dumps([report for maxima in (False, True) for truncated in (False, True)
        for report in compare_chunk_statistics(round_maxima=maxima,
            truncate_correction_scale=truncated)], indent=2))
