"""CPU precision ablation on the exact small simulator fixture, not device acceptance."""

import json

import torch

from dspark_ladder_fixtures import fixture_probe


def compare_score_precision():
    reports = []
    with fixture_probe(128, 256) as probe:
        values = probe.fixtures()[0]
        query = values['query'][0:1].float()
        key = probe.joined(values, 'key')[0:1].float().repeat_interleave(4, 1)
        value = probe.joined(values, 'value')[0:1].float().repeat_interleave(4, 1)
        raw = query @ key.transpose(-1, -2)
        expected = probe.reference(values, 0)
        variants = (('fp32', raw), ('bf16_rne', raw.bfloat16().float()),
            ('bf16_truncate', (raw.contiguous().view(torch.int32) & -65536).view(torch.float32)))
        for name, scores in variants:
            result = torch.softmax(scores * (128 ** -.5) + values['mask'].float(), dim=-1) @ value
            close = torch.isclose(result, expected, rtol=.01, atol=.01)
            reports.append(dict(mode=name, failed_elements=int((~close).sum()),
                max_abs=float((result - expected).abs().max()), first_output=result[0, 0, 0, :4].tolist(),
                device_qualified=False))
    return reports


if __name__ == '__main__':
    print(json.dumps(compare_score_precision(), indent=2))
