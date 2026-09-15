"""CPU reference attribution for the reproduced 32K failure; not a device emulator."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch
import dspark_full_attention


def main():
    torch.set_num_threads(1)
    path = Path(__file__).with_name('dspark-native-8k-attention-probe.py')
    spec = importlib.util.spec_from_file_location('row_fixture', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.CAPACITY = 33792
    fixture.POSITIONS = (32768, 33777)
    with patch.object(dspark_full_attention, 'MAX_CONTEXT', fixture.CAPACITY):
        values = fixture.fixtures()[0]
    query = values['query'][1, 15, 5].float()
    keys = torch.cat((values['history_key'][1, 3], values['query_key'][1, 3, :15])).float()
    vectors = torch.cat((values['history_value'][1, 3], values['query_value'][1, 3, :15])).float()
    scores = keys @ query * (128 ** -.5)
    scores += values['mask'][0, 0, 5, :keys.shape[0]].float()
    probabilities = torch.softmax(scores, dim=0)
    columns = [8, 38, 54, 77]
    output = probabilities @ vectors[:, columns]
    top = torch.topk(probabilities, 8)
    print(json.dumps(dict(scope=__doc__, context=32768, chip=1, head=15, query_row=5,
        folded_row=23, columns=columns, reference=output.tolist(),
        reference_bf16=output.bfloat16().float().tolist(),
        largest_weights=[dict(key=int(index), probability=float(weight),
            score=float(scores[index]), contribution=(weight * vectors[index, columns]).tolist())
            for index, weight in zip(top.indices, top.values, strict=True)],
        remaining_probability=float(1 - top.values.sum()), performance_qualified=False), indent=2))


if __name__ == '__main__':
    main()
