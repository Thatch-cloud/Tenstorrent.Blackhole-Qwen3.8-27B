"""Bounded CPU comparison with one hash-pinned upstream confidence class."""

import ast
import hashlib
import json
from pathlib import Path
from typing import Optional
import urllib.request

import torch

from dspark_confidence_fixture import TENSORS, fetch
from dspark_confidence_reference import UPSTREAM_REVISION, evaluate


SOURCE_SHA256 = 'd70ebbbbb81b93c0ad9e5baf1cec73d8fea5ab90851d14ca9128afa23fb1d75a'
SOURCE_URL = f'https://raw.githubusercontent.com/sgl-project/sglang/{UPSTREAM_REVISION}/python/sglang/srt/models/dspark.py'


def upstream_class(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Exact reviewed upstream source required before class extraction')
    selected = [node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == 'DSparkConfidenceHead']
    if len(selected) != 1:
        raise ValueError('Exactly one upstream confidence class required')
    namespace = dict(torch=torch, nn=torch.nn, Optional=Optional)
    exec(compile(ast.Module(body=selected, type_ignores=[]), SOURCE_URL, 'exec'), namespace)
    return namespace['DSparkConfidenceHead']


def run():
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read(262145)
    constructor = upstream_class(source)
    weights = fetch()
    generator = torch.Generator().manual_seed(71)
    checks = []
    with torch.no_grad():
        head = constructor(hidden_size=5120, markov_rank=256)
        head.load_state_dict({name.removeprefix('confidence_head.'): value.float()
            for name, value in weights.items()})
        for width in (1, 7, 15, 31):
            for dtype in (torch.float32, torch.bfloat16):
                for calibrated in (False, True):
                    hidden = torch.randn(2, width, 5120, generator=generator).to(dtype)
                    embedding = torch.randn(64, 256, generator=generator).bfloat16()
                    anchors = torch.tensor([0, 63])
                    drafts = torch.randint(0, 64, (2, width), generator=generator)
                    previous = torch.cat((anchors[:, None], drafts[:, :-1]), dim=1)
                    temperatures = torch.linspace(0.5, 2, width) if calibrated else torch.tensor(1.)
                    head.sts_temperatures = temperatures
                    logits = head(hidden, embedding[previous])
                    probabilities = head.apply_sts(logits)
                    actual = evaluate(hidden, anchors, drafts, embedding, head.proj.weight, head.proj.bias, temperatures)
                    expected = dict(logits=logits, probabilities=probabilities,
                        survival=probabilities.cumprod(1), predecessors=previous)
                    if any(not torch.equal(actual[name], value) for name, value in expected.items()):
                        raise AssertionError('Learned confidence reference differs from upstream')
                    checks.append(dict(width=width, hidden_dtype=str(dtype), calibrated=calibrated, exact=True))
    return dict(passed=True, checks=checks, upstream_revision=UPSTREAM_REVISION,
        upstream_sha256=SOURCE_SHA256, tensor_sha256={name: value[0] for name, value in TENSORS.items()},
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('dspark_confidence_parity.py', 'dspark_confidence_reference.py', 'dspark_confidence_fixture.py')},
        learned_head=True, synthetic_features=True, hardware_qualified=False, performance_qualified=False)


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(json.dumps(run(), indent=2))
