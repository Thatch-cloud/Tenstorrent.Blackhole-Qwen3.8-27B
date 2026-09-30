"""Compare captured candidate outputs against native-score eager proposal outputs."""

import os

from dspark_t32_markov import execute as native_markov
from dspark_t32_prepared import execute


def compare(proposal):
    import torch

    if (os.environ.get('QWEN_SIM_ONLY') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))
            or proposal.closed or proposal.trace is None):
        raise ValueError('Live simulator proposal replay required for score comparison')
    device = proposal.device
    if not hasattr(device, 'proposal_markov') or device.proposal_markov is native_markov:
        raise ValueError('Explicit candidate score backend required')
    candidate = device.proposal_markov
    scope = proposal.output_owner()
    try:
        device.proposal_markov = native_markov
        expected = proposal.snapshot(execute(device, proposal.inputs, proposal.history, scope.retain))
        actual = proposal.snapshot(proposal.outputs)
        if len(actual) != 6 or len(expected) != 6:
            raise AssertionError('Hidden, logits and tokens on both replicas required')
        if any(not torch.equal(value, reference) for value, reference in zip(actual, expected, strict=True)):
            raise AssertionError('Fused proposal differs from native-score complete proposal')
        return dict(position=device.position, tensors=6, exact=True, reference='native-score-eager')
    finally:
        device.proposal_markov = candidate
        scope.release()
