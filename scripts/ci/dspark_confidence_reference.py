"""CPU confidence oracle; not a serving policy or device qualification."""

import torch
import torch.nn.functional as functional


UPSTREAM_REVISION = '2733afe54e4efe142cfdd01efb672eabf603c9a4'


def evaluate(hidden, anchors, drafts, embedding, weight, bias, temperatures):
    if hidden.ndim != 3 or drafts.shape != hidden.shape[:2] or anchors.shape != hidden.shape[:1]:
        raise ValueError('Aligned batch, proposal positions and anchors required')
    batch, width, channels = hidden.shape
    if batch < 1 or width < 1 or embedding.ndim != 2 or embedding.shape[0] < 1:
        raise ValueError('Nonempty proposals and predecessor embedding required')
    if weight.shape != (1, channels + embedding.shape[1]) or bias.shape != (1,):
        raise ValueError('Hidden plus Markov feature projection required')
    if temperatures.shape not in (torch.Size([]), torch.Size([width])):
        raise ValueError('Scalar or per-position calibration required')
    tensors = (hidden, anchors, drafts, embedding, weight, bias, temperatures)
    if any(value.device.type != 'cpu' for value in tensors):
        raise ValueError('Reference is CPU-only')
    if anchors.dtype != torch.int64 or drafts.dtype != torch.int64:
        raise ValueError('Integer token identifiers required')
    if any(not value.is_floating_point() or not torch.isfinite(value).all()
            for value in (hidden, embedding, weight, bias, temperatures)):
        raise ValueError('Finite floating point features and parameters required')
    if not (temperatures > 0).all() or bias.dtype != weight.dtype:
        raise ValueError('Positive calibration and matching projection dtypes required')
    if any((value < 0).any() or (value >= embedding.shape[0]).any() for value in (anchors, drafts)):
        raise ValueError('Tokens outside predecessor vocabulary')
    predecessors = torch.cat((anchors[:, None], drafts[:, :-1]), dim=1)
    latent = functional.embedding(predecessors, embedding).to(hidden.dtype)
    features = torch.cat((hidden, latent), dim=-1).to(weight.dtype)
    logits = functional.linear(features, weight, bias).squeeze(-1)
    probabilities = torch.sigmoid(logits.float() / temperatures.float())
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise ValueError('Confidence arithmetic overflow')
    return dict(logits=logits, probabilities=probabilities,
        survival=torch.cumprod(probabilities, dim=1), predecessors=predecessors)
