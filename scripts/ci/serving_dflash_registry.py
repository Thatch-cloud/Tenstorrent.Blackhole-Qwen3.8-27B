"""Metadata admission for the DFlash2 model executed by the TT fast worker."""

from torch import nn


class DFlash2DraftModel(nn.Module):
    """Registry metadata only; execution belongs to the combined TT runtime."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError('DFlash2 execution requires the explicit TT combined fast runtime')

    def forward(self, *args, **kwargs):
        raise RuntimeError('DFlash2 metadata adapter cannot execute a forward pass')

    def embed_input_ids(self, *args, **kwargs):
        raise RuntimeError('DFlash2 metadata adapter cannot embed tokens')

    def compute_logits(self, *args, **kwargs):
        raise RuntimeError('DFlash2 metadata adapter cannot compute logits')

    def load_weights(self, *args, **kwargs):
        raise RuntimeError('DFlash2 metadata adapter cannot load weights')


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model('DFlash2DraftModel', DFlash2DraftModel)
