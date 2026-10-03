"""Defer compact entry refresh to prefix-zero publication, retaining the original DMA kernel."""

from gdn_commit_dma import prepare as prepare_compact
from gdn_commit_dma import validate_shapes
from gdn_state_copy import copy_active


def prepare(mesh, layers, prefix, *, experimental=False):
    if experimental is not True:
        raise ValueError('Explicit commit-only native-slot publication experiment required')
    rows = validate_shapes([[tuple(value.shape) for value in layer] for layer in layers], prefix)
    if rows != 16:
        raise ValueError('Native-slot publication is restricted to T16')
    publication = prepare_compact(mesh, layers, prefix)
    def execute():
        if prefix == 0:
            for layer in layers:
                copy_active(layer[10:15], layer[:5])
        publication()
    return execute
