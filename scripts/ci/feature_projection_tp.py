"""feature_projection's tap packing at any served width.

feature_projection.py is frozen evidence (its sha256 is pinned by the TP2 attach) and packs the drafter's feature
projection for TWO hidden-sharded chips: projection_shards splits the tap-major fc weight into two halves of the hidden
size, concatenate_local_features expects chip-local taps of hidden / 2. The four-card drafter shards the same fc weight
over four chips (1,280 columns per tap and chip), so this twin is the same two functions with the chip count from
tp_shapes; at the pair each is call for call the pinned one (test_draft_tp4.FeatureProjectionTests holds them equal). dflash_device
imports these directly; the pinned module stays for the experiment scripts that import it.

Stdlib only (torch arrives with the weights).
"""

import tp_shapes


def projection_shards(weight, *, tap_count=5, hidden_size=5120):
    """The fc weight (outputs, tap_count * hidden) as one (tap_count * hidden / tp, outputs) matrix per chip: chip c holds
    the c-th hidden slice of every tap, taps in order, transposed for the projection matmul."""
    chips = tp_shapes.chip_count()
    if (type(tap_count) is not int or tap_count < 1 or type(hidden_size) is not int
            or hidden_size < chips or hidden_size % chips or weight.ndim != 2
            or weight.shape[0] < 1 or weight.shape[1] != tap_count * hidden_size):
        raise ValueError('Output-by-tap-major-hidden projection and even TP%d hidden size required' % chips)
    outputs = weight.shape[0]
    grouped = weight.reshape(outputs, tap_count, hidden_size)
    width = hidden_size // chips
    return tuple(grouped[:, :, chip * width:(chip + 1) * width].reshape(outputs, tap_count * width)
        .transpose(0, 1).contiguous() for chip in range(chips))


def concatenate_local_features(operations, features, *, tap_count=5, hidden_size=5120):
    """The five chip-local taps [1, 1, T, hidden / tp] joined along the width."""
    chips = tp_shapes.chip_count()
    features = tuple(features)
    if (type(tap_count) is not int or tap_count < 1 or type(hidden_size) is not int
            or hidden_size < chips or hidden_size % chips or len(features) != tap_count):
        raise ValueError('Complete ordered feature taps and even TP%d hidden size required' % chips)
    shape = tuple(features[0].shape)
    if (len(shape) != 4 or shape[:2] != (1, 1) or shape[2] < 1 or shape[3] != hidden_size // chips
            or any(tuple(value.shape) != shape for value in features)):
        raise ValueError('Matching chip-local [1,1,T,hidden/%d] features required' % chips)
    return operations.concat(features, dim=-1, memory_config=operations.DRAM_MEMORY_CONFIG)
