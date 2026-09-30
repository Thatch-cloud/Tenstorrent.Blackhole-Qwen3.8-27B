"""Untraced native TTSampling force-argmax composition for an outer verifier trace.

The sampler takes one 32-row tile of logits: `tt_sampling.max_batch_size` is 32, which
is the plugin sampler's own padded batch (models/common/sampling, tt_sampling.py in the
tt-metal checkout), not this repo's constant and not `model.args.max_batch_size`, which
serving pins at 8 (serving_cache_owner.py). Narrower blocks are padded up to that tile;
the 64-row M3 block (four T16 users) is served as two 32-row tiles through the same
pinned sampler, `sample_tiles`, and the ids are joined on their row axis. That is two
sampler launches inside one outer verify trace, and it has not run on hardware.
"""

from gdn_multitoken_conv import addresses, release_owned

SAMPLER_ROWS = 32
SAMPLE_WIDTHS = (1, 2, 4, 8, 16, 32, 64)


def selected_ids(result):
    if isinstance(result, tuple):
        if len(result) != 2 or result[1] is not None:
            raise ValueError('Greedy verifier does not consume log-probability outputs')
        return result[0]
    return result


def row_axis(shape):
    """The axis the sampler laid its 32 ids along, whatever its layout."""
    axes = [index for index, size in enumerate(shape) if size == SAMPLER_ROWS]
    if len(axes) != 1:
        raise ValueError('Sampler ids must carry exactly one %d-row axis' % SAMPLER_ROWS)
    return axes[0]


def sample_tiles(sampler, logits, rows, operations, shape):
    """Rows beyond the sampler's tile: one sampler call per 32-row slice, ids joined."""
    owned, tiles = [], []
    try:
        for start in range(0, rows, SAMPLER_ROWS):
            piece = operations.slice(logits, (0, 0, start, 0), (1, 1, start + SAMPLER_ROWS, shape[3]))
            owned.append(piece)
            tiles.append(selected_ids(sampler.sample(piece, enable_trace=False)))
            owned.append(tiles[-1])
        if len({addresses(operations, tile) for tile in tiles}) != len(tiles):
            raise ValueError('Sampler reused its output across tiles; the joined ids would be one tile twice')
        axis = row_axis(tuple(tiles[0].shape))
        if any(tuple(tile.shape) != tuple(tiles[0].shape) for tile in tiles):
            raise ValueError('Every sampled tile must share one id layout')
        return operations.concat(tiles, dim=axis)
    finally:
        release_owned(operations, owned)


def sample_rows(sampler, logits, rows, operations, *, native_rows=False):
    if type(native_rows) is not bool:
        raise ValueError('Explicit boolean native-row sampling required')
    if type(rows) is not int or rows not in SAMPLE_WIDTHS:
        raise ValueError('Supported verifier width required')
    shape = tuple(logits.shape)
    if len(shape) != 4 or shape[:3] != (1, 1, rows):
        raise ValueError('Expected pre-gather vocab-sharded verifier logits')
    if not sampler.tt_sampling.force_argmax_sampling or sampler.tt_sampling.max_batch_size != SAMPLER_ROWS:
        raise ValueError('Pinned 32-row force-argmax sampler required')
    if sampler.seed_manager.has_active_request_seed():
        raise ValueError('Seeded sampling is outside the greedy verifier contract')
    if native_rows and (shape[-1] != 124160 or sampler.tt_sampling.vocab_size != 248320
            or sampler.tt_sampling.padded_vocab_size != 248320
            or sampler._penalties_active or getattr(sampler, '_log_probs_active', False)):
        raise ValueError('Native-row experiment requires unpadded Qwen TP2 vocabulary without penalties or logprobs')
    if rows > SAMPLER_ROWS:
        return sample_tiles(sampler, logits, rows, operations, shape)
    padded = logits
    owns_padding = False
    try:
        if rows < SAMPLER_ROWS and not native_rows:
            padded = operations.pad(logits, [(0, 0), (0, 0), (0, SAMPLER_ROWS - rows), (0, 0)], value=0.0)
            original_addresses, padded_addresses = addresses(operations, logits), addresses(operations, padded)
            if original_addresses != padded_addresses:
                if any(original == current for original, current in zip(original_addresses, padded_addresses, strict=True)):
                    raise ValueError('Padding must not partially alias input across chips')
                owns_padding = True
        return selected_ids(sampler.sample(padded, enable_trace=False))
    finally:
        if owns_padding:
            operations.deallocate(padded)
