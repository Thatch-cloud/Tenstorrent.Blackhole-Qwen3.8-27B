"""Untraced native TTSampling force-argmax composition for an outer verifier trace."""

from gdn_multitoken_conv import addresses


def sample_rows(sampler, logits, rows, operations, *, native_rows=False):
    if type(native_rows) is not bool:
        raise ValueError('Explicit boolean native-row sampling required')
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Supported verifier width required')
    shape = tuple(logits.shape)
    if len(shape) != 4 or shape[:3] != (1, 1, rows):
        raise ValueError('Expected pre-gather vocab-sharded verifier logits')
    if not sampler.tt_sampling.force_argmax_sampling or sampler.tt_sampling.max_batch_size != 32:
        raise ValueError('Pinned 32-row force-argmax sampler required')
    if sampler.seed_manager.has_active_request_seed():
        raise ValueError('Seeded sampling is outside the greedy verifier contract')
    if native_rows and (shape[-1] != 124160 or sampler.tt_sampling.vocab_size != 248320
            or sampler.tt_sampling.padded_vocab_size != 248320
            or sampler._penalties_active or getattr(sampler, '_log_probs_active', False)):
        raise ValueError('Native-row experiment requires unpadded Qwen TP2 vocabulary without penalties or logprobs')
    padded = logits
    owns_padding = False
    try:
        if rows < 32 and not native_rows:
            padded = operations.pad(logits, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], value=0.0)
            original_addresses, padded_addresses = addresses(operations, logits), addresses(operations, padded)
            if original_addresses != padded_addresses:
                if any(original == current for original, current in zip(original_addresses, padded_addresses, strict=True)):
                    raise ValueError('Padding must not partially alias input across chips')
                owns_padding = True
        result = sampler.sample(padded, enable_trace=False)
        if isinstance(result, tuple):
            if len(result) != 2 or result[1] is not None:
                raise ValueError('Greedy verifier does not consume log-probability outputs')
            return result[0]
        return result
    finally:
        if owns_padding:
            operations.deallocate(padded)
