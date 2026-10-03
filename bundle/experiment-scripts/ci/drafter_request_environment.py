"""Shared target callbacks for a loaded two-card drafter comparison."""

import math
import os
from pathlib import Path

from dspark_projection import tensor_digest
from gdn_multitoken_conv import addresses


def prepare(operations, generator, model, collectives, pages, kv_cache):
    import torch
    from full_request import terminal_ids
    from gdn_snapshot import ActiveSnapshot
    from models.common.sampling.generator import SamplingGenerator

    sampler = SamplingGenerator(args=model.args, mesh_device=model.mesh_device, tt_ccl=collectives)
    sampler.set_trace_bucket(1)
    layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    helpers = [ActiveSnapshot(layer, operations, direct=True) for layer in layers]
    recurrent = [value for layer in layers for value in (layer.rec_state, *layer.conv_states)]
    caches = [value for pair in model._paged_kv_caches for value in pair]
    if (len(helpers), len(recurrent), len(caches)) != (48, 240, 32):
        raise ValueError('Complete two-card hybrid target state required')
    bindings = [addresses(operations, value) for value in (*recurrent, *caches)]

    def check_bindings():
        if [addresses(operations, value) for value in (*recurrent, *caches)] != bindings:
            raise ValueError('Target cache bindings changed between drafters')

    def host(value):
        shards = operations.get_device_tensors(value)
        if len(shards) != 2:
            raise ValueError('Both physical chips must participate')
        return [operations.to_torch(shard).clone() for shard in shards]

    def live_digest():
        check_bindings()
        return [tensor_digest(shard) for value in recurrent for shard in host(value)]

    def inactive_digest():
        check_bindings()
        return [tensor_digest(shard[1:] if index % 5 == 0 else shard[:, 1:])
            for index, value in enumerate(recurrent) for shard in host(value)]

    def kv_digest(valid):
        check_bindings()
        if type(valid) is not int or not 1 <= valid <= 4352:
            raise ValueError('Initial 4K comparison requires a bounded valid KV prefix')
        result = []
        for value in caches:
            for start in range(0, math.ceil(valid / 64), 64):
                end = min(start + 64, math.ceil(valid / 64))
                sliced = operations.slice(value, (start, 0, 0, 0), (end, value.shape[1], 64, value.shape[3]))
                try:
                    for shard in host(sliced):
                        logical = shard.permute(1, 0, 2, 3).reshape(shard.shape[1], -1, shard.shape[3])
                        result.append(tensor_digest(logical[:, :min((end - start) * 64, valid - start * 64)]))
                finally:
                    if addresses(operations, sliced) != addresses(operations, value):
                        operations.deallocate(sliced)
        return result

    def prefill(tokens):
        generator.prev_page_table = None
        logits, unused = generator.prefill_forward(torch.tensor([tokens], dtype=torch.int32), pages, kv_cache,
            [len(tokens)], empty_slots=[0], enable_trace=False)
        return int(logits.reshape(-1, model.args.vocab_size)[0].float().argmax())

    def decode(token, position, traced):
        if traced and not generator.trace_ids_decode[False]:
            raise ValueError('Cold native trace capture cannot occur inside a control decode')
        output = generator.decode_forward(tokens=torch.tensor([[token]], dtype=torch.int32),
            start_pos=torch.tensor([position], dtype=torch.int32), page_table=pages, kv_cache=kv_cache,
            enable_trace=traced, read_from_device=True)
        return (output[0] if isinstance(output, tuple) else output).clone()

    return sampler, helpers, dict(prefill=prefill, decode=decode, live_digest=live_digest,
        kv_digest=kv_digest, inactive_digest=inactive_digest,
        eos_ids=terminal_ids(Path(os.environ['MODEL_WEIGHTS_DIR']), model.args.vocab_size))
