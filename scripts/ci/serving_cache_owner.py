"""Verify that serving and captured target execution share the same physical KV."""

import os

from gdn_multitoken_conv import addresses
import tp_shapes


class ServingCacheOwner:
    def __init__(self, operations, runner, model):
        self.operations, self.runner, self.model = operations, runner, model
        if (len(runner.model.model) != 1 or runner.model.model[0] is not model
                or model.num_devices != tp_shapes.chip_count() or model.args.max_batch_size != 8):
            raise ValueError('One TP%d target with native eight-slot GDN required' % tp_shapes.chip_count())
        self.bindings = self.inspect()
        self.physical_pages = self.bindings[0][1][0]
        # NP1 (QWEN_FAST_SOLO_PACKED=1, idle_scratch.py; default off, and then nothing is imported): the cache holds vLLM's pool plus the scratch blocks the lone-user padded round
        # parks its idle segments' K/V in. `physical_pages` stays the POOL (the ids a request's block table may name, which the bridge factory and the page binding bound-check), and
        # `scratch_pages` the blocks past it. A cache that is not exactly that fails the attach here.
        self.scratch_pages = ()
        if os.environ.get('QWEN_FAST_SOLO_PACKED', '0') != '0':
            import idle_scratch

            self.scratch_pages = idle_scratch.claim(self.physical_pages)
            self.physical_pages = self.scratch_pages[0]

    def inspect(self):
        caches = self.model._paged_kv_caches
        attention = [layer.attention for layer in self.model.layers if layer.is_full_attention]
        if len(caches) != 16 or len(attention) != 16 or len(self.runner.kv_caches) != 16:
            raise ValueError('All sixteen target attention-layer cache pairs required')
        bindings = []
        pages = None
        for layer, pair, serving_pair in zip(attention, caches, self.runner.kv_caches, strict=True):
            if len(pair) != 2 or len(serving_pair) != 2 or not layer.use_paged:
                raise ValueError('Complete model-bound paged K/V required')
            for target, serving, bound in zip(pair, serving_pair, (layer.paged_k, layer.paged_v), strict=True):
                if target is not serving or target is not bound:
                    raise ValueError('Serving, model and attention caches do not share ownership')
                shape = tuple(target.shape)
                if (len(shape) != 4 or shape[0] < 68 or shape[1:] != (tp_shapes.active().attn_kv_heads, 64, 256)
                        or target.dtype != self.operations.bfloat8_b):
                    raise ValueError('Qualified BF8 TP%d paged KV geometry required' % tp_shapes.chip_count())
                if pages is not None and shape[0] != pages:
                    raise ValueError('Inconsistent physical KV capacity')
                pages = shape[0]
                bindings.append((id(target), shape, tuple(addresses(self.operations, target))))
        return tuple(bindings)

    def validate(self):
        if (self.runner.model.model[0] is not self.model or self.inspect() != self.bindings):
            raise ValueError('Serving KV ownership or captured physical addresses changed')
