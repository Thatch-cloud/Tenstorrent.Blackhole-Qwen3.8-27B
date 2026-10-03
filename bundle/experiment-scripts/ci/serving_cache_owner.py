"""Verify that serving and captured target execution share the same physical KV."""

from gdn_multitoken_conv import addresses


class ServingCacheOwner:
    def __init__(self, operations, runner, model):
        self.operations, self.runner, self.model = operations, runner, model
        if (len(runner.model.model) != 1 or runner.model.model[0] is not model
                or model.num_devices != 2 or model.args.max_batch_size != 8):
            raise ValueError('One TP2 target with native eight-slot GDN required')
        self.bindings = self.inspect()
        self.physical_pages = self.bindings[0][1][0]

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
                if (len(shape) != 4 or shape[0] < 68 or shape[1:] != (2, 64, 256)
                        or target.dtype != self.operations.bfloat8_b):
                    raise ValueError('Qualified BF8 TP2 paged KV geometry required')
                if pages is not None and shape[0] != pages:
                    raise ValueError('Inconsistent physical KV capacity')
                pages = shape[0]
                bindings.append((id(target), shape, tuple(addresses(self.operations, target))))
        return tuple(bindings)

    def validate(self):
        if (self.runner.model.model[0] is not self.model or self.inspect() != self.bindings):
            raise ValueError('Serving KV ownership or captured physical addresses changed')
