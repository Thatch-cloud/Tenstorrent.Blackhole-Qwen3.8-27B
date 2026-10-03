"""Unqualified T16 GDN partial-output placement experiment; no serving route."""

from contextlib import contextmanager

from model_batch import instance_overrides


class GDNOutputL1Arm:
    def __init__(self, operations, model, matmul):
        self.operations, self.matmul = operations, matmul
        self.layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        if len(self.layers) != 48 or not callable(matmul):
            raise ValueError('All 48 GDN layers and the native decode matmul helper required')
        if any(getattr(layer.args, 'proj_1d_decode', False) is not True for layer in self.layers):
            raise ValueError('Existing native 1D decode projection required')
        self.originals = [layer._row_proj for layer in self.layers]
        self.active = False
        self.hits = [0] * 48

    def forward(self, index, value, weight):
        if not self.active:
            raise RuntimeError('Explicit experimental scope required')
        layer = self.layers[index]
        if tuple(value.shape) != (1, 16, 3072) or weight is not layer.tw['out']:
            return self.originals[index](value, weight)
        result = self.matmul(value, weight, layer.args.gdn_out_decode_1d_progcfg,
            layer.cfg, out_memory_config=self.operations.L1_MEMORY_CONFIG)
        self.hits[index] += 1
        return result

    @contextmanager
    def install(self):
        if self.active:
            raise RuntimeError('Nested GDN output placement scope is unsupported')
        self.active = True
        try:
            with instance_overrides([(layer, '_row_proj',
                    lambda value, weight, index=index: self.forward(index, value, weight))
                    for index, layer in enumerate(self.layers)]):
                yield self
        finally:
            self.active = False
            if any(layer._row_proj != original for layer, original in zip(self.layers, self.originals, strict=True)):
                raise AssertionError('Native GDN projection binding was not restored')
