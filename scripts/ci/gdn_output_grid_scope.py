"""Opt-in GDN output-column redistribution with unchanged native DRAM output."""

from gdn_output_l1_scope import GDNOutputL1Arm
from output_projection_grid import widen


class GDNOutputGridArm(GDNOutputL1Arm):
    def __init__(self, operations, model, matmul):
        super().__init__(operations, model, matmul)
        self.programs = [widen(operations, layer.args.gdn_out_decode_1d_progcfg) for layer in self.layers]

    def forward(self, index, value, weight):
        if not self.active:
            raise RuntimeError('Explicit experimental scope required')
        layer = self.layers[index]
        if tuple(value.shape) != (1, 16, 3072) or weight is not layer.tw['out']:
            return self.originals[index](value, weight)
        result = self.matmul(value, weight, self.programs[index], layer.cfg,
            out_memory_config=self.operations.DRAM_MEMORY_CONFIG)
        self.hits[index] += 1
        return result
