"""Experimental fixed-history T32 drafter; requires an explicitly prepared trace."""

from dspark_device import DSparkDevice
from dspark_inputs import VOCABULARY
from dspark_stable_history import StableHistoryKV
from dspark_t32_inputs import geometry


class T32DSparkDevice(DSparkDevice):
    def __init__(self, operations, target, collectives, parameters, layer_weights, predecessor, successor,
            chunks, rotary, *, position, proposals=31, history_capacity=None):
        geometry(position, proposals)
        if (type(history_capacity) is not int or not position <= history_capacity <= 8192
                or history_capacity % 32 or target.num_devices != 2 or target.vocab_size != VOCABULARY
                or getattr(target, '_lmhead_vocab_sharded', False) is not True):
            raise ValueError('Fixed complete history and vocabulary-sharded TP2 target required')
        self.operations, self.target, self.mesh, self.collectives = operations, target, target.mesh_device, collectives
        self.parameters, self.layer_weights = parameters, layer_weights
        self.predecessor, self.successor, self.rotary = predecessor, successor, rotary
        self.max_drafts, self.closed = proposals, False
        self.history = StableHistoryKV(operations, self.mesh, collectives, parameters, layer_weights, chunks, rotary,
            position=position, capacity=history_capacity)

    def propose(self, anchor, count):
        raise ValueError('T32 proposals require an explicitly prepared fixed-history trace')
