"""Allocate prefix snapshots before native decode/prefill traces are captured."""

from prefill_gdn_checkpoint import PrefillGDNCheckpoint


def no_traces(value):
    if isinstance(value, dict):
        return all(no_traces(child) for child in value.values())
    return value is None


class CheckpointAllocation:
    def __init__(self, operations, checkpoint):
        self.operations, self.checkpoint = operations, checkpoint
        self.buffers = tuple(checkpoint.buffers)
        self.closed = False

    def close(self):
        if self.closed:
            return
        if self.checkpoint.phase in ('copying', 'restoring'):
            raise ValueError('Cannot release an in-flight prefix checkpoint')
        self.operations.synchronize_device(self.checkpoint.mesh)
        self.closed = True
        self.checkpoint.phase = 'closed'
        first_error = None
        for value in reversed(self.buffers):
            try:
                self.operations.deallocate(value)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def prepare(operations, generator, model):
    if (getattr(generator, 'model', None) != [model]
            or not isinstance(getattr(generator, 'trace_ids_decode', None), dict)
            or not no_traces(generator.trace_ids_decode)
            or getattr(model, '_chunked_trace_id', 'missing') is not None
            or getattr(model, '_bucket_trace_id', 'missing') is not None):
        raise ValueError('Single native model before all decode and prefill trace capture required')
    buffers = []
    previous = model._bind_gdn_prefill_scratch()
    try:
        layers = tuple(layer.attention for layer in model.layers if not layer.is_full_attention)
        if len(layers) != 48:
            raise ValueError('All native GDN prefill layers required')
        for layer in layers:
            for value in (layer.rec_state, *layer.conv_states, layer.conv_carry):
                buffers.append(operations.clone(value, memory_config=value.memory_config()))
        operations.synchronize_device(model.device)
        checkpoint = PrefillGDNCheckpoint(operations, model.device, layers, buffers)
        return CheckpointAllocation(operations, checkpoint)
    except BaseException:
        operations.synchronize_device(model.device)
        for value in reversed(buffers):
            operations.deallocate(value)
        raise
    finally:
        model._unbind_gdn_prefill_scratch(previous)
