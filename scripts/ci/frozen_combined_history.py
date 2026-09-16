"""Admitted 32K constructor bounds; retain historical capture and projection math."""


def require_position(position):
    from dspark_8k_admission import history_limit
    if history_limit() != 33024 or type(position) is not int or position != 32768:
        raise ValueError('Admitted exact 32K full-history frontier required')


def initialise_history(instance, operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, position):
    from dspark_history import project_chunks
    require_position(position)
    instance.operations, instance.mesh, instance.collectives = operations, mesh, collectives
    instance.parameters, instance.layer_weights, instance.rotary = parameters, layer_weights, rotary
    instance.position, instance.pending, instance.closed = position, None, False
    instance.layers = project_chunks(operations, mesh, collectives, parameters, layer_weights,
        chunks, rotary, start=0, rows=position)


def prefill_capture_class(original):
    class AdmittedCapture(original):
        def __init__(self, operations, model, position):
            require_position(position)
            if not callable(getattr(model, '_forward_prefill_chunk_masked_tp', None)):
                raise ValueError('Pinned native masked prefill boundary required')
            self.operations, self.model, self.position = operations, model, position
            self.children, self.chunks = [], []
            self.cursor = 0
            self.started = self.active = self.complete = self.closed = False

    return AdmittedCapture
