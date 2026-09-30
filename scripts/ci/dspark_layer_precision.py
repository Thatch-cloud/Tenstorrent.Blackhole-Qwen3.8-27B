"""Explicit proposal-layer precision adapter; no global backend replacement."""

from dspark_projection_precision import ProjectionOperations


DIMENSIONS = ((5120, 2048), (5120, 512), (5120, 512), (2048, 5120),
    (5120, 8704), (5120, 8704), (8704, 5120))


class LayerOperations(ProjectionOperations):
    def matmul(self, value, weight, **options):
        shape, matrix = tuple(value.shape), tuple(weight.shape)
        if (self.calls >= len(DIMENSIONS) or len(shape) != 4 or shape[:3] != (1, 1, 32)
                or len(matrix) != 4 or matrix[:2] != (1, 1)
                or shape[-1] != matrix[-2] or matrix[-2:] != DIMENSIONS[self.calls]):
            raise ValueError('Exact ordered seven-projection proposal layer required')
        return super().matmul(value, weight, **options)


def execute(backend, operations, *args, **kwargs):
    if not callable(backend):
        raise ValueError('Explicit admitted proposal-layer backend required')
    selected = LayerOperations(operations)
    result = backend(selected, *args, **kwargs)
    if selected.calls != len(DIMENSIONS):
        raise ValueError('Every proposal-layer projection must execute exactly once')
    return result
