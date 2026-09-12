"""Prepare a verified prefix from borrowed captured outputs without advancing history."""

from dspark_history import TensorScope, leaves
from dspark_projection import require_tensor


def prepare(history, projection, features, tables, prefix, *, position):
    history.check_prefix(prefix, position)
    if projection.operations is not history.operations or projection.mesh is not history.mesh:
        raise ValueError('Projection and history must share the same runtime and mesh')
    outputs = projection.project(features, tables)
    if len(outputs) != 5 or any(len(pair) != 2 for pair in outputs):
        raise ValueError('All five captured K/V pairs required')
    operations = history.operations
    for value in leaves(outputs):
        require_tensor(operations, value, (1, 4, 32, 128), operations.bfloat16)
    scope = TensorScope(operations, leaves(outputs) + leaves(history.layers) + leaves(history.spare_layers))
    try:
        added = tuple(tuple(scope.retain(operations.slice(value, (0, 0, 0, 0), (1, 4, prefix, 128)))
            if prefix < 32 else value for value in pair) for pair in outputs)
        return history.prepare_projected(added, prefix, position=position)
    finally:
        scope.release()
