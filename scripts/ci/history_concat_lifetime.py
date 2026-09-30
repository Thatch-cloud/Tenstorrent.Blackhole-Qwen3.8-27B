"""Retire owned concat inputs after consumption; preserve borrowed history storage."""

from dspark_history import TensorScope
from gdn_multitoken_conv import addresses


def join_rows(operations, values, retain):
    scope = getattr(retain, '__self__', None)
    if not isinstance(scope, TensorScope) or scope.operations is not operations:
        raise ValueError('Explicit matching history TensorScope required')
    values = tuple(values)
    if not values:
        raise ValueError('Nonempty ordered history pieces required')
    identities = [addresses(operations, value) for value in values]
    if any(len({identity[chip] for identity in identities}) != len(identities) for chip in (0, 1)):
        raise ValueError('Consumed history pieces must have distinct storage')
    while len(values) > 1:
        next_values = []
        for start in range(0, len(values), 8):
            group = values[start:start + 8]
            if len(group) == 1:
                next_values.append(group[0])
                continue
            output = retain(operations.concat(list(group), dim=2,
                memory_config=operations.DRAM_MEMORY_CONFIG))
            output_ids = set(enumerate(addresses(operations, output)))
            for value in group:
                identity = addresses(operations, value)
                if set(enumerate(identity)) & (scope.protected | output_ids):
                    continue
                owned = scope.owned.pop(identity, None)
                if owned is not None:
                    operations.deallocate(owned)
            next_values.append(output)
        values = tuple(next_values)
    return values[0]
