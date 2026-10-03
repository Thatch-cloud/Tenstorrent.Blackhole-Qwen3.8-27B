"""T8 target-precision MLP with one small-tile conversion at each boundary."""


def execute(operations, source, weights, programs, compute, retain, *, tiny, retile=None, multiply=None):
    if type(tiny) is not bool or tuple(source.shape) != (1, 1, 8, 5120) or source.dtype != operations.bfloat16:
        raise ValueError('Native T8 BF16 MLP input required')
    if (set(weights) != {'gate', 'up', 'down'} or set(programs) != set(weights)
            or any(weights[name].dtype != expected for name, expected in (
                ('gate', operations.bfloat4_b), ('up', operations.bfloat4_b), ('down', operations.bfloat8_b)))):
        raise ValueError('Complete unchanged target-precision MLP weights and programs required')
    if tiny and (not callable(retile) or not callable(multiply)):
        raise ValueError('Prepared DMA boundaries and small-tile product required')
    value = retile(source, 16) if tiny else source
    state = {}
    for name in ('gate', 'up'):
        state[name] = retain(operations.linear(value, weights[name], program_config=programs[name],
            compute_kernel_config=compute, memory_config=operations.L1_MEMORY_CONFIG))
    state['hidden'] = (multiply(state['gate'], state['up']) if tiny else
        retain(operations.mul(state['gate'], state['up'], memory_config=operations.L1_MEMORY_CONFIG)))
    partial = retain(operations.linear(state['hidden'], weights['down'], program_config=programs['down'],
        compute_kernel_config=compute, memory_config=operations.L1_MEMORY_CONFIG))
    state['partial'] = retile(partial, 32) if tiny else partial
    return state
