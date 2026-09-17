"""Unqualified T16 recurrence reading native B8 slot zero without a compact copy."""

import os
from pathlib import Path

import gdn_vsplit as split
from gdn_multitoken_conv import addresses, release_owned


def validate_inputs(operations, mesh, inputs):
    expected = ((1, 16, 5120), (1, 16, 24), (1, 16, 24),
        (8, 24, 128, 128), (1, 16, 3072), (1, 1, 128))
    if len(inputs) != len(expected) or tuple(tuple(value.shape) for value in inputs) != expected:
        raise ValueError('T16 TP2 inputs and full native B8 recurrent state required')
    if tuple(mesh.shape) != (1, 2):
        raise ValueError('Two-chip mesh required')
    if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT
            or value.memory_config() != operations.DRAM_MEMORY_CONFIG for value in inputs):
        raise ValueError('Interleaved DRAM BF16 TILE inputs required')
    bindings = [addresses(operations, value) for value in inputs]
    if any(len({binding[chip] for binding in bindings}) != len(inputs) for chip in range(2)):
        raise ValueError('Independent read-only input buffers required')
    grid = mesh.compute_with_storage_grid_size()
    split.core_coordinates(grid.x, grid.y, 96)
    return bindings


def execute(operations, mesh, qkv, beta, gate, native_state, *, z, norm_w, root, experimental=False):
    if experimental is not True:
        raise ValueError('Native-slot recurrence is an unqualified explicit experiment')
    inputs = [qkv, beta, gate, native_state, z, norm_w]
    bindings = validate_inputs(operations, mesh, inputs)
    from gdn_vsplit_norm_batch import load_kernels, validate_runtime
    runtime = Path(os.environ.get('TT_METAL_HOME', str(root)))
    split.validate_runtime(runtime)
    validate_runtime(runtime)
    kernels = load_kernels(root)
    owned = []

    def allocate(shape, dtype, layout, memory):
        value = operations.empty(shape, device=mesh, dtype=dtype, layout=layout, memory_config=memory)
        owned.append(value)
        return value

    try:
        bridge = allocate((16, 1, 96, 32), operations.float32, operations.ROW_MAJOR_LAYOUT, operations.DRAM_MEMORY_CONFIG)
        states = allocate((16, 24, 128, 128), operations.bfloat16, operations.TILE_LAYOUT, operations.DRAM_MEMORY_CONFIG)
        output = allocate((1, 16, 3072), operations.bfloat16, operations.TILE_LAYOUT, operations.L1_MEMORY_CONFIG)
        tensors = inputs[:4] + [bridge, states] + inputs[4:] + [output]
        all_bindings = [addresses(operations, value) for value in tensors]
        if any(len({binding[chip] for binding in all_bindings}) != len(tensors) for chip in range(2)):
            raise ValueError('Prefix/output buffers must not alias native state or other operands')
        shards = [operations.get_device_tensors(value) for value in tensors]
        for stage in ('recurrence', 'norm_gate'):
            program = split.build_program(operations, mesh, shards, kernels, stage, 16)
            operations.generic_op(tensors, program)
        if [addresses(operations, value) for value in inputs] != bindings:
            raise AssertionError('Native-slot recurrence changed input bindings')
        return output, states, bridge
    except BaseException:
        release_owned(operations, [value for value in owned
            if all(address not in {binding[chip] for binding in bindings}
                for chip, address in enumerate(addresses(operations, value)))])
        raise
