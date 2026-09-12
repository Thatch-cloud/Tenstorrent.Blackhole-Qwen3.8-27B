"""Request-scoped T16 pipeline with preparation storage retained through trace release."""

from contextlib import contextmanager

import gdn_vsplit as split
from gdn_shared_qk_pipeline import build


@contextmanager
def scoped_shared_qk(operations, admission):
    original = split.execute
    if getattr(original, '_shared_qk_override', False):
        raise ValueError('Nested shared-Q/K experiments are forbidden')
    retained = []
    audit = dict(admission=admission, loads=[], restored=False, released=False)

    def execute(mesh, qkv, beta, gate, initial, *, z, norm_w, root=split.DEFAULT_ROOT,
                experimental=False, batch_norm=False, synchronize=True, output_memory=None):
        if tuple(qkv.shape) != (1, 16, 5120):
            return original(mesh, qkv, beta, gate, initial, z=z, norm_w=norm_w, root=root,
                experimental=experimental, batch_norm=batch_norm, synchronize=synchronize,
                output_memory=output_memory)
        if (experimental is not True or batch_norm is not True or synchronize is not False
                or output_memory != operations.L1_MEMORY_CONFIG):
            raise ValueError('Shared Q/K requires unfenced T16 batch norm with L1 output')
        owned = []
        def allocate(shape, dtype, layout, memory):
            tensor = operations.empty(shape, device=mesh, dtype=dtype, layout=layout, memory_config=memory)
            owned.append(tensor)
            return tensor
        try:
            bridge = allocate((16, 1, 96, 32), operations.float32, operations.ROW_MAJOR_LAYOUT, operations.DRAM_MEMORY_CONFIG)
            states = allocate((16, 24, 128, 128), operations.bfloat16, operations.TILE_LAYOUT, operations.DRAM_MEMORY_CONFIG)
            output = allocate((1, 16, 3072), operations.bfloat16, operations.TILE_LAYOUT, operations.L1_MEMORY_CONFIG)
            query = allocate((1, 16, 1024), operations.float32, operations.TILE_LAYOUT, operations.DRAM_MEMORY_CONFIG)
            key = allocate((1, 16, 1024), operations.float32, operations.TILE_LAYOUT, operations.DRAM_MEMORY_CONFIG)
            tensors = [qkv, beta, gate, initial, bridge, states, z, norm_w, output, query, key]
            programs = build(operations, mesh, tensors, root=root)
            for operands, program in programs:
                operations.generic_op(operands, program)
        except BaseException:
            retained.extend((mesh, tensor) for tensor in owned)
            raise
        retained.extend(((mesh, query), (mesh, key)))
        audit['loads'].append(dict(rows=16, programs=len(programs), retained_preparation_buffers=2))
        return output, states, bridge

    execute._shared_qk_override = True
    split.execute = execute
    try:
        yield audit
    finally:
        unchanged = split.execute is execute
        if unchanged:
            split.execute = original
            audit['restored'] = True
        meshes = {id(mesh): mesh for mesh, tensor in retained}
        for mesh in meshes.values():
            operations.synchronize_device(mesh)
        while retained:
            operations.deallocate(retained[-1][1])
            retained.pop()
        audit['released'] = True
        if not unchanged:
            raise ValueError('GDN executor changed externally during experiment')
