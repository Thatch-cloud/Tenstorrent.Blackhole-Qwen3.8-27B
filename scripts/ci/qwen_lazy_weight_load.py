"""Defer TP weight transposes until TTNN actually needs a cache rebuild."""

from contextlib import ExitStack, contextmanager
from functools import wraps
import inspect
import os
from unittest.mock import patch


def lazy_shard_loader(original, operations, torch_module, record):
    @wraps(original)
    def load(torch_tensor, mesh, dim, memory_config, cache_path, dtype=operations.bfloat8_b):
        record['calls'] += 1

        def preprocess(tensor):
            record['materializations'] += 1
            return tensor.to(torch_module.bfloat16).T.contiguous()

        return operations.as_tensor(torch_tensor, dtype=dtype, device=mesh,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=dim),
            layout=operations.TILE_LAYOUT, memory_config=memory_config,
            cache_file_name=cache_path, preprocess=preprocess)

    return load


def lazy_gate_up_loader(original, operations, torch_module, prepare, record):
    @wraps(original)
    def load(gate_w, up_w, mesh, tp, cache_path):
        record['packed_calls'] += 1

        def preprocess(gate):
            record['packed_materializations'] += 1
            gate_transposed = gate.to(torch_module.bfloat16).T.contiguous()
            up_transposed = up_w.to(torch_module.bfloat16).T.contiguous()
            packed = torch_module.cat([gate_transposed, up_transposed], dim=-1)
            return prepare(packed, ndev=tp, gate_is_first=True)

        return operations.as_tensor(gate_w, dtype=operations.bfloat4_b, device=mesh,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=-1),
            layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_path, preprocess=preprocess)

    return load


@contextmanager
def lazy_weight_load(module, operations, torch_module, records, mlp=None, prepare=None):
    if os.environ.get('QWEN_LAZY_WEIGHT_LOAD') != '1':
        raise ValueError('Explicit lazy-weight experiment required')
    original = module.shard_w
    parameters = tuple(inspect.signature(original).parameters)
    if parameters != ('torch_tensor', 'mesh', 'dim', 'memory_config', 'cache_path', 'dtype'):
        raise ValueError('Pinned TP shard loader signature required')
    if (mlp is None) != (prepare is None):
        raise ValueError('Packed loader and native packing function must be provided together')
    packed_original = mlp._build_gate_up if mlp is not None else None
    if packed_original is not None and tuple(inspect.signature(packed_original).parameters) != (
            'gate_w', 'up_w', 'mesh', 'tp', 'cache_path'):
        raise ValueError('Pinned packed MLP loader signature required')
    record = dict(calls=0, materializations=0, packed_calls=0, packed_materializations=0, restored=False,
        correctness_qualified=False, performance_qualified=False)
    records.append(record)
    replacement = lazy_shard_loader(original, operations, torch_module, record)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, 'shard_w', replacement))
            if mlp is not None:
                stack.enter_context(patch.object(mlp, '_build_gate_up', lazy_gate_up_loader(
                    packed_original, operations, torch_module, prepare, record)))
            yield
            if record['calls'] == 0:
                raise ValueError('Lazy shard loader was not exercised')
    finally:
        record['restored'] = module.shard_w is original and (
            mlp is None or mlp._build_gate_up is packed_original)
