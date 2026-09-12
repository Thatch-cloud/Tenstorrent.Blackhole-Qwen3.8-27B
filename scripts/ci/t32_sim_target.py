"""Target embedding/head only, with the same TP2 shard axes as the request path."""

from types import SimpleNamespace

from dspark_inputs import VOCABULARY
from gdn_multitoken_conv import release_owned
from t32_target_weights import HIDDEN_WIDTH, TargetWeights


def load(operations, mesh, directory, owned):
    if list(mesh.shape) != [1, 2] or not isinstance(owned, list):
        raise ValueError('Explicit TP2 mesh and caller-owned tensors required')
    reader = TargetWeights(directory)
    created = []
    try:
        embedding = reader.tensor('embedding').reshape(1, 1, VOCABULARY, HIDDEN_WIDTH)
        embedding_weight = operations.from_torch(embedding, device=mesh, dtype=operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=3))
        created.append(embedding_weight)
        del embedding
        head = reader.tensor('head').T.contiguous().reshape(1, 1, HIDDEN_WIDTH, VOCABULARY)
        head_weight = operations.from_torch(head, device=mesh, dtype=operations.bfloat16,
            layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG,
            mesh_mapper=operations.ShardTensorToMesh(mesh, dim=3))
        created.append(head_weight)
        del head
        operations.synchronize_device(mesh)
        reader.check_unchanged()

        def embed(identifiers, memory_config=None):
            return operations.embedding(identifiers, embedding_weight, layout=operations.TILE_LAYOUT,
                memory_config=memory_config)

        target = SimpleNamespace(mesh_device=mesh, num_devices=2, vocab_size=VOCABULARY,
            _lmhead_vocab_sharded=True, lm_head_weight=head_weight, embd=embed)
        owned.extend(created)
        return target, reader.manifest
    except BaseException:
        release_owned(operations, created)
        raise
