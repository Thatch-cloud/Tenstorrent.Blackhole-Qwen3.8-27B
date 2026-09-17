"""Simulator-only transaction audit with synthetic initial history and learned deltas."""

from unittest.mock import patch

from dspark_captured_publication import prepare
from dspark_history import leaves
from dspark_stable_history import StableHistoryKV
from gdn_multitoken_conv import addresses


class PublicationBankAudit:
    def __init__(self, operations, mesh, *, position=4096, capacity=4384):
        import os
        import torch

        if os.environ.get('QWEN_SIM_ONLY') != '1':
            raise ValueError('Synthetic bank initialization is simulator-only')
        self.operations, self.mesh = operations, mesh
        initial = []
        try:
            for operand in range(10):
                initial.append(operations.from_torch(torch.zeros(1, 4, position, 128, dtype=torch.bfloat16),
                    device=mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                    memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(mesh)))
        except BaseException:
            for value in initial:
                operations.deallocate(value)
            raise
        pairs = tuple(tuple(initial[index:index + 2]) for index in range(0, 10, 2))
        with patch('dspark_history.project_chunks', return_value=pairs):
            self.cache = StableHistoryKV(operations, mesh, None, {}, (), (), None,
                position=position, capacity=capacity)
        self.expected = self.snapshot(self.cache.layers)
        self.bindings = self.bank_addresses()
        self.checks = []

    def snapshot(self, layers):
        self.operations.synchronize_device(self.mesh)
        result = []
        for value in leaves(layers):
            shards = self.operations.get_device_tensors(value)
            if len(shards) != 2:
                raise AssertionError('Two bank shards required')
            result.extend(self.operations.to_torch(shard).clone() for shard in shards)
        return tuple(result)

    def bank_addresses(self):
        return sorted(addresses(self.operations, value)
            for value in leaves(self.cache.layers) + leaves(self.cache.spare_layers))

    def exercise(self, projection, features, tables, prefix, commit):
        import torch

        position = self.cache.position
        publication = prepare(self.cache, projection, features, tables, prefix, position=position)
        delta = projection.snapshot(projection.outputs)
        expected = tuple(value.clone() for value in self.expected)
        for value, added in zip(expected, delta, strict=True):
            value[..., position:position + prefix, :] = added[..., :prefix, :]
        for actual, golden in ((self.snapshot(self.cache.layers), self.expected),
                (self.snapshot(publication.layers), expected)):
            if any(not torch.equal(left, right) for left, right in zip(actual, golden, strict=True)):
                raise AssertionError('Captured publication changed active history or prepared the wrong prefix')
        if commit:
            self.cache.commit_publication(publication)
            self.expected = expected
        else:
            self.cache.discard_publication(publication)
        if self.cache.position != position + (prefix if commit else 0) or self.bank_addresses() != self.bindings:
            raise AssertionError('Publication frontier or persistent bank addresses changed incorrectly')
        self.checks.append(dict(position=position, prefix=prefix, committed=commit, exact=True))

    def close(self):
        self.cache.close()
