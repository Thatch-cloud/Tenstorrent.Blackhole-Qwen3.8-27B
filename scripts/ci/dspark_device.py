"""Opt-in full-history DSpark request drafter; eager integration prototype, not a serving default."""

from dspark_cached_layer import execute as layer
from dspark_full_attention import full_mask, geometry, validate_mask
from dspark_history import FullHistoryKV, TensorScope, leaves
from dspark_inputs import VOCABULARY
from dspark_layer import norm
from dspark_markov_device import execute as markov
from dspark_wide_target import noise_embeddings, pack_tokens, query_inputs, shared_head_logits


class DSparkDevice:
    def __init__(self, operations, target, collectives, parameters, layer_weights, predecessor, successor,
            chunks, rotary, *, position, proposals=15):
        geometry(position, proposals)
        if (target.num_devices != 2 or target.vocab_size != VOCABULARY
                or getattr(target, '_lmhead_vocab_sharded', False) is not True):
            raise ValueError('Borrowed complete TP2 target required')
        self.operations, self.target, self.mesh, self.collectives = operations, target, target.mesh_device, collectives
        self.parameters, self.layer_weights = parameters, layer_weights
        self.predecessor, self.successor, self.rotary = predecessor, successor, rotary
        self.max_drafts, self.closed = proposals, False
        self.history = FullHistoryKV(operations, self.mesh, collectives, parameters, layer_weights, chunks, rotary, position=position)

    @property
    def position(self):
        return self.history.position

    def propose(self, anchor, count):
        import torch

        if self.closed or self.history.pending is not None or type(count) is not int or not 1 <= count <= self.max_drafts:
            raise ValueError('Idle full-history drafter and bounded proposal count required')
        identifiers = query_inputs(anchor, self.position, self.max_drafts)
        mask = full_mask(self.position, self.max_drafts)
        validate_mask(mask, self.position, self.max_drafts)
        operations = self.operations
        scope = TensorScope(operations, [*self.parameters.values(),
            *(value for weights in self.layer_weights for value in weights.values()), *leaves(self.history.layers),
            self.predecessor, self.successor, self.target.lm_head_weight])

        def upload(value, *, dtype=None, row_major=False):
            return scope.retain(operations.from_torch(value, dtype=operations.bfloat16 if dtype is None else dtype,
                layout=operations.ROW_MAJOR_LAYOUT if row_major else operations.TILE_LAYOUT, device=self.mesh,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh), memory_config=operations.DRAM_MEMORY_CONFIG))

        try:
            device_ids = upload(identifiers, dtype=operations.uint32, row_major=True)
            device_anchor = upload(torch.tensor([[[[anchor]]]], dtype=torch.int64), dtype=operations.uint32, row_major=True)
            device_mask = upload(mask)
            host_tables = [torch.ones(1, 1, 32, 128, dtype=torch.bfloat16), torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)]
            for table, source in zip(host_tables, self.rotary.tables(self.position, self.max_drafts), strict=True):
                table[:, :, :self.max_drafts] = source
            tables = tuple(upload(value) for value in host_tables)
            live = torch.zeros(1, 1, 32, 1, dtype=torch.float32)
            live[:, :, :self.max_drafts] = 1
            device_live = upload(live, dtype=operations.float32)
            hidden = noise_embeddings(operations, self.target, self.mesh, self.collectives, device_ids,
                scope.retain, proposals=self.max_drafts)
            for weights, cached in zip(self.layer_weights, self.history.layers, strict=True):
                hidden = layer(operations, self.mesh, self.collectives, hidden, cached, weights, tables, device_mask,
                    device_live, scope.retain, position=self.position, proposals=self.max_drafts, mask_validated=True)['finish']['output']
            normalized = norm(operations, hidden, self.parameters['norm.weight'], scope.retain)
            logits = shared_head_logits(operations, self.target, self.mesh, self.collectives, normalized,
                scope.retain, proposals=self.max_drafts)
            owned = []
            try:
                records = markov(operations, device_anchor, logits, self.predecessor, self.successor, owned)
            finally:
                for value in owned:
                    scope.retain(value)
            packed = pack_tokens(operations, records, scope.retain)
            operations.synchronize_device(self.mesh)
            shards = operations.get_device_tensors(packed)
            if len(shards) != 2:
                raise ValueError('Both independently computed proposal replicas required')
            values = [tuple(int(token) for token in operations.to_torch(shard).reshape(-1).tolist()) for shard in shards]
            if (values[0] != values[1] or len(values[0]) != self.max_drafts
                    or any(not 0 <= token < VOCABULARY for token in values[0])):
                raise AssertionError('Incomplete, invalid or divergent full-vocabulary DSpark proposals')
            return values[0][:count]
        finally:
            scope.release()

    def prepare_publication(self, features, prefix, *, position):
        return self.history.prepare_publication(features, prefix, position=position)

    def commit_publication(self, publication):
        return self.history.commit_publication(publication)

    def discard_publication(self, publication):
        return self.history.discard_publication(publication)

    def close(self):
        if not self.closed:
            self.closed = True
            self.history.close()
