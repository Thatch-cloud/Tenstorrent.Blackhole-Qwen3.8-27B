"""Experimental full learned DSpark KV history, projected once and extended only at publication."""

from types import SimpleNamespace

from dspark_intake import TAPS
from dspark_layer import SPECIFICATIONS, linear, norm
from dspark_mesh import gather_partials
from dspark_prefill import FeatureChunk, validate_chunks
from dspark_projection import normalize_partials, project, require_tensor
from dspark_rotary_device import execute as rotate
from gdn_multitoken_conv import addresses


class TensorScope:
    def __init__(self, operations, borrowed):
        self.operations = operations
        self.protected = {identity for value in borrowed for identity in enumerate(addresses(operations, value))}
        self.owned = {}

    def retain(self, value):
        identity = addresses(self.operations, value)
        overlap = set(enumerate(identity)) & self.protected
        if overlap and len(overlap) != len(identity):
            raise ValueError('Temporary partially aliases borrowed mesh storage')
        if not overlap:
            self.owned[identity] = value
        return value

    def release(self, keep=()):
        protected = self.protected | {identity for value in keep for identity in enumerate(addresses(self.operations, value))}
        owned, self.owned = self.owned, {}
        first_error = None
        for identity, value in owned.items():
            if set(enumerate(identity)) & protected:
                continue
            try:
                self.operations.deallocate(value)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def leaves(layers):
    return tuple(value for pair in layers for value in pair)


def join_rows(operations, values, retain):
    values = tuple(values)
    if not values:
        raise ValueError('Nonempty ordered history pieces required')
    while len(values) > 1:
        values = tuple(values[start] if len(values[start:start + 8]) == 1 else
            retain(operations.concat(list(values[start:start + 8]), dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
            for start in range(0, len(values), 8))
    return values[0]


def project_block(operations, mesh, collectives, features, parameters, layer_weights, tables, retain):
    projected = project(operations, dict(zip(TAPS, features, strict=True)), parameters['fc.weight'], retain)
    parts = gather_partials(operations, mesh, collectives, projected['partial'], retain)
    context = normalize_partials(operations, *parts, parameters['hidden_norm.weight'], retain, composed_norm=True)['context']
    layers = []
    for weights in layer_weights:
        pair = []
        for name in ('k', 'v'):
            hidden = linear(operations, context, weights[f'self_attn.{name}_proj.weight'], retain)
            shaped = retain(operations.reshape(hidden, (1, 32, 4, 128)))
            heads = retain(operations.transpose(shaped, 1, 2))
            if name == 'k':
                normalized = norm(operations, heads, weights['self_attn.k_norm.weight'], retain)
                rotary_owned = []
                try:
                    heads = rotate(operations, normalized, *tables, rotary_owned, composed=True)
                finally:
                    for value in rotary_owned:
                        retain(value)
            pair.append(heads)
        layers.append(tuple(pair))
    return tuple(layers)


def project_chunks(operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, start, rows):
    import torch

    validate_chunks(chunks, start=start, rows=rows)
    if (list(mesh.shape) != [1, 2] or len(layer_weights) != 5
            or any(set(weights) != set(SPECIFICATIONS) for weights in layer_weights)
            or not {'fc.weight', 'hidden_norm.weight'} <= set(parameters)):
        raise ValueError('Complete learned TP2 feature projection and five attention parameter sets required')
    borrowed = [*parameters.values(), *(value for weights in layer_weights for value in weights.values()),
        *(value for chunk in chunks for value in chunk.features)]
    output_scope = TensorScope(operations, borrowed)
    pieces = [[] for layer in range(5)]
    output = ()
    try:
        for chunk in chunks:
            for value in chunk.features:
                require_tensor(operations, value, tuple(value.shape), operations.bfloat16)
            for offset in range(0, chunk.rows, 32):
                valid = min(32, chunk.rows - offset)
                scope = TensorScope(operations, borrowed)
                retained = ()
                try:
                    features = [scope.retain(operations.slice(value, (0, 0, offset, 0), (1, 1, offset + 32, 2560)))
                        for value in chunk.features]
                    host_tables = [torch.ones(1, 1, 32, 128, dtype=torch.bfloat16),
                        torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)]
                    for table, actual in zip(host_tables, rotary.tables(chunk.start + offset, valid), strict=True):
                        table[:, :, :valid] = actual
                    tables = [scope.retain(operations.from_torch(table, dtype=operations.bfloat16,
                        layout=operations.TILE_LAYOUT, device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG,
                        mesh_mapper=operations.ReplicateTensorToMesh(mesh))) for table in host_tables]
                    layers = project_block(operations, mesh, collectives, features, parameters, layer_weights, tables, scope.retain)
                    layers = tuple(tuple(scope.retain(operations.slice(value, (0, 0, 0, 0), (1, 4, valid, 128)))
                        if valid < 32 else value for value in pair) for pair in layers)
                    retained = leaves(layers)
                    for layer, pair in enumerate(layers):
                        pieces[layer].append(pair)
                    for value in retained:
                        output_scope.retain(value)
                finally:
                    scope.release(keep=retained)
        output = tuple(tuple(join_rows(operations, [pair[index] for pair in layer], output_scope.retain)
            for index in range(2)) for layer in pieces)
        for value in leaves(output):
            require_tensor(operations, value, (1, 4, rows, 128), operations.bfloat16)
        operations.synchronize_device(mesh)
    except BaseException:
        output_scope.release()
        raise
    output_scope.release(keep=leaves(output))
    return output


class FullHistoryKV:
    def __init__(self, operations, mesh, collectives, parameters, layer_weights, chunks, rotary, *, position):
        if type(position) is not int or not 1 <= position <= 8192:
            raise ValueError('Full-history prefill frontier within the experimental capacity required')
        self.operations, self.mesh, self.collectives = operations, mesh, collectives
        self.parameters, self.layer_weights, self.rotary = parameters, layer_weights, rotary
        self.position, self.pending, self.closed = position, None, False
        self.layers = project_chunks(operations, mesh, collectives, parameters, layer_weights, chunks, rotary, start=0, rows=position)

    def prepare_publication(self, features, prefix, *, position):
        if (self.closed or self.pending is not None or type(position) is not int or position != self.position
                or type(prefix) is not int or not 1 <= prefix <= 32 or position + prefix > 8192):
            raise ValueError('One bounded feature publication at the exact committed frontier required')
        operations = self.operations
        features = tuple(features)
        if len(features) != len(TAPS):
            raise ValueError('All five complete verifier feature taps required')
        scope = TensorScope(operations, [*features, *leaves(self.layers)])
        next_layers = ()
        try:
            prepared = []
            for value in features:
                shape = tuple(value.shape)
                if len(shape) != 4 or shape[:2] != (1, 1) or shape[-1] != 2560 or not prefix <= shape[2] <= 32:
                    raise ValueError('Complete verified prefix features within one padded block required')
                require_tensor(operations, value, shape, operations.bfloat16)
                prepared.append(scope.retain(operations.pad(value, [(0, 0), (0, 0), (0, 32 - shape[2]), (0, 0)], 0.0))
                    if shape[2] != 32 else value)
            added = project_chunks(operations, self.mesh, self.collectives, self.parameters, self.layer_weights,
                (FeatureChunk(position, prefix, tuple(prepared)),), self.rotary, start=position, rows=prefix)
            for value in leaves(added):
                scope.retain(value)
            next_layers = tuple(tuple(join_rows(operations, [old, new], scope.retain)
                for old, new in zip(previous, delta, strict=True)) for previous, delta in zip(self.layers, added, strict=True))
            for value in leaves(next_layers):
                require_tensor(operations, value, (1, 4, position + prefix, 128), operations.bfloat16)
            operations.synchronize_device(self.mesh)
        except BaseException:
            scope.release()
            raise
        scope.release(keep=leaves(next_layers))
        self.pending = SimpleNamespace(owner=self, position=position, prefix=prefix, layers=next_layers, status='prepared')
        return self.pending

    def commit_publication(self, publication):
        self.validate_publication(publication)
        previous = self.layers
        self.layers, self.position = publication.layers, self.position + publication.prefix
        publication.status, self.pending = 'committed', None
        self.release_layers(previous)

    def discard_publication(self, publication):
        if getattr(publication, 'owner', None) is self and publication.status == 'committed':
            return
        self.validate_publication(publication)
        publication.status, self.pending = 'discarded', None
        self.release_layers(publication.layers)

    def validate_publication(self, publication):
        if (self.closed or publication is None or publication is not self.pending
                or publication.status != 'prepared' or publication.position != self.position):
            raise ValueError('Only the current prepared full-history publication may resolve')

    def release_layers(self, layers):
        scope = TensorScope(self.operations, ())
        for value in leaves(layers):
            scope.retain(value)
        scope.release()

    def close(self):
        if self.closed:
            return
        try:
            if self.pending is not None:
                self.discard_publication(self.pending)
        finally:
            self.closed = True
            layers, self.layers = self.layers, ()
            self.release_layers(layers)
