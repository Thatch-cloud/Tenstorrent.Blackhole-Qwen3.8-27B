"""Single-stream MLP composition with two reusable weight FIFOs, not per-layer FIFO allocation."""

from types import SimpleNamespace

from attention_batch import Overlay
from gdn_multitoken_conv import addresses
from tensix_stream_projection import execute_projection, prepare_projection, validate_inputs
from tensix_weight_stream import stream_geometry


class StreamBufferPool:
    def __init__(self, operations, mesh, producers=8):
        if type(producers) is not int or producers not in (8, 16):
            raise ValueError('Explicit eight or sixteen producer pool required')
        self.operations = operations
        self.mesh = mesh
        self._producers = producers
        self.entries = {}

    @property
    def producers(self):
        return self._producers

    def expected_mapping(self, size):
        if type(size) is not int or size not in (36864, 34816):
            raise ValueError('Only the two reviewed full-MLP FIFO geometries are allowed')
        projection, blocks = ('gate', 20) if size == 36864 else ('down', 34)
        geometry = stream_geometry(projection, blocks, self.producers)
        operations = self.operations
        return [(operations.CoreCoord(*sender), operations.CoreRangeSet([
            operations.CoreRange(operations.CoreCoord(*geometry['coordinates'][index]),
                operations.CoreCoord(*geometry['coordinates'][index])) for index in indices]))
            for sender, indices in geometry['mapping']]

    def acquire(self, mesh, mapping, size):
        if mesh is not self.mesh or mapping != self.expected_mapping(size):
            raise ValueError('Shared FIFO must retain the original mesh and exact sender/receiver mapping')
        if size not in self.entries:
            result = self.operations.create_global_circular_buffer(mesh, mapping, size)
            if result.sender_core_type() != 'worker':
                raise ValueError('Worker-sender FIFO required')
            self.entries[size] = result
        return self.entries[size]


def prepare_mlp(operations, mesh, source, weights, buffers, pool, native_root):
    if set(weights) != {'gate', 'up', 'down'} or set(buffers) != {'gate', 'up', 'hidden', 'partial'}:
        raise ValueError('All three native weights and four caller-owned MLP buffers required')
    if not isinstance(pool, StreamBufferPool) or pool.mesh is not mesh or pool.operations is not operations:
        raise ValueError('Explicit same-mesh single-stream FIFO pool required')
    geometries = {name: stream_geometry(name, 34 if name == 'down' else 20, pool.producers) for name in weights}
    for name in ('gate', 'up', 'down'):
        validate_inputs(operations, mesh, buffers['hidden'] if name == 'down' else source, weights[name],
            buffers['partial'] if name == 'down' else buffers[name], geometries[name])
    references = [source, *(weights[name] for name in ('gate', 'up', 'down')),
        *(buffers[name] for name in ('gate', 'up', 'hidden', 'partial'))]
    bindings = [addresses(operations, value) for value in references]
    if any(len({binding[chip] for binding in bindings}) != len(references) for chip in range(2)):
        raise ValueError('All MLP input, weight and output buffers must be disjoint')
    pooled_operations = Overlay(operations, create_global_circular_buffer=pool.acquire)
    projections = {name: prepare_projection(pooled_operations, mesh,
        buffers['hidden'] if name == 'down' else source, weights[name],
        buffers['partial'] if name == 'down' else buffers[name], geometries[name], native_root)
        for name in ('gate', 'up', 'down')}
    if projections['gate'].gcb is not projections['up'].gcb or len(pool.entries) != 2:
        raise ValueError('Gate/up must share one FIFO; down must use the second')
    return SimpleNamespace(projections=projections, buffers=dict(buffers), references=references,
        bindings=bindings, pool=pool)


def execute_mlp(operations, prepared):
    if operations is not prepared.pool.operations:
        raise ValueError('Prepared operation namespace changed')
    if [addresses(operations, value) for value in prepared.references] != prepared.bindings:
        raise ValueError('Prepared MLP bindings changed')
    gate = execute_projection(operations, prepared.projections['gate'])
    up = execute_projection(operations, prepared.projections['up'])
    hidden = operations.multiply(gate, up, memory_config=operations.L1_MEMORY_CONFIG,
        output_tensor=prepared.buffers['hidden'])
    if addresses(operations, hidden) != addresses(operations, prepared.buffers['hidden']):
        raise ValueError('Native multiply replaced its caller-owned output')
    execute_projection(operations, prepared.projections['down'])
    return prepared.buffers


def execute_from_dram(operations, source, prepared):
    destination = prepared.references[0]
    if (operations is not prepared.pool.operations or tuple(source.shape) != (1, 1, 8, 5120)
            or source.dtype != operations.bfloat16 or source.layout != operations.TILE_LAYOUT
            or source.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Explicit T8 BF16 interleaved DRAM input required')
    if [addresses(operations, value) for value in prepared.references] != prepared.bindings:
        raise ValueError('Prepared MLP bindings changed before input copy')
    source_bindings = addresses(operations, source)
    if any(source_bindings[chip] in {binding[chip] for binding in prepared.bindings} for chip in range(2)):
        raise ValueError('Caller-owned DRAM input must not alias any MLP buffer')
    copied = operations.copy(source, destination)
    if addresses(operations, copied) != prepared.bindings[0]:
        raise ValueError('Native copy replaced the preallocated L1 input')
    return execute_mlp(operations, prepared)
