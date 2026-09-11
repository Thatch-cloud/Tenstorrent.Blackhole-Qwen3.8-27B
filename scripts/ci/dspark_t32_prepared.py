"""Opt-in fixed-capacity proposal trace; target request integration and hardware speed are not yet qualified."""

from attention_batch import capture_operation
from dspark_t32_layer import execute as layer
from dspark_t32_device import T32DSparkDevice as DSparkDevice
from dspark_t32_inputs import proposal_inputs
from dspark_history import TensorScope, leaves
from dspark_inputs import VOCABULARY
from dspark_layer import norm
from dspark_t32_markov import execute as markov
from dspark_t32_target import noise_embeddings, pack_tokens, shared_head_logits
from gdn_multitoken_conv import addresses


def execute(device, inputs, history, retain):
    operations, mesh = device.operations, device.mesh
    layer_backend = getattr(device, 'proposal_layer', layer)
    hidden = noise_embeddings(operations, device.target, mesh, device.collectives, inputs['identifiers'],
        retain, proposals=device.max_drafts)
    for weights, cached in zip(device.layer_weights, history, strict=True):
        hidden = layer_backend(operations, mesh, device.collectives, hidden, cached, weights,
            (inputs['cosine'], inputs['sine']), inputs['mask'], inputs['live'], retain,
            position=device.history.capacity, proposals=device.max_drafts, mask_validated=True)['finish']['output']
    normalized = norm(operations, hidden, device.parameters['norm.weight'], retain)
    logits = shared_head_logits(operations, device.target, mesh, device.collectives, normalized,
        retain, proposals=device.max_drafts)
    owned = []
    try:
        records = markov(operations, inputs['anchor'], logits, device.predecessor, device.successor, owned)
    finally:
        for value in owned:
            retain(value)
    return dict(normalized=normalized, logits=logits, tokens=pack_tokens(operations, records, retain))


class PreparedDSparkProposal:
    def __init__(self, device, anchor, *, audit=False, defer_capture=False):
        if (type(audit) is not bool or type(defer_capture) is not bool or device.closed or device.history.pending is not None
                or not hasattr(device.history, 'capacity') or device.max_drafts != 31):
            raise ValueError('Idle fixed-bank drafter and explicit capture audit policy required')
        self.device, self.operations, self.mesh = device, device.operations, device.mesh
        self.audit, self.closed = audit, False
        self.trace = self.output_scope = None
        self.outputs = None
        self.checks = []
        borrowed = [*device.parameters.values(), *(value for weights in device.layer_weights for value in weights.values()),
            *leaves(device.history.layers), *leaves(device.history.spare_layers),
            device.predecessor, device.successor, device.target.lm_head_weight]
        self.input_scope = TensorScope(self.operations, borrowed)
        self.borrowed = borrowed
        self.inputs, self.history = {}, ()
        try:
            host = proposal_inputs(anchor, device.position, device.history.capacity, device.rotary)
            for name, value in host.items():
                integer = name in ('identifiers', 'anchor')
                self.inputs[name] = self.input_scope.retain(self.operations.from_torch(value,
                    dtype=self.operations.uint32 if integer else self.operations.float32 if name == 'live' else self.operations.bfloat16,
                    layout=self.operations.ROW_MAJOR_LAYOUT if integer else self.operations.TILE_LAYOUT,
                    device=self.mesh, mesh_mapper=self.operations.ReplicateTensorToMesh(self.mesh),
                    memory_config=self.operations.DRAM_MEMORY_CONFIG))
            self.history = self.allocate_history()
            self.bindings = [addresses(self.operations, value) for value in self.input_values()]
            warm = self.output_owner()
            try:
                self.expected_warmup = self.read_tokens(execute(device, self.inputs, self.history, warm.retain)['tokens'])
            finally:
                warm.release()
            if not defer_capture:
                self.capture()
        except BaseException:
            self.close()
            raise

    def capture(self):
        if self.closed or self.device.closed or self.trace is not None or self.output_scope is not None:
            raise ValueError('One capture for a live prepared proposal required')
        try:
            self.output_scope = self.output_owner()
            self.trace, self.outputs = capture_operation(self.operations, self.mesh,
                lambda: execute(self.device, self.inputs, self.history, self.output_scope.retain))
            self.operations.execute_trace(self.mesh, self.trace, cq_id=0, blocking=True)
            if self.read_tokens(self.outputs['tokens']) != self.expected_warmup:
                raise AssertionError('Prepared proposal capture differs from its fixed-layout eager warmup')
        except BaseException:
            self.close()
            raise

    def input_values(self):
        return [*self.inputs.values(), *leaves(self.history)]

    def allocate_history(self):
        return tuple(tuple(self.input_scope.retain(self.operations.clone(value,
            memory_config=self.operations.DRAM_MEMORY_CONFIG)) for value in pair) for pair in self.device.history.layers)

    def update_history(self):
        for actual, destination in zip(leaves(self.device.history.layers), leaves(self.history), strict=True):
            self.operations.copy(actual, destination)

    def output_owner(self):
        return TensorScope(self.operations, [*self.borrowed, *self.input_values()])

    def read_tokens(self, value):
        self.operations.synchronize_device(self.mesh)
        shards = self.operations.get_device_tensors(value)
        if len(shards) != 2:
            raise AssertionError('Both independently computed full-vocabulary proposal replicas required')
        replicas = [tuple(int(token) for token in self.operations.to_torch(shard).reshape(-1).tolist()) for shard in shards]
        if (replicas[0] != replicas[1] or len(replicas[0]) != self.device.max_drafts
                or any(not 0 <= token < VOCABULARY for token in replicas[0])):
            raise AssertionError('Incomplete, invalid or divergent prepared proposal IDs')
        return replicas[0]

    def update(self, anchor):
        device, operations = self.device, self.operations
        if self.closed or device.closed or device.history.pending is not None:
            raise ValueError('Proposal replay requires committed, live history')
        if [addresses(operations, value) for value in self.input_values()] != self.bindings:
            raise AssertionError('Prepared proposal input addresses changed')
        host = proposal_inputs(anchor, device.position, device.history.capacity, device.rotary)
        for name, value in host.items():
            destination = self.inputs[name]
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        self.update_history()

    def snapshot(self, outputs):
        import torch

        values = []
        for name in ('normalized', 'logits', 'tokens'):
            shards = self.operations.get_device_tensors(outputs[name])
            if len(shards) != 2:
                raise AssertionError('Both fixed-layout eager/replay output shards required')
            for shard in shards:
                value = self.operations.to_torch(shard).clone()
                if not torch.isfinite(value).all():
                    raise AssertionError('Nonfinite prepared proposal output')
                values.append(value)
        return tuple(values)

    def propose(self, anchor, count):
        import torch

        if self.trace is None:
            raise ValueError('Proposal trace must be captured before replay')
        if type(count) is not int or not 1 <= count <= self.device.max_drafts:
            raise ValueError('Bounded actual proposal count required')
        self.update(anchor)
        expected = None
        if self.audit:
            scope = self.output_owner()
            try:
                expected = self.snapshot(execute(self.device, self.inputs, self.history, scope.retain))
            finally:
                scope.release()
        self.operations.execute_trace(self.mesh, self.trace, cq_id=0, blocking=True)
        if expected is not None:
            actual = self.snapshot(self.outputs)
            if any(not torch.equal(value, reference) for value, reference in zip(actual, expected, strict=True)):
                raise AssertionError('Changing-input proposal replay differs from fixed-layout eager execution')
            self.checks.append(dict(position=self.device.position, tensors=len(actual), exact=True))
        return self.read_tokens(self.outputs['tokens'])[:count]

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.operations.synchronize_device(self.mesh)
        if self.trace is not None:
            self.operations.release_trace(self.mesh, self.trace)
            self.trace = None
        if self.output_scope is not None:
            self.output_scope.release()
        self.input_scope.release()


class TracedDSparkDevice(DSparkDevice):
    def __init__(self, *args, native_attention=True, **options):
        if native_attention is not True:
            raise ValueError('Explicit native proposal attention policy required')
        if native_attention:
            from dspark_t32_layer import execute as native_layer
            self.proposal_layer = native_layer
        self.prepared = None
        super().__init__(*args, **options)

    def prepare_trace(self, anchor, *, audit=False):
        if self.closed or self.prepared is not None:
            raise ValueError('One prepared proposal trace per live drafter required')
        self.prepared = PreparedDSparkProposal(self, anchor, audit=audit)

    def propose(self, anchor, count):
        return super().propose(anchor, count) if self.prepared is None else self.prepared.propose(anchor, count)

    def close(self):
        try:
            if self.prepared is not None:
                self.prepared.close()
        finally:
            super().close()
