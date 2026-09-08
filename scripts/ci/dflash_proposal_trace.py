"""Pre-verifier capture of fixed-context five-layer proposal computation."""

from types import SimpleNamespace

from attention_batch import capture_operation
from dflash_proposal_inputs import proposal_contexts, proposal_inputs
from gdn_multitoken_conv import addresses, release_owned


class PreparedDFlashProposal:
    def __init__(self, device, *, max_new_tokens):
        import torch

        self.device, self.operations, self.mesh = device, device.operations, device.mesh
        self.buckets, self.owned, self.checks = {}, [], []
        self.closed = False
        operations = self.operations
        def upload(value, *, identifiers=False):
            tensor = operations.from_torch(value, device=self.mesh,
                dtype=operations.uint32 if identifiers else operations.bfloat16,
                layout=operations.ROW_MAJOR_LAYOUT if identifiers else operations.TILE_LAYOUT,
                memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            self.owned.append(tensor)
            return tensor
        try:
            for context in proposal_contexts(device.position, max_new_tokens):
                host = proposal_inputs(0, device.position, device.history_rows, device.block_rows, context)
                bucket = SimpleNamespace(context=context, identifiers=upload(host['identifiers'], identifiers=True),
                    history=upload(torch.zeros((1, 1, context + 32, 5120), dtype=torch.bfloat16)),
                    mask=upload(host['mask']), rope={name: tuple(upload(value) for value in host['rope'][name]) for name in ('q', 'k')},
                    trace=None, outputs=None, owned=[])
                bucket.inputs = [bucket.identifiers, bucket.history, bucket.mask, *bucket.rope['q'], *bucket.rope['k']]
                bucket.addresses = [addresses(operations, value) for value in bucket.inputs]
                self.buckets[context] = bucket
            for bucket in self.buckets.values():
                self.update(bucket, 0)
                transient, retain = device.temporaries([*device.owned, *self.owned])
                try:
                    self.execute(bucket, transient, retain)
                    operations.synchronize_device(self.mesh)
                finally:
                    release_owned(operations, transient)
                bucket.owned, retain = device.temporaries([*device.owned, *self.owned])
                bucket.trace, bucket.outputs = capture_operation(operations, self.mesh,
                    lambda: self.execute(bucket, bucket.owned, retain))
                operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
        except BaseException:
            self.close()
            raise

    def execute(self, bucket, owned, retain):
        return self.device.execute_proposal(bucket.identifiers, bucket.history, bucket.mask, bucket.rope,
            context=bucket.context, owned=owned, retain=retain, stage=lambda name, **values: None, audit=False)

    def update(self, bucket, seed):
        operations, device = self.operations, self.device
        host = proposal_inputs(seed, device.position, device.history_rows, device.block_rows, bucket.context)
        sources = [host['identifiers'], host['mask'], *host['rope']['q'], *host['rope']['k']]
        destinations = [bucket.identifiers, bucket.mask, *bucket.rope['q'], *bucket.rope['k']]
        for value, destination in zip(sources, destinations, strict=True):
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        owned, retain = device.temporaries([device.history, device.spare_history, *self.owned])
        try:
            history = retain(operations.slice(device.history, (0, 0, 0, 0), (1, 1, bucket.context, 5120)))
            padded = retain(operations.pad(history, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0))
            operations.copy(padded, bucket.history)
            operations.synchronize_device(self.mesh)
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared proposal input addresses moved')
        finally:
            release_owned(operations, owned)

    def propose(self, seed, count):
        import torch

        if self.closed:
            raise ValueError('Closed proposal capture cannot replay')
        device, operations = self.device, self.operations
        bucket = next((value for context, value in self.buckets.items() if context >= device.history_rows), None)
        if bucket is None:
            raise ValueError('Committed history exceeds prepared request contexts')
        self.update(bucket, seed)
        expected = None
        if device.progress is not None:
            owned, retain = device.temporaries([*device.owned, *self.owned])
            try:
                outputs = self.execute(bucket, owned, retain)
                operations.synchronize_device(self.mesh)
                expected = device.proposal_snapshot(outputs)
            finally:
                release_owned(operations, owned)
        operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
        if expected is not None:
            actual = device.proposal_snapshot(bucket.outputs)
            if len(actual) != len(expected) or any(not torch.equal(left, right) for left, right in zip(actual, expected, strict=True)):
                raise AssertionError('Captured learned proposal differs from the same fixed-context eager execution')
            self.checks.append(dict(position=device.position, context=bucket.context, tensors=len(actual), exact=True))
        return device.select_proposal(bucket.outputs, seed, count)

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        for bucket in self.buckets.values():
            if bucket.trace is not None:
                self.operations.release_trace(self.mesh, bucket.trace)
                bucket.trace = None
        for bucket in self.buckets.values():
            release_owned(self.operations, bucket.owned)
            bucket.owned.clear()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True
