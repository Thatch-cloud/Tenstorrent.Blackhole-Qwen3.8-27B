"""Captured native MTP proposal chain with device embedding feedback."""

from attention_batch import capture_operation
from force_argmax import sample_rows
from gdn_multitoken_conv import addresses, release_owned
from verifier_inputs import host_inputs


def feedback_embedding(operations, embed, gather, identifiers, owned):
    """The native gather callback consumes its DRAM input allocation."""
    if (tuple(identifiers.shape) not in ((1, 1, 1), (1, 1, 1, 1)) or identifiers.dtype != operations.uint32
            or identifiers.layout != operations.ROW_MAJOR_LAYOUT):
        raise ValueError('One native global UINT32 token required for device embedding feedback')
    indices = operations.reshape(identifiers, (1, 1))
    original, reshaped = addresses(operations, identifiers), addresses(operations, indices)
    if original != reshaped:
        if any(first == second for first, second in zip(original, reshaped, strict=True)):
            raise ValueError('Embedding index view must not partially alias its borrowed token')
        owned.append(indices)
    embedded = embed(indices, memory_config=operations.DRAM_MEMORY_CONFIG)
    local = operations.reshape(embedded, (1, 1, 1, embedded.shape[-1]))
    embedded_ids, local_ids = addresses(operations, embedded), addresses(operations, local)
    if embedded_ids != local_ids:
        if any(first == second for first, second in zip(embedded_ids, local_ids, strict=True)):
            raise ValueError('Embedding reshape must not partially alias its consuming gather input')
        owned.append(embedded)
    gathered = gather(local)
    owned.append(gathered)
    if gathered.shape[-1] != 5120:
        raise ValueError('Complete replicated target embedding width required')
    result = operations.reshape(gathered, (1, 1, 1, 5120))
    owned.append(result)
    return result


def collect_tokens(operations, identifiers, owned):
    if not identifiers or any(tuple(value.shape) not in ((1, 1, 1), (1, 1, 1, 1)) for value in identifiers):
        raise ValueError('Complete native single-row proposal outputs required')
    rows = [operations.reshape(value, (1, 1, 1, 1)) for value in identifiers]
    owned.extend(rows)
    result = operations.concat(rows, dim=3, memory_config=operations.DRAM_MEMORY_CONFIG)
    owned.append(result)
    return result


class MTPDeviceChain:
    def __init__(self, device_step, *, max_drafts=7):
        import torch

        if (not device_step.ready or device_step.closed or not device_step.native_sampling_rows
                or device_step.shortlist is not None or type(max_drafts) is not int
                or max_drafts not in (1, 3, 7)):
            raise ValueError('Prepared native-row full-vocabulary MTP step and bounded chain required')
        self.step = device_step
        self.operations, self.mesh = device_step.operations, device_step.mesh
        self.max_drafts = max_drafts
        self.inputs, self.metadata, self.outputs, self.traces, self.owned = {}, [], {}, {}, []
        self.ready = self.closed = False
        operations = self.operations
        mapper = operations.ReplicateTensorToMesh(self.mesh)
        try:
            for name, value, dtype, layout in (
                    ('seed', torch.zeros((1, 1, 1, 1), dtype=torch.int32), operations.uint32, operations.ROW_MAJOR_LAYOUT),
                    ('hidden', torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16), operations.bfloat16, operations.TILE_LAYOUT)):
                self.inputs[name] = operations.from_torch(value, device=self.mesh, dtype=dtype, layout=layout,
                    memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            for offset in range(max_drafts):
                _, positions, cosine, sine = host_inputs([0], offset, device_step.model.args.rope_head_dim,
                                                         device_step.model.args.rope_theta)
                row = {}
                self.metadata.append(row)
                for name, value in (('positions', positions), ('cosine', cosine), ('sine', sine)):
                    row[name] = operations.from_torch(value, device=self.mesh,
                        dtype=operations.int32 if name == 'positions' else operations.bfloat16,
                        layout=operations.ROW_MAJOR_LAYOUT if name == 'positions' else operations.TILE_LAYOUT,
                        memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        except BaseException:
            self.close()
            raise

    def execute(self, count):
        operations, step = self.operations, self.step
        hidden, identifiers = self.inputs['hidden'], self.inputs['seed']
        selected = []
        for metadata in self.metadata[:count]:
            embedded = feedback_embedding(operations, step.model.embd, step.mtp._gather, identifiers, self.owned)
            hidden = step.mtp.forward(embedded, hidden, metadata['positions'], metadata['cosine'], metadata['sine'],
                                      page_table=step.pages)
            self.owned.append(hidden)
            logits = operations.linear(hidden, step.model.lm_head_weight)
            self.owned.append(logits)
            identifiers = sample_rows(step.sampler, logits, 1, operations, native_rows=True)
            self.owned.append(identifiers)
            selected.append(identifiers)
        return collect_tokens(operations, selected, self.owned)

    def prepare(self):
        if self.ready or self.closed:
            raise ValueError('Prepare proposal chains once, before MTP prompt initialization')
        try:
            for count in range(1, self.max_drafts + 1):
                self.execute(count)
                self.operations.synchronize_device(self.mesh)
                release_owned(self.operations, self.owned)
                self.owned.clear()
            for count in range(1, self.max_drafts + 1):
                self.traces[count], self.outputs[count] = capture_operation(
                    self.operations, self.mesh, lambda count=count: self.execute(count))
            self.ready = True
        except BaseException:
            self.close()
            raise

    def __call__(self, token, hidden, position, count):
        import torch

        if (not self.ready or self.closed or type(count) is not int or count not in self.traces
                or type(token) is not int or not 0 <= token < 248320
                or type(position) is not int or position < 1 or position - 1 + count > self.step.pages.shape[1] * 64
                or tuple(hidden.shape) != (1, 1, 1, 5120)):
            raise ValueError('Prepared chain at a valid target-to-MTP aligned request position required')
        operations = self.operations
        values = [(self.inputs['seed'], torch.tensor([[[[token]]]], dtype=torch.int32), operations.uint32,
                   operations.ROW_MAJOR_LAYOUT)]
        for offset, row in enumerate(self.metadata[:count]):
            _, positions, cosine, sine = host_inputs([0], position - 1 + offset,
                self.step.model.args.rope_head_dim, self.step.model.args.rope_theta)
            values.extend((row[name], value,
                           operations.int32 if name == 'positions' else operations.bfloat16,
                           operations.ROW_MAJOR_LAYOUT if name == 'positions' else operations.TILE_LAYOUT)
                          for name, value in (('positions', positions), ('cosine', cosine), ('sine', sine)))
        for destination, value, dtype, layout in values:
            source = operations.from_torch(value, dtype=dtype, layout=layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(source, destination)
        operations.copy(hidden, self.inputs['hidden'])
        operations.execute_trace(self.mesh, self.traces[count], cq_id=0, blocking=True)
        selected = operations.to_torch(operations.get_device_tensors(self.outputs[count])[0]).reshape(-1).tolist()
        if len(selected) != count or any(type(identifier) is not int or not 0 <= identifier < 248320 for identifier in selected):
            raise ValueError('Device chain returned invalid global proposal tokens')
        return tuple(selected)

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        for trace in self.traces.values():
            self.operations.release_trace(self.mesh, trace)
        release_owned(self.operations, self.owned)
        release_owned(self.operations, list(self.inputs.values()) + [value for row in self.metadata for value in row.values()])
        self.closed = True
