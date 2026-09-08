"""Prepared native MTP layer and optional shortlisted head; no target mutation."""

from attention_batch import capture_operation
from draft_shortlist_device import select_token
from force_argmax import sample_rows
from gdn_multitoken_conv import release_owned
from verifier_inputs import host_inputs


class MTPDeviceStep:
    def __init__(self, operations, model, mtp, embedding, page_table, sampler, *, shortlist=None):
        import torch

        if (model.num_devices != 2 or tuple(embedding.shape) != (248320, 5120)
                or embedding.device.type != 'cpu' or mtp.mesh is not model.mesh_device):
            raise ValueError('Pinned TP2 native MTP and full CPU embedding table required')
        self.operations, self.model, self.mtp = operations, model, mtp
        self.embedding, self.pages, self.sampler = embedding, page_table, sampler
        self.shortlist = shortlist
        self.mesh = model.mesh_device
        self.inputs, self.outputs, self.owned, self.traces = {}, {}, [], {}
        self.ready = self.closed = False
        mapper = operations.ReplicateTensorToMesh(self.mesh)
        _, positions, cosine, sine = host_inputs([0], 0, model.args.rope_head_dim, model.args.rope_theta)
        fixtures = dict(embedding=torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16),
                        hidden=torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16),
                        positions=positions, cosine=cosine, sine=sine)
        try:
            for name, value in fixtures.items():
                tensor = operations.from_torch(value, device=self.mesh,
                    dtype=operations.int32 if name == 'positions' else operations.bfloat16,
                    layout=operations.ROW_MAJOR_LAYOUT if name == 'positions' else operations.TILE_LAYOUT,
                    memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
                self.inputs[name] = tensor
        except BaseException:
            release_owned(operations, list(self.inputs.values()))
            raise

    def execute(self, select):
        operations = self.operations
        hidden = self.mtp.forward(self.inputs['embedding'], self.inputs['hidden'], self.inputs['positions'],
                                  self.inputs['cosine'], self.inputs['sine'], page_table=self.pages)
        self.owned.append(hidden)
        if not select:
            return hidden, None
        if self.shortlist is not None:
            tokens = select_token(operations, hidden, *self.shortlist, self.owned)
        else:
            logits = operations.linear(hidden, self.model.lm_head_weight)
            self.owned.append(logits)
            tokens = sample_rows(self.sampler, logits, 1, operations)
            self.owned.append(tokens)
        return hidden, tokens

    def prepare(self):
        if self.ready or self.closed:
            raise ValueError('Prepare MTP once before its prefill and before target verifier capture')
        try:
            for selected in (True, False):
                self.execute(selected)
                self.operations.synchronize_device(self.mesh)
                release_owned(self.operations, self.owned)
                self.owned.clear()
            for selected in (True, False):
                self.traces[selected], self.outputs[selected] = capture_operation(
                    self.operations, self.mesh, lambda selected=selected: self.execute(selected))
            self.ready = True
        except BaseException:
            self.close()
            raise

    def __call__(self, token, hidden, position, *, select):
        if (not self.ready or self.closed or type(select) is not bool
                or type(token) is not int or not 0 <= token < 248320
                or type(position) is not int or not 0 <= position < self.pages.shape[1] * 64
                or tuple(hidden.shape) != (1, 1, 1, 5120)):
            raise ValueError('Prepared MTP with valid token, position and one hidden row required')
        operations = self.operations
        _, positions, cosine, sine = host_inputs([token], position, self.model.args.rope_head_dim, self.model.args.rope_theta)
        fixtures = dict(embedding=self.embedding[token].reshape(1, 1, 1, 5120),
                        positions=positions, cosine=cosine, sine=sine)
        for name, value in fixtures.items():
            source = operations.from_torch(value,
                dtype=operations.int32 if name == 'positions' else operations.bfloat16,
                layout=operations.ROW_MAJOR_LAYOUT if name == 'positions' else operations.TILE_LAYOUT,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(source, self.inputs[name])
        operations.copy(hidden, self.inputs['hidden'])
        operations.execute_trace(self.mesh, self.traces[select], cq_id=0, blocking=True)
        output, identifiers = self.outputs[select]
        if not select:
            return output, None
        identifier = int(operations.to_torch(operations.get_device_tensors(identifiers)[0]).reshape(-1)[0])
        return output, identifier

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        for trace in self.traces.values():
            self.operations.release_trace(self.mesh, trace)
        release_owned(self.operations, self.owned)
        release_owned(self.operations, list(self.inputs.values()))
        self.closed = True
