"""Opt-in append-only vLLM page allocation binding for captured T16 verifier metadata."""

from gdn_multitoken_conv import addresses


def validate_initial_capture_pages(pages, blocks, *, position, output_budget, rows=16):
    blocks = tuple(blocks)
    if (type(position) is not int or position < 1 or type(output_budget) is not int
            or output_budget < 2 or type(rows) is not int or rows != 16
            or not blocks or len(set(blocks)) != len(blocks)
            or any(type(block) is not int or block < 0 for block in blocks)):
        raise ValueError('Explicit unique scheduler allocation and T16 request bounds required')
    if (pages.ndim != 2 or pages.shape[0] != 1
            or pages.shape[1] * 64 < position + output_budget - 1
            or len(blocks) > pages.shape[1]
            or len(blocks) * 64 < position + min(rows, output_budget - 1)):
        raise ValueError('Scheduler must own capture warmup pages before verifier allocation')
    values = pages[0].tolist()
    if (tuple(values[:len(blocks)]) != blocks
            or any(type(value) is not int or value not in blocks for value in values)):
        raise ValueError('Captured page table must reference only this request allocation')


class VerifierPageBinding:
    def __init__(self, engine, initial_blocks, *, physical_pages):
        self.engine = engine
        self.operations, self.mesh = engine.operations, engine.mesh
        self.physical_pages = physical_pages
        self.failed = False
        if (engine.phase != 'idle' or type(physical_pages) is not int or physical_pages < 1
                or engine.pages.ndim != 2 or engine.pages.shape[0] != 1):
            raise ValueError('Idle verifier and one bounded host page table required')
        self.capacity = engine.pages.shape[1]
        self.blocks = self.validate_blocks(initial_blocks)
        if tuple(engine.pages[0, :len(self.blocks)].tolist()) != self.blocks:
            raise ValueError('Captured initial pages differ from scheduler-owned blocks')
        tensors = []
        for bucket in engine.buckets.values():
            fixture = bucket['fixture']
            tensors.extend((fixture.pages, fixture.singleton_pages))
            replay = getattr(fixture, 'replay_reader', None)
            if fixture.grouped_readers != ([] if replay is None else [replay]):
                raise ValueError('Unrecognized grouped attention page ownership')
            if replay is not None:
                if replay.audit is not None:
                    raise ValueError('Instrumented attention is not a serving page-binding path')
                tensors.extend(entry[1] for entry in replay.metadata)
            for reader in fixture.readers:
                if reader is replay:
                    continue
                if any(value is not fixture.singleton_pages for value in reader.pages):
                    raise ValueError('Unexpected singleton attention page binding')
            for writer in fixture.writers:
                if hasattr(writer, 'pages') and any(value is not fixture.singleton_pages for value in writer.pages):
                    raise ValueError('Unexpected singleton cache-writer page binding')
        if not tensors:
            raise ValueError('Captured verifier page metadata required')
        self.bindings = {}
        for tensor in tensors:
            shape = tuple(tensor.shape)
            identity = tuple(addresses(self.operations, tensor))
            if (len(shape) != 2 or shape[0] < 1 or not 1 <= shape[1] <= self.capacity
                    or len(identity) != 2):
                raise ValueError('Bounded two-chip page metadata required')
            if identity in self.bindings and self.bindings[identity][1] != shape:
                raise ValueError('Aliased page metadata has conflicting geometry')
            self.bindings[identity] = (tensor, shape)

    def validate_blocks(self, blocks):
        blocks = tuple(blocks)
        if (not blocks or len(blocks) > self.capacity or len(set(blocks)) != len(blocks)
                or any(type(block) is not int or not 0 <= block < self.physical_pages for block in blocks)):
            raise ValueError('Unique bounded scheduler-owned physical pages required')
        return blocks

    def refresh(self, blocks, *, position, rows):
        import torch

        if self.failed or self.engine.phase != 'idle':
            raise ValueError('Only an idle unpoisoned verifier may bind pages')
        blocks = self.validate_blocks(blocks)
        if (type(position) is not int or position < 0 or type(rows) is not int or rows not in (1, 2, 4, 8, 16)
                or (position + rows + 63) // 64 > len(blocks)
                or blocks[:len(self.blocks)] != self.blocks):
            raise ValueError('Append-only allocation must cover every scheduled verifier row')
        for identity, (tensor, shape) in self.bindings.items():
            if tuple(addresses(self.operations, tensor)) != identity or tuple(tensor.shape) != shape:
                self.failed = True
                self.engine.phase = 'failed'
                raise ValueError('Captured page metadata addresses changed')
        if blocks == self.blocks:
            return False
        host = torch.full((1, self.capacity), blocks[0], dtype=torch.int32)
        host[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        try:
            for tensor, shape in self.bindings.values():
                values = host[:, :shape[1]].repeat(shape[0], 1).contiguous()
                source = self.operations.from_torch(values, dtype=self.operations.int32,
                    layout=self.operations.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.operations.ReplicateTensorToMesh(self.mesh))
                self.operations.copy_host_to_device_tensor(source, tensor)
            self.operations.synchronize_device(self.mesh)
            for identity, (tensor, _) in self.bindings.items():
                if tuple(addresses(self.operations, tensor)) != identity:
                    raise ValueError('Page upload replaced a captured device buffer')
            self.engine.pages.copy_(host)
            self.blocks = blocks
            return True
        except BaseException:
            self.failed = True
            self.engine.phase = 'failed'
            raise
