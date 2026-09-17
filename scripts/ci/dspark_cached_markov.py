"""Experimental request-owned exact Markov bias cache; not a serving default."""

import math

from dspark_markov_device import validate
from dspark_score_layout import execute as score_layout
from markov_cache_program import build as controller
from markov_cache_pipeline import build as plumbing


class BiasCache:
    def __init__(self, operations, mesh, predecessor, successor):
        import torch

        self.operations, self.mesh = operations, mesh
        self.predecessor, self.successor = predecessor, successor
        self.width = int(successor.shape[-1])
        if (self.width not in (64, 248320) or list(mesh.shape) != [1, 2]
                or tuple(predecessor.shape) != (1, 1, self.width, 256)
                or tuple(successor.shape) != (1, 1, 256, self.width)):
            raise ValueError('One fixed rank-256 Markov owner on two chips required')
        self.epoch, self.closed, self.owned = 1, False, []
        mapper = operations.ReplicateTensorToMesh(mesh)

        def allocate(host, dtype):
            result = operations.from_torch(host, dtype=dtype, layout=operations.ROW_MAJOR_LAYOUT,
                device=mesh, mesh_mapper=mapper, memory_config=operations.DRAM_MEMORY_CONFIG)
            self.owned.append(result)
            return result

        try:
            self.state = allocate(torch.zeros((1, 1, 65, 8), dtype=torch.int64), operations.uint32)
            self.request = allocate(torch.tensor([0, 248320, self.epoch] + [0] * 5).reshape(1, 1, 1, 8), operations.uint32)
            self.commit_request = allocate(torch.tensor([1] + [0] * 7).reshape(1, 1, 1, 8), operations.uint32)
            self.payload = allocate(torch.full((1, 1, 64, self.width), float('nan')), operations.float32)
        except BaseException:
            for tensor in reversed(self.owned):
                operations.deallocate(tensor)
            self.closed = True
            raise

    def reset(self):
        import torch

        if self.closed or self.epoch >= 0xffffffff:
            raise ValueError('Live cache with an unused epoch required')
        self.operations.synchronize_device(self.mesh)
        self.epoch += 1
        state = torch.zeros((1, 1, 65, 8), dtype=torch.int64)
        state[..., 0, 0] = self.epoch
        request = torch.tensor([0, 248320, self.epoch] + [0] * 5).reshape(1, 1, 1, 8)
        for host, destination in ((state, self.state), (request, self.request)):
            payload = self.operations.from_torch(host, dtype=self.operations.uint32,
                layout=self.operations.ROW_MAJOR_LAYOUT, mesh_mapper=self.operations.ReplicateTensorToMesh(self.mesh))
            self.operations.copy_host_to_device_tensor(payload, destination)
        self.operations.synchronize_device(self.mesh)

    def dot(self, anchor, latent, retain):
        if self.closed:
            raise ValueError('Live request-owned cache required')
        operations, mesh, width = self.operations, self.mesh, self.width
        if (tuple(latent.shape) not in ((1, 1, 256), (1, 1, 1, 256))
                or latent.dtype != operations.bfloat16 or latent.layout != operations.TILE_LAYOUT
                or latent.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('One native BF16 rank-256 embedding row required')

        def allocate(shape, dtype, layout):
            return retain(operations.empty(shape, dtype=dtype, layout=layout,
                device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG))

        decision, status = [allocate((1, 1, 1, 8), operations.uint32, operations.ROW_MAJOR_LAYOUT) for unused in range(2)]
        mask = allocate((1, 1, 1, 1), operations.bfloat16, operations.ROW_MAJOR_LAYOUT)
        bias, output = [allocate((1, 1, 1, width), operations.float32, operations.TILE_LAYOUT) for unused in range(2)]
        lookup = controller(mesh, self.state, self.request, decision, status, anchor=anchor)
        commit = controller(mesh, self.state, self.commit_request, decision, status)
        mask_program, payload_program = plumbing(mesh, self.state, decision, bias, self.payload, output, mask)
        grid = (2, 1) if width == 64 else (10, 10)
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
            in0_block_w=1, out_subblock_h=1, out_subblock_w=1, per_core_M=1,
            per_core_N=math.ceil(width / (32 * math.prod(grid))), fuse_batch=True, fused_activation=None, mcast_in0=True)
        kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        operations.generic_op([self.state, self.request, decision, status, anchor], lookup)
        operations.generic_op([decision, mask], mask_program)
        operations.sparse_matmul(latent, self.successor, sparsity=mask, program_config=program,
            is_input_a_sparse=True, is_input_b_sparse=True, compute_kernel_config=kernel,
            dtype=operations.float32, memory_config=operations.DRAM_MEMORY_CONFIG, optional_output_tensor=bias)
        operations.generic_op([self.state, decision, bias, self.payload, output], payload_program)
        operations.generic_op([self.state, self.commit_request, decision, status], commit)
        return output

    def close(self, *, traces_released=False):
        if self.closed:
            return
        if traces_released is not True:
            raise ValueError('Release every borrowing proposal trace before cache buffers')
        self.operations.synchronize_device(self.mesh)
        for tensor in reversed(self.owned):
            self.operations.deallocate(tensor)
        self.closed = True


def execute(cache, operations, mesh, anchor, base_logits, predecessor, successor, owned, *, on_step_enqueued=None):
    steps, width = validate(operations, anchor, base_logits, predecessor, successor)
    if (cache.closed or cache.operations is not operations or cache.mesh is not mesh
            or cache.predecessor is not predecessor or cache.successor is not successor or cache.width != width):
        raise ValueError('Exact request cache and weight allocation owner required')
    if on_step_enqueued is not None and not callable(on_step_enqueued):
        raise ValueError('Step observer must be callable')

    def retain(value):
        owned.append(value)
        return value

    previous, records = anchor, []
    for step in range(steps):
        embedding = retain(operations.embedding(previous, predecessor, layout=operations.ROW_MAJOR_LAYOUT))
        latent = retain(operations.to_layout(embedding, operations.TILE_LAYOUT))
        bias = cache.dot(previous, latent, retain)
        scores = score_layout(operations, mesh, base_logits, bias, step, retain)
        token = retain(operations.argmax(scores, dim=-1, keepdim=False))
        previous = retain(operations.reshape(token, (1, 1, 1, 1)))
        records.append(dict(token=token, scores=scores))
        if on_step_enqueued is not None:
            on_step_enqueued(step)
    return records
