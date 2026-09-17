"""Opt-in T8 full-verifier MLP scope; no class-global patch, serving default or collective-policy change."""

from contextlib import contextmanager

from gdn_multitoken_conv import addresses
from model_batch import instance_overrides
from tensix_mlp_collective import reduce_partial
from tensix_stream_mlp import StreamBufferPool, execute_mlp, prepare_mlp


class StreamedMlpScope:
    def __init__(self, operations, model, pool, workspace, native_root):
        self.operations, self.model, self.pool = operations, model, pool
        self.mesh, self.native_root = model.mesh_device, native_root
        self.workspace = dict(workspace)
        if (list(self.mesh.shape) != [1, 2] or len(model.layers) != 64
                or set(workspace) != {'gate', 'up', 'hidden', 'partial'}
                or not isinstance(pool, StreamBufferPool) or pool.mesh is not self.mesh
                or pool.operations is not operations or set(pool.entries) != {36864, 34816}):
            raise ValueError('Explicit two-FIFO pool, four caller-owned buffers and complete TP2 target required')
        self.feed_forwards = tuple(layer.feed_forward for layer in model.layers)
        if len({id(module) for module in self.feed_forwards}) != 64:
            raise ValueError('All 64 distinct native MLP instances required')
        for module in self.feed_forwards:
            compute, args = module.compute_kernel_config_decode, module.args
            if (module.device is not self.mesh or module.num_devices != 2 or not module._mlp_1d_decode
                    or module._dram_sharded or module.tt_ccl is not model.tt_ccl
                    or (args.dim, args.hidden_dim, args.decode_grid_w) != (5120, 17408, 11)
                    or compute.math_fidelity != operations.MathFidelity.LoFi
                    or not compute.math_approx_mode or not compute.fp32_dest_acc_en or not compute.packer_l1_acc
                    or args.ccl_topology() != operations.Topology.Linear):
                raise ValueError('Every MLP must retain the reviewed native decode mode, math and shared collective')
        if any(model.tt_ccl.get_num_links(axis) != 4 for axis in (None, 0, 1)):
            raise ValueError('Control and candidate must already share an explicit four-link target policy')
        self.bindings = {name: addresses(operations, value) for name, value in self.workspace.items()}
        if any(len({binding[chip] for binding in self.bindings.values()}) != 4 for chip in range(2)):
            raise ValueError('Shared MLP outputs must be independent physical buffers')
        self.active, self.failed = False, False
        self.visited, self.completed_forwards, self.converted_inputs = 0, 0, 0

    def validate_workspace(self):
        if (set(self.pool.entries) != {36864, 34816}
                or self.bindings != {name: addresses(self.operations, value) for name, value in self.workspace.items()}):
            raise ValueError('Shared workspace or FIFO pool changed')

    def forward(self, index, source):
        operations = self.operations
        if (not self.active or self.failed or type(index) is not int or index != self.visited or not 0 <= index < 64
                or tuple(source.shape) != (1, 1, 8, 5120) or source.dtype != operations.bfloat16
                or source.layout != operations.TILE_LAYOUT):
            raise ValueError('One ordered T8 call per target MLP is required inside the explicit scope')
        self.validate_workspace()
        original_bindings = addresses(operations, source)
        interleaved = operations.to_memory_config(source, operations.L1_MEMORY_CONFIG)
        converted_bindings = addresses(operations, interleaved)
        aliases = [converted_bindings[chip] == original_bindings[chip] for chip in range(2)]
        if any(aliases) and not all(aliases):
            raise ValueError('Mixed per-chip conversion ownership is unsupported')
        try:
            module = self.feed_forwards[index]
            weights = dict(gate=module.weights.w1, up=module.weights.w3, down=module.weights.w2)
            prepared = prepare_mlp(operations, self.mesh, interleaved, weights, self.workspace, self.pool, self.native_root)
            result = execute_mlp(operations, prepared)
            output = reduce_partial(operations, self.mesh, module.tt_ccl, result['partial'], module.args.ccl_topology())
            self.visited += 1
            self.converted_inputs += int(not all(aliases))
            return output
        finally:
            if not all(aliases):
                operations.deallocate(interleaved)

    @contextmanager
    def capture(self):
        if self.active or self.failed or hasattr(self.model, '_qwen_streamed_mlp_scope'):
            raise RuntimeError('Only one healthy streamed-MLP scope may own this target forward')
        self.validate_workspace()
        self.active, self.visited = True, 0
        bindings = [(self.model, '_qwen_streamed_mlp_scope', self)]
        bindings.extend((module, 'forward', lambda source, index=index: self.forward(index, source))
            for index, module in enumerate(self.feed_forwards))
        try:
            with instance_overrides(bindings):
                yield
                if self.visited != 64:
                    raise AssertionError('The captured target forward must engage all 64 streamed MLPs')
                self.validate_workspace()
            self.completed_forwards += 1
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    def summary(self):
        return dict(rows=8, layers=64, pool_buffers=len(self.pool.entries),
            completed_forwards=self.completed_forwards, converted_inputs=self.converted_inputs,
            failed=self.failed, serving_defaults_changed=False,
            scope='Host engagement of full-verifier MLP adapter; not device correctness or performance evidence')
