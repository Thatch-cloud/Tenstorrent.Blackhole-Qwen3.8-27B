"""Prepared, experimental GDN V-split operation for externally owned CQ0 traces.

Construct PreparedVSplit before capture, warm up with run(), and fence outside
run() before capture/readback as required by the caller. run() uses generic_op's
default CQ0 for both stages, in order. The shared queue and existing writer NOC
barriers supply the bridge dependency; run() does not fence or allocate device
storage. It returns the same (output, states, bridge) tuple on every invocation.

The caller owns all inputs, trace capture/replay/release, and inter-operation
fences. Keep this object and every bound tensor alive and address-stable while
any external trace refers to them. Finish capture, fence/release ALL such traces,
then close() before closing the mesh. This object cannot discover external traces
and never releases them. Trace replay bypasses Python binding/re-entry guards;
the caller must enforce the same lifetime and CQ0 ordering rules for replay.
Input contents may be updated in place on CQ0, but bindings must not change.

close() fences before releasing only its three owned tensors. Fence failure
retains all owned storage; release failure retains failed handles for another
close() attempt. No destructor performs implicit device work. Constructor errors
keep their original type and expose cleanup_errors; prepared_operation provides
a retryable cleanup handle if any allocated tensors could not be released.

The default output is DRAM, matching the existing eager prototype's placement.
output_memory=ttnn.L1_MEMORY_CONFIG changes only the final output placement and
requires NEW simulator and hardware validation. Prepared trace execution itself
also needs central validation; unchanged kernels do not certify trace behavior.
"""

import os
from pathlib import Path
from threading import Lock

import gdn_vsplit as split


class PreparedCleanupError(RuntimeError):
    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__('Owned output cleanup failed: ' + '; '.join(str(error) for error in errors))


def tensor_signature(tensor, operations):
    config = tensor.memory_config()
    if config == operations.DRAM_MEMORY_CONFIG:
        memory = 'dram'
    elif config == operations.L1_MEMORY_CONFIG:
        memory = 'l1'
    else:
        raise ValueError('Fixed tensor memory must be interleaved DRAM or L1')
    return (tuple(tensor.shape), tuple(tensor.padded_shape), tensor.dtype,
            tensor.layout, memory)


class PreparedVSplit:
    """Prepare once outside capture; explicitly opt in with experimental=True."""

    def __init__(self, mesh, qkv, beta, gate, initial, *, z, norm_w,
                 root=split.DEFAULT_ROOT, experimental=False, output_memory=None, operations=None):
        if experimental is not True:
            raise ValueError('Prepared V-split requires explicit experimental=True')
        if operations is None:
            import ttnn as operations

        self._operations = operations
        self._mesh = mesh
        self._lock = Lock()
        self._closed = False
        self._poisoned = False
        self._owned = []
        self._programs = ()
        self._shards = ()
        self._tensors = []
        self._result = None
        self._inputs = (qkv, beta, gate, initial, z, norm_w)
        try:
            split.validate_runtime(Path(os.environ.get('TT_METAL_HOME', str(split.DEFAULT_ROOT))))
            kernels = split.load_kernels(Path(root))
            self._rows = split.native.validate_geometry(*(tuple(value.shape) for value in self._inputs[:4]))
            if tuple(z.shape) != (1, self.rows, 3072) or tuple(norm_w.shape) != (1, 1, 128):
                raise ValueError('Expected z [1,T,3072] and norm_w [1,1,128]')
            if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT or
                   value.memory_config() != operations.DRAM_MEMORY_CONFIG for value in self._inputs):
                raise ValueError('Expected interleaved DRAM BF16 TILE inputs')
            self._mesh_signature = self._mesh_binding()
            if self._mesh_signature[0] != (1, 2):
                raise ValueError('Expected a 1x2 mesh')
            spec = split.stage_spec('recurrence', self.rows)
            split.core_coordinates(*self._mesh_signature[1:], spec['workers'])
            self._bindings(self._inputs)
            if output_memory is None:
                output_memory = operations.DRAM_MEMORY_CONFIG
            if output_memory not in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG):
                raise ValueError('Output memory must be interleaved DRAM or L1')
            pre_norm = self._allocate((self.rows, 1, 96, 32), operations.float32,
                                      operations.ROW_MAJOR_LAYOUT, operations.DRAM_MEMORY_CONFIG)
            states = self._allocate((self.rows, 24, 128, 128), operations.bfloat16,
                                    operations.TILE_LAYOUT, operations.DRAM_MEMORY_CONFIG)
            output = self._allocate((1, self.rows, 3072), operations.bfloat16,
                                    operations.TILE_LAYOUT, output_memory)
            self._tensors = list(self._inputs[:4]) + [pre_norm, states] + list(self._inputs[4:]) + [output]
            self._tensor_ids = tuple(id(value) for value in self._tensors)
            self._signatures, self._shards = self._bindings(self._tensors)
            self._programs = tuple(split.build_program(operations, mesh, self._shards, kernels, stage, self.rows)
                                   for stage in ('recurrence', 'norm_gate'))
            self._validate_bindings()
            self._result = (output, states, pre_norm)
        except BaseException as error:
            self._poisoned = True
            cleanup_errors = self._release_owned()
            error.cleanup_errors = tuple(cleanup_errors)
            if self._owned:
                error.prepared_operation = self
            for cleanup_error in cleanup_errors:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Constructor cleanup also failed: {cleanup_error!r}')
            if not self._owned:
                self._drop_references()
                self._closed = True
            if cleanup_errors and not hasattr(error, 'add_note'):
                raise error from PreparedCleanupError(cleanup_errors)
            raise

    @property
    def rows(self):
        return self._rows

    @property
    def poisoned(self):
        return self._poisoned

    @property
    def closed(self):
        return self._closed

    @property
    def output(self):
        return self._outputs()[0]

    @property
    def states(self):
        return self._outputs()[1]

    @property
    def pre_norm(self):
        return self._outputs()[2]

    @property
    def bridge(self):
        return self.pre_norm

    def _outputs(self):
        if self._result is None:
            raise RuntimeError('Prepared outputs are unavailable')
        return self._result

    def _allocate(self, shape, dtype, layout, memory):
        value = self._operations.empty(shape, device=self._mesh, dtype=dtype,
                                       layout=layout, memory_config=memory)
        if any(value is tensor for tensor in self._inputs):
            raise ValueError('Allocator returned a caller-owned input')
        if any(value is tensor for tensor in self._owned):
            raise ValueError('Allocator returned a duplicate owned tensor')
        self._owned.append(value)
        if tuple(value.shape) != shape or value.dtype != dtype or value.layout != layout or value.memory_config() != memory:
            raise ValueError('Allocated output signature differs from requested placement/geometry')
        return value

    def _mesh_binding(self):
        grid = self._mesh.compute_with_storage_grid_size()
        return tuple(self._mesh.shape), grid.x, grid.y

    def _bindings(self, tensors):
        signatures = []
        shards = []
        addresses = [set(), set()]
        for tensor in tensors:
            signature = tensor_signature(tensor, self._operations)
            local = tuple(self._operations.get_device_tensors(tensor))
            if len(local) != 2:
                raise ValueError('Both chips required for every fixed tensor')
            local_signatures = []
            for chip, shard in enumerate(local):
                local_signature = tensor_signature(shard, self._operations)
                if local_signature != signature:
                    raise ValueError(f'Tensor signature differs on chip {chip}')
                address = shard.buffer_address()
                if address <= 0 or address in addresses[chip]:
                    raise ValueError(f'Invalid or aliased fixed address on chip {chip}')
                addresses[chip].add(address)
                local_signatures.append((local_signature, address))
            signatures.append((signature, tuple(local_signatures)))
            shards.append(local)
        return tuple(signatures), tuple(shards)

    def _validate_bindings(self):
        if self._mesh_binding() != self._mesh_signature:
            raise ValueError('Prepared mesh signature changed')
        if tuple(id(value) for value in self._tensors) != self._tensor_ids:
            raise ValueError('Prepared tensor bindings changed')
        signatures, unused = self._bindings(self._tensors)
        if signatures != self._signatures:
            raise ValueError('Prepared tensor address/signature changed')

    def _enter(self):
        if not self._lock.acquire(blocking=False):
            self._poisoned = True
            raise RuntimeError('Prepared operation re-entry; object poisoned')

    def _require_ready(self):
        if self._closed:
            raise RuntimeError('Prepared operation is closed')
        if self._poisoned:
            raise RuntimeError('Prepared operation is poisoned')

    def run(self):
        """Enqueue two fixed programs on default CQ0; caller fences outside."""
        self._enter()
        try:
            self._require_ready()
            for program in self._programs:
                self._validate_bindings()
                self._require_ready()
                self._operations.generic_op(self._tensors, program)
                self._require_ready()
            return self._result
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._lock.release()

    def _release_owned(self):
        errors = []
        retained = []
        for value in reversed(self._owned):
            try:
                self._operations.deallocate(value)
            except BaseException as error:
                retained.append(value)
                errors.append(error)
        self._owned = list(reversed(retained))
        return errors

    def _drop_references(self):
        self._programs = ()
        self._shards = ()
        self._tensors = []
        self._inputs = ()
        self._result = None

    def close(self):
        """After caller releases external traces, fence and free owned outputs.

        Idempotent after success. Retryable on fence/cleanup failure, but run()
        stays poisoned. Never call during capture, enqueueing, or trace replay.
        """
        self._enter()
        try:
            if self._closed:
                return
            self._poisoned = True
            self._operations.synchronize_device(self._mesh)
            errors = self._release_owned()
            if errors:
                raise PreparedCleanupError(errors) from errors[0]
            self._drop_references()
            self._closed = True
        finally:
            self._lock.release()
