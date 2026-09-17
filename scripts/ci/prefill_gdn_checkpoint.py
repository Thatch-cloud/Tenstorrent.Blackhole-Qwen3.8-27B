"""Preallocated B=1 GDN checkpoint component; not a complete prefix cache."""


class PrefillGDNCheckpoint:
    def __init__(self, operations, mesh, layers, buffers):
        self.operations, self.mesh = operations, mesh
        self.layers, self.buffers = tuple(layers), tuple(buffers)
        self.phase, self.position = 'empty', None
        self.live = self.current()
        if len(self.buffers) != len(self.live):
            raise ValueError('Preallocated recurrent, convolution and carry buffers required')
        for live, saved in zip(self.live, self.buffers, strict=True):
            if self.metadata(live) != self.metadata(saved):
                raise ValueError('Checkpoint must preserve tensor geometry, layout and precision')
        self.bindings = self.storage((*self.live, *self.buffers))
        for chip in (0, 1):
            addresses = [binding[chip] for binding in self.bindings]
            if len(set(addresses)) != len(addresses):
                raise ValueError('Checkpoint and live state must own distinct storage on each chip')

    def current(self):
        if len(self.layers) != 48 or any(layer.B != 1 or not layer._stable_state
                or len(layer.conv_states) != 4 or layer.conv_carry is None for layer in self.layers):
            raise ValueError('All 48 bound B=1 stable GDN prefill layers required')
        return tuple(value for layer in self.layers
            for value in (layer.rec_state, *layer.conv_states, layer.conv_carry))

    @staticmethod
    def metadata(value):
        return tuple(value.shape), value.dtype, value.layout

    def storage(self, values):
        result = []
        for value in values:
            parts = self.operations.get_device_tensors(value)
            if len(parts) != 2:
                raise ValueError('Both physical chip allocations required')
            result.append(tuple(part.buffer_address() for part in parts))
        return tuple(result)

    def validate(self):
        if self.phase not in ('empty', 'ready'):
            raise ValueError('Busy or failed checkpoint cannot be reused')
        live = self.current()
        if self.storage((*live, *self.buffers)) != self.bindings:
            raise ValueError('Prefill scratch or checkpoint storage changed')
        if any(self.metadata(current) != self.metadata(saved)
                for current, saved in zip(live, self.buffers, strict=True)):
            raise ValueError('Checkpoint metadata changed')
        return live

    def capture(self, position):
        if type(position) is not int or not 0 < position <= 262144 or position % 128:
            raise ValueError('Aligned completed GDN prefill boundary required')
        live = self.validate()
        self.phase, self.position = 'copying', None
        try:
            self.operations.synchronize_device(self.mesh)
            for source, destination in zip(live, self.buffers, strict=True):
                self.operations.copy(source, destination)
            self.operations.synchronize_device(self.mesh)
            self.position, self.phase = position, 'ready'
        except BaseException:
            self.phase = 'failed'
            raise

    def restore(self, position):
        live = self.validate()
        if self.phase != 'ready' or type(position) is not int or position != self.position:
            raise ValueError('Exact completed checkpoint boundary required')
        self.phase = 'restoring'
        try:
            self.operations.synchronize_device(self.mesh)
            for source, destination in zip(self.buffers, live, strict=True):
                self.operations.copy(source, destination)
            self.operations.synchronize_device(self.mesh)
            self.phase = 'ready'
        except BaseException:
            self.phase = 'failed'
            raise
