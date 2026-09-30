"""One-request diagnostic ownership; caller must preallocate before trace capture."""

from contextlib import contextmanager
from unittest.mock import patch


class CombinedClockCapture:
    def __init__(self, model, captures, candidate):
        if len(model.layers) != 64 or len(captures) != 64 or not callable(candidate):
            raise ValueError('All 64 target MLP layers and preallocated captures required')
        self.model, self.captures, self.candidate = model, tuple(captures), candidate
        self.weights = tuple(layer.feed_forward.weights.w_gate_up for layer in model.layers)
        self.projections = []
        self.engines = []
        self.records = []
        self.active = self.failed = False
        identities = set()
        for capture in self.captures:
            if capture.mesh is not model.mesh_device or capture.pending or capture.records:
                raise ValueError('Fresh capture ownership on the target mesh required')
            for addresses in capture.addresses():
                if len(addresses) != 2:
                    raise ValueError('Two-chip sample storage required for every layer')
                for chip, address in enumerate(addresses):
                    identity = chip, address
                    if identity in identities:
                        raise ValueError('Sample storage aliases another layer or reader')
                    identities.add(identity)

    def projection(self, mesh, weights, **kwargs):
        index = len(self.projections)
        if (not self.active or self.failed or index >= 64 or mesh is not self.model.mesh_device
                or weights is not self.weights[index] or kwargs.get('token_rows') != 16
                or kwargs.get('pairs_per_worker') != 3 or kwargs.get('math_approx_mode') is not True
                or 'sample_buffers' in kwargs):
            raise ValueError('Ordered native-weight T16 projections required')
        projection = self.candidate(mesh, weights, sample_buffers=self.captures[index].buffers, **kwargs)
        self.projections.append(projection)
        return projection

    def register_engine(self, engine, model):
        if not self.active or model is not self.model or self.engines or len(self.projections) != 64:
            raise ValueError('Exactly one owned verifier after all projection bindings required')
        self.engines.append(engine)

    def verify(self, engine, ticket, execute):
        if not self.active or self.failed or self.engines != [engine]:
            raise ValueError('Active owned diagnostic verifier required')
        if len(ticket.tokens) != 16 or len(self.records) == 2:
            return execute()
        try:
            for capture in self.captures:
                capture.prepare()
            result = execute()
            layers = [dict(layer=index, capture=capture.collect(f'verify-{len(self.records)}-layer-{index}'))
                for index, capture in enumerate(self.captures)]
            self.records.append(dict(position=ticket.position, rows=16, layers=layers))
            return result
        except BaseException:
            self.failed = True
            for capture in self.captures:
                capture.pending = False
            raise

    def assert_releasable(self):
        if self.active or any(getattr(engine, 'phase', None) != 'closed' for engine in self.engines):
            raise ValueError('Release verifier traces before freeing diagnostic sample buffers')

    def summary(self):
        self.assert_releasable()
        if self.failed or len(self.projections) != 64 or len(self.engines) != 1 or len(self.records) != 2:
            raise ValueError('Complete closed two-replay combined diagnostic required')
        return dict(diagnostic_only=True, committed_tg=None, layers=64,
            sampled_verifier_replays=2, records=list(self.records), sample_buffers_releasable=True)

    @contextmanager
    def install(self, fusion_module, engine_class):
        if self.active or self.engines or self.projections or self.records or self.failed:
            raise ValueError('Fresh one-request diagnostic scope required')
        original_init, original_verify = engine_class.__init__, engine_class.verify

        def initialize(engine, model, *args, **kwargs):
            self.register_engine(engine, model)
            try:
                original_init(engine, model, *args, **kwargs)
            except BaseException:
                self.failed = True
                raise

        def verify(engine, ticket):
            return self.verify(engine, ticket, lambda: original_verify(engine, ticket))

        self.active = True
        try:
            with patch.object(fusion_module, 'FusedProjection', self.projection), \
                    patch.object(engine_class, '__init__', initialize), patch.object(engine_class, 'verify', verify):
                yield self
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False
