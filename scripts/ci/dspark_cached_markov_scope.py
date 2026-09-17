"""Request-owned cached feedback hook with explicit post-warmup invalidation."""

from contextlib import contextmanager
from pathlib import Path

from dspark_cached_markov import BiasCache, execute
from dspark_cached_markov_build import completed
from dspark_cached_markov_gate import qualify
from dspark_score_layout_scope import ScoreLayoutArm


class CachedMarkovArm(ScoreLayoutArm):
    def __init__(self, device, *, hardware_audit, build_evidence, factory_root):
        super().__init__(device, hardware_audit=hardware_audit)
        self.cache = None
        self.reset_epoch = None
        self.admission = qualify(Path(__file__).parent, Path(__file__).parent)
        inputs = build_evidence['factory_inputs']
        if inputs['admission'] != self.admission:
            raise ValueError('Hardware build must use the accepted cache sources')
        binaries = build_evidence['binaries']
        checked = completed(factory_root, inputs, binaries['build_Release/lib/_ttnncpp.so'],
            import_passed=build_evidence.get('import_passed'))
        if any(build_evidence.get(key) != value for key, value in checked.items()):
            raise ValueError('Cache hardware build evidence differs from loaded runtime')

    def execute_feedback(self, *arguments, **keywords):
        return execute(self.cache, *arguments, **keywords)

    def reset_after_warmup(self):
        if not self.installed or self.reset_epoch is not None or self.calls < 1:
            raise ValueError('One cold reset after proposal warmup required')
        self.cache.reset()
        self.reset_epoch = self.cache.epoch

    @contextmanager
    def install(self, prepared_module=None):
        device = self.device
        with super().install(prepared_module):
            self.cache = BiasCache(device.operations, device.mesh, device.predecessor, device.successor)
            try:
                yield self
            finally:
                prepared = getattr(device, 'prepared', None)
                if prepared is not None:
                    prepared.close()
                self.cache.close(traces_released=True)

    def summary(self):
        result = super().summary()
        if self.reset_epoch is None or self.cache is None or not self.cache.closed:
            raise ValueError('Cold reset and trace-safe cache release required')
        result['bias_cache'] = dict(admission=self.admission, reset_epoch=self.reset_epoch,
            released=True, performance_qualified=False)
        return result
