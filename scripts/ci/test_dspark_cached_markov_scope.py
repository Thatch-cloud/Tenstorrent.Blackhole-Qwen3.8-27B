from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dspark_cached_markov_scope as scope
import dspark_score_layout_scope as base


class CachedScopeTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.cache = SimpleNamespace(epoch=1, closed=False)
        self.cache.reset = Mock(side_effect=lambda: setattr(self.cache, 'epoch', 2))
        def close_cache(**keywords):
            self.assertEqual(keywords, dict(traces_released=True))
            self.events.append('cache')
            self.cache.closed = True
        self.cache.close = Mock(side_effect=close_cache)
        self.device = SimpleNamespace(closed=False, max_drafts=15, mesh=SimpleNamespace(shape=[1, 2]),
            operations=object(), predecessor=object(), successor=object(),
            prepared=SimpleNamespace(close=Mock(side_effect=lambda: self.events.append('trace'))))
        self.module = SimpleNamespace(markov=base.native)
        self.arguments = (self.device.operations, object(), SimpleNamespace(shape=(1, 1, 15, 248320)),
            self.device.predecessor, self.device.successor, [])
        for module, name, value in ((scope, 'qualify', {}), (scope, 'completed', {}),
                (scope, 'BiasCache', self.cache), (scope, 'execute', []),
                (base, 'qualify', {}), (base, 'validate_hardware', 'digest'), (base, 'addresses', [1, 2])):
            patcher = patch.object(module, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.arm = scope.CachedMarkovArm(self.device,
            hardware_audit=dict(weight_bindings=[[1, 2], [1, 2]]),
            build_evidence=dict(factory_inputs=dict(admission={}),
                binaries={'build_Release/lib/_ttnncpp.so': 'digest'}, import_passed=True),
            factory_root='/unused')

    def test_reset_requires_warmup_and_is_single_use(self):
        with self.arm.install(self.module):
            with self.assertRaises(ValueError):
                self.arm.reset_after_warmup()
            self.module.markov(*self.arguments)
            self.arm.reset_after_warmup()
            with self.assertRaises(ValueError):
                self.arm.reset_after_warmup()
        self.assertEqual(self.events, ['trace', 'cache'])
        self.assertEqual(self.arm.summary()['bias_cache']['reset_epoch'], 2)
        self.assertIs(self.module.markov, base.native)

    def test_exception_releases_trace_before_cache(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            with self.arm.install(self.module):
                raise RuntimeError('request failed')
        self.assertEqual(self.events, ['trace', 'cache'])
        self.assertIs(self.module.markov, base.native)
        with self.assertRaises(ValueError):
            self.arm.summary()

    def test_trace_release_failure_does_not_free_borrowed_buffers(self):
        self.device.prepared.close.side_effect = RuntimeError('trace release failed')
        with self.assertRaisesRegex(RuntimeError, 'trace release failed'):
            with self.arm.install(self.module):
                pass
        self.cache.close.assert_not_called()
        self.assertIs(self.module.markov, base.native)

    def test_missing_reset_cannot_qualify(self):
        with self.arm.install(self.module):
            self.module.markov(*self.arguments)
        with self.assertRaisesRegex(ValueError, 'Cold reset'):
            self.arm.summary()
