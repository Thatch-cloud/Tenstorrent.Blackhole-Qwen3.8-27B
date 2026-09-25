from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_direct_window_hardware_sources import GUARD, HARDWARE_GUARD
from gdn_direct_window_t32_adapter import payloads as simulator_payloads
from gdn_direct_window_t32_hardware_sources import payloads
from gdn_direct_window_t32_gate import qualify
import gdn_direct_window_t32_scope as scope


class T32WindowIntegrationTests(unittest.TestCase):
    def test_hardware_adapter_changes_guard_not_device_math(self):
        directory = Path(__file__).parent
        result = payloads(directory)
        simulator = simulator_payloads({name: (directory / name).read_text()
            for name in ('gdn_direct_window.py', 'gdn_direct_window_device.py')})
        restored = result['gdn_direct_window_t32_hardware_device.py'].replace(HARDWARE_GUARD, GUARD)
        restored = restored.replace('Allocated direct-window hardware experiment required',
            'Direct convolution windows are simulator-only')
        self.assertEqual(restored, simulator['gdn_direct_window_t32_device.py'])
        self.assertEqual(result['gdn_direct_window_t32.py'], simulator['gdn_direct_window_t32.py'])
        native = (directory / 'gdn_batched_conv.py').read_text()
        suffix = '        prefixes = [None] * rows\n'
        candidate = result['gdn_direct_window_t32_hardware_batch.py']
        self.assertEqual(candidate[candidate.index(suffix):], native[native.index(suffix):])
        self.assertIn('rows == 32 and dma_windows', candidate)

    def test_scope_dispatch_counts_and_cleanup_on_failure(self):
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                names = ('gdn_direct_window_t32', 'gdn_direct_window_t32_hardware_device',
                    'gdn_direct_window_t32_hardware_batch')
                generated = {name + '.py': 'fixture' for name in names}
                modules = {}
                for name in names:
                    path = directory / (name + '.py')
                    path.write_text('fixture')
                    modules[name] = SimpleNamespace(__file__=str(path))
                candidate = Mock(side_effect=RuntimeError('execution failed') if failed else None, return_value='candidate')
                modules[names[-1]].run_batched_projected = candidate
                native = Mock(return_value='native')
                native._direct_window_override = False
                batch = SimpleNamespace(run_batched_projected=native)
                loop = SimpleNamespace(run_batched_projected=native)
                modules.update(gdn_batched_conv=batch, gdn_device_loop_state=loop,
                    ttnn=SimpleNamespace(L1_MEMORY_CONFIG='l1'))
                projected = SimpleNamespace(shape=(1, 32, 8240), memory_config=lambda: 'l1')
                options = dict(dma_windows=True, packed_checkpoints=True, norm_batch=True, defer_conv_publication=True)
                with patch.dict('sys.modules', modules), patch.object(scope, 'qualify', return_value={'rows': 32}), \
                        patch.object(scope, 'payloads', return_value=generated):
                    try:
                        with scope.scoped_direct_windows_t32(directory, directory, directory) as audit:
                            self.assertEqual(batch.run_batched_projected(None, SimpleNamespace(shape=(1, 16, 8240))), 'native')
                            self.assertEqual(loop.run_batched_projected(None, projected, **options), 'candidate')
                    except RuntimeError:
                        self.assertTrue(failed)
                self.assertIs(batch.run_batched_projected, native)
                self.assertIs(loop.run_batched_projected, native)
                self.assertTrue(audit['restored'])
                self.assertEqual(audit['hits'], 0 if failed else 1)
                self.assertEqual(audit['fallbacks'], 1)

    def test_unretained_report_cannot_admit_hardware_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'gdn-output-grid.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exact retained T32'):
                qualify(temporary, temporary, temporary)


if __name__ == '__main__':
    unittest.main()
