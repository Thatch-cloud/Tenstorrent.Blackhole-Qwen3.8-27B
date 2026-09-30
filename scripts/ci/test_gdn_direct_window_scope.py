from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_direct_window_gate import REPORT_SHA256
from gdn_direct_window_scope import scoped_direct_windows


class DirectScopeTests(unittest.TestCase):
    def test_native_tails_owned_t16_and_exception_restoration(self):
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sources = {name: 'qualified' for name in
                    ('gdn_direct_window_hardware_device.py', 'gdn_direct_window_hardware_batch.py')}
                for name, source in sources.items():
                    (directory / name).write_bytes(source.encode())
                native = Mock(return_value='native')
                native._direct_window_override = False
                batch, state = (SimpleNamespace(run_batched_projected=native) for _ in range(2))
                callback = Mock(side_effect=RuntimeError('device') if failed else None, return_value='direct')
                modules = {'gdn_direct_window_hardware_device': SimpleNamespace(
                    __file__=directory / 'gdn_direct_window_hardware_device.py'),
                    'gdn_direct_window_hardware_batch': SimpleNamespace(
                        __file__=directory / 'gdn_direct_window_hardware_batch.py', run_batched_projected=callback)}
                admission = dict(report_sha256=REPORT_SHA256, passed=True, report={})
                with patch.dict('sys.modules', dict(gdn_batched_conv=batch, gdn_device_loop_state=state,
                                                   ttnn=SimpleNamespace(L1_MEMORY_CONFIG='L1'))), \
                        patch('gdn_direct_window_scope.validate'), \
                        patch('gdn_direct_window_scope.payloads', return_value=sources), \
                        patch('gdn_direct_window_scope.importlib.import_module', side_effect=modules.__getitem__):
                    try:
                        with scoped_direct_windows(admission, directory) as audit:
                            self.assertIs(batch.run_batched_projected, state.run_batched_projected)
                            self.assertEqual(state.run_batched_projected(None, SimpleNamespace(shape=(1, 8, 8240))), 'native')
                            with self.assertRaises(ValueError):
                                state.run_batched_projected(None, SimpleNamespace(shape=(1, 16, 8256)))
                            flags = {name: True for name in ('dma_windows', 'packed_checkpoints', 'norm_batch', 'defer_conv_publication')}
                            result = state.run_batched_projected(None,
                                SimpleNamespace(shape=(1, 16, 8240), memory_config=lambda: 'L1'), **flags)
                            self.assertEqual(result, 'direct')
                    except RuntimeError:
                        self.assertTrue(failed)
                    self.assertTrue(audit['restored'])
                    self.assertIs(batch.run_batched_projected, native)
                    self.assertIs(state.run_batched_projected, native)
                    self.assertEqual(audit['hits'], 0 if failed else 1)
                    self.assertEqual(audit['fallbacks'], 1)
