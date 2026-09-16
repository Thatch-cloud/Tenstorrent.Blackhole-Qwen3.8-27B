import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from frozen_combined_history import initialise_history, prefill_capture_class


class CombinedHistoryTests(unittest.TestCase):
    def test_capture_retains_methods_and_requires_live_admission(self):
        class Original:
            capture = object()
        capture = prefill_capture_class(Original)
        model = SimpleNamespace(_forward_prefill_chunk_masked_tp=lambda: None)
        with patch('dspark_8k_admission.history_limit', return_value=8192):
            with self.assertRaises(ValueError):
                capture(None, model, 32768)
        with patch('dspark_8k_admission.history_limit', return_value=33024):
            instance = capture(None, model, 32768)
            self.assertIs(instance.capture, Original.capture)
            self.assertEqual(instance.position, 32768)
            self.assertFalse(instance.complete)
            with self.assertRaises(ValueError):
                capture(None, model, 65536)
            with self.assertRaises(ValueError):
                capture(None, object(), 32768)

    def test_projection_covers_full_requested_history(self):
        project = Mock(return_value='retained layers')
        instance = SimpleNamespace()
        with patch.dict('sys.modules', {'dspark_history': SimpleNamespace(project_chunks=project)}), \
                patch('dspark_8k_admission.history_limit', return_value=33024):
            initialise_history(instance, 'ops', 'mesh', 'ccl', 'parameters', 'weights', 'chunks', 'rope', position=32768)
        project.assert_called_once_with('ops', 'mesh', 'ccl', 'parameters', 'weights', 'chunks', 'rope', start=0, rows=32768)
        self.assertEqual(instance.layers, 'retained layers')
        self.assertFalse(instance.closed)


if __name__ == '__main__':
    unittest.main()
