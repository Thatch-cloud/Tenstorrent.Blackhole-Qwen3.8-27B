import sys
import unittest

from qwen_load_sample import snapshot


class LoadSampleTests(unittest.TestCase):
    def test_sample_has_stack_and_selected_status_without_locals(self):
        result = snapshot(sys._getframe(), 10, lambda name:
            'VmRSS:\t123 kB\nVmSwap:\t0 kB\nThreads:\t4\nName:\tprivate\n')
        self.assertEqual(result['elapsed_seconds'], 10)
        self.assertFalse(result['performance_qualified'])
        self.assertNotIn('private', result['metrics']['/proc/self/status'])
        self.assertEqual(result['stack'][-1]['function'],
            'test_sample_has_stack_and_selected_status_without_locals')
        self.assertEqual(set(result['stack'][-1]), {'file', 'line', 'function'})

    def test_missing_proc_files_are_reported_without_losing_stack(self):
        def missing(name):
            raise FileNotFoundError(name)
        result = snapshot(sys._getframe(), 20, missing)
        self.assertTrue(result['stack'])
        self.assertEqual(set(result['metrics'].values()), {'FileNotFoundError'})


if __name__ == '__main__':
    unittest.main()
