from pathlib import Path
import unittest

from full_window_geometry_admission import ORIGINAL, EXTENDED, admit, adapt, staged_geometry


class GeometryAdmissionTests(unittest.TestCase):
    def test_non_window_staging_restores_exact_qualified_source(self):
        import hashlib

        payload = Path(__file__).with_name('frozen_context_geometry.py').read_bytes()
        original = staged_geometry(payload, full_window=False)
        self.assertEqual(hashlib.sha256(original).hexdigest(), ORIGINAL)
        self.assertEqual(staged_geometry(original, full_window=False), original)
        self.assertEqual(hashlib.sha256(staged_geometry(payload, full_window=True)).hexdigest(), EXTENDED)
        for value, window in ((payload + b'\n', False), (original, True), (payload, 1)):
            with self.assertRaises(ValueError):
                staged_geometry(value, full_window=window)
        before, after = {}, {}
        exec(compile(original, 'original-geometry', 'exec'), before)
        exec(compile(payload, 'extended-geometry', 'exec'), after)
        for context in before['CONTEXTS']:
            self.assertEqual(before['geometry'](context), after['geometry'](context))

    def test_only_context_list_extension_admitted(self):
        payload = Path(__file__).with_name('frozen_context_geometry.py').read_bytes()
        self.assertFalse(admit(payload, ORIGINAL, 261888)['performance_qualified'])
        for altered, expected, context in ((payload + b'\n', ORIGINAL, 261888),
                (payload.replace(b'capacity = context + 256', b'capacity = context + 128'), ORIGINAL, 261888),
                (payload, '0' * 64, 261888), (payload, ORIGINAL, 131072)):
            with self.assertRaises(ValueError):
                admit(altered, expected, context)

    def test_adapter_preserves_other_source_rejection(self):
        source = Path(__file__).with_name('frozen_combined_runtime.py').read_text()
        result = adapt(source)
        self.assertIn("if name != 'frozen_context_geometry.py':", result)
        self.assertIn("result['geometry_source_extension'] = admit", result)
        with self.assertRaises(ValueError):
            adapt(result)


if __name__ == '__main__':
    unittest.main()
