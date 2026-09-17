from pathlib import Path
import unittest

from full_window_geometry_admission import ORIGINAL, admit, adapt


class GeometryAdmissionTests(unittest.TestCase):
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
