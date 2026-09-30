from pathlib import Path
import unittest
from unittest.mock import patch

from dflash_t32_comparison_experiment import measure_width


class WidthDispatchTests(unittest.TestCase):
    def test_same_model_pool_and_options_with_width_specific_evidence(self):
        arguments = [object() for unused in range(6)]
        streams, fixtures = object(), object()
        for rows in (16, 32):
            with self.subTest(rows=rows), \
                    patch('dflash_combined_request.measure_combined_dflash', return_value='control') as control, \
                    patch('dflash_t32_combined_request.measure_combined_t32', return_value='candidate') as candidate:
                result = measure_width(rows, *arguments, directory='scripts/ci', runtime_root='runtime',
                    streams=streams, fixtures=fixtures, audit_features=True, max_new_tokens=256)
                selected, unused = (control, candidate) if rows == 16 else (candidate, control)
                unused.assert_not_called()
                self.assertEqual(result, 'control' if rows == 16 else 'candidate')
                self.assertEqual(selected.call_args.args, tuple(arguments))
                options = selected.call_args.kwargs
                self.assertIs(options['fixtures'], fixtures)
                self.assertEqual(options['max_new_tokens'], 256)
                self.assertTrue(options['audit_features'])
                if rows == 16:
                    self.assertIs(options['block_stream']['streams'], streams)
                    self.assertEqual(options['native_attention_evidence'], Path('scripts/ci/dflash-t16-native-evidence'))
                else:
                    self.assertIs(options['streams'], streams)
                    self.assertEqual(set(options['evidence']), {'attention', 'cache', 'gdn', 'windows', 'down', 'stream'})

    def test_unknown_width_cannot_select_an_arm(self):
        for rows in (True, 8, 64, '32'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                measure_width(rows, *([None] * 6), directory='scripts/ci', runtime_root='runtime', streams=[])


if __name__ == '__main__':
    unittest.main()
