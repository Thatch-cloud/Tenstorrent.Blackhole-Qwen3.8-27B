from pathlib import Path
import tempfile
import unittest

from dspark_rope_reference import compare, reviewed_function


class DSparkRopeReferenceTests(unittest.TestCase):
    def test_unreviewed_or_changed_source_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('unreviewed.py', 'modeling_rope_utils.py.source.txt'):
                source = Path(directory) / name
                source.write_text("raise RuntimeError('must not execute')")
                with self.assertRaises(ValueError):
                    reviewed_function(source, '_compute_yarn_parameters')

    def test_changed_config_fails_before_source_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            config.write_text('{}')
            with self.assertRaises(ValueError):
                compare(config, Path(directory) / 'not-loaded')


if __name__ == '__main__':
    unittest.main()
