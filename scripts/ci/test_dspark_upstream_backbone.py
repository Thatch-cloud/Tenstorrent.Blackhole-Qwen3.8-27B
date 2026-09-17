from pathlib import Path
import tempfile
import unittest

from dspark_upstream_backbone import reviewed_functions


class DSparkUpstreamBackboneTests(unittest.TestCase):
    def test_changed_checkpoint_source_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dflash.py.source.txt'
            path.write_text("raise RuntimeError('not executable')")
            with self.assertRaises(ValueError):
                reviewed_functions(path, Path(directory))


if __name__ == '__main__':
    unittest.main()
