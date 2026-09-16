from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from frozen_stage_evidence import stage


class StageEvidenceTests(unittest.TestCase):
    def test_existing_evidence_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Fresh evidence'):
                stage(root, root, root, root, root)

    def test_unpinned_report_does_not_publish_partial_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dspark-native-8k-attention.json').write_text('{"passed": true}')
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, 'Pinned report'):
                stage(root, root, root, root, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
