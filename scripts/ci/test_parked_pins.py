"""Stage E: the sources the branch must not edit are the base bytes, and every file it does edit reaches the C2 image.
Kept apart from test_parked_census so that module (which runs inside the image) imports nothing the image lacks: these
read the checkout's git history and the overlay manifest's parser."""
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import serving_parked_engines as parked  # noqa: E402
from test_parked_census import BASE_COMMIT, git  # noqa: E402


class SourcePinTests(unittest.TestCase):
    def test_no_stage_e_edit_is_a_pinned_source_and_each_reaches_the_image(self):
        import c2_overlay

        path = ROOT / 'docker' / 'qwen-c2-overlay.txt'
        if not path.is_file():
            self.skipTest('no checkout')
        sources = {entry.source for entry in c2_overlay.parse_manifest(path.read_text(encoding='utf-8'))}
        for name in parked.STAGE_E_EDITS:
            self.assertNotIn(name, c2_overlay.FORBIDDEN)
            self.assertNotIn(name, parked.NEVER_EDITED)
            self.assertIn('scripts/ci/' + name, sources, '%s must be overlaid to reach the C2 image' % name)

    def test_the_never_edited_sources_are_the_branch_base_bytes(self):
        for name in parked.NEVER_EDITED:
            base = git('show', '%s:scripts/ci/%s' % (BASE_COMMIT, name))
            self.assertEqual((HERE / name).read_bytes().replace(b'\r\n', b'\n'), base.replace(b'\r\n', b'\n'),
                             '%s must stay byte-identical' % name)

    def test_the_kv_slide_adapter_still_matches_draft_kv_history_prepare(self):
        text = (HERE / 'draft_kv_slide_scope.py').read_text(encoding='utf-8')
        self.assertIn('DraftKVHistory', text)


if __name__ == '__main__':
    unittest.main()
