"""The pair's pinned sources are untouched, and the width switch leaves the pair's path alone.

Recorded evidence pins the bytes of the sources the TP2 fast path loads (the 42 sources of the frozen component
reports, the evidence-pinned extent reader, gdn_multitoken through gdn_vsplit.HELPER_HASH, the quad conv kernel),
and the fast-path attach hashes the image's copies of them. Editing one voids that evidence and stops every TP2
attach, so the four-card port adds sibling modules and selects them by width (tp_shapes) in unpinned callers.
tp2_pinned_sources.json records each pinned file's sha256 (line endings normalised); a diff to one fails here
until someone re-records the file on purpose, which is the review this test forces.

The 42 sources' frozen sha256 (the evidence's own) is recorded beside the repo's where they agree: five of them
differ in scripts/ci today and still attach, because none is overlaid onto the image (the plan's section 2).
"""

import hashlib
import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = json.load(open(os.path.join(HERE, 'tp2_pinned_sources.json')))


def digest(name):
    with open(os.path.join(HERE, name), 'rb') as handle:
        return hashlib.sha256(handle.read().replace(b'\r\n', b'\n')).hexdigest()


def constant(name, pattern):
    with open(os.path.join(HERE, name), encoding='utf-8') as handle:
        return re.search(pattern, handle.read()).group(1)


class PinnedSourceTests(unittest.TestCase):
    def test_the_frozen_evidence_sources_are_the_recorded_bytes(self):
        self.assertEqual(len(PINS['frozen_evidence_sources']), 42)
        changed = sorted(name for name, found in PINS['frozen_evidence_sources'].items()
                         if digest(name) != found['repo'])
        self.assertEqual(changed, [], 'a TP2-pinned source was edited: add a sibling module instead')

    def test_where_the_repo_still_holds_the_evidence_bytes_the_record_says_so(self):
        for name, found in PINS['frozen_evidence_sources'].items():
            if found['pinned'] is not None:
                self.assertEqual(found['pinned'], found['repo'], name)
        # the five that already differ are not overlaid onto the image (docker/qwen-c2-overlay.txt)
        unpinned = sorted(name for name, found in PINS['frozen_evidence_sources'].items() if found['pinned'] is None)
        self.assertEqual(unpinned, ['attention_mask_replay.py', 'dspark_attention_chunk_trial.py',
                                    'dspark_stats_pack.py', 'frozen_context_geometry.py', 'native_draft_sdpa.py'])
        with open(os.path.join(HERE, '..', '..', 'docker', 'qwen-c2-overlay.txt'), encoding='utf-8') as handle:
            overlaid = handle.read()
        for name in unpinned:
            self.assertNotIn('scripts/ci/%s\n' % name, overlaid + '\n', name)

    def test_the_other_pinned_files_are_the_recorded_bytes(self):
        changed = sorted(name for name, recorded in PINS['other_pinned'].items() if digest(name) != recorded)
        self.assertEqual(changed, [], 'a TP2-pinned source was edited: add a sibling module instead')

    def test_the_recorded_pins_match_the_constants_the_code_checks(self):
        self.assertEqual(digest('gdn_multitoken.py'), constant('gdn_vsplit.py', r"HELPER_HASH = '(\w+)'"))
        self.assertEqual(PINS['other_pinned']['quad_conv_io.cpp'],
                         constant('quad_draft.py', r"CONV_KERNEL_SHA256 = '(\w+)'"))

    def test_the_t16_gate_sources_are_all_pinned(self):
        with open(os.path.join(HERE, 'target_t16_attention_gate.py'), encoding='utf-8') as handle:
            listed = re.search(r'SOURCES = \{(.*?)\}', handle.read(), re.S).group(1)
        names = set(re.findall(r"'([^']+)'", listed)) - {'target-t16-attention-probe.py'}
        self.assertEqual(sorted(names - set(PINS['frozen_evidence_sources'])), [])

    def test_the_evidence_pinned_extent_reader_is_recorded(self):
        with open(os.path.join(HERE, 'packed_any_evidence.json'), encoding='utf-8') as handle:
            evidence = json.load(handle)
        self.assertEqual(evidence['sources'], {'extent_attention_replay.py': PINS['other_pinned']['extent_attention_replay.py']})
        for name, recorded in evidence['sections']['CB2b']['pinned_modules'].items():
            found = PINS['frozen_evidence_sources'].get(name)
            if found is not None and found['pinned'] is not None:
                self.assertEqual(recorded, found['repo'], name)


if __name__ == '__main__':
    unittest.main()
