"""The pair's pinned sources are untouched, and the width switch leaves the pair's path alone.

Recorded evidence pins the bytes of the sources the TP2 fast path loads (the 42 sources of the frozen component
reports, the evidence-pinned extent reader, gdn_multitoken through gdn_vsplit.HELPER_HASH, the quad conv kernel),
and the fast-path attach hashes the image's copies of them. Editing one voids that evidence and stops every TP2
attach, so the four-card port adds sibling modules and selects them by width (tp_shapes) in unpinned callers.
tp2_pinned_sources.json records each pinned file's sha256 (line endings normalised); a diff to one fails here
until someone re-records the file on purpose, which is the review this test forces.

The 42 sources' frozen sha256 (the evidence's own) is recorded beside the repo's where they agree: five of them
differ in scripts/ci today and still attach, because none is overlaid onto the image (the plan's section 2).

The set is the files the attach actually hashes, not only the 42: gdn_direct_window_report hashes gdn_conv_windows.py and
gdn_direct_window_hardware_sources.BATCH_SHA256 hashes gdn_batched_conv.py at every fast profile's attach, and the down-grid,
shared-QK, register-epilogue, native T16, K/V slide and block-stream gates each hash their own closure. The first version of
this record held only the 42 and the port edited gdn_batched_conv.py / gdn_conv_windows.py, which no test noticed while every
two-card attach would have refused ("Exact frozen batched GDN implementation required"). test_every_file_the_attach_hashes_is_recorded
derives the set from the gate modules themselves, so a gate that starts hashing another file fails here until it is recorded.
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


ROOT = os.path.join(HERE, '..', '..')


def read_root(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as handle:
        return handle.read().replace(chr(13) + chr(10), chr(10))


def attach_hashed_names():
    """Every scripts/ci file the serving attach hashes, read from the gate modules' own constants and source lists
    (minus the 42 frozen sources, which their own test covers)."""
    names = set()
    report = read_root('scripts', 'ci', 'gdn_direct_window_report.py')
    names |= set(re.findall(r"'([\w\-\.]+\.(?:py|cpp))'", re.search(r'names = \((.*?)\)\n', report, re.S).group(1)))
    names.add('gdn-direct-window-probe.py')           # the report's 'gdn-output-grid-probe.py' is this local file
    names.add('gdn_batched_conv.py')                  # gdn_direct_window_hardware_sources.BATCH_SHA256
    import dflash_t16_native_attention_gate
    import gdn_shared_qk_gate
    import mlp_down_grid_gate
    import mlp_register_epilogue_gate
    import shared_qk_norm_scatter_gate
    names |= set(mlp_down_grid_gate.LOCAL_SOURCES.values())
    names |= set(gdn_shared_qk_gate.LOCAL_SOURCES) | set(shared_qk_norm_scatter_gate.DEPENDENCIES)
    names |= set(mlp_register_epilogue_gate.HELPERS)
    names |= set(dflash_t16_native_attention_gate.SOURCES)
    names |= {'draft_kv_slide.py', 'draft_kv_slide.cpp'}    # draft_kv_slide_gate (the pair's K/V slide scope)
    stream = read_root('scripts', 'ci', 'mlp_block_stream_gate.py')
    names |= set(re.findall(r"'(\w+\.(?:py|cpp))'", re.search(r'for name in \((.*?)\):', stream, re.S).group(1)))
    return names - set(PINS['frozen_evidence_sources'])


def image_list_membership():
    """name -> the lists it is in now: D (fast-serving Dockerfile COPY), W (image workflow context loop), O (C2 overlay)."""
    found = {}
    for line in read_root('docker', 'qwen-fast-serving.Dockerfile').splitlines():
        if line.startswith('COPY '):
            for token in line.split():
                match = re.match(r'^scripts/ci/([\w\-\.]+\.(?:py|cpp))$', token)
                if match:
                    found.setdefault(match.group(1), set()).add('D')
    for match in re.finditer(r'for name in ([^;]+); do', read_root('.github', 'workflows', 'qwen-fast-serving-image.yml')):
        for token in match.group(1).split():
            found.setdefault(token, set()).add('W')
    for line in read_root('docker', 'qwen-c2-overlay.txt').splitlines():
        for token in line.split('#', 1)[0].split():
            match = re.match(r'^scripts/ci/([\w\-\.]+\.(?:py|cpp))$', token)
            if match:
                found.setdefault(match.group(1), set()).add('O')
    return found


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
        # the attach's exact-bytes check of the batched GDN implementation, and the direct-window report's gdn_conv_windows.py
        self.assertEqual(PINS['attach_hashed']['gdn_batched_conv.py'],
                         constant('gdn_direct_window_hardware_sources.py', r"BATCH_SHA256 = '(\w+)'"))
        self.assertEqual(digest('gdn_batched_conv.py'), PINS['attach_hashed']['gdn_batched_conv.py'])
        self.assertEqual(digest('gdn_conv_windows.py'), PINS['attach_hashed']['gdn_conv_windows.py'])
        self.assertEqual(PINS['other_pinned']['quad_conv_io.cpp'],
                         constant('quad_draft.py', r"CONV_KERNEL_SHA256 = '(\w+)'"))

    def test_every_file_the_attach_hashes_is_recorded(self):
        derived = attach_hashed_names()
        recorded = set(PINS['attach_hashed']) | set(PINS['other_pinned'])
        self.assertEqual(sorted(derived - recorded), [], 'the attach hashes a file this record does not hold')
        self.assertGreater(len(derived), 30)
        for required in ('gdn_batched_conv.py', 'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'attention_batch.py'):
            self.assertTrue(required in recorded or required in PINS['frozen_evidence_sources'], required)

    def test_the_attach_hashed_files_are_the_recorded_bytes(self):
        changed = sorted(name for name, recorded in PINS['attach_hashed'].items() if digest(name) != recorded)
        self.assertEqual(changed, [], 'a file the attach hashes was edited: the image would refuse every attach; add a sibling')

    def test_no_pinned_file_reaches_the_image_by_a_list_it_was_not_in_at_the_base(self):
        pinned = set(PINS['frozen_evidence_sources']) | set(PINS['other_pinned']) | set(PINS['attach_hashed'])
        now = image_list_membership()
        allowed = {name: set(lists) for name, lists in PINS['image_lists_at_base'].items()}
        newly = sorted('%s (%s)' % (name, ''.join(sorted(now[name] - allowed.get(name, set()))))
                       for name in pinned if now.get(name, set()) - allowed.get(name, set()))
        self.assertEqual(newly, [], 'a pinned source is newly in an image copy list: the image would serve bytes the '
                                    'evidence does not hash (or a bundle copy the evidence pinned would be replaced)')

    def test_the_four_card_twins_of_the_pinned_helpers_reach_the_image(self):
        import tp_addresses

        now = image_list_membership()
        for _, _, twin, _ in tp_addresses.TWINS:
            if twin in ('tp_addresses',):
                continue
            lists = now.get(twin + '.py', set())
            self.assertIn('O', lists, twin)                       # the C2 overlay: the image that serves four cards
            self.assertEqual('D' in lists, 'W' in lists, twin)    # and the fast image's two lists agree

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
