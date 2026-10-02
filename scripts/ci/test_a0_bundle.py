"""a0_bundle on a synthetic tau-lab data and results directory in the REAL layout (the lab's own test fixtures): refusals, opaque
ids, prefix groups, the schedule and cap checks, file modes, and that no lab id or sentinel text reaches the bundle."""
import gzip
import json
import os
import random
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a0_bundle as bundle  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
from extent_attention_replay import accept_limit  # noqa: E402
from test_tau_lab import make_data, packed_lines, write_jsonl, TEXT  # noqa: E402


def slurp(path, binary=False):
    with open(path, 'rb' if binary else 'r', **({} if binary else {'encoding': 'utf-8'})) as handle:
        return handle.read()


def make_results(data, directory, skip=None, schedule_break=None, finish='stop'):
    """turns.jsonl, outputs.jsonl and server.log for every A1 and A2 turn of `data` (the lab's own record shapes)."""
    turns, outputs, log, number = [], [], [], 0
    for arm, set_name, conv_file, ids_file in bundle.SETS:
        for entry in lab.conversation_records(os.path.join(data, conv_file), os.path.join(data, ids_file), set_name):
            number += 1
            if skip and entry['id'] == skip:
                continue
            rng = random.Random(number)
            emitted = [rng.randint(2, 15) for _ in range(10)]
            request_id = 'cmpl-%016x' % number
            log.extend(packed_lines(request_id, emitted, number))
            total = 1 + sum(emitted) + (3 if schedule_break == entry['id'] else 0)
            turns.append(dict(arm=arm, id=entry['id'], set=set_name, cluster=entry['cluster'], turn=entry['turn'], status='ok',
                              finish=finish, request_id=request_id, think_tokens=4, tool_at=None, prompt_tokens=entry['tokens'],
                              completion_tokens=total, thinking=True, chunk_tokens=[total]))
            outputs.append(dict(arm=arm, id=entry['id'], output_ids=[900 + at % 50 for at in range(total)], finish=finish))
    os.makedirs(directory, exist_ok=True)
    write_jsonl(os.path.join(directory, 'turns.jsonl'), turns)
    write_jsonl(os.path.join(directory, 'outputs.jsonl'), outputs)
    with open(os.path.join(directory, 'server.log'), 'w') as handle:
        handle.write('\n'.join(log) + '\n')


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.data = os.path.join(self.root, 'data')
        self.results = os.path.join(self.root, 'results')
        self.out = os.path.join(self.root, 'bundle')
        self.map = os.path.join(self.root, 'private', 'map.json')
        os.makedirs(self.data)
        os.makedirs(os.path.dirname(self.map))
        make_data(self.data, swe=3, per=2, own=2)

    def tearDown(self):
        shutil.rmtree(self.root)

    def build(self, **kwargs):
        return bundle.build(self.data, self.results, self.out, self.map, expect=kwargs.pop('expect', dict(swe=6, own=4)), **kwargs)

    def test_round_trip_counts_and_manifest(self):
        make_results(self.data, self.results)
        manifest = self.build()
        self.assertEqual(manifest['counts'], dict(swe=6, own=4, turns=10))
        self.assertEqual(manifest['checks']['scheduled'], 10)
        self.assertEqual(bundle.verify_bundle(self.out, 10)['format'], bundle.FORMAT)
        records = list(bundle.read_bundle(self.out))
        self.assertEqual(sorted(record['k'] for record in records), list(range(10)))
        for record in records:
            self.assertEqual(sum(record['schedule']), len(record['output_ids']) - 1)
            self.assertIsNotNone(record['weight'])

    def test_no_lab_id_or_text_reaches_the_bundle_or_the_meta(self):
        make_results(self.data, self.results)
        self.build()
        for name in (bundle.BUNDLE_NAME, bundle.META_NAME, bundle.MANIFEST_NAME):
            path = os.path.join(self.out, name)
            raw = gzip.decompress(slurp(path, True)) if name.endswith('.gz') else slurp(path, True)
            text = raw.decode('utf-8')
            for needle in ('a1-0000', 'a2-0000', 'cmpl-', TEXT, 'openhands', 'all-hands', 'zqsource', 'turn_id', 'request_id'):
                self.assertNotIn(needle, text, (name, needle))

    def test_the_map_stays_outside_the_bundle_and_covers_every_turn(self):
        make_results(self.data, self.results)
        self.build()
        self.assertFalse(os.path.exists(os.path.join(self.out, 'map.json')))
        with open(self.map) as handle:
            mapping = json.load(handle)
        self.assertEqual(sorted(int(k) for k in mapping), list(range(10)))
        self.assertTrue(all(value[0] in ('A1', 'A2') for value in mapping.values()))

    def test_indices_are_a_seeded_shuffle(self):
        make_results(self.data, self.results)
        self.build(seed=1)
        first = json.loads(slurp(self.map))
        os.remove(self.map)
        shutil.rmtree(self.out)
        self.build(seed=1)
        self.assertEqual(json.loads(slurp(self.map)), first)
        os.remove(self.map)
        shutil.rmtree(self.out)
        self.build(seed=2)
        self.assertNotEqual(json.loads(slurp(self.map)), first)
        lab_order = sorted(first.items(), key=lambda item: int(item[0]))
        self.assertNotEqual([value for _, value in lab_order], sorted(value for _, value in lab_order))

    @unittest.skipIf(os.name == 'nt', 'POSIX modes')
    def test_files_are_private(self):
        make_results(self.data, self.results)
        self.build()
        self.assertEqual(stat.S_IMODE(os.stat(self.out).st_mode), 0o700)
        for name in os.listdir(self.out):
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.out, name)).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.map).st_mode), 0o600)

    def test_prefix_groups_chain_prompts_that_extend_each_other(self):
        make_results(self.data, self.results)
        self.build()
        meta = [json.loads(line) for line in slurp(os.path.join(self.out, bundle.META_NAME)).splitlines()]
        by_group = {}
        for entry in meta:
            by_group.setdefault(entry['group'], []).append(entry)
        # the fixture's prompts are range(n): within a conversation each is a prefix of the next
        self.assertTrue(all(len(entries) >= 1 for entries in by_group.values()))
        self.assertLess(len(by_group), len(meta))
        for entries in by_group.values():
            self.assertEqual(sorted(entry['order'] for entry in entries), list(range(len(entries))))
            lengths = [entry['prompt_tokens'] for entry in sorted(entries, key=lambda e: e['order'])]
            self.assertEqual(lengths, sorted(lengths))

    def test_a_turn_that_is_not_a_prefix_starts_its_own_group(self):
        records = [dict(set='swe', cluster=0, turn=0, prompt_ids=[1, 2, 3]), dict(set='swe', cluster=0, turn=1, prompt_ids=[1, 2, 9, 4]),
                   dict(set='swe', cluster=0, turn=2, prompt_ids=[1, 2, 9, 4, 5])]
        groups, chained = bundle.prefix_groups(records)
        self.assertEqual((groups, chained), (2, 1))
        self.assertNotEqual(records[0]['group'], records[1]['group'])
        self.assertEqual(records[1]['group'], records[2]['group'])
        self.assertEqual([record['order'] for record in records], [0, 0, 1])

    def test_refusals(self):
        make_results(self.data, self.results)
        with self.assertRaises(bundle.BundleError):                       # wrong count
            self.build(expect=dict(swe=240, own=96))
        with open(os.path.join(self.data, 'a1_swe_heldout.jsonl'), 'a') as handle:   # the manifest sha256 no longer matches
            handle.write('\n')
        with self.assertRaises(lab.LabError):
            self.build()

    def test_a_missing_result_is_refused(self):
        make_results(self.data, self.results, skip='a1-0000-t0')
        with self.assertRaises(bundle.BundleError):
            self.build()

    def test_missing_ids_are_refused(self):
        make_results(self.data, self.results)
        os.remove(os.path.join(self.data, 'a2_own_sessions.ids.jsonl.gz'))
        with self.assertRaises(lab.LabError):
            self.build()

    def test_a_schedule_that_misses_the_answer_is_not_scheduled(self):
        make_results(self.data, self.results, schedule_break='a1-0000-t0')
        manifest = self.build()
        self.assertEqual(manifest['checks']['scheduled'], 9)
        records = {record['k']: record for record in bundle.read_bundle(self.out)}
        self.assertEqual(sum(1 for record in records.values() if record['schedule'] is None), 1)

    def test_an_existing_output_is_refused(self):
        make_results(self.data, self.results)
        os.makedirs(self.out)
        slurp_path = os.path.join(self.out, 'x')
        with open(slurp_path, 'w'):
            pass
        with self.assertRaises(bundle.BundleError):
            self.build()

    def test_a_tampered_bundle_fails_verification(self):
        make_results(self.data, self.results)
        self.build()
        with open(os.path.join(self.out, bundle.META_NAME), 'a') as handle:
            handle.write('{}\n')
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(self.out)

    def test_cli_prints_counts_only_and_exit_codes(self):
        make_results(self.data, self.results)
        lines = []
        code = bundle.main(['--data', self.data, '--results', self.results, '--out', self.out, '--map', self.map,
                            '--expect', 'swe=6,own=4'], say=lines.append)
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 1)
        self.assertNotIn('a1-0000', lines[0])
        os.remove(self.map)
        shutil.rmtree(self.out)
        lines = []
        code = bundle.main(['--data', self.data, '--results', self.results, '--out', self.out, '--map', self.map,
                            '--expect', 'swe=240,own=96'], say=lines.append)
        self.assertEqual(code, 2)
        self.assertNotIn('a1-0000', lines[0])


class CapCheckTests(unittest.TestCase):
    def test_caps_that_follow_the_extent_rule_have_no_mismatch(self):
        positions = [300, 310, 500, 511, 512]
        caps = [min(16, accept_limit(position, 16)) for position in positions]
        self.assertEqual(bundle.cap_mismatches(250, [1] * 5, positions, caps), 0)

    def test_a_larger_cap_is_a_mismatch_and_a_budget_cut_only_at_the_tail(self):
        positions = [300, 310, 500, 511, 520]
        caps = [min(16, accept_limit(p, 16)) for p in positions]
        caps[0] += 1
        self.assertEqual(bundle.cap_mismatches(250, [1] * 5, positions, caps), 1)
        caps = [min(16, accept_limit(p, 16)) for p in positions]
        caps[1] = 3            # smaller, mid-turn: not a budget cut
        self.assertEqual(bundle.cap_mismatches(250, [1] * 5, positions, caps), 1)
        caps = [min(16, accept_limit(p, 16)) for p in positions]
        caps[4] = 2            # smaller in the last rounds: a budget cut, allowed
        self.assertEqual(bundle.cap_mismatches(250, [1] * 5, positions, caps), 0)

    def test_the_alternative_start_convention_is_counted_separately(self):
        positions = [255, 100, 400]
        caps = [min(16, accept_limit(position, 16)) for position in positions]
        self.assertEqual(bundle.cap_mismatches(0, [1] * 3, positions, caps), 0)
        self.assertGreater(bundle.cap_mismatches(0, [1] * 3, positions, caps, shift=1), 0)


if __name__ == '__main__':
    unittest.main()
