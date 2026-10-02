"""ft_select: the guard runs before anything is written; the directory is the lab's FORMAT and c2_tau_lab drives it as arm G, with the
target's answers kept in outputs.jsonl (the lab's own fakes: no engine, no sockets)."""
import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_tau_lab as lab  # noqa: E402
import ft_select as sel  # noqa: E402
import ft_split  # noqa: E402
import tau_lab_report as rep  # noqa: E402
from test_tau_lab import TEXT, run_lab  # noqa: E402


def conversations(source='swe', count=4, per=3, tokens=500, first=0):
    out = []
    for n in range(first, first + count):
        out.append(dict(id='%s-%d' % (source, n), source=source, group='g-%s-%d' % (source, n // 2), repo='repo-%d' % (n // 2),
                        turns=[dict(turn_id='%s-%d-t%d' % (source, n, k), ids=list(range(tokens + 40 * k + n))) for k in range(per)]))
    return out


def slurp(path, binary=False):
    with open(path, 'rb' if binary else 'r', **({} if binary else dict(encoding='utf-8'))) as handle:
        return handle.read()


class SelectTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.out = os.path.join(self.root, 'g')

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_the_mix_decides_the_turns_per_source_and_whole_conversations_are_kept(self):
        candidates = conversations('swe', 10) + conversations('chained', 10) + conversations('code', 10)
        chosen = sel.select_conversations(candidates, ft_split.DEFAULT_MIX, 30, seed=1)
        counts = {}
        for conversation in chosen:
            counts[conversation['source']] = counts.get(conversation['source'], 0) + len(conversation['turns'])
        self.assertEqual(counts, dict(swe=15, chained=9, code=6))
        self.assertEqual(chosen, sel.select_conversations(candidates, ft_split.DEFAULT_MIX, 30, seed=1))
        self.assertNotEqual([c['id'] for c in chosen], [c['id'] for c in sel.select_conversations(candidates, ft_split.DEFAULT_MIX, 30, seed=2)])

    def test_a_source_short_of_its_quota_is_refused(self):
        with self.assertRaises(sel.SelectError):
            sel.select_conversations(conversations('swe', 1, 2) + conversations('chained', 5) + conversations('code', 5),
                                     ft_split.DEFAULT_MIX, 30, seed=1)

    def test_own_is_not_selectable_without_d4(self):
        with self.assertRaises(ft_split.SplitError):
            sel.select_conversations(conversations('own', 5), [('own', 1.0)], 5, seed=1)

    def test_the_directory_is_the_labs_format(self):
        convs = conversations('swe', 3) + conversations('code', 2, first=10)
        manifest = sel.write_training_directory(self.out, convs)
        self.assertEqual(manifest['arms'], dict(G=dict(conversations=5, turns=15)))
        loaded, read_back, scrub = lab.load_data(self.out, ['G'])
        self.assertEqual(len(loaded['G']), 15)
        self.assertEqual(scrub, {})
        record = loaded['G'][0]
        self.assertEqual((record['set'], record['weight']), ('train', 1.0))
        self.assertEqual(record['bucket'], rep.bucket_of(record['tokens']))

    def test_the_lab_refuses_an_edited_file(self):
        sel.write_training_directory(self.out, conversations('swe', 2))
        with open(os.path.join(self.out, sel.CONVERSATIONS), 'a') as handle:
            handle.write('\n')
        with self.assertRaises(lab.LabError):
            lab.load_data(self.out, ['G'])

    def test_the_guard_runs_before_anything_is_written(self):
        convs = conversations('swe', 2)
        with self.assertRaises(ft_split.SplitError):
            sel.write_training_directory(self.out, convs, a1=[dict(repo='repo-0')])
        self.assertFalse(os.path.exists(self.out))
        with self.assertRaises(ft_split.SplitError):
            sel.write_training_directory(self.out, conversations('own', 1))
        self.assertFalse(os.path.exists(self.out))

    def test_duplicate_turn_ids_and_an_occupied_directory_are_refused(self):
        convs = conversations('swe', 2)
        convs[1]['turns'][0]['turn_id'] = convs[0]['turns'][0]['turn_id']
        with self.assertRaises(sel.SelectError):
            sel.write_training_directory(self.out, convs)
        os.makedirs(self.out)
        with open(os.path.join(self.out, 'x'), 'w'):
            pass
        with self.assertRaises(sel.SelectError):
            sel.write_training_directory(self.out, conversations('swe', 2))

    @unittest.skipIf(os.name == 'nt', 'POSIX modes')
    def test_files_are_private(self):
        sel.write_training_directory(self.out, conversations('swe', 2))
        for name in os.listdir(self.out):
            self.assertEqual(os.stat(os.path.join(self.out, name)).st_mode & 0o777, 0o600)

    def test_no_text_in_the_directory(self):
        sel.write_training_directory(self.out, conversations('swe', 2))
        for name in os.listdir(self.out):
            raw = slurp(os.path.join(self.out, name), True)
            text = gzip.decompress(raw).decode('utf-8') if name.endswith('.gz') else raw.decode('utf-8')
            self.assertNotIn(TEXT, text)
            self.assertNotIn('"messages"', text)


class ArmGTests(unittest.TestCase):
    def test_arm_g_is_known_but_not_in_the_default_run(self):
        self.assertEqual(lab.parse_arms('G'), ['G'])
        self.assertEqual(lab.parse_arms('A1 G'), ['A1', 'G'])
        self.assertNotIn('G', lab.ARMS)
        self.assertEqual(lab.parse_arms(' '.join(lab.ARMS)), list(lab.ARMS))
        with self.assertRaises(lab.LabError):
            lab.parse_arms('G G')
        with self.assertRaises(lab.LabError):
            lab.parse_arms('A9')

    def test_the_lab_drives_arm_g_and_keeps_every_answer(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        data = os.path.join(root, 'train')
        sel.write_training_directory(data, conversations('swe', 3, per=2, tokens=300))
        code, out, results, public, engine, _ = run_lab(self, ['--arms', 'G'], data=data)
        self.assertEqual(code, 0, '\n'.join(out))
        turns = [json.loads(line) for line in slurp(os.path.join(results, 'turns.jsonl')).splitlines()]
        outputs = [json.loads(line) for line in slurp(os.path.join(results, 'outputs.jsonl')).splitlines()]
        self.assertEqual(len(turns), 6)
        self.assertTrue(all(turn['arm'] == 'G' and turn['status'] == 'ok' and turn['thinking'] is True for turn in turns))
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(entry['arm'] == 'G' and entry['output_ids'] for entry in outputs))   # the answers are KEPT
        self.assertEqual(sorted(entry['id'] for entry in outputs), sorted(turn['id'] for turn in turns))

    def test_a_g_only_manifest_does_not_need_the_a_tables_and_an_a_manifest_still_does(self):
        manifest = dict(format='FORMAT.md', files={}, arms=dict(G=dict(conversations=1, turns=1)))
        self.assertEqual(lab.expected_counts(manifest), dict(G=1))
        with self.assertRaises(lab.LabError):
            lab.expected_counts(dict(format='FORMAT.md', files={}, arms=dict(A1=dict(turns=1))))
        with self.assertRaises(lab.LabError):
            lab.check_counts({'A1': []}, manifest, ['A1'])               # an arm the manifest has no table for


class JobTests(unittest.TestCase):
    def test_the_job_parser_accepts_g_only_when_named(self):
        import c2_serving_job as job
        names = sorted(lab_profiles())
        outputs = job.read_job(dict(C2_IMAGE_TAG='tp4-serve-2', C2_CARDS='quad', C2_ACTIONS='taulab', C2_TAULAB_ARMS='G',
                                    C2_TAULAB_DATA='kwork64/taulab/train'), names)
        self.assertEqual(outputs['taulab_arms'], 'G')
        default = job.read_job(dict(C2_IMAGE_TAG='tp4-serve-2', C2_CARDS='quad', C2_ACTIONS='taulab'), names)
        self.assertNotIn('G', default['taulab_arms'].split())


def lab_profiles():
    here = os.path.dirname(os.path.abspath(__file__))
    return json.loads(slurp(os.path.join(here, 'qwen_c2_profiles.json')))['profiles']


if __name__ == '__main__':
    unittest.main()
