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


HELD = dict(a1=[dict(repo='held-a1')], a2=[dict(conversation='held-c', project='held-p')],
            eval_tier=[dict(repo='held-eval'), dict(group='held-group')])


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
        manifest = sel.write_training_directory(self.out, convs, **HELD)
        self.assertEqual(manifest['arms'], dict(G=dict(conversations=5, turns=15)))
        loaded, read_back, scrub = lab.load_data(self.out, ['G'])
        self.assertEqual(len(loaded['G']), 15)
        self.assertEqual(scrub, {})
        record = loaded['G'][0]
        self.assertEqual((record['set'], record['weight']), ('train', 1.0))
        self.assertEqual(record['bucket'], rep.bucket_of(record['tokens']))

    def test_the_lab_refuses_an_edited_file(self):
        sel.write_training_directory(self.out, conversations('swe', 2), **HELD)
        with open(os.path.join(self.out, sel.CONVERSATIONS), 'a') as handle:
            handle.write('\n')
        with self.assertRaises(lab.LabError):
            lab.load_data(self.out, ['G'])

    def test_the_guard_runs_before_anything_is_written(self):
        convs = conversations('swe', 2)
        with self.assertRaises(ft_split.SplitError):
            sel.write_training_directory(self.out, convs, **dict(HELD, a1=[dict(repo='repo-0')]))
        self.assertFalse(os.path.exists(self.out))
        with self.assertRaises(ft_split.SplitError):
            sel.write_training_directory(self.out, conversations('own', 1), **HELD)
        self.assertFalse(os.path.exists(self.out))

    def test_duplicate_turn_ids_and_an_occupied_directory_are_refused(self):
        convs = conversations('swe', 2)
        convs[1]['turns'][0]['turn_id'] = convs[0]['turns'][0]['turn_id']
        with self.assertRaises(sel.SelectError):
            sel.write_training_directory(self.out, convs, **HELD)
        os.makedirs(self.out)
        with open(os.path.join(self.out, 'x'), 'w'):
            pass
        with self.assertRaises(sel.SelectError):
            sel.write_training_directory(self.out, conversations('swe', 2), **HELD)

    @unittest.skipIf(os.name == 'nt', 'POSIX modes')
    def test_files_are_private(self):
        sel.write_training_directory(self.out, conversations('swe', 2), **HELD)
        for name in os.listdir(self.out):
            self.assertEqual(os.stat(os.path.join(self.out, name)).st_mode & 0o777, 0o600)

    def test_no_text_in_the_directory(self):
        sel.write_training_directory(self.out, conversations('swe', 2), **HELD)
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
        sel.write_training_directory(data, conversations('swe', 3, per=2, tokens=300), **HELD)
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


class ScrubGateForGTests(unittest.TestCase):
    """Arm G reads training conversations; any that could be our own traces must pass the scrub report's final check first."""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def own_dir(self, hits=None, with_report=True):
        data = os.path.join(self.root, 'own')
        held = dict(HELD, a2=[dict(conversation='held-c', project='held-p')])
        sel.write_training_directory(data, [dict(c, conversation='c-%s' % c['id'], project='p-%s' % c['id']) for c in conversations('own', 2)],
                                     d4_cleared=True, **held)
        if with_report:
            report = os.path.join(data, 'scrub_report.json')
            with open(report, 'w') as handle:
                json.dump(dict(conversations_kept=2, conversations_dropped=0, final_detector_hits_on_written_file=hits or {}), handle)
            manifest_path = os.path.join(data, sel.MANIFEST)
            manifest = json.loads(slurp(manifest_path))
            manifest['files']['scrub_report.json'] = dict(bytes=os.path.getsize(report), sha256=sel.sha256_file(report))
            with open(manifest_path, 'w') as handle:
                json.dump(manifest, handle)
        return data

    def test_g_with_own_conversations_is_refused_without_a_scrub_report(self):
        data = self.own_dir(with_report=False)
        self.assertTrue(lab.g_needs_scrub(data))
        with self.assertRaises(lab.LabError):
            lab.load_data(data, ['G'])

    def test_g_with_own_conversations_is_refused_on_a_residual_detector_hit(self):
        data = self.own_dir(hits=dict(email=1))
        with self.assertRaises(lab.LabError):
            lab.load_data(data, ['G'])

    def test_g_with_own_conversations_loads_with_a_clean_report_and_counts_it(self):
        data = self.own_dir()
        loaded, manifest, scrub = lab.load_data(data, ['G'])
        self.assertEqual(scrub, dict(kept=2, dropped=0))
        self.assertEqual(len(loaded['G']), 6)

    def test_g_of_public_sources_alone_needs_no_report(self):
        data = os.path.join(self.root, 'public')
        sel.write_training_directory(data, conversations('swe', 2) + conversations('chained', 2, first=10), **HELD)
        self.assertFalse(lab.g_needs_scrub(data))
        self.assertEqual(lab.load_data(data, ['G'])[2], {})
        self.assertNotIn(lab.SCRUB_REPORT_NAME, lab.files_needed(['G'], data))

    def test_a_conversation_with_no_source_is_treated_as_own(self):
        data = os.path.join(self.root, 'nosource')
        os.makedirs(data)
        with open(os.path.join(data, 'g_train.jsonl'), 'w') as handle:
            handle.write(json.dumps(dict(id='x', group='g', turns=[])) + chr(10))
        self.assertTrue(lab.g_needs_scrub(data))
        self.assertTrue(lab.g_needs_scrub(os.path.join(self.root, 'absent')))

    def test_hiding_the_own_source_by_editing_the_file_fails_the_manifest_check(self):
        data = self.own_dir()
        path = os.path.join(data, 'g_train.jsonl')
        edited = slurp(path).replace('"source": "own"', '"source": "swe"')
        with open(path, 'w') as handle:
            handle.write(edited)
        with self.assertRaises(lab.LabError):
            lab.load_data(data, ['G'])

    def test_a2_still_needs_its_report_and_the_default_arms_are_unchanged(self):
        self.assertIn(lab.SCRUB_REPORT_NAME, lab.files_needed(['A2']))
        self.assertIn(lab.SCRUB_REPORT_NAME, lab.files_needed(['A1', 'A2']))
        self.assertNotIn(lab.SCRUB_REPORT_NAME, lab.files_needed(['A1']))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.lab_dir = os.path.join(self.root, 'lab')
        os.makedirs(self.lab_dir)
        for key, rows in (('a1', [dict(repo='held-a1')]), ('a2', [dict(conversation='hc', project='hp')]),
                          ('eval_tier', [dict(repo='held-eval'), dict(group='held-group')])):
            with open(os.path.join(self.lab_dir, ft_split.HELD_OUT_FILES[key]), 'w') as handle:
                for row in rows:
                    handle.write(json.dumps(row) + chr(10))
        self.candidates = os.path.join(self.root, 'candidates.jsonl')
        with open(self.candidates, 'w') as handle:
            for c in conversations('swe', 12) + conversations('chained', 12) + conversations('code', 12):
                handle.write(json.dumps(c) + chr(10))
        self.out = os.path.join(self.root, 'g')

    def run_cli(self, *extra):
        lines = []
        code = sel.main(['--lab-dir', self.lab_dir, '--candidates', self.candidates, '--out', self.out, '--total-turns', '30'] + list(extra),
                        say=lines.append)
        return code, lines

    def test_the_cli_reads_the_real_held_out_sets_and_writes_the_directory(self):
        code, lines = self.run_cli()
        self.assertEqual(code, 0)
        self.assertTrue(lines[0].startswith('wrote '))
        self.assertTrue(os.path.exists(os.path.join(self.out, sel.MANIFEST)))

    def test_a_held_out_overlap_in_the_lab_dir_refuses_and_names_a_type_only(self):
        with open(os.path.join(self.lab_dir, ft_split.HELD_OUT_FILES['a1']), 'w') as handle:
            handle.write(json.dumps(dict(repo='repo-0')) + chr(10))
            handle.write(json.dumps(dict(repo='repo-1')) + chr(10))
            handle.write(json.dumps(dict(repo='repo-2')) + chr(10))
            handle.write(json.dumps(dict(repo='repo-3')) + chr(10))
            handle.write(json.dumps(dict(repo='repo-4')) + chr(10))
            handle.write(json.dumps(dict(repo='repo-5')) + chr(10))
        code, lines = self.run_cli('--seed', '1')
        self.assertEqual((code, lines), (2, ['refused: SplitError']))
        self.assertFalse(os.path.exists(self.out))

    def test_a_missing_held_out_file_refuses(self):
        os.remove(os.path.join(self.lab_dir, ft_split.HELD_OUT_FILES['eval_tier']))
        self.assertEqual(self.run_cli(), (2, ['refused: SplitError']))


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
