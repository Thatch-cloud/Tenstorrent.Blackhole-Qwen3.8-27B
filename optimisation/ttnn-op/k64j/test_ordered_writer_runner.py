"""run_card_b.sh K64J_HARNESS=ordered_writer (E1, the 262k window's first card evidence), dry run: the launch mounts this checkout's
scripts/ci, carries QWEN_FAST_TP=4 and runs ordered_writer_tp4_card_test.py with arguments it parses; the harness's source files
are required; the other harnesses never see it.

    py -3.11 -B -m unittest test_ordered_writer_runner      (from this directory; scripts/ci on the path)
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_extent_reader_card_b as base  # noqa: E402

CI = base.CI
FILES = ('ordered_writer_tp4_card_test.py', 'ordered_cache_hw_plan.py', 'ordered_cache.py', 'ordered_cache_tp.py',
         'packed_ordered_cache.py', 'page_width_tp4.py', 'ordered_writer_evidence_tp4.json', 'chip_view.py', 'tp_shapes.py',
         'verify_trace_t1.py', 'verify_trace_t2.py')
TEMPLATES = CI / 'references' / 'tp4-262k8-jobs'


@unittest.skipUnless(base.BASH, 'bash not found')
class OrderedWriterRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.graft = base.card_tests.make_graft(self.dir)
        self.helper = base.RunnerTests('test_the_extent_readers_dry_run')
        self.helper.dir = self.dir
        self.helper.graft = self.graft
        self.runner, self.ci = base.make_tree(self.dir)
        for name in FILES:
            if not (self.ci / name).exists():
                shutil.copyfile(CI / name, self.ci / name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_runner(self, **env):
        env.setdefault('K64J_HARNESS', 'ordered_writer')
        return self.helper.run_runner(self.runner, **env)

    def environment(self, argv):
        return [argv[index + 1] for index, word in enumerate(argv) if word == '-e']

    def harness_words(self, argv):
        return argv[argv.index('card') + 1:] if 'card' in argv else argv[argv.index('--out'):]

    def parse(self, argv):
        import ordered_writer_tp4_card_test as card

        words = argv[argv.index('--out'):]
        return card.parse(words)

    def test_the_dry_run_launches_this_checkouts_test_on_four_cards(self):
        result = self.run_runner()
        argv = self.helper.argv(result)
        self.assertEqual(result.stderr, '')
        self.assertIn(' harness=ordered_writer', result.stdout)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-k64j-ordered-' + base.CARD_X)
        environment = self.environment(argv)
        for value in ('QWEN_FAST_TP=4', 'QWEN_C2_SERVING=0', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS=1'):
            self.assertIn(value, environment)
        pythonpath = [value for value in environment if value.startswith('PYTHONPATH=')]
        self.assertEqual(len(pythonpath), 1)
        self.assertTrue(pythonpath[0].startswith('PYTHONPATH=/bench/ci:'), pythonpath)
        self.assertIn('/opt/tt-metal/ttnn', pythonpath[0])
        self.assertFalse([value for value in environment if value.startswith('QWEN_FAST_SDPA_MODES')])
        mounts = self.helper.mounts(argv)
        ci = [mount for mount in mounts if mount['dst'] == '/bench/ci']
        self.assertEqual(len(ci), 1)
        self.assertTrue(ci[0]['src'].endswith('/scripts/ci') and ci[0].get('readonly'), ci)
        self.assertNotIn('/experiment-scripts/ci', ' '.join(mount.get('src', '') for mount in mounts))
        inner = argv[argv.index('--entrypoint') + 4]
        self.assertTrue(inner.endswith('exec python3 -B /bench/ci/ordered_writer_tp4_card_test.py "$@"'), inner)
        arguments = self.parse(argv)
        self.assertRegex(arguments.out.as_posix(), r'^/results/ordered-[0-9]{8}T[0-9]{6}\.json$')
        self.assertEqual((arguments.widths, arguments.writers, arguments.seeds, arguments.modes),
                         ([2052, 4096], ['chained64', 'tiles32'], [0, 1, 2],
                          ['eager', 'replay_changed', 'replay_unchanged']))
        self.assertEqual(arguments.expect_binary_sha256, base.sha(base.card_tests.BINARY))
        self.assertGreater(arguments.deadline_s, 0)

    def test_the_watcher_pass_is_the_reduced_scope_under_the_sanitiser(self):
        argv = self.helper.argv(self.run_runner(WATCHER='1'))
        self.assertIn('TT_METAL_WATCHER=5', self.environment(argv))
        arguments = self.parse(argv)
        self.assertEqual((arguments.seeds, arguments.modes), ([0], ['eager', 'replay_changed']))
        self.assertEqual(arguments.watchdog, 120.0)

    def test_extra_arguments_are_appended_last(self):
        argv = self.helper.argv(self.run_runner(CARD_B_ARGS='--widths 4096 --writers tiles32 --seeds 0'))
        arguments = self.parse(argv)
        self.assertEqual((arguments.widths, arguments.writers, arguments.seeds), ([4096], ['tiles32'], [0]))

    def test_a_missing_source_is_refused_before_launch(self):
        for name in FILES:
            moved = self.ci / (name + '.moved')
            (self.ci / name).rename(moved)
            try:
                result = self.run_runner()
                self.assertEqual(result.returncode, 1, name)
                self.assertIn('%s missing (the ordered-writer test runs this checkout' % name, result.stderr, name)
                self.assertNotIn('### argv: ', result.stdout)
            finally:
                moved.rename(self.ci / name)
        self.helper.argv(self.run_runner())

    def test_the_other_harnesses_never_see_it(self):
        for harness in ('card', 'extent_reader'):
            argv = self.helper.argv(self.run_runner(K64J_HARNESS=harness))
            inner = argv[argv.index('--entrypoint') + 4]
            self.assertNotIn('ordered_writer', inner)
            self.assertNotIn('QWEN_C2_SERVING=0', self.environment(argv)) if harness == 'card' else None

    def test_an_unknown_harness_still_names_the_new_one(self):
        result = self.run_runner(K64J_HARNESS='k2')
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: K64J_HARNESS=k2 is none of card', result.stderr)
        self.assertIn('ordered_writer', result.stderr)

    def test_the_window_templates_call_it_with_flags_it_parses(self):
        names = sorted(path.name for path in TEMPLATES.glob('E1*.env')) if TEMPLATES.is_dir() else []
        self.assertEqual(names, ['E1-ordered-writer.env', 'E1w-ordered-writer-watcher.env'])
        import c2_serving_job as job

        for name, watcher in (('E1w-ordered-writer-watcher.env', True), ('E1-ordered-writer.env', False)):
            values = dict(line.split('=', 1) for line in (TEMPLATES / name).read_text().splitlines()
                          if line.strip() and not line.startswith('#'))
            text = values['C2_CARDM_ENV']
            for placeholder, value in (('@K64J_GRAFT_DIR@', self.graft.as_posix()),
                                       ('@K64J_TTNNCPP_SHA256@', base.sha(base.card_tests.BINARY)),
                                       ('@SERVED_IMAGE@', 'tt-vllm:four-card-test')):
                text = text.replace(placeholder, value)
            self.assertNotIn('@', text)
            env = dict(pair.split('=', 1) for pair in text.split())
            self.assertEqual((env['K64J_HARNESS'], env.get('WATCHER') == '1'), ('ordered_writer', watcher))
            env['CARD_B_ARGS'] = values['C2_CARDM_ARGS']
            plain = values['C2_CARDM_ENV'].replace('@K64J_GRAFT_DIR@', '/opt/graft-K64j').replace(
                '@K64J_TTNNCPP_SHA256@', 'ab' * 32).replace('@SERVED_IMAGE@', 'tt-vllm:four-card-test')
            harness, words, pairs = job.read_cardm(dict(values, C2_CARDM_ENV=plain), True, root=str(base.ROOT))
            self.assertEqual(harness, 'optimisation/ttnn-op/k64j/run_card_b.sh')
            argv = self.helper.argv(self.helper.run_runner(self.runner, **env))
            arguments = self.parse(argv)
            self.assertEqual(argv[argv.index('--entrypoint') + 2], 'tt-vllm:four-card-test')
            if watcher:
                self.assertEqual(arguments.seeds, [0])
                self.assertIn('TT_METAL_WATCHER=5', self.environment(argv))
            else:
                self.assertEqual((arguments.widths, arguments.writers, arguments.seeds, arguments.modes),
                                 ([2052, 4096], ['chained64', 'tiles32'], [0, 1, 2],
                                  ['eager', 'replay_changed', 'replay_unchanged']))
                import ordered_writer_tp4_card_test as card

                self.assertEqual(card.scope_of(dict(requested=dict(widths=arguments.widths, writers=arguments.writers,
                                                                   seeds=arguments.seeds, modes=arguments.modes))), 'full')


if __name__ == '__main__':
    unittest.main()
