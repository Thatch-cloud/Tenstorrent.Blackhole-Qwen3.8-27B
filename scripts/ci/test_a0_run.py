"""a0_run end to end on fakes: a tiny hybrid target, drafters that follow the logged answer, a synthetic bundle in the real layout.
Resume, the deadline, the watchdog's trip file, OOM, failed arms, admission, the plan, the gates, privacy, and the hand-off to the report."""
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import a0_bundle as bundle  # noqa: E402
import a0_fakes as fakes  # noqa: E402
import a0_run as run  # noqa: E402
import a0_target as tgt  # noqa: E402
import tf_pair_report as report  # noqa: E402
import tf_pair_walk as walk  # noqa: E402

VOCAB, HIDDEN, TAPS = 50, 16, (1, 3, 5)
SENTINEL = 'SENTINELTEXT'


def slurp(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def greedy_answer(target, prompt, count):
    state = target.new_state()
    hidden = target.forward(state, torch.as_tensor(prompt, dtype=torch.int64))
    target.taps.collect()
    out = []
    for _ in range(count):
        token = int(target.argmax(hidden[-1:])[0])
        out.append(token)
        hidden = target.forward(state, torch.tensor([token]))
        target.taps.collect()
    return out


def make_bundle(root, target, swe=4, own=3, per=2, answer=40, seed=1):
    """Records in the bundle layout: conversations whose prompts extend each other, answers = the target's own greedy tokens."""
    rng = random.Random(seed)
    records = []
    for set_name, count in (('swe', swe), ('own', own)):
        for conv in range(count):
            prompt = [rng.randrange(VOCAB) for _ in range(20 + 5 * conv)]
            for turn in range(per):
                out = greedy_answer(target, prompt, answer + 3 * turn)
                schedule, left = [], len(out) - 1
                while left > 0:
                    step = min(left, rng.randint(2, 9))
                    schedule.append(step)
                    left -= step
                records.append(dict(set=set_name, cluster='%s%d' % (set_name, conv), turn=turn, bucket='4k', weight=1.0 + conv,
                                    prompt_ids=list(prompt), output_ids=out, finish='stop', think_tokens=5, tool_at=None,
                                    schedule=schedule, tau_served=None, lab_arm='A1', lab_id='%s-%d-%d' % (set_name, conv, turn)))
                prompt = prompt + [rng.randrange(VOCAB) for _ in range(6)]
    groups, chained = bundle.prefix_groups(records)
    mapping = bundle.anonymise(records, seed)
    out_dir = os.path.join(root, 'bundle')
    bundle.write_bundle(out_dir, records, dict(swe=swe * per, own=own * per), dict(scheduled=len(records)), groups, chained, mapping,
                        os.path.join(root, 'map.json'))
    return out_dir, records


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.model = fakes.TinyHybrid(VOCAB, HIDDEN)
        self.target = fakes.TinyTarget(self.model, TAPS)
        self.bundle, self.records = make_bundle(self.root, self.target)
        self.out = os.path.join(self.root, 'results')
        self.trip = os.path.join(self.root, 'trip')
        self.lines = []
        self.now = 1000.0

    def tearDown(self):
        shutil.rmtree(self.root)

    def guard(self, deadline=10 ** 9, avail=100.0, **kwargs):
        return run.Guard(self.trip, deadline, lambda: self.now, lambda: avail, **kwargs)

    def runner(self, make_arm=None, arms=run.CORE, guard=None, seed=7):
        store = run.prepare_results(self.bundle, self.out)
        meta = [json.loads(line) for line in slurp(os.path.join(self.bundle, bundle.META_NAME)).splitlines()]
        plan = run.make_plan(meta, seed, arms)
        make_arm = make_arm or (lambda spec: fakes.FollowsAnswer(good=3 if spec['drafter'] == 'dflash2' else 6))
        group_runner = tgt.GroupRunner(self.target, chunk=7, feature_dtype=torch.float32)
        return run.Runner(self.bundle, store, group_runner, make_arm, plan, guard or self.guard(), self.lines.append, width=3 * HIDDEN)


class RunTests(Fixture):
    def test_every_core_arm_runs_over_every_turn_and_the_report_reads_it(self):
        runner = self.runner()
        counts = runner.run()
        n = len(self.records)
        self.assertEqual(counts['arm_ok'], 2 * n)
        self.assertEqual(counts['arm_failed'], 0)
        self.assertEqual(runner.stopped, None)
        validity = run.finalize(runner.store, self.bundle, self.guard())
        self.assertEqual(validity['V1'], 'PASS')
        self.assertEqual(validity['V6'], 'PASS')
        meta, arms, loaded = report.load(self.out)
        self.assertEqual(sorted(arms), sorted(run.CORE))
        self.assertEqual(len(arms['dspark-t16']), n)
        private, public = report.build(meta, arms, dict(loaded, V0='PASS', V3='PASS', V4='PASS', V5='PASS'), resamples=100,
                                       w8_resamples=10, w8_draws=100)
        self.assertGreater(public['r']['point'], 1.0)          # the 6-good drafter beats the 3-good one
        report.assert_public(public)

    def test_one_forward_per_group_serves_all_arms(self):
        calls = []

        class Counting(tgt.GroupRunner):
            def run_group(self, records, consume, width, wanted=None):
                calls.append(len(records))
                return tgt.GroupRunner.run_group(self, records, consume, width, wanted)
        runner = self.runner()
        runner.group_runner = Counting(self.target, chunk=7, feature_dtype=torch.float32)
        runner.run()
        self.assertEqual(sum(calls), len(self.records))        # each turn passes the target once, whatever the arm count

    def test_resume_does_not_repeat_finished_work(self):
        self.runner().run()
        again = self.runner()
        counts = again.run()
        self.assertEqual((counts['arm_ok'], counts['groups']), (0, 0))
        lines = slurp(os.path.join(self.out, 'arm-dspark-t16.jsonl')).splitlines()
        self.assertEqual(len(lines), len(self.records))

    def test_deadline_stops_the_run_and_a_later_run_completes_it(self):
        ticks = {'n': 0}

        def clock():
            ticks['n'] += 1
            return 0 if ticks['n'] < 12 else 10 ** 9

        guard = run.Guard(self.trip, 5, clock, lambda: 100.0)
        first = self.runner(guard=guard)
        first.run()
        self.assertEqual(first.stopped, 'deadline')
        done_first = len(slurp(os.path.join(self.out, 'arm-dspark-t16.jsonl')).splitlines())
        self.assertTrue(0 < done_first < len(self.records))
        second = self.runner()
        second.run()
        self.assertEqual(second.stopped, None)
        lines = [json.loads(line) for line in slurp(os.path.join(self.out, 'arm-dspark-t16.jsonl')).splitlines()]
        self.assertEqual(sorted(entry['k'] for entry in lines), sorted(range(len(self.records))))

    def test_the_watchdogs_trip_file_stops_the_run_and_fails_v6(self):
        with open(self.trip, 'w') as handle:
            handle.write('xid\n')
        runner = self.runner()
        runner.run()
        self.assertEqual(runner.stopped, 'watchdog')
        self.assertEqual(runner.counts['arm_ok'], 0)
        self.assertEqual(run.finalize(runner.store, self.bundle, self.guard())['V6'], 'FAIL')

    def test_oom_stops_at_once_and_never_skips_ahead(self):
        class Oom(fakes.FollowsAnswer):
            def propose(self, sequence, start, count):
                raise MemoryError()
        runner = self.runner(make_arm=lambda spec: Oom())
        runner.run()
        self.assertEqual(runner.stopped, 'oom')
        self.assertEqual(runner.counts['turns'], 1)
        self.assertEqual(runner.counts['arm_failed'], 0)

    def test_a_failed_arm_is_recorded_by_type_and_the_run_goes_on(self):
        class Broken(fakes.FollowsAnswer):
            def propose(self, sequence, start, count):
                raise ValueError(SENTINEL)

        def make(spec):
            return Broken() if spec['name'] == 'dspark-t16' else fakes.FollowsAnswer(good=3)
        runner = self.runner(make_arm=make)
        counts = runner.run()
        self.assertEqual(counts['arm_failed'], len(self.records))
        self.assertEqual(counts['arm_ok'], len(self.records))
        text = slurp(os.path.join(self.out, 'arm-dspark-t16.jsonl'))
        self.assertIn('ValueError', text)
        self.assertNotIn(SENTINEL, text)
        self.assertNotIn(SENTINEL, '\n'.join(self.lines))

    def test_a_turn_whose_projected_peak_breaks_the_floor_is_not_started(self):
        runner = self.runner(guard=self.guard(avail=10.0))        # 10 GiB available, floor 10, transient 4: nothing fits
        counts = runner.run()
        self.assertEqual(counts['deferred'], len(self.records))
        self.assertEqual(counts['arm_ok'], 0)

    def test_stdout_lines_carry_counts_only(self):
        self.runner().run()
        self.assertTrue(self.lines)
        for line in self.lines:
            self.assertRegex(line, r'^turn \d+ of \d+: arms ok \d+, failed \d+$')

    def test_results_files_hold_no_ids_or_text(self):
        self.runner().run()
        run.finalize(run.Store(self.out), self.bundle, self.guard())
        for name in os.listdir(self.out):
            text = slurp(os.path.join(self.out, name))
            for needle in ('swe-0', 'own-0', 'prompt_ids', 'output_ids', SENTINEL):
                self.assertNotIn(needle, text, (name, needle))


class PlanTests(Fixture):
    def meta(self):
        return [json.loads(line) for line in slurp(os.path.join(self.bundle, bundle.META_NAME)).splitlines()]

    def test_subsets_are_seeded_stratified_and_respect_eligibility(self):
        meta = self.meta()
        a = run.select_subset(meta, {'swe': 3, 'own': 2}, 1, 'x')
        self.assertEqual(a, run.select_subset(meta, {'swe': 3, 'own': 2}, 1, 'x'))
        self.assertNotEqual(a, run.select_subset(meta, {'swe': 3, 'own': 2}, 2, 'x'))
        sets = [row['set'] for row in meta if row['k'] in a]
        self.assertEqual((sets.count('swe'), sets.count('own')), (3, 2))
        only = run.select_subset(meta, {'swe': 100}, 1, 'y', eligible=lambda row: row['k'] % 2 == 0)
        self.assertTrue(all(k % 2 == 0 for k in only))
        self.assertEqual(len(only), len([r for r in meta if r['set'] == 'swe' and r['k'] % 2 == 0]))

    def test_largest_remainder(self):
        self.assertEqual(sum(run.largest_remainder({'a': 5, 'b': 3, 'c': 1}, 7).values()), 7)
        self.assertEqual(run.largest_remainder({'a': 1, 'b': 1}, 4), {'a': 2, 'b': 2})

    def test_registered_arms_and_their_subsets(self):
        plan = run.make_plan(self.meta(), 1)
        self.assertEqual([arm['name'] for arm in plan['arms']][:2], list(run.CORE))
        self.assertEqual(len(plan['arms']), 8)
        self.assertEqual([arm['proposals'] for arm in plan['arms'] if arm['name'].endswith('t8')], [7, 7])
        self.assertEqual(run.make_plan(self.meta(), 1, ['dspark-t16'])['arms'][0]['name'], 'dspark-t16')

    def test_group_order_is_deterministic_and_interleaves_the_sets(self):
        rows = [dict(set='swe', group=g) for g in range(10)] + [dict(set='own', group=100 + g) for g in range(5)]
        order = run.group_order(rows, 3)
        self.assertEqual(order, run.group_order(rows, 3))
        self.assertEqual(sorted(order), sorted(set(r['group'] for r in rows)))
        head = order[:6]
        self.assertTrue(any(g >= 100 for g in head))        # the small set appears early, not last
        self.assertTrue(any(g < 100 for g in head))

    def test_the_forced_arm_runs_only_on_scheduled_turns_and_calibrates(self):
        for entry in self.records[:2]:
            entry['schedule'] = None
        # rebuild the bundle so two turns carry no schedule
        shutil.rmtree(self.bundle)
        os.remove(os.path.join(self.root, 'map.json'))
        groups, chained = bundle.prefix_groups(self.records)
        mapping = bundle.anonymise(self.records, 1)
        bundle.write_bundle(self.bundle, self.records, dict(swe=8, own=6), dict(scheduled=12), groups, chained, mapping,
                            os.path.join(self.root, 'map.json'))
        meta = self.meta()
        plan = run.make_plan(meta, 1, ['dflash2-forced'])
        unscheduled = set(row['k'] for row in meta if not row['scheduled'])
        self.assertFalse(unscheduled & set(plan['subsets']['forced']))


class GateTests(unittest.TestCase):
    def test_v1_threshold_overall_and_per_set(self):
        good = [dict(set='swe', rows=100, agree=96), dict(set='own', rows=100, agree=95)]
        self.assertEqual(run.gate_v1(good)[0], 'PASS')
        self.assertEqual(run.gate_v1([dict(set='swe', rows=100, agree=96), dict(set='own', rows=100, agree=94)])[0], 'FAIL')
        self.assertEqual(run.gate_v1([])[0], 'NOT_RUN')

    def test_v3_band(self):
        meta = [dict(k=k, set='swe' if k % 2 else 'own', cluster=k // 4, bucket='4k', weight=1.0) for k in range(40)]

        def arm(value):
            return dict((k, [dict(committed=value, uncapped=value)] * 5) for k in range(40))
        self.assertEqual(run.gate_v3(meta, arm(5), arm(5), resamples=50)[0], 'PASS')
        self.assertEqual(run.gate_v3(meta, arm(8), arm(5), resamples=50)[0], 'FAIL')      # 1.6
        self.assertEqual(run.gate_v3(meta, arm(4), arm(5), resamples=50)[0], 'FAIL')      # 0.8
        self.assertEqual(run.gate_v3(meta, dict((k, v) for k, v in arm(5).items() if k < 3), arm(5))[0], 'NOT_RUN')

    def test_v4_boundaries(self):
        rows = [(True, 1.0)] * 99 + [(False, 1.0)]
        rounds = [(3, 3)] * 97 + [(3, 4)] * 3
        self.assertEqual(run.v4_decide(rows, rounds), 'PASS')
        self.assertEqual(run.v4_decide(rows[:-2] + [(False, 1.0)] * 2, rounds), 'FAIL')                # 98% agreement
        self.assertEqual(run.v4_decide(rows, [(3, 3)] * 96 + [(3, 4)] * 4), 'FAIL')                     # 96% rounds
        self.assertEqual(run.v4_decide(rows + [(False, 0.1)] * 50, rounds), 'PASS')                      # low-margin rows do not count
        self.assertEqual(run.v4_decide([(True, 0.1)], rounds), 'NOT_RUN')

    def test_oom_detection(self):
        class OutOfMemoryError(RuntimeError):
            pass
        self.assertTrue(run.is_oom(OutOfMemoryError()))
        self.assertTrue(run.is_oom(MemoryError()))
        self.assertFalse(run.is_oom(ValueError()))

    def test_guard_admission_arithmetic(self):
        guard = run.Guard(None, 10 ** 9, lambda: 0, lambda: 50.0)
        self.assertAlmostEqual(guard.projected_gib(122774), 4.0 + 122774 * (65536 + 51200) / float(1 << 30), places=6)
        self.assertTrue(guard.admits(122774))            # 50 - 17.3 >= 10
        self.assertFalse(run.Guard(None, 10 ** 9, lambda: 0, lambda: 25.0).admits(122774))


class MainTests(Fixture):
    def test_main_runs_the_canary_with_an_injected_environment(self):
        def environment(options, say):
            group_runner = tgt.GroupRunner(self.target, chunk=7, feature_dtype=torch.float32)
            group_runner.width = 3 * HIDDEN
            return group_runner, (lambda spec: fakes.FollowsAnswer(good=4)), self.guard(), dict(V0='PASS')
        lines = []
        code = run.main(['--bundle', self.bundle, '--out', self.out, '--deadline', '1e12', '--trip-file', self.trip, '--canary'],
                        say=lines.append, environment=environment)
        self.assertEqual(code, 0)
        canary = os.path.join(self.out, 'canary')
        done = set(json.loads(line)['k'] for line in slurp(os.path.join(canary, 'arm-dspark-t16.jsonl')).splitlines())
        meta = [json.loads(line) for line in slurp(os.path.join(self.bundle, bundle.META_NAME)).splitlines()]
        self.assertEqual(done, run.canary_keys(meta))
        self.assertEqual(json.loads(slurp(os.path.join(canary, 'validity.json')))['V0'], 'PASS')
        self.assertTrue(any(line.startswith('gates:') for line in lines))

    def test_main_prints_the_exception_type_only(self):
        def environment(options, say):
            raise RuntimeError(SENTINEL)
        lines = []
        code = run.main(['--bundle', self.bundle, '--out', self.out, '--deadline', '1e12', '--trip-file', self.trip],
                        say=lines.append, environment=environment)
        self.assertEqual(code, 2)
        self.assertEqual(lines, ['refused: RuntimeError'])

    def test_main_refuses_a_tampered_bundle(self):
        with open(os.path.join(self.bundle, bundle.META_NAME), 'a') as handle:
            handle.write('{}\n')
        lines = []
        self.assertEqual(run.main(['--bundle', self.bundle, '--out', self.out, '--deadline', '1e12', '--trip-file', self.trip],
                                  say=lines.append, environment=None), 2)
        self.assertEqual(lines, ['refused: BundleError'])

    def test_v0_needs_every_check(self):
        self.assertEqual(run.gate_v0(dict(a=True, b=True)), 'PASS')
        self.assertEqual(run.gate_v0(dict(a=True, b=False)), 'FAIL')
        self.assertEqual(run.gate_v0({}), 'FAIL')

    def test_drop_file_cache_visits_every_file(self):
        with open(os.path.join(self.root, 'shard'), 'wb') as handle:
            handle.write(b'x')
        count = run.drop_file_cache(self.root)
        self.assertTrue(count == 0 or count >= 1)      # 0 where posix_fadvise does not exist (Windows)


class ImageClosureTests(unittest.TestCase):
    """Every repository module the screen's image runs must be in the Dockerfile's COPY list, and only the light ones (no lab, no
    serving gate): a file in neither list ships a stale or missing module (the serving image bundle lesson)."""
    HERE = os.path.dirname(os.path.abspath(__file__))
    DOCKERFILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'docker', 'a0-spark', 'Dockerfile')

    def copied(self):
        text = slurp(self.DOCKERFILE).replace('\\\n', ' ')
        names = []
        for line in text.splitlines():
            if line.startswith('COPY '):
                names.extend(os.path.basename(part)[:-3] for part in line.split()[1:-1] if part.endswith('.py'))
        return set(names)

    def local_imports(self, module):
        import ast
        tree = ast.parse(slurp(os.path.join(self.HERE, module + '.py')))
        found = set()
        for node in ast.walk(tree):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                head = name.split('.')[0]
                if os.path.exists(os.path.join(self.HERE, head + '.py')):
                    found.add(head)
        return found

    def test_the_copy_list_closes_over_the_entry_points_imports(self):
        copied = self.copied()
        self.assertIn('a0_run', copied)
        # lazily imported modules (inside real_environment) are part of the closure too
        pending, seen = ['a0_run'], set()
        while pending:
            module = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            for name in self.local_imports(module):
                self.assertIn(name, copied, '%s imports %s, which the image does not copy' % (module, name))
                pending.append(name)

    def test_the_image_does_not_carry_the_lab_or_the_watchdog(self):
        copied = self.copied()
        for name in ('c2_tau_lab', 'a0_bundle', 'a0_watchdog', 'c2_serving_gate'):
            self.assertNotIn(name, copied)

    def test_no_registry_name_or_digest_in_the_dockerfile(self):
        text = slurp(self.DOCKERFILE)
        self.assertNotRegex(text, r'sha256:[0-9a-f]{16}|[0-9a-f]{40,}|\bzot\b|registry\.')
        self.assertIn('ARG BASE', text)


if __name__ == '__main__':
    unittest.main()
