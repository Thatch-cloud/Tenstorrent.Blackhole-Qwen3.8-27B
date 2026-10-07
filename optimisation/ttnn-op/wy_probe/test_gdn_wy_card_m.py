"""CPU checks for the window-WY card probe (gdn_wy_card_m.py, run_card_m.sh); no device, no ttnn.

  - the pure helpers: the commit plan, the 32-row inputs, the later-row variants, the raw-byte prefix compare (rows 0..n-1 and the
    committed state; a one-bit difference, a signed zero and a NaN payload are seen), the timing rule (kill above 90 us for two
    windows), the SRAM rule, the scope accounting, the verdict (KILL outranks NO-DECISION, CONTINUE only at full scope with a kernel);
  - run_card_m.sh: it parses, sources the card library, mounts every module from this checkout, pins the launched environment and
    names no host or serial; it refuses without a card;
  - the BASELINE flow on the fake ttnn of test_gdn_v5_card_m (the page-level memory model drives the real K5-A builder): with no
    window kernel the probe runs selftest, the controls and the timing of A and A2, and ends NO-DECISION (exit 4) saying so.
    The W arms are not run on a fake: the kernel does not exist, and a stub would test the stub.
The device half runs on the rig only.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
V5 = HERE.parent / 'v5split'
for path in (HERE, V5, CI):
    sys.path.insert(0, str(path))

import torch  # noqa: E402

import gdn_seq_block_device_test as dev  # noqa: E402
import gdn_wy_card_m as probe  # noqa: E402
import test_gdn_v5_card_m as v5_tests  # noqa: E402
import tp_shapes  # noqa: E402

SCRIPT = HERE / 'run_card_m.sh'
BASH = shutil.which('bash')
FOUR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_VERIFY_T1': '1'}


def found():
    with mock.patch.dict(os.environ, FOUR):
        return tp_shapes.geometry(4)


class PureHelpers(unittest.TestCase):
    def test_commit_plan_splits_n_over_two_windows(self):
        self.assertEqual(probe.commit_plan(1), (1, 0))
        self.assertEqual(probe.commit_plan(16), (16, 0))
        self.assertEqual(probe.commit_plan(17), (16, 1))
        self.assertEqual(probe.commit_plan(31), (16, 15))

    def test_window_inputs_are_two_windows_of_the_same_user(self):
        norm_w, users, poison = probe.window_inputs(torch, 'R1', 17, 2, found(), 2)
        self.assertEqual(len(users), 2)
        self.assertEqual(tuple(users[0]['qkv'].shape), (1, 32, 2560))
        self.assertEqual(tuple(users[0]['beta'].shape), (1, 32, 12))
        self.assertEqual(tuple(users[0]['z'].shape), (1, 32, 1536))
        self.assertEqual(tuple(users[0]['initial'].shape), (1, 12, 128, 128))
        again = probe.window_inputs(torch, 'R1', 17, 2, found(), 2)
        self.assertTrue(torch.equal(users[1]['qkv'], again[1][1]['qkv']))
        self.assertFalse(torch.equal(users[0]['qkv'][:, :16], users[0]['qkv'][:, 16:]))
        one = probe.window_inputs(torch, 'R1', 17, 2, found(), 1)
        self.assertTrue(torch.equal(one[1][0]['qkv'], users[0]['qkv'][:, :16]))

    def test_variants_change_exactly_the_later_rows(self):
        norm_w, users, poison = probe.window_inputs(torch, 'R1', 17, 1, found(), 2)
        other = probe.window_inputs(torch, 'R1', 5017, 1, found(), 2)[1]
        n = 7
        for variant in ('zeros', 'other', 'nan'):
            changed = probe.vary_rows(torch, users, n, variant, other)[0]
            for name in ('qkv', 'beta', 'gate', 'z'):
                self.assertTrue(torch.equal(changed[name][:, :n], users[0][name][:, :n]), (variant, name))
                self.assertFalse(torch.equal(changed[name][:, n:], users[0][name][:, n:]) and variant != 'other', (variant, name))
            self.assertTrue(torch.equal(changed['initial'], users[0]['initial']))
        self.assertEqual(float(probe.vary_rows(torch, users, n, 'zeros')[0]['qkv'][:, n:].abs().max()), 0.0)
        self.assertTrue(bool(torch.isnan(probe.vary_rows(torch, users, n, 'nan')[0]['gate'][:, n:]).all()))
        earlier = probe.vary_rows(torch, users, n, 'earlier', other)[0]
        self.assertFalse(torch.equal(earlier['qkv'][:, 0], users[0]['qkv'][:, 0]))
        self.assertTrue(torch.equal(earlier['qkv'][:, 1:], users[0]['qkv'][:, 1:]))
        self.assertTrue(torch.equal(probe.vary_rows(torch, users, n, 'real')[0]['qkv'], users[0]['qkv']))
        with self.assertRaises(ValueError):
            probe.vary_rows(torch, users, n, 'bogus')

    def images(self, n_states):
        gen = torch.Generator().manual_seed(3)
        output = torch.randn(1, 32, 1536, generator=gen).bfloat16()
        states = torch.randn(n_states, 12, 128, 128, generator=gen).bfloat16()
        return output, states

    def test_prefix_compare_is_raw_bytes_on_the_committed_rows_only(self):
        for kind, n_states, commit, n in (('W', 2, (16, 5), 21), ('A', 16, None, 11)):
            a = self.images(n_states)
            b = (a[0].clone(), a[1].clone())
            self.assertTrue(probe.prefix_bytes_equal(torch, a, b, n, kind, commit or (16, 5))['exact'])
            b[0][0, n + 4, 3] = 1.0   # a row past the committed prefix: not compared
            self.assertTrue(probe.prefix_bytes_equal(torch, a, b, n, kind, commit or (16, 5))['exact'])
            b = (a[0].clone(), a[1].clone())
            b[0][0, n - 1, 3] = b[0][0, n - 1, 3] + 1   # a committed row
            self.assertFalse(probe.prefix_bytes_equal(torch, a, b, n, kind, commit or (16, 5))['parts']['output'])

    def test_a_state_window_that_was_not_committed_is_not_compared(self):
        a = self.images(2)
        b = (a[0].clone(), a[1].clone())
        b[1][1, 0, 0, 0] = b[1][1, 0, 0, 0] + 1
        self.assertTrue(probe.prefix_bytes_equal(torch, a, b, 10, 'W', (10, 0))['exact'])
        self.assertFalse(probe.prefix_bytes_equal(torch, a, b, 20, 'W', (16, 4))['exact'])

    def test_signed_zero_and_nan_payload_are_differences(self):
        a = self.images(2)
        zero, negative = a[0].clone(), a[0].clone()
        zero[0, 0, 0], negative[0, 0, 0] = 0.0, -0.0
        self.assertTrue(bool((zero == negative).all()))
        self.assertFalse(probe.prefix_bytes_equal(torch, (zero, a[1]), (negative, a[1]), 4, 'W', (4, 0))['exact'])
        one, two = a[0].clone(), a[0].clone()
        one[0, 0, 0] = torch.tensor(0x7FC1, dtype=torch.int16).view(torch.bfloat16)
        two[0, 0, 0] = torch.tensor(0x7FC2, dtype=torch.int16).view(torch.bfloat16)
        self.assertFalse(probe.prefix_bytes_equal(torch, (one, a[1]), (two, a[1]), 4, 'W', (4, 0))['exact'])

    def test_timing_rule_kills_above_ninety_us_for_two_windows(self):
        self.assertEqual(probe.timing_verdict(193.0, 387.0, 60.0, 89.9)['label'], 'continue')
        self.assertEqual(probe.timing_verdict(193.0, 387.0, 60.0, 90.1)['label'], 'kill')
        self.assertEqual(probe.timing_verdict(193.0, 387.0, None, None)['label'], 'not-run')
        got = probe.timing_verdict(193.0, 387.0, 60.0, 80.0)
        self.assertAlmostEqual(got['per_layer_saving_us'], 307.0)
        self.assertAlmostEqual(got['verify_ms_saving_est'], 48 * 307.0 / 1000)
        self.assertAlmostEqual(got['speedup_vs_a2'], 387.0 / 80.0)

    def test_sram_rule(self):
        self.assertEqual(probe.sram_verdict({1: 50000, 2: 70000})['label'], 'ok')
        self.assertEqual(probe.sram_verdict({1: 50000, 2: probe.L1_BYTES + 1})['label'], 'kill')
        self.assertEqual(probe.sram_verdict({})['label'], 'not-run')

    def test_the_cpu_plan_fits_l1(self):
        plan = probe.model.sram_plan(windows=2)
        self.assertLess(plan['chained_total_bytes'], probe.L1_BYTES)


class VerdictTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = dict(sections=list(probe.SECTIONS), regimes=list(probe.REGIMES), seeds=list(probe.PLAN_SEEDS),
                      commits=list(probe.PLAN_COMMITS), users=4, timing_launches=48, timing_rounds=25)
        values.update(overrides)
        return type('Arguments', (), values)()

    def good(self):
        return dict(a_qualified=True, kernel_present=True, unwritten=[], inputs_moved=[], missing_scope=[],
                    sections=dict(causality=dict(error=None, w_failures=[], control_a_causal=True, control_blind=False),
                                  packing=dict(error=None, w_exact=True), timing=dict(error=None, label='continue', w2_us=70.0),
                                  sram=dict(error=None, label='ok', cb_bytes={2: 1}, l1_bytes=2), accuracy=dict(error=None, nonfinite=False)))

    def test_the_plan_is_full_scope(self):
        self.assertEqual(probe.scope_missing(self.arguments()), [])

    def test_each_reduction_is_named(self):
        self.assertIn('commit n=17', probe.scope_missing(self.arguments(commits=[1, 4, 8, 15, 16, 24, 31])))
        self.assertIn('section timing', probe.scope_missing(self.arguments(sections=['selftest', 'sram', 'causality', 'packing', 'accuracy'])))
        self.assertIn('users 2 of 4', probe.scope_missing(self.arguments(users=2)))
        self.assertTrue(any(item.startswith('timing 2 launches') for item in probe.scope_missing(self.arguments(timing_launches=2))))

    def test_continue_only_with_a_kernel_at_full_scope(self):
        self.assertEqual(probe.decide(self.good()), ('CONTINUE', []))
        report = self.good()
        report['kernel_present'] = False
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')
        report = self.good()
        report['missing_scope'] = ['section timing']
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')

    def test_each_kill_rule_kills_even_at_reduced_scope_or_with_an_error(self):
        for mutate in (lambda r: r['sections']['causality'].update(w_failures=['x']),
                       lambda r: r['sections']['packing'].update(w_exact=False, first_failure='u1'),
                       lambda r: r['sections']['timing'].update(label='kill', w2_us=91.0),
                       lambda r: r['sections']['sram'].update(label='kill'),
                       lambda r: r.update(unwritten=[dict(arm='W2')]),
                       lambda r: r.update(inputs_moved=[dict(tensor='qkv')]),
                       lambda r: r['sections']['accuracy'].update(nonfinite=True)):
            report = self.good()
            mutate(report)
            report['missing_scope'] = ['section timing']
            report['sections']['sram']['error'] = report['sections']['sram']['error'] or None
            verdict, problems = probe.decide(report)
            self.assertEqual(verdict, 'KILL', problems)

    def test_a_harness_control_that_fails_is_no_decision_not_a_finding(self):
        report = self.good()
        report['sections']['causality']['control_a_causal'] = False
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')
        report = self.good()
        report['sections']['causality']['control_blind'] = True
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')
        report = self.good()
        report['sections']['timing']['error'] = 'boom'
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')
        report = self.good()
        report['a_qualified'] = False
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')

    def test_exit_codes_and_the_verdict_line(self):
        self.assertEqual([probe.exit_code(v) for v in ('CONTINUE', 'KILL', 'NO-DECISION')], [0, 1, 4])
        report = self.good()
        report['verdict'] = 'CONTINUE'
        line = probe.verdict_line(report)
        self.assertTrue(line.startswith('GDN_WY verdict=CONTINUE scope=full kernel=present'))
        self.assertIn('never a serving licence', line)
        report['kernel_present'] = False
        self.assertIn('kernel=absent causality=control-only', probe.verdict_line(report))


class ParseTests(unittest.TestCase):
    def test_defaults_are_the_full_plan(self):
        arguments = probe.parse(['--out', 'x.json'])
        self.assertEqual(probe.scope_missing(arguments), [])
        self.assertEqual(arguments.users, 4)

    def test_bad_arguments_are_usage_errors(self):
        for bad in (['--sections', 'bogus'], ['--regimes', 'R9'], ['--timing-arms', 'W2'], ['--commits', '32'], ['--commits', '0'],
                    ['--timing-launches', '0']):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    probe.parse(['--out', 'x.json'] + bad)


class ScriptTests(unittest.TestCase):
    def text(self):
        return SCRIPT.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))

    def test_every_module_is_mounted_from_this_checkout(self):
        text = self.text()
        for name in probe.MODULES:
            if name in ('gdn_wy_block.py',):
                continue   # the planned kernel: mounted by glob when it exists
            self.assertTrue(name in text or name in ('gdn_wy_card_m.py',), name)
        for name in ('gdn_seq_block.py', 'gdn_seq_block_split.py', 'gdn_wy_model.py', 'gdn_seq_block_device_test.py'):
            self.assertTrue((CI / name).exists(), name)
        self.assertIn('gdn_v5_card_m.py', text)
        self.assertIn('gdn_wy_block*', text)

    def test_the_launched_environment_is_pinned_and_the_card_comes_first(self):
        text = self.text()
        self.assertIn('-e QWEN_FAST_TP=4 -e QWEN_FAST_VERIFY_T1=1', text)
        self.assertIn('--network none', text)
        self.assertIn('# >>> qual_card.sh', text)
        self.assertEqual(len(re.findall(r'^timeout -k 30 "\$timeout_s" docker run', text, flags=re.M)), 1)
        self.assertLess(text.index('qual_card_select'), text.index('docker images'))
        self.assertLess(text.index('qual_refuse_holders'), text.index('docker images'))
        self.assertLess(text.index('qual_card_recheck'), text.index('timeout -k 30 "$timeout_s" docker run'))
        self.assertIn('3|124|137)', text)

    def test_no_registry_host_serial_or_address_is_named_beyond_the_embedded_block(self):
        text = self.text()
        start, end = text.index('# >>> qual_card.sh'), text.index('# <<< qual_card.sh')
        for path_text in (text[:start] + text[end:], (HERE / 'gdn_wy_card_m.py').read_text(encoding='utf-8')):
            for needle in ('zot.', '.local:', 'blackhole-', 'thatch', '192.168.', '10.0.', '/home/', 'C:' + chr(92)):
                self.assertNotIn(needle, path_text)

    def test_the_embedded_card_library_is_the_canonical_one(self):
        library = (CI / 'qual_card.sh').read_text(encoding='utf-8')
        text = self.text()
        start = text.index('# >>> qual_card.sh')
        nl = chr(10)
        end = text.index(nl, text.index('# <<< qual_card.sh')) + 1
        self.assertEqual(text[start:end], library)
        self.assertTrue(text[end:].startswith('qual_card_select' + nl))

    @unittest.skipUnless(BASH, 'needs bash')
    def test_the_script_parses(self):
        subprocess.run([BASH, '-n', str(SCRIPT)], check=True)

    @unittest.skipUnless(BASH, 'needs bash')
    def test_it_refuses_without_a_card(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith('QUAL_')}
        env.pop('ALLOW_SERVING_CARD', None)
        env['REPO'] = str(ROOT)
        done = subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True, env=env, cwd=str(HERE))
        self.assertEqual(done.returncode, 1)
        self.assertIn('QUAL_CARD is not set', done.stderr)

    def test_the_probe_has_no_kernel_yet(self):
        self.assertIsNone(probe.load_window_kernel())


class BaselineFlow(unittest.TestCase):
    def run_probe(self, extra=()):
        with tempfile.TemporaryDirectory() as directory, v5_tests.fake_runtime() as (fake, root):
            out = Path(directory) / 'report.json'
            argv = ['--out', str(out), '--root', str(root), '--users', '1', '--seeds', '17', '--regimes', 'R1', '--commits', '4,16',
                    '--timing-launches', '2', '--timing-rounds', '2', '--call-timeout', '60', *extra]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = probe.main(argv)
            return code, json.loads(out.read_text()), stdout.getvalue().strip().splitlines()

    def test_without_a_kernel_the_baseline_runs_and_the_verdict_says_why(self):
        code, report, lines = self.run_probe(extra=('--sections', 'selftest,sram,causality,packing,accuracy,timing'))
        self.assertEqual(code, 4)
        self.assertEqual(report['verdict'], 'NO-DECISION')
        self.assertFalse(report['kernel_present'])
        self.assertTrue(any('not built' in problem for problem in report['problems']))
        self.assertTrue(report['a_qualified'])
        self.assertTrue(report['sections']['selftest']['raw_round_trip_exact'])
        self.assertEqual(report['sections']['sram']['label'], 'not-run')
        self.assertIsNone(report['sections']['packing']['w_exact'])
        self.assertEqual({name: section['error'] for name, section in report['sections'].items()},
                         {name: None for name in report['sections']})
        timing = report['sections']['timing']
        self.assertEqual(sorted(timing['per_launch']), ['A', 'A2'])
        self.assertEqual(timing['label'], 'not-run')
        self.assertGreater(timing['per_launch']['A2']['median_us'], 0)
        self.assertTrue(lines[-2].startswith('GDN_WY verdict=NO-DECISION'))
        self.assertEqual(json.loads(lines[-1])['kind'], 'gdn-wy-probe')

    def test_the_probe_refuses_a_process_without_the_four_card_environment(self):
        with tempfile.TemporaryDirectory() as directory, v5_tests.fake_runtime() as (fake, root):
            env = {k: v for k, v in os.environ.items() if k not in FOUR}
            with mock.patch.dict(os.environ, env, clear=True):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = probe.main(['--out', str(Path(directory) / 'r.json'), '--root', str(root), '--sections', 'selftest'])
            self.assertEqual(code, 4)


if __name__ == '__main__':
    unittest.main()
