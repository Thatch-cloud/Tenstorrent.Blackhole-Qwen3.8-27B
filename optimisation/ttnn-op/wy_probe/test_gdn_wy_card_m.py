"""CPU checks for the window-WY card probe (gdn_wy_card_m.py, run_card_m.sh); no device, no ttnn.

  - the pure helpers: the commit plan, the 32-row inputs, the later-row variants, the raw-byte prefix compare (rows 0..n-1 and the
    committed state; a one-bit difference, a signed zero and a NaN payload are seen), the timing rule (kill above 90 us for two
    windows), the SRAM rule, the scope accounting, the verdict (KILL outranks NO-DECISION, CONTINUE only at full scope with a kernel);
  - run_card_m.sh: it parses, sources the card library, mounts every module from this checkout, pins the launched environment and
    names no host or serial; it refuses without a card;
  - the BASELINE flow on the fake ttnn of test_gdn_v5_card_m (the page-level memory model drives the real K5-A builder): with no
    window kernel the probe runs selftest, the controls and the timing of A and A2, and ends NO-DECISION (exit 4) saying so;
  - the WINDOW flow on the same fake, with K5-A and a fake gdn_wy_block that COMPUTE (gdn_wy_model, bf16 class) instead of writing
    recipes: this tests the harness, not a kernel. A correct fake reaches CONTINUE at full scope; each broken fake is caught by the
    section that must catch it: a later row that leaks into an earlier one, a packed launch that differs from a solo one, a state page never
    written, an input page overwritten (in a causality launch and in a timing replay), an output that is causal and exact but wrong (zeros, a
    wrong decay sign), a timing over 90 us (a KILL only at the full timing scope), and a kernel module that raises on import.
The device half (a real kernel, a real card) runs on the rig only; no kernel exists yet.
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
import types
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


class KernelLoading(unittest.TestCase):
    def test_an_absent_kernel_module_is_none(self):
        with mock.patch.dict(sys.modules, {'gdn_wy_block': None}):
            self.assertIsNone(probe.load_window_kernel())

    def test_a_present_kernel_module_that_cannot_import_is_an_error_not_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'gdn_wy_block.py').write_text('import a_module_that_does_not_exist_for_this_test\n', encoding='utf-8')
            sys.path.insert(0, directory)
            sys.modules.pop('gdn_wy_block', None)
            try:
                with self.assertRaises(ModuleNotFoundError) as caught:
                    probe.load_window_kernel()
                self.assertEqual(caught.exception.name, 'a_module_that_does_not_exist_for_this_test')
            finally:
                sys.path.remove(directory)
                sys.modules.pop('gdn_wy_block', None)

    def test_json_never_holds_nan_or_infinity(self):
        text = probe.dump_report(dict(a=float('nan'), b=[float('inf'), 1.5, dict(c=float('-inf'))], d=(1, float('nan'))))
        self.assertEqual(json.loads(text), dict(a=None, b=[None, 1.5, dict(c=None)], d=[1, None]))
        json.loads(text, parse_constant=lambda name: self.fail(name))


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
                      commits=list(probe.PLAN_COMMITS), users=4, timing_launches=48, timing_rounds=25, timing_arms=list(probe.TIMING_ARMS))
        values.update(overrides)
        return type('Arguments', (), values)()

    def good(self):
        return dict(a_qualified=True, kernel_present=True, unwritten=[], inputs_moved=[], missing_scope=[],
                    sections=dict(causality=dict(error=None, w_failures=[], control_a_causal=True, control_blind=False),
                                  packing=dict(error=None, w_exact=True), timing=dict(error=None, label='continue', w2_us=70.0),
                                  sram=dict(error=None, label='ok', cb_bytes={2: 1}, l1_bytes=2),
                                  accuracy=dict(error=None, nonfinite=False, correct=True, correctness_failures=[], reference_failed=None)))

    def test_the_plan_is_full_scope(self):
        self.assertEqual(probe.scope_missing(self.arguments()), [])

    def test_each_reduction_is_named(self):
        self.assertIn('commit n=17', probe.scope_missing(self.arguments(commits=[1, 4, 8, 15, 16, 24, 31])))
        self.assertIn('section timing', probe.scope_missing(self.arguments(sections=['selftest', 'sram', 'causality', 'packing', 'accuracy'])))
        self.assertIn('users 2 of 4', probe.scope_missing(self.arguments(users=2)))
        self.assertTrue(any(item.startswith('timing 2 launches') for item in probe.scope_missing(self.arguments(timing_launches=2))))

    def test_a_run_without_the_w2_arm_is_reduced_scope_and_never_continues(self):
        arguments = self.arguments(timing_arms=['A', 'A2', 'W1'])
        self.assertEqual(probe.scope_missing(arguments), ['timing arm W2'])
        report = self.good()
        report['missing_scope'] = probe.scope_missing(arguments)
        report['sections']['timing'].update(label='not-run', w2_us=None)
        verdict, problems = probe.decide(report)
        self.assertEqual(verdict, 'NO-DECISION')
        self.assertTrue(any('W2 arm was not measured' in problem for problem in problems))
        # the same holds when the scope accounting is bypassed: a kernel present and a timing label of not-run is never CONTINUE
        report['missing_scope'] = []
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')
        # without the timing section the arms are not asked for (the section itself is the missing item)
        self.assertEqual(probe.scope_missing(self.arguments(sections=['selftest'], timing_arms=['A'])),
                         ['section %s' % name for name in probe.SECTIONS if name != 'selftest'] )

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
                       lambda r: r['sections']['accuracy'].update(correct=False, correctness_failures=['R1 seed 17 user 0: output W 1 > 1.5 x K5-A 0.1']),
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

    def test_a_timing_kill_needs_the_full_timing_scope(self):
        report = self.good()
        report['sections']['timing'].update(label='kill', w2_us=91.0)
        self.assertEqual(probe.decide(report)[0], 'KILL')
        report['missing_scope'] = ['regime R2']   # another reduction does not excuse a measured kill at the plan's timing scope
        self.assertEqual(probe.decide(report)[0], 'KILL')
        report['missing_scope'] = ['timing 2 launches x 2 rounds (plan 48 x 25)']
        verdict, problems = probe.decide(report)
        self.assertEqual(verdict, 'NO-DECISION')
        self.assertTrue(any('reduced timing scope' in problem for problem in problems))

    def test_a_kernel_module_that_raised_on_import_is_no_decision_with_its_error(self):
        report = self.good()
        report.update(kernel_present=False, kernel_import_error='ModuleNotFoundError: No module named \'x\'')
        verdict, problems = probe.decide(report)
        self.assertEqual(verdict, 'NO-DECISION')
        self.assertTrue(any('raised on import' in problem for problem in problems))
        self.assertFalse(any('is not built' in problem for problem in problems))

    def test_a_reference_that_fails_is_no_decision_and_a_wrong_answer_is_a_kill(self):
        report = self.good()
        report['sections']['accuracy'].update(correct=None, reference_failed='K5-A output error against fp64 is 0.9')
        self.assertEqual(probe.decide(report)[0], 'NO-DECISION')
        report['sections']['accuracy'].update(correct=False, reference_failed=None, correctness_failures=['wrong'])
        verdict, problems = probe.decide(report)
        self.assertEqual((verdict, problems[0]), ('KILL', 'correctness: wrong'))

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
        self.assertEqual([probe.exit_code(v) for v in ('CONTINUE', 'KILL', 'NO-DECISION')], [0, 10, 4])
        self.assertEqual(probe.EXIT_KILL, 10)   # 1 is what a launcher refusal and an import-time crash return
        report = self.good()
        report['verdict'] = 'CONTINUE'
        line = probe.verdict_line(report)
        self.assertTrue(line.startswith('GDN_WY verdict=CONTINUE scope=full kernel=present'))
        self.assertIn('correctness=within-bound', line)
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

    def test_the_serials_and_the_image_reference_stay_out_of_the_log(self):
        text = self.text()
        tail = text[text.index('# <<< qual_card.sh'):]
        self.assertIn('qual_card_resolve >/dev/null', tail)
        self.assertIn('qual_refuse_holders >/dev/null', tail)
        self.assertIn('qual_card_recheck >/dev/null', tail)
        for needle in ('card=$QUAL_CARD', 'node=$node', 'image=$IMAGE', 'qual_reset_hint >&2', '$R/gdn-wy-$stamp.json"'):
            self.assertNotIn(needle, tail)
        self.assertIn('image-tag=${IMAGE##*:}', tail)
        self.assertIn('10 KILL', tail)


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


# ---- the WINDOW flow: K5-A and a fake gdn_wy_block that compute (gdn_wy_model, bf16 class) on the page-level fake ----

def _logical(tensor):
    """A fake device tensor's logical bf16 values, from its pages."""
    image = dev.untile_image(torch, tensor.pages, dev.padded_shape(tensor.shape))
    return image[tuple(slice(0, size) for size in tensor.shape)]


def _store(tensor, value, at=None):
    """Write `value` (bf16, logical) into the tensor's pages; `at` = (first page, count) of a window of a stacked tensor."""
    pages = dev.tile_image(torch, dev.pad_image(torch, value))
    if at is None:
        tensor.pages = pages
    else:
        merged = tensor.pages.clone() if tensor.pages is not None else torch.full(
            (dev.page_count(tensor.shape), 512), dev.SENTINEL_WORD, dtype=torch.int32)
        merged[at[0]:at[0] + at[1]] = pages
        tensor.pages = merged


def _user_tensors(flat, user):
    start = 0 if user == 0 else 7 * user + 1
    return flat[start:start + 7]   # qkv, beta, gate, initial, output, states, z (the shared norm_w is flat[7])


class ModelBench(v5_tests.FakeBench):
    """The V5 fake, except that K5-A's launch computes the sequential recurrence (bf16 class: causal, per-token snapshots) instead of
    writing a recipe, so the harness' controls and its correctness yardstick have something true to measure."""

    def launch(self, tensors, chip_program):
        text = next((re.search(r'// gdn_seq_block(_split)? generated build: (.*)\n', kernel.kernel_source)
                     for kernel in chip_program.kernels if re.search(r'// gdn_seq_block(_split)? generated build: (.*)\n', kernel.kernel_source)),
                    None)
        if text is None or ('variant=' in text.group(2) and 'level=' not in text.group(2)):
            return super().launch(tensors, chip_program)
        flat = list(tensors)
        weight = _logical(flat[7])[0, 0].float()
        bf = probe.model.MODELS['bf16']
        for user in range((len(flat) - 1) // 7):
            qkv, beta, gate, initial, output, states, z = _user_tensors(flat, user)
            raw = probe.to_raw(torch, _logical(qkv), _logical(beta), _logical(gate), _logical(z), probe.ROWS)
            p = probe.model.prep(raw, weight, bf.dtype)
            o, snaps, _ = probe.model.seq_round(_logical(initial).float(), p, probe.ROWS, bf)
            y = probe.model.gated(o, p, probe.ROWS).transpose(1, 2).reshape(1, probe.ROWS, 1536)
            _store(output, y.bfloat16())
            _store(states, torch.stack([snap[0] for snap in snaps]).bfloat16())


class FakeBuild:
    def __init__(self, flags):
        self.flags = flags

    def cb_bytes(self, windows):
        return 100000 * windows

    def dram_bytes(self, windows):
        return 400000 * windows

    def sha256(self):
        return '0' * 64


class FakeWindowKernel:
    """The planned gdn_wy_block interface, computing with gdn_wy_model. `flags` break it in one way each."""

    def __init__(self, **flags):
        self.flags = flags
        self.launches = 0

    def load_kernels(self, root, unqualified=True):
        return FakeBuild(self.flags)

    def execute(self, device, groups, operations, output_memory=None, kernels=None, windows=2, commit_rows=None):
        self.launches += 1
        flags, bf = self.flags, probe.model.MODELS['bf16']
        rows = probe.ROWS * windows
        produced = []
        for user, (qkv, beta, gate, initial, z, norm_w) in enumerate(groups):
            output = operations.empty((1, rows, 1536), device=device, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                                      memory_config=output_memory)
            states = operations.empty((windows, 12, 128, 128), device=device, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                                      memory_config=operations.DRAM_MEMORY_CONFIG)
            produced.append((output, states))
            host = dict(qkv=_logical(qkv), beta=_logical(beta), gate=_logical(gate), z=_logical(z))
            gate_values = -host['gate'].float() if flags.get('wrong_decay_sign') else host['gate']
            raw = probe.to_raw(torch, host['qkv'], host['beta'], gate_values, host['z'], rows)
            p = probe.model.prep(raw, _logical(norm_w)[0, 0].float(), bf.dtype)
            state, outputs = _logical(initial).float(), []
            for window in range(windows):
                window_rows = probe.model.rows_of(p, probe.ROWS * window, probe.ROWS * (window + 1))
                o, context = probe.model.wy_window(state, window_rows, bf)
                outputs.append(probe.model.gated(o, window_rows, probe.ROWS).transpose(1, 2).reshape(1, probe.ROWS, 1536))
                take = probe.ROWS if commit_rows is None else commit_rows[window]
                if take > 0:
                    state = probe.model.wy_commit(state, context, take, bf)
                    if not (flags.get('unwritten_state') and window == 0):
                        _store(states, state.bfloat16(), at=(192 * window, 192))
            y = torch.cat(outputs, dim=1).bfloat16()
            if flags.get('zero_output'):
                y = torch.zeros_like(y)
            if flags.get('leak_later_row'):   # row 0 depends on the block's last row: a kernel that reads rows it must not
                y[:, 0] = (y[:, 0].float() + 0.5 * host['qkv'][:, rows - 1, :1536].float()).bfloat16()
            if flags.get('depends_on_batch') and len(groups) > 1:   # packed differs from solo by one ulp
                y[0, 0, 0] = (y[0, 0, 0].float() * 1.01 + 0.01).bfloat16()
            _store(output, y)
            if flags.get('move_input') and user == 0:
                pages = qkv.pages.clone()
                pages[0, 0] = 0x12345678   # a fixed word: idempotent, so a warm-up launch and a captured one cannot cancel
                qkv.pages = pages
        return produced


@contextlib.contextmanager
def window_runtime(kernel_flags=None, kernel=True):
    fake = ModelBench()
    synthetic = v5_tests.SyntheticRoot()
    root = synthetic.__enter__()
    modules = {'ttnn': fake}
    if kernel:
        modules['gdn_wy_block'] = FakeWindowKernel(**(kernel_flags or {}))
    patches = [mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, FOUR),
               mock.patch('gdn_multitoken.validate_handoff_runtime'),
               mock.patch.dict(v5_tests.seq.QUALIFIED, {0: v5_tests.seq.sha256(v5_tests.seq.generate(root, 0))})]
    for patch in patches:
        patch.start()
    v5_tests.split._ENGAGED.clear()
    v5_tests.verify_trace_t1.take()
    try:
        yield fake, root, modules.get('gdn_wy_block')
    finally:
        for patch in reversed(patches):
            patch.stop()
        synthetic.__exit__(None, None, None)
        v5_tests.verify_trace_t1.take()


# the whole plan, shrunk to what a fake can run in seconds: scope=full then means this
SMALL_PLAN = dict(PLAN_USERS=1, PLAN_SEEDS=(17,), PLAN_COMMITS=(4, 16, 17), PLAN_TIMING_LAUNCHES=2, PLAN_TIMING_ROUNDS=2, REGIMES=('R1',))


class WindowFlow(unittest.TestCase):
    def run_window(self, flags=None, extra=(), plan=None, kernel=True, kill_us=None):
        patches = [mock.patch.multiple(probe, **dict(SMALL_PLAN, **(plan or {})))]
        if kill_us is not None:
            patches.append(mock.patch.object(probe, 'KILL_TWO_WINDOWS_US', kill_us))
        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as directory, \
                window_runtime(flags, kernel) as (fake, root, module):
            for patch in patches:
                stack.enter_context(patch)
            out = Path(directory) / 'report.json'
            argv = ['--out', str(out), '--root', str(root), '--call-timeout', '60', *extra]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = probe.main(argv)
            text = out.read_text()
            json.loads(text, parse_constant=lambda name: self.fail('%s in the report' % name))
            return code, json.loads(text), stdout.getvalue().strip().splitlines(), module

    def test_a_correct_fake_kernel_reaches_continue_at_full_scope(self):
        code, report, lines, module = self.run_window(kill_us=1e9)
        self.assertEqual(report['problems'], [])
        self.assertEqual((code, report['verdict']), (0, 'CONTINUE'))
        self.assertEqual((report['unwritten'], report['inputs_moved']), ([], []))
        self.assertTrue(report['kernel_present'])
        sections = report['sections']
        self.assertEqual({name: section['error'] for name, section in sections.items()}, {name: None for name in sections})
        self.assertEqual(sections['causality']['w_failures'], [])
        self.assertTrue(sections['causality']['control_a_causal'])
        self.assertFalse(sections['causality']['control_blind'])
        self.assertTrue(sections['packing']['w_exact'])
        self.assertTrue(sections['accuracy']['correct'])
        self.assertEqual(sections['accuracy']['correctness_failures'], [])
        for case in sections['accuracy']['cases']:
            for name, metric in case['gate']['metrics'].items():
                self.assertTrue(metric['ok'], (name, metric))
                self.assertLessEqual(metric['w_vs_fp64'], 1.5 * metric['a_vs_fp64'])
                self.assertLess(metric['w_vs_bf16_model'], 1e-6)   # the fake computes exactly the bf16 model
        self.assertEqual(sections['sram']['label'], 'ok')
        self.assertEqual(sections['sram']['dram_kernel'], {'1': 400000, '2': 800000})
        self.assertIn('dram_plan_note', sections['sram'])
        self.assertEqual(sorted(sections['timing']['per_launch']), ['A', 'A2', 'W1', 'W2'])
        self.assertGreater(module.launches, 0)
        self.assertGreater(report['launches']['input_checks'], report['launches']['run_arm'])   # the inputs were read back after every launch and the replays
        self.assertTrue(lines[-2].startswith('GDN_WY verdict=CONTINUE scope=full kernel=present causality=clean packing=exact correctness=within-bound'))

    def test_a_later_row_that_leaks_into_an_earlier_one_is_a_kill(self):
        code, report, lines, _ = self.run_window(dict(leak_later_row=True), extra=('--sections', 'selftest,causality'))
        self.assertEqual((code, report['verdict']), (10, 'KILL'))
        self.assertTrue(report['sections']['causality']['w_failures'])
        self.assertTrue(report['sections']['causality']['control_a_causal'])
        self.assertIn('causality=MISMATCH', lines[-2])

    def test_a_packed_launch_that_differs_from_a_solo_launch_is_a_kill(self):
        code, report, lines, _ = self.run_window(dict(depends_on_batch=True), extra=('--sections', 'selftest,packing', '--users', '2'),
                                                 plan=dict(PLAN_USERS=2))
        self.assertEqual((code, report['verdict']), (10, 'KILL'))
        self.assertFalse(report['sections']['packing']['w_exact'])
        self.assertIn('packing=DIFFERS', lines[-2])

    def test_a_state_page_the_kernel_never_wrote_is_a_kill(self):
        code, report, _, _ = self.run_window(dict(unwritten_state=True), extra=('--sections', 'selftest,causality', '--commits', '4'))
        self.assertEqual((code, report['verdict']), (10, 'KILL'))
        self.assertTrue(any(item['tensor'] == 'states0' and item['arm'] == 'W2' for item in report['unwritten']))

    def test_a_launch_that_overwrites_an_input_is_a_kill(self):
        code, report, _, _ = self.run_window(dict(move_input=True), extra=('--sections', 'selftest,causality', '--commits', '4'))
        self.assertEqual((code, report['verdict']), (10, 'KILL'))
        self.assertTrue(report['inputs_moved'])
        self.assertTrue(all(item['arm'] == 'W2' and item['tensor'] == 'qkv' for item in report['inputs_moved']))
        self.assertTrue(any('an input moved' in problem for problem in report['problems']))
        self.assertEqual(report['sections']['causality']['w_failures'], [])   # only the input read-back sees it

    def test_a_kernel_that_moves_an_input_during_the_timing_replays_is_a_kill(self):
        code, report, _, _ = self.run_window(dict(move_input=True), extra=('--sections', 'selftest,timing'))
        self.assertEqual((code, report['verdict']), (10, 'KILL'))
        self.assertTrue(any(item['case'] == 'timing' and item['arm'] == 'timing replays' for item in report['inputs_moved']))

    def test_a_fast_causal_exact_but_wrong_kernel_is_a_kill_by_the_correctness_gate_alone(self):
        for flags in (dict(zero_output=True), dict(wrong_decay_sign=True)):
            with self.subTest(flags=flags):
                code, report, lines, _ = self.run_window(flags, extra=('--sections', 'selftest,causality,packing,accuracy', '--commits', '4'))
                self.assertEqual((code, report['verdict']), (10, 'KILL'))
                self.assertEqual(report['sections']['causality']['w_failures'], [])
                self.assertTrue(report['sections']['packing']['w_exact'])
                self.assertFalse(report['sections']['accuracy']['correct'])
                self.assertTrue(any(problem.startswith('correctness:') for problem in report['problems']))
                self.assertIn('correctness=WRONG', lines[-2])

    def test_a_timing_over_the_rule_kills_only_at_the_full_timing_scope(self):
        code, report, _, _ = self.run_window(extra=('--sections', 'selftest,timing'), kill_us=-1.0)
        self.assertEqual((code, report['verdict']), (10, 'KILL'))
        self.assertEqual(report['sections']['timing']['label'], 'kill')
        # the same measurement at a quarter of the plan's launches is reported, never a KILL
        code, report, _, _ = self.run_window(extra=('--sections', 'selftest,timing', '--timing-launches', '1'), kill_us=-1.0)
        self.assertEqual((code, report['verdict']), (4, 'NO-DECISION'))
        self.assertTrue(any('reduced timing scope' in problem for problem in report['problems']))

    def test_a_run_without_the_w2_arm_never_reaches_continue(self):
        code, report, _, _ = self.run_window(extra=('--sections', 'selftest,timing', '--timing-arms', 'A,A2,W1'), kill_us=1e9)
        self.assertEqual((code, report['verdict']), (4, 'NO-DECISION'))
        self.assertIn('timing arm W2', report['missing_scope'])

    def test_a_kernel_module_that_cannot_import_is_recorded_not_read_as_not_built(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'gdn_wy_block.py').write_text('import a_module_that_does_not_exist_for_this_test\n', encoding='utf-8')
            sys.path.insert(0, directory)
            try:
                code, report, _, _ = self.run_window(kernel=False, extra=('--sections', 'selftest'))
            finally:
                sys.path.remove(directory)
                sys.modules.pop('gdn_wy_block', None)
        self.assertEqual((code, report['verdict']), (4, 'NO-DECISION'))
        self.assertFalse(report['kernel_present'])
        self.assertIn('a_module_that_does_not_exist_for_this_test', report['kernel_import_error'])
        self.assertTrue(any('raised on import' in problem for problem in report['problems']))
        self.assertFalse(any('is not built' in problem for problem in report['problems']))


if __name__ == '__main__':
    unittest.main()
