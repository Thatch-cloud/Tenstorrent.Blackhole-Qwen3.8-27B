"""tp4_sampdraft: the flags (strict, refused at the pair), the profiles (each its control plus exactly its flags), the smoke rules that
prove each lever engaged, and the image copy lists and CPU allowlist that carry the lever files."""

import ast
import json
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import c2_smoke_check
import tp4_sampdraft as sd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']

CONTROL, CONTROL_GATE = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-gate'
SAMP, D2 = 'c2-packed-tp4-best-samp', 'c2-packed-tp4-best-d2'
SAMP_GATE, D2_GATE = 'c2-packed-tp4-best-gate-samp', 'c2-packed-tp4-best-gate-d2'
# profile -> (control, exact env additions)
DELTAS = {SAMP: (CONTROL, {sd.SHARD_ARGMAX: '1'}),
          D2: (CONTROL, {sd.DRAFT_CONV: '1', sd.DRAFT_HEADS: '1'}),
          SAMP_GATE: (CONTROL_GATE, {sd.SHARD_ARGMAX: '1', sd.SHARD_ARGMAX_AUDIT: '1'}),
          D2_GATE: (CONTROL_GATE, {sd.DRAFT_CONV: '1', sd.DRAFT_HEADS: '1', sd.DRAFT_CONV_AUDIT: '1', sd.DRAFT_HEADS_AUDIT: '1'})}
FOUR = {'QWEN_FAST_TP': '4'}


def clean_environment(**extra):
    environ = {key: value for key, value in os.environ.items() if key not in sd.ALL_FLAGS and key != 'QWEN_FAST_TP'}
    environ.update(extra)
    return environ


class FlagTests(unittest.TestCase):
    def test_every_flag_is_strict(self):
        for flag in sd.LEVERS:
            self.assertFalse(sd.enabled(flag, dict(FOUR)))
            self.assertFalse(sd.enabled(flag, dict(FOUR, **{flag: '0'})))
            self.assertTrue(sd.enabled(flag, dict(FOUR, **{flag: '1'})))
            for bad in ('', '2', 'true', 'on', ' 1'):
                with self.subTest(flag=flag, value=bad), self.assertRaises(ValueError):
                    sd.enabled(flag, dict(FOUR, **{flag: bad}))

    def test_every_flag_is_refused_at_the_pair_and_off_is_fine_there(self):
        for environ in ({}, {'QWEN_FAST_TP': '2'}):
            for flag in sd.ALL_FLAGS:
                with self.subTest(flag=flag, environ=environ), self.assertRaisesRegex(ValueError, 'TP4 levers'):
                    if flag in sd.LEVERS:
                        sd.enabled(flag, dict(environ, **{flag: '1'}))
                    else:
                        sd.audit_enabled(flag, dict(environ, **{flag: '1', sd.AUDITS[flag]: '1'}))
            for flag in sd.LEVERS:
                self.assertFalse(sd.enabled(flag, dict(environ)))
            for flag in sd.AUDITS:
                self.assertFalse(sd.audit_enabled(flag, dict(environ)))

    def test_an_audit_without_its_lever_is_a_misconfigured_arm_and_raises(self):
        for audit, lever in sd.AUDITS.items():
            self.assertTrue(sd.audit_enabled(audit, dict(FOUR, **{audit: '1', lever: '1'})))
            with self.assertRaisesRegex(ValueError, 'would compare nothing'):
                sd.audit_enabled(audit, dict(FOUR, **{audit: '1'}))
            with self.assertRaisesRegex(ValueError, 'would compare nothing'):
                sd.audit_enabled(audit, dict(FOUR, **{audit: '1', lever: '0'}))

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            sd.enabled('QWEN_FAST_TP4_NOPE', dict(FOUR))
        with self.assertRaises(ValueError):
            sd.audit_enabled(sd.DRAFT_HEADS, dict(FOUR))

    def test_the_flags_are_read_at_call_time_not_import(self):
        with patch.dict(os.environ, clean_environment(QWEN_FAST_TP='4'), clear=True):
            self.assertFalse(sd.enabled(sd.SHARD_ARGMAX))
            os.environ[sd.SHARD_ARGMAX] = '1'
            self.assertTrue(sd.enabled(sd.SHARD_ARGMAX))

    def test_audit_names_end_in_audit_so_the_real_text_compare_calls_them_arithmetic_neutral(self):
        import real_text_compare
        for audit in sd.AUDITS:
            self.assertTrue(audit.endswith(real_text_compare.ARITHMETIC_NEUTRAL_SUFFIXES), audit)
        for lever in sd.LEVERS:
            self.assertFalse(lever.endswith(real_text_compare.ARITHMETIC_NEUTRAL_SUFFIXES), lever)

    def test_the_markers_are_distinct_and_pindiag_lines(self):
        markers = [getattr(sd, name) for name in dir(sd) if name.endswith(('_ENGAGED', '_FALLBACK', '_AUDIT', '_MISMATCH')) and
                   name.startswith(('SARG_', 'CONV_', 'HEADS_'))]
        self.assertEqual(len(markers), len(set(markers)))
        self.assertTrue(all(marker.startswith('[PINDIAG] tp4 ') for marker in markers))


class ProfileTests(unittest.TestCase):
    def test_each_profile_is_its_control_plus_exactly_its_flags(self):
        for name, (control, additions) in DELTAS.items():
            with self.subTest(profile=name):
                mine, base = PROFILES[name], PROFILES[control]
                self.assertEqual(mine['env'], dict(base['env'], **additions))
                self.assertEqual(sorted(key for key in mine['env'] if key not in base['env']), sorted(additions))
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertIs(mine['gate_only'], True)
                for flag in additions:
                    self.assertIn(flag + '=1', mine['description'])
                self.assertIn(control, mine['description'])
                self.assertIn('UNVERIFIED on hardware', mine['description'])

    def test_the_timed_twins_keep_the_audits_off_and_the_hang_fix_the_audited_twins_keep_the_audits_on(self):
        for name in (SAMP, D2):
            env = PROFILES[name]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
            self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
            self.assertEqual(env['QWEN_FAST_BUDGET_CAP'], '1')
            self.assertNotIn(sd.SHARD_ARGMAX_AUDIT, env)
            self.assertNotIn(sd.DRAFT_CONV_AUDIT, env)
            self.assertNotIn(sd.DRAFT_HEADS_AUDIT, env)
        for name in (SAMP_GATE, D2_GATE):
            env = PROFILES[name]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
            self.assertEqual(env['QWEN_FAST_TP4_VGLUE_AUDIT'], '1')
            self.assertEqual(env['QWEN_FAST_DRAFT_SINGLES_AUDIT'], 'all')

    def test_every_flag_is_set_by_these_four_profiles_and_no_other(self):
        # plus tp4/262k8-x's eight-seat 262k twins (test_tp4_262k8_x holds each as its control plus exactly these flags)
        timed = 'c2-packed-tp4-8x262k-best-time-gate-'
        X262K_FLAGS = {timed + 's1': (sd.SHARD_ARGMAX,), timed + 'd2': (sd.DRAFT_CONV, sd.DRAFT_HEADS),
                       timed + 'stack': (sd.SHARD_ARGMAX, sd.DRAFT_CONV, sd.DRAFT_HEADS),
                       'c2-packed-tp4-8x262k-best-stack-audit': sd.ALL_FLAGS,
                       # tp4/w1: the D2 pair (conv, heads) without S1, and the D2 audits on the audited twin (test_tp4_w1)
                       'c2-packed-tp4-8x262k-w1': (sd.DRAFT_CONV, sd.DRAFT_HEADS),
                       'c2-packed-tp4-8x262k-w1-audit': (sd.DRAFT_CONV, sd.DRAFT_HEADS, sd.DRAFT_CONV_AUDIT, sd.DRAFT_HEADS_AUDIT),
                       'c2-packed-tp4-8x262k-w1-lite': (sd.DRAFT_CONV, sd.DRAFT_HEADS), 'c2-packed-tp4-8x262k-w1-nod1': (sd.DRAFT_CONV, sd.DRAFT_HEADS),
                       'c2-packed-tp4-8x262k-w1-audit-nod1': (sd.DRAFT_CONV, sd.DRAFT_HEADS, sd.DRAFT_CONV_AUDIT, sd.DRAFT_HEADS_AUDIT),
                       # tp4/w2: wave 1's D2 flags, plus its audits on the audited twin
                       'c2-packed-tp4-8x262k-w2': (sd.DRAFT_CONV, sd.DRAFT_HEADS),
                       'c2-packed-tp4-8x262k-w2-audit': (sd.DRAFT_CONV, sd.DRAFT_HEADS, sd.DRAFT_CONV_AUDIT, sd.DRAFT_HEADS_AUDIT)}
        for name, profile in PROFILES.items():
            for flag in sd.ALL_FLAGS:
                wanted = (name in DELTAS and flag in DELTAS[name][1]) or flag in X262K_FLAGS.get(name, ())
                self.assertEqual(flag in profile['env'], wanted, (name, flag))

    def test_the_samp_pair_and_the_d2_pair_never_share_a_lever(self):
        self.assertFalse(set(DELTAS[SAMP][1]) & set(DELTAS[D2][1]))
        self.assertFalse(set(DELTAS[SAMP_GATE][1]) & set(DELTAS[D2_GATE][1]))

    def test_the_four_card_levers_name_the_four_card_mesh(self):
        for name in DELTAS:
            self.assertEqual(PROFILES[name]['env']['QWEN_FAST_TP'], '4')
            self.assertEqual(PROFILES[name]['mesh_device'], 'P150x4')


class SmokeRuleTests(unittest.TestCase):
    ENV_SAMP = dict(PROFILES[SAMP]['env'])
    ENV_D2 = dict(PROFILES[D2]['env'])
    ENV_SAMP_GATE = dict(PROFILES[SAMP_GATE]['env'])
    ENV_D2_GATE = dict(PROFILES[D2_GATE]['env'])

    def test_the_smoke_checkers_markers_are_the_modules(self):
        by_flag = {flag: (engaged, fell) for flag, engaged, fell, _ in c2_smoke_check.SAMPDRAFT_LEVERS}
        self.assertEqual(by_flag, {sd.SHARD_ARGMAX: (sd.SARG_ENGAGED, sd.SARG_FALLBACK),
                                   sd.DRAFT_CONV: (sd.CONV_ENGAGED, sd.CONV_FALLBACK),
                                   sd.DRAFT_HEADS: (sd.HEADS_ENGAGED, sd.HEADS_FALLBACK)})
        self.assertEqual({flag: marker for flag, marker, _ in c2_smoke_check.SAMPDRAFT_AUDITS},
                         {sd.SHARD_ARGMAX_AUDIT: sd.SARG_AUDIT, sd.DRAFT_CONV_AUDIT: sd.CONV_AUDIT,
                          sd.DRAFT_HEADS_AUDIT: sd.HEADS_AUDIT})

    def log(self, *lines):
        return '\n'.join(lines) + '\n'

    def test_a_profile_that_asks_for_a_lever_and_logs_no_engaged_line_fails(self):
        for env, flags in ((self.ENV_SAMP, [sd.SHARD_ARGMAX]), (self.ENV_D2, [sd.DRAFT_CONV, sd.DRAFT_HEADS])):
            problems = c2_smoke_check.sampdraft_problems('nothing engaged\n', env)
            self.assertEqual(len(problems), len(flags))
            for flag in flags:
                self.assertTrue(any(flag in problem and 'never ran' in problem for problem in problems), (flag, problems))

    def test_every_engaged_line_present_is_clean(self):
        self.assertEqual(c2_smoke_check.sampdraft_problems(self.log(sd.SARG_ENGAGED + ' rows=64 workers=110 tasks=220 fold=1 audit=0'),
                                                          self.ENV_SAMP), [])
        both = self.log(sd.CONV_ENGAGED + ' site=quad rows=64 pages=320 workers=110', sd.HEADS_ENGAGED + ' site=pair op=split')
        self.assertEqual(c2_smoke_check.sampdraft_problems(both, self.ENV_D2), [])

    def test_a_fall_back_fails_even_when_the_lever_also_engaged_elsewhere(self):
        text = self.log(sd.SARG_ENGAGED + ' rows=64', sd.SARG_FALLBACK + ' rows=16 reason=logits are not interleaved DRAM')
        problems = c2_smoke_check.sampdraft_problems(text, self.ENV_SAMP)
        self.assertEqual(len(problems), 1)
        self.assertIn('fell back', problems[0])
        self.assertIn('rows=16', problems[0])
        # a fall-back fails even without a profile (env=None: the checker is run without one)
        self.assertEqual(len(c2_smoke_check.sampdraft_problems(self.log(sd.HEADS_FALLBACK + ' site=quad'), None)), 1)

    def test_an_audited_arm_needs_a_passing_audit_line_and_a_mismatch_line_never_counts_as_one(self):
        engaged = self.log(sd.SARG_ENGAGED + ' rows=64', sd.CONV_ENGAGED + ' site=pair', sd.HEADS_ENGAGED + ' site=pair')
        problems = c2_smoke_check.sampdraft_problems(engaged, self.ENV_SAMP_GATE)
        self.assertEqual(len(problems), 1)
        self.assertIn(sd.SHARD_ARGMAX_AUDIT, problems[0])
        passing = engaged + self.log(sd.SARG_AUDIT + ' 3 exact=True rows=64 chips=4')
        self.assertEqual(c2_smoke_check.sampdraft_problems(passing, self.ENV_SAMP_GATE), [])
        mismatch = engaged + self.log(sd.SARG_MISMATCH + ' round=1 chip=0 rows=[3] kernel=[5] served=[6]')
        self.assertEqual(len(c2_smoke_check.sampdraft_problems(mismatch, self.ENV_SAMP_GATE)), 1)
        conv = engaged + self.log(sd.CONV_AUDIT + ' exact=True convs=20 round=1 site=quad')
        heads = engaged + self.log(sd.HEADS_AUDIT + ' exact=True tensors=3 round=1 site=pair')
        both = conv + self.log(sd.HEADS_AUDIT + ' exact=True tensors=3 round=1 site=pair')
        self.assertEqual(c2_smoke_check.sampdraft_problems(both, self.ENV_D2_GATE), [])
        for partial, missing in ((conv, sd.DRAFT_HEADS_AUDIT), (heads, sd.DRAFT_CONV_AUDIT)):
            problems = c2_smoke_check.sampdraft_problems(partial, self.ENV_D2_GATE)
            self.assertEqual(len(problems), 1)
            self.assertIn(missing, problems[0])
        self.assertEqual(len(c2_smoke_check.sampdraft_problems(engaged, self.ENV_D2_GATE)), 2)

    def test_a_mismatch_line_trips_the_checkers_existing_audit_mismatch_rule(self):
        for line in (sd.SARG_MISMATCH + ' round=1', sd.CONV_MISMATCH + ' round=1 pairs=20', sd.HEADS_MISMATCH + ' site=pair round=1 pairs=3'):
            self.assertTrue(c2_smoke_check.MISMATCH.search(line), line)
        for line in (sd.SARG_AUDIT + ' 1 exact=True', sd.CONV_AUDIT + ' exact=True convs=1', sd.HEADS_AUDIT + ' exact=True tensors=3'):
            self.assertFalse(c2_smoke_check.MISMATCH.search(line), line)

    def test_check_runs_the_rules_with_the_served_profiles_environment(self):
        problems, _ = c2_smoke_check.check('', 'no markers\n', slide=False, env=dict(self.ENV_SAMP))
        self.assertTrue(any(sd.SHARD_ARGMAX in problem and 'never ran' in problem for problem in problems), problems)

    def test_a_control_profile_is_never_asked_for_a_marker(self):
        for name in (CONTROL, CONTROL_GATE, 'c2-packed-tp4'):
            self.assertEqual(c2_smoke_check.sampdraft_problems('nothing\n', PROFILES[name]['env']), [], name)


class ShippingTests(unittest.TestCase):
    """Every file a lever needs at run time reaches the image through both copy lists, and its tests run in CI."""

    def read(self, relative):
        return (ROOT / relative).read_text(encoding='utf-8')

    def test_the_runtime_files_exist_and_are_all_in_scripts_ci(self):
        for name in sd.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)

    def test_every_runtime_file_is_in_the_overlay_the_dockerfile_and_the_image_workflow(self):
        overlay = self.read('docker/qwen-c2-overlay.txt').splitlines()
        dockerfile = self.read('docker/qwen-fast-serving.Dockerfile')
        workflow = self.read('.github/workflows/qwen-fast-serving-image.yml')
        for name in sd.RUNTIME_FILES:
            self.assertIn('scripts/ci/' + name, overlay, name)
            self.assertIn('scripts/ci/' + name, dockerfile, name)
            self.assertRegex(workflow, r'(?<![A-Za-z0-9_.])%s(?![A-Za-z0-9_])' % re.escape(name), name)

    def test_the_modules_that_import_the_levers_are_in_the_copy_lists_too(self):
        overlay = self.read('docker/qwen-c2-overlay.txt').splitlines()
        dockerfile = self.read('docker/qwen-fast-serving.Dockerfile')
        for name in ('verify_trace_t1.py', 'packed_verifier.py', 'draft_convolution_fused_tp.py', 'draft_head_layout_tp.py',
                     'quad_draft_tp.py'):
            self.assertTrue('scripts/ci/' + name in overlay or 'scripts/ci/' + name in dockerfile, name)

    def test_the_cpu_allowlist_names_this_branchs_test_modules(self):
        workflow = self.read('.github/workflows/qwen-integration-cpu.yml')
        for module in ('test_tp4_sampdraft', 'test_tp4_shard_argmax', 'test_tp4_draft_conv', 'test_tp4_draft_heads',
                       'test_tp4_samp_draft_report', 'test_tp4_samp_draft_window'):
            self.assertRegex(workflow, r'unittest [^\n]*\b%s\b' % module, module)
            self.assertTrue((HERE / (module + '.py')).is_file(), module)

    def test_runtime_modules_import_nothing_heavy_at_import_time(self):
        for name in ('tp4_sampdraft.py', 'tp4_shard_argmax.py', 'tp4_draft_conv.py', 'tp4_draft_heads.py'):
            tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
            imported = {alias.name.split('.')[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
            imported |= {node.module.split('.')[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
            self.assertEqual(imported - {'os', 'sys', 'pathlib', 'tp_shapes', 'tp4_sampdraft'}, set(), name)

    def test_the_pinned_modules_are_not_edited(self):
        import hashlib
        import quad_draft
        digest = hashlib.sha256((HERE / 'quad_conv_io.cpp').read_bytes()).hexdigest()
        self.assertEqual(digest, quad_draft.CONV_KERNEL_SHA256)
        # the served I/O kernels and pinned modules the pair keeps: no word of them mentions the new levers
        for name in ('draft_convolution_fused_io.cpp', 'draft_convolution_fused.py', 'draft_head_layout.py', 'quad_draft.py'):
            self.assertNotIn('tp4_sampdraft', (HERE / name).read_text(encoding='utf-8'), name)


if __name__ == '__main__':
    unittest.main()
