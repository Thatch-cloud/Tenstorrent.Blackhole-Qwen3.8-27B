"""tp4/w2: the fusion audit's wave 2 on the eight-seat 262k stack: the multi-user SDPA launch rebased onto wave 1, the conv-gates spread (F1), the job pack.

W2 is tp4/w1 (nosamp, D2a, D2c, the host-gap stage 1, U1, the drafter ring gathers) merged with tp4/sdpa-multi (QWEN_FAST_TP4_SDPA=multi, never timed beside
wave 1 before) plus F1 (QWEN_FAST_TP4_CONV_GATES_SPREAD, gdn_conv_gates_spread). The V5 recurrence split came in with wave 1 (tp4/262k8-x carries it) and stays off:
its qualified triple is empty and its flag refuses until the card-M byte gate passes. Two gate-only profiles switch the new levers on:

  c2-packed-tp4-8x262k-w2        c2-packed-tp4-8x262k-w1 plus QWEN_FAST_TP4_SDPA=multi and QWEN_FAST_TP4_CONV_GATES_SPREAD=1
  c2-packed-tp4-8x262k-w2-audit  c2-packed-tp4-8x262k-w1-audit plus those two, QWEN_FAST_TP4_SDPA_AUDIT=1 and QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT=1

This module pins the two profiles' exact deltas, that the levers' own flag readers and smoke rules accept the stack together, the merge's semantic seams
(profile level, the flagged-twin table), the job pack, and the shipping lists. F1's own tests are test_gdn_conv_gates_spread; the multi launch's are
test_sdpa_multi_tp and test_tp4_sdpa_multi_window.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_w2` from scripts/ci."""

import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import c2_serving_job as job
import c2_smoke_check
import gdn_conv_gates_spread as spread
import sdpa_long_tp
import sdpa_multi_tp
import tp4_vglue
import tp_addresses
import test_tp4_w1 as w1_tests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
FOLDER = HERE / 'references' / 'tp4-w2-jobs'
IMAGE = 'tp4-w2-1'
CONTROL = 'c2-packed-tp4-8x262k-best-time-gate'
W1, W1_AUDIT = w1_tests.W1, w1_tests.W1_AUDIT
W2, W2_AUDIT = 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit'
SDPA_TIMED, SDPA_AUDITED = 'c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit'
V5_FLAG = 'QWEN_FAST_GDN_SPLIT_V'
SDPA = {sdpa_long_tp.FLAG: 'multi'}
SDPA_AUDIT = {sdpa_multi_tp.AUDIT_FLAG: '1'}
F1 = {spread.FLAG: '1'}
F1_AUDIT = {spread.AUDIT_FLAG: '1'}
NEW_LEVERS = dict(SDPA, **F1)
NEW_AUDITS = dict(SDPA_AUDIT, **F1_AUDIT)
BANNED = w1_tests.BANNED


def load():
    return w1_tests.load()


def flat(profile):
    return w1_tests.flat(profile)


class ProfileTests(unittest.TestCase):
    def test_the_timed_arm_is_wave_1_plus_exactly_the_two_new_levers(self):
        found = load()['profiles']
        expected = flat(found[W1])
        expected['env'].update(NEW_LEVERS)
        self.assertEqual(flat(found[W2]), expected)

    def test_the_audited_arm_is_wave_1s_audited_arm_plus_the_two_levers_and_their_audits(self):
        found = load()['profiles']
        expected = flat(found[W1_AUDIT])
        expected['env'].update(NEW_LEVERS)
        expected['env'].update(NEW_AUDITS)
        self.assertEqual(flat(found[W2_AUDIT]), expected)

    def test_the_audited_arm_is_the_timed_arm_plus_every_audit_of_wave_1_and_wave_2(self):
        found = load()['profiles']
        timed, audited = found[W2]['env'], found[W2_AUDIT]['env']
        extra = {key: value for key, value in audited.items() if timed.get(key) != value}
        wave_1_audits = {key: value for key, value in found[W1_AUDIT]['env'].items() if found[W1]['env'].get(key) != value}
        self.assertEqual(extra, dict(wave_1_audits, **NEW_AUDITS))
        self.assertEqual({key for key in timed if key not in audited}, set())

    def test_both_are_gate_only_unqualified_and_nothing_else_moved(self):
        data = load()
        self.assertEqual(data['default'], 'c2-packed-tp4')
        for name in (W2, W2_AUDIT):
            with self.subTest(name=name):
                self.assertTrue(data['profiles'][name]['gate_only'])
                self.assertEqual(data['profiles'][name]['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
        for name in (CONTROL, W1, W1_AUDIT, SDPA_TIMED, SDPA_AUDITED):
            for flag in (*NEW_LEVERS, *NEW_AUDITS) if name in (CONTROL, W1, W1_AUDIT) else (spread.FLAG, spread.AUDIT_FLAG):
                self.assertNotIn(flag, data['profiles'][name]['env'], (name, flag))

    def test_the_v5_split_is_in_no_w2_profile(self):
        found = load()['profiles']
        for name in (W2, W2_AUDIT):
            self.assertNotIn(V5_FLAG, found[name]['env'])

    def test_the_levers_new_flags_are_in_no_traffic_profile_and_in_no_other_gate_arm(self):
        for name, profile in load()['profiles'].items():
            env = profile.get('env', {})
            if not profile.get('gate_only'):
                with self.subTest(name=name):
                    self.assertFalse((set(NEW_LEVERS) | set(NEW_AUDITS)) & set(env))
        carriers = sorted(name for name, profile in load()['profiles'].items() if spread.FLAG in profile.get('env', {}))
        self.assertEqual(carriers, [W2, W2_AUDIT])
        multi = sorted(name for name, profile in load()['profiles'].items() if profile.get('env', {}).get(sdpa_long_tp.FLAG) == 'multi')
        self.assertEqual(multi, sorted([SDPA_TIMED, SDPA_AUDITED, W2, W2_AUDIT]))

    def test_sdpa_multi_s_own_arms_are_untouched_by_the_merge(self):
        found = load()['profiles']
        control = flat(found[CONTROL])
        timed = flat(found[SDPA_TIMED])
        control['env'].update(SDPA)
        self.assertEqual(timed, control)
        audited = flat(found[SDPA_AUDITED])
        control['env'].update(SDPA_AUDIT)
        self.assertEqual(audited, control)

    def test_the_fabric_engine_and_mesh_are_the_controls_in_both(self):
        found = load()['profiles']
        for name in (W2, W2_AUDIT):
            with self.subTest(name=name):
                self.assertEqual(found[name]['engine'], found[CONTROL]['engine'])
                self.assertEqual(found[name]['mesh_graph_descriptor'], found[CONTROL]['mesh_graph_descriptor'])
                self.assertEqual(found[name]['env']['QWEN_FAST_CCL_TOPOLOGY'], 'ring')       # wave 1's D1: the drafter side only

    def test_wave_1_s_in_trace_sampler_is_still_out_and_the_request_warm_still_in(self):
        found = load()['profiles']
        for name in (W2, W2_AUDIT):
            self.assertNotIn('QWEN_FAST_PACKED_SAMPLER_IN_TRACE', found[name]['env'])
            self.assertEqual(found[name]['env']['QWEN_FAST_M3_REQUEST_WARM'], '1')


class FlagTests(unittest.TestCase):
    """The levers' own strict readers accept the two arms' environments, and the audited one is the only one that audits."""

    def env(self, name):
        return dict(load()['profiles'][name]['env'], QWEN_FAST_PRESTAGE='1')

    def test_every_new_flag_reads_clean_from_the_timed_arm(self):
        env = self.env(W2)
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(spread.enabled())
            self.assertFalse(spread.audit_enabled())
            self.assertTrue(sdpa_long_tp.enabled(env))
            self.assertFalse(sdpa_multi_tp.audit_enabled(env))
            self.assertFalse(tp4_vglue.audit_enabled())

    def test_every_new_flag_and_its_audit_read_clean_from_the_audited_arm(self):
        env = self.env(W2_AUDIT)
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(spread.audit_enabled())
            self.assertTrue(sdpa_multi_tp.audit_enabled(env))
            self.assertTrue(tp4_vglue.audit_enabled())
            self.assertTrue(all(tp4_vglue.enabled(name) for name in tp4_vglue.LEVERS))

    def test_the_control_and_wave_1_leave_the_new_levers_off(self):
        for name in (CONTROL, W1, W1_AUDIT):
            with self.subTest(name=name), patch.dict(os.environ, self.env(name), clear=True):
                self.assertFalse(spread.enabled())
                self.assertFalse(sdpa_long_tp.enabled(self.env(name)) and self.env(name).get(sdpa_long_tp.FLAG) == 'multi')

    def test_the_split_recurrence_is_off_in_both_and_its_qualified_triple_is_still_empty(self):
        import gdn_seq_block_split

        self.assertEqual(gdn_seq_block_split.QUALIFIED, {})      # a flag set to 2 would refuse to build (test_gdn_seq_block_split holds that)
        for name in (W2, W2_AUDIT):
            with patch.dict(os.environ, self.env(name), clear=True):
                self.assertFalse(gdn_seq_block_split.enabled(self.env(name)))
                self.assertEqual(gdn_seq_block_split.factor(self.env(name)), 1)


class MergeSeamTests(unittest.TestCase):
    """What a textual merge of two branches that both touched the same tables could have lost."""

    def test_the_flagged_twin_table_holds_both_branches_rows(self):
        found = tp_addresses.FLAGGED_TWINS
        self.assertEqual(found[('extent_attention_replay_tp', 'PackedExtentReplayReader')][0], 'QWEN_FAST_TP4_ATTN_FOLD or QWEN_FAST_TP4_SDPA')
        self.assertEqual(found[('gdn_seq_block', 'execute')][0], V5_FLAG)
        self.assertEqual(found[('gdn_device_loop_state', 'DeviceLoopState')][0], 'QWEN_FAST_TP4_GDN_GLUE')

    def test_the_sdpa_flag_alone_binds_the_attention_twin_and_the_split_flag_does_not(self):
        four = {'QWEN_FAST_TP': '4'}
        rows, modules = tp_addresses.bound_twins(four)
        names = {row[:2] for row in rows}
        self.assertNotIn(('extent_attention_replay_tp', 'PackedExtentReplayReader'), names)
        rows, modules = tp_addresses.bound_twins(dict(four, **SDPA))
        self.assertIn(('extent_attention_replay_tp', 'PackedExtentReplayReader'), {row[:2] for row in rows})

    def test_the_packed_verifier_runs_both_audits_after_the_replay(self):
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        for marker in ('tile_collective_tp.audit_round(self.operations, self.fixture, self.rounds + 1)', 'sdpa_audit(self.rounds + 1)',
                       'tp4_vglue.audit_round(self.operations, self.fixture.retained.records, self.rounds + 1)',
                       'gdn_conv_gates_spread.audit_round(self.operations, self.fixture.retained.records, self.rounds + 1)'):
            self.assertEqual(source.count(marker), 1, marker)
        self.assertLess(source.index('tp4_vglue.audit_round('), source.index('gdn_conv_gates_spread.audit_round('))

    def test_the_smoke_check_runs_every_branchs_rule(self):
        source = (HERE / 'c2_smoke_check.py').read_text(encoding='utf-8')
        for call in ('problems += sampdraft_problems(container_text, env)', 'problems += u1_problems(env, container_text)',
                     'problems += sdpa_long_problems(env, container_text)', 'problems += sdpa_multi_problems(env, container_text)',
                     'problems += spread_problems(env, container_text)'):
            self.assertEqual(source.count(call), 1, call)
        self.assertIn("GDN_SPLIT_FLAG = 'QWEN_FAST_GDN_SPLIT_V'", source)


SDPA_ENGAGED = '[PINDIAG] tp4 sdpa engaged config=multi flags=0x21 users=4 cores=64'
SDPA_CALL = '[PINDIAG] tp4 sdpa multi call layer=3 users=4'
SDPA_AUDIT_LINE = '[PINDIAG] tp4 sdpa audit 7 exact=True'
SPREAD_ENGAGED = spread.ENGAGED + ' rows=64 conv_cores=80 gate_cores=2 per_core=2 served_per_core=2 audit=1'
SPREAD_AUDIT_LINE = spread.AUDIT_MARKER + ' 2 exact=True layers=0,24 entries=14'


def wave_2_log(*, sdpa=True, sdpa_call=True, sdpa_audit=True, f1=True, f1_audit=True, **wave_1):
    lines = [w1_tests.stack_log(**wave_1)]
    if sdpa:
        lines.append(SDPA_ENGAGED)
    if sdpa_call:
        lines.append(SDPA_CALL)
    if sdpa_audit:
        lines.append(SDPA_AUDIT_LINE)
    if f1:
        lines.append(SPREAD_ENGAGED)
    if f1_audit:
        lines.append(SPREAD_AUDIT_LINE)
    return '\n'.join(lines)


class SmokeUnionTests(unittest.TestCase):
    """The smoke rules of wave 1's three branches, sdpa-multi's and F1's judge the stack together."""

    def env(self, name=W2_AUDIT):
        env = dict(load()['profiles'][name]['env'])
        env.setdefault('QWEN_FAST_TP', '4')
        return env

    def judge(self, env, log):
        problems = list(c2_smoke_check.hostgap_problems(env, log, True)[0])
        problems += c2_smoke_check.u1_problems(env, log)
        problems += c2_smoke_check.sampdraft_problems(log, env)
        problems += c2_smoke_check.sdpa_long_problems(env, log)
        problems += c2_smoke_check.sdpa_multi_problems(env, log)
        problems += c2_smoke_check.spread_problems(env, log)
        return problems

    def test_a_clean_log_of_all_five_levers_passes_the_audited_arm(self):
        self.assertEqual(self.judge(self.env(), wave_2_log()), [])

    def test_each_wave_2_lever_missing_from_the_log_is_caught_by_its_own_rule(self):
        env = self.env()
        for what, log in (('sdpa engaged', wave_2_log(sdpa=False)), ('sdpa call', wave_2_log(sdpa_call=False)),
                          ('sdpa audit', wave_2_log(sdpa_audit=False)), ('f1 engaged', wave_2_log(f1=False)),
                          ('f1 audit', wave_2_log(f1_audit=False))):
            with self.subTest(missing=what):
                self.assertTrue(self.judge(env, log), what)

    def test_each_wave_1_lever_missing_is_still_caught_beside_wave_2(self):
        env = self.env()
        for what in ('hostgap', 'u1', 'd2'):
            with self.subTest(missing=what):
                self.assertTrue(self.judge(env, wave_2_log(**{what: False})), what)

    def test_a_fell_back_or_mismatch_line_fails_the_arm(self):
        env = self.env()
        for line in (spread.FELL_BACK + ' reason=an operand is not interleaved DRAM', spread.AUDIT_MISMATCH + ' round=1 layers=0 x',
                     '[PINDIAG] tp4 sdpa audit MISMATCH layer=2'):
            with self.subTest(line=line[:60]):
                self.assertTrue(self.judge(env, wave_2_log() + '\n' + line))

    def test_the_timed_arm_needs_the_engaged_lines_alone(self):
        env = self.env(W2)
        timed = '\n'.join([w1_tests.stack_log(u1=True).replace(w1_tests.AUDIT_U1 % 'capture3', '').replace(w1_tests.AUDIT_U1 % 'capture4', ''),
                           SDPA_ENGAGED, SDPA_CALL, SPREAD_ENGAGED.replace('audit=1', 'audit=0')])
        found = [problem for problem in self.judge(env, timed) if 'audit' not in problem.lower()]
        self.assertEqual(found, [])
        self.assertTrue(self.judge(env, w1_tests.stack_log() + '\n' + SDPA_ENGAGED + '\n' + SDPA_CALL))

    def test_wave_1_arms_log_none_of_the_wave_2_lines(self):
        env = self.env(W1)
        self.assertEqual([p for p in c2_smoke_check.spread_problems(env, '') + c2_smoke_check.sdpa_multi_problems(env, '')], [])
        self.assertTrue(c2_smoke_check.spread_problems(env, SPREAD_ENGAGED))


CARD_M = {'F1-cardm-conv-gates-spread': ('cardm', 'soft', 'optimisation/ttnn-op/gdn_conv_gates_spread/run_card_m.sh'),
          'V1a-cardm-watcher': ('cardm', 'soft', 'optimisation/ttnn-op/v5split/run_card_m.sh'),
          'V1b-cardm-full': ('cardm', 'soft', 'optimisation/ttnn-op/v5split/run_card_m.sh'),
          'MR-cardm-mesh-read-probe': ('cardm', 'soft', 'optimisation/ttnn-op/mr_probe/run_card_m.sh')}
EXPECTED = {
    'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    **{name: (value[0], None, value[1]) for name, value in CARD_M.items()},
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'),
    'S0c-control-attach-smoke': ('reset smoke', CONTROL, 'stop'),
    'A1-audited-attach-smoke': ('reset smoke', W2_AUDIT, 'stop'),
    'H1-hang-shapes-w2': ('reset smoke', W2, 'stop'), 'H2-hang-shapes-w2': ('reset smoke', W2, 'stop'),
    'H3-hang-shapes-w2': ('reset smoke', W2, 'stop'), 'H4-hang-shapes-w2': ('reset smoke', W2, 'stop'),
    'H5-hang-shapes-w2': ('reset smoke', W2, 'stop'),
    'T1-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T2-timed-B-w2': ('reset smoke', W2, 'soft'),
    'T3-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T4-timed-B-w2': ('reset smoke', W2, 'soft'),
    'T5-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T6-timed-B-w2': ('reset smoke', W2, 'soft'),
    'T7-timed-C-w1-alone': ('reset smoke', W1, 'soft'),
    'P1-w2-8-user-profile': ('status reset gate', W2, 'soft'),
    'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority'}


def pack_text(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def pack_job(name):
    return job.read_job(job.parse_env(pack_text(name)), sorted(load()['profiles']), root=ROOT)


def order_text():
    return (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')


def order_lines():
    return [line.split() for line in order_text().splitlines() if line.strip() and not line.startswith('#')]


def tests_of(name):
    return pack_job(name)['tests'].replace(' ', ',').split(',')


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], list(EXPECTED))
        self.assertEqual(sorted(path.stem for path in FOLDER.glob('*.env')), sorted(EXPECTED))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image), (EXPECTED[name][2], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_with_its_actions_profile_and_the_one_image(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertEqual((result['actions'], result['tag']), (actions, IMAGE))
                self.assertEqual(result['cards'], 'pair' if name in CARD_M else 'quad')
                if profile:
                    self.assertEqual(result['profile'], profile)

    def test_the_build_comes_first_then_card_m_then_the_quad_jobs_and_x0_is_the_first_quad_job(self):
        names = [line[0] for line in order_lines()]
        self.assertEqual(names[0], 'B0-build')
        self.assertEqual(names[1:5], list(CARD_M))
        self.assertEqual(names[5], 'X0-status-rescan-reset')
        self.assertEqual(pack_job('X0-status-rescan-reset')['actions'], 'status rescan reset')
        self.assertEqual(names[-1], 'Z-reset')
        quad = [name for name in names[5:] if pack_job(name)['cards'] == 'quad']
        self.assertEqual(quad, names[5:])

    def test_the_card_m_jobs_run_card_m_harnesses_that_turn_the_serving_hook_off_and_unset_the_mesh_descriptor(self):
        for name, (_actions, _mode, harness) in CARD_M.items():
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertEqual(result['cardm_harness'], harness)
                self.assertIn('IMAGE_TAG=' + IMAGE, result['cardm_env'])
                text = (ROOT / harness).read_text(encoding='utf-8')
                self.assertIn('-e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3', text)
                self.assertIn('qual_card_select', text)

    def test_the_v5_byte_gate_jobs_are_the_v5split_packs_with_this_windows_image(self):
        v5 = HERE / 'references' / 'tp4-v5split-jobs'
        for mine, theirs in (('V1a-cardm-watcher', 'V1a-cardm-watcher'), ('V1b-cardm-full', 'V1b-cardm-full')):
            ours = job.parse_env(pack_text(mine))
            original = job.parse_env((v5 / (theirs + '.env')).read_text(encoding='utf-8'))
            for key in ('C2_ACTIONS', 'C2_CARDS', 'C2_CARDM_HARNESS', 'C2_CARDM_ARGS'):
                self.assertEqual(ours.get(key), original.get(key), (mine, key))
            self.assertEqual(ours['C2_CARDM_ENV'], original['C2_CARDM_ENV'].replace('tp4-v5split-1', IMAGE))

    def test_no_job_touches_the_agent_or_production(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                result = pack_job(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))

    def test_every_timed_and_attach_smoke_runs_concurrent8_steady(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            if not profile or 'smoke' not in actions:
                continue
            with self.subTest(name=name):
                self.assertIn('concurrent8_steady', tests_of(name))

    def test_the_control_and_the_audited_attaches_run_the_same_tests_and_the_32k_and_equal_hashes_are_recorded(self):
        self.assertEqual(tests_of('S0c-control-attach-smoke'), tests_of('A1-audited-attach-smoke'))
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_equal'} <= set(tests_of('A1-audited-attach-smoke')))

    def test_five_hang_shape_runs_on_the_audits_off_stack_carry_the_eight_seat_shapes(self):
        names = [name for name in EXPECTED if name.startswith('H')]
        self.assertEqual(len(names), 5)
        self.assertEqual(len({pack_job(name)['tests'] for name in names}), 1)
        for shape in ('concurrent8_steady', 'steady_resend', 'replay_concurrent8', 'replay_concurrent4', 'concurrent8_code_equal', 'concurrent8_drain'):
            self.assertIn(shape, tests_of(names[0]))
        self.assertEqual({pack_job(name)['profile'] for name in names}, {W2})

    def test_the_timing_jobs_alternate_abab_then_wave_1_alone_on_the_same_tests_at_32k_and_128k(self):
        timed = [name for name in EXPECTED if name.startswith('T')]
        self.assertEqual([pack_job(name)['profile'] for name in timed], [CONTROL, W2, CONTROL, W2, CONTROL, W2, W1])
        self.assertEqual(len({pack_job(name)['tests'] for name in timed}), 1)
        self.assertTrue({'warmup', 'coding', 'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_128k'} <= set(tests_of(timed[0])))

    def test_the_device_profile_copies_the_w1_packs_conventions_on_the_wave_2_profile(self):
        mine = job.parse_env(pack_text('P1-w2-8-user-profile'))
        theirs = job.parse_env((HERE / 'references' / 'tp4-w1-jobs' / 'P1-w1-8-user-profile.env').read_text(encoding='utf-8'))
        for key in ('C2_CARDS', 'C2_ACTIONS', 'C2_GATE_PLAN', 'C2_GATE_JIT'):
            self.assertEqual(mine[key], theirs[key], key)
        self.assertEqual(mine['C2_PROFILE'], W2)

    def test_the_dependencies_and_the_read_rules(self):
        order = order_text()
        for line in ('# NEEDS F1 V1a V1b MR X0 <- B0', '# NEEDS V1b <- V1a', '# NEEDS S0c <- X0', '# NEEDS A1 <- S0c', '# NEEDS H1 H2 H3 H4 H5 <- A1',
                     '# NEEDS T1 T2 T3 T4 T5 T6 T7 <- A1 H1 H2 H3 H4 H5', '# NEEDS P1 <- A1 H1'):
            self.assertIn(line, order)
        for text in ('about 159 ms', 'PAIRED', 'ZERO audit mismatches', 'five consecutive', 'KILL LINE',
                     'U2', 'NO HARNESS EXISTS', 'GDN_CG_SPREAD verdict=PASS', 'INCONCLUSIVE-SINGLE-CHIP', 'conv gates spread engaged', 'tp4 sdpa multi call', 'T7'):
            self.assertIn(text, order)

    def test_every_job_named_by_a_needs_line_exists(self):
        names = {line[0].split('-')[0] for line in order_lines()}
        for line in order_text().splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for group in match.groups():
                    for token in group.split():
                        self.assertIn(token, names, line)

    def test_no_hostname_address_registry_or_digest_and_lf_endings(self):
        for path in FOLDER.iterdir():
            text = path.read_text(encoding='utf-8')
            self.assertIsNone(BANNED.search(text), path.name)
            self.assertNotIn('\r', text, path.name)


class ShippingTests(unittest.TestCase):
    RUNTIME = spread.RUNTIME_FILES + ('sdpa_long_tp.py', 'sdpa_multi_tp.py', 'sdpa_multi_mask_tp.cpp', 'sdpa_multi_gather_tp.cpp', 'sdpa_multi_audit_tp.cpp',
                                      'gdn_block_conv_tp.py', 'tp4_vglue.py', 'packed_verifier.py', 'c2_smoke_check.py')

    def test_every_new_served_file_is_in_both_image_copy_lists_and_the_overlay(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml').read_text(encoding='utf-8')
        dockerfile = (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile').read_text(encoding='utf-8')
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
        for name in spread.RUNTIME_FILES + ('sdpa_multi_tp.py', 'sdpa_multi_mask_tp.cpp', 'sdpa_multi_gather_tp.cpp', 'sdpa_multi_audit_tp.cpp'):
            with self.subTest(name=name):
                self.assertIn(name, workflow)
                self.assertIn(' scripts/ci/' + name, dockerfile)
                self.assertIn('scripts/ci/' + name, overlay)

    def test_the_suites_run_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for suite in ('test_tp4_w2', 'test_gdn_conv_gates_spread', 'test_sdpa_multi_tp', 'test_tp4_sdpa_multi_window', 'test_tp4_w1'):
            self.assertRegex(workflow, r'python -B -m unittest [^\n]*\b%s\b' % suite)
        self.assertIn("discover -s optimisation/ttnn-op/mr_probe -p 'test_*.py'", workflow)
        self.assertIn("discover -s optimisation/ttnn-op/gdn_conv_gates_spread -p 'test_*.py'", workflow)

    def test_the_mr_probe_harness_is_on_the_qual_card_lists(self):
        text = (HERE / 'test_qual_card.py').read_text(encoding='utf-8')
        self.assertEqual(text.count("OPS / 'mr_probe' / 'run_card_m.sh'"), 2)
        self.assertEqual(text.count("OPS / 'gdn_conv_gates_spread' / 'run_card_m.sh'"), 2)


if __name__ == '__main__':
    unittest.main()
