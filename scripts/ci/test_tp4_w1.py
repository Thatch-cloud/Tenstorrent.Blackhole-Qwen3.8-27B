"""tp4/w1: the fusion audit's wave 1, the built levers gated together on the eight-seat 262k stack (every flag default off).

W1 is not a new lever: it is tp4/262k8-x (nosamp, the drafter ops D2a and D2c with the warm-pass fix) with tp4/hostgap (the host-gap
stage 1, the epochs arm) and tp4/u1 (the unit-major reduce-scatter) merged in, and two gate-only profiles that switch them on together:

  c2-packed-tp4-8x262k-w1        the timed control (c2-packed-tp4-8x262k-best-time-gate) MINUS QWEN_FAST_PACKED_SAMPLER_IN_TRACE plus the
                                 D2 flags, the five host-gap flags of hostgap-2 and QWEN_FAST_TP4_RS_UNIT_MAJOR
  c2-packed-tp4-8x262k-w1-audit  that plus the audit of every one of those levers and the fused-commit, verify-glue and draft-singles
                                 audits of the best-audit profile

D1 (QWEN_FAST_CCL_TOPOLOGY=ring) is in: only the drafter-side collectives read it and the engine's fabric stays FABRIC_1D as in the control
(the older ring arms that also set FABRIC_1D_RING are other profiles). S1 is left out (no measured gain). Three fallback twins exist:
-w1-lite (no block epochs), -w1-nod1 and -w1-audit-nod1 (no D1). This module pins the two profiles' exact deltas, that the levers' own
flag readers and smoke rules accept the stack together (the union of their engaged and audit rules), the job pack, and the shipping lists."""

import json
from pathlib import Path
import re
import unittest

import c2_serving_job as job
import c2_smoke_check
import mesh_link_policy
import tile_collective_tp
import tp4_sampdraft
import verify_prestage as vp
import test_tp4_hostgap as hostgap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
FOLDER = HERE / 'references' / 'tp4-w1-jobs'
IMAGE = 'tp4-w1-1'
CONTROL = 'c2-packed-tp4-8x262k-best-time-gate'
U1_ALONE = 'c2-packed-tp4-8x262k-best-u1-audit'
BEST_AUDIT = 'c2-packed-tp4-8x262k-best-audit'
W1, W1_AUDIT = 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit'
W1_LITE, W1_NOD1, W1_AUDIT_NOD1 = W1 + '-lite', W1 + '-nod1', W1_AUDIT + '-nod1'
SAMPLER = 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')

D2 = {tp4_sampdraft.DRAFT_CONV: '1', tp4_sampdraft.DRAFT_HEADS: '1'}
HOSTGAP = {'QWEN_FAST_TP4_HOSTGAP_LOG': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_WINDOW_VALIDATE': '1',
           'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1', 'QWEN_FAST_TP4_ENTRY_DIET': '1'}
U1 = {tile_collective_tp.UNIT_MAJOR_FLAG: '1'}
D1 = {mesh_link_policy.TOPOLOGY_SWITCH: 'ring'}
STACK = dict(D2, **HOSTGAP, **U1, **D1)
AUDITS = {tp4_sampdraft.DRAFT_CONV_AUDIT: '1', tp4_sampdraft.DRAFT_HEADS_AUDIT: '1', vp.FULL_AUDIT_FLAG: '1',
          tile_collective_tp.UNIT_MAJOR_AUDIT_FLAG: '1', 'QWEN_FAST_FUSED_COMMIT_AUDIT': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1',
          'QWEN_FAST_DRAFT_SINGLES_AUDIT': 'all'}
# The lever deliberately not in the stack, by flag.
LEFT_OUT = (tp4_sampdraft.SHARD_ARGMAX, tp4_sampdraft.SHARD_ARGMAX_AUDIT)


def load():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def flat(profile):
    profile = json.loads(json.dumps(profile))
    profile.pop('description')
    return profile


class ProfileTests(unittest.TestCase):
    def test_the_timed_arm_is_the_control_minus_the_in_trace_sampler_plus_the_stack_and_nothing_else(self):
        found = load()['profiles']
        expected = flat(found[CONTROL])
        del expected['env'][SAMPLER]
        expected['env'].update(STACK)
        self.assertEqual(flat(found[W1]), expected)
        self.assertNotIn(SAMPLER, found[W1]['env'])
        self.assertEqual(found[CONTROL]['env'][SAMPLER], '1')

    def test_the_audited_arm_is_the_timed_arm_plus_every_levers_audit_and_nothing_else(self):
        found = load()['profiles']
        expected = flat(found[W1])
        expected['env'].update(AUDITS)
        self.assertEqual(flat(found[W1_AUDIT]), expected)

    def test_the_audited_arms_audits_include_the_best_audit_profiles_lever_audits(self):
        found = load()['profiles']
        carried = {key: value for key, value in found[BEST_AUDIT]['env'].items() if key not in found[CONTROL]['env']
                   or found[CONTROL]['env'][key] != value}
        # the verify T1 and T2 audits are the best-audit's own and stay 0 here: the shard audit would bring the reference sampler back
        # into the verify trace, which the timed arm does not carry
        shared = {key: value for key, value in carried.items() if key not in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT')}
        for key, value in shared.items():
            self.assertEqual(found[W1_AUDIT]['env'][key], value, key)
        for name in (W1, W1_AUDIT):
            self.assertEqual((found[name]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], found[name]['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))

    def test_both_are_gate_only_unqualified_and_the_control_and_the_default_are_untouched(self):
        data = load()
        self.assertEqual(data['default'], 'c2-packed-tp4')
        for name in (W1, W1_AUDIT):
            with self.subTest(name=name):
                self.assertTrue(data['profiles'][name]['gate_only'])
                self.assertEqual(data['profiles'][name]['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
        control = data['profiles'][CONTROL]['env']
        self.assertFalse((set(STACK) | set(AUDITS)) & set(control))

    def test_the_fallback_twins_are_the_timed_and_audited_arms_minus_one_thing_each(self):
        found = load()['profiles']
        for name, base, dropped in ((W1_LITE, W1, 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS'), (W1_NOD1, W1, mesh_link_policy.TOPOLOGY_SWITCH),
                                    (W1_AUDIT_NOD1, W1_AUDIT, mesh_link_policy.TOPOLOGY_SWITCH)):
            with self.subTest(name=name):
                expected = flat(found[base])
                del expected['env'][dropped]
                self.assertEqual(flat(found[name]), expected)
                self.assertTrue(found[name]['gate_only'])
                self.assertNotEqual(found[name]['description'], found[base]['description'])

    def test_s1_is_in_no_w1_profile_d1_is_ring_in_the_stack_arms_and_the_fabric_is_the_controls(self):
        found = load()['profiles']
        self.assertEqual({found[name]['env'].get(mesh_link_policy.TOPOLOGY_SWITCH) for name in (W1, W1_AUDIT, W1_LITE)}, {'ring'})
        self.assertNotIn(mesh_link_policy.TOPOLOGY_SWITCH, found[CONTROL]['env'])
        for name in (W1, W1_AUDIT, W1_LITE, W1_NOD1, W1_AUDIT_NOD1):
            with self.subTest(name=name):
                for flag in LEFT_OUT:
                    self.assertNotIn(flag, found[name]['env'])
                self.assertEqual(found[name]['mesh_graph_descriptor'], found[CONTROL]['mesh_graph_descriptor'])
                self.assertEqual(found[name]['engine'], found[CONTROL]['engine'])
                self.assertNotEqual(found[name]['engine']['additional-config']['tt'].get('fabric_config'), 'FABRIC_1D_RING')

    def test_no_traffic_profile_carries_a_stack_flag_and_only_the_two_w1_arms_carry_all_of_them(self):
        for name, profile in load()['profiles'].items():
            env = profile.get('env', {})
            if not profile.get('gate_only') and name != 'c2-packed-tp4-8x262k-ship-prefix':     # ship/262k-prefix: the one traffic profile that carries the stack (test_ship_262k_prefix)
                with self.subTest(name=name):
                    self.assertFalse((set(STACK) | set(AUDITS)) & set(env))
            if set(STACK) <= set(env) and name.startswith(W1):
                self.assertIn(name, (W1, W1_AUDIT))

    def test_the_flag_names_the_profiles_use_are_the_ones_the_runtime_reads(self):
        for name in HOSTGAP:
            self.assertIn(name, vp.HOSTGAP_FLAGS)
        self.assertIn(vp.FULL_AUDIT_FLAG, vp.HOSTGAP_FLAGS)
        for name in D2:
            self.assertIn(name, tp4_sampdraft.LEVERS)
        for name in (tp4_sampdraft.DRAFT_CONV_AUDIT, tp4_sampdraft.DRAFT_HEADS_AUDIT):
            self.assertIn(name, tp4_sampdraft.AUDITS)
        self.assertEqual(U1, {'QWEN_FAST_TP4_RS_UNIT_MAJOR': '1'})


class CompositionTests(unittest.TestCase):
    """What the levers' own readers say about the stack's environment: each accepts it, none reads a flag of another, and the
    interaction rules written in the profiles' descriptions hold in code."""

    def env(self, name):
        # QWEN_FAST_PRESTAGE=1 is the image's (docker/qwen-c2-serving.Dockerfile), not the profile's; the host-gap flags need it
        return dict(load()['profiles'][name]['env'], QWEN_FAST_PRESTAGE='1')

    def test_every_levers_strict_reader_accepts_the_audited_arms_environment(self):
        env = self.env(W1_AUDIT)
        tp4_sampdraft.validate(env)
        self.assertEqual(tile_collective_tp.unit_major_settings(env), (True, tile_collective_tp.DEFAULT_AUDIT_CALLS))
        for name in vp.HOSTGAP_FLAGS:
            vp._flag(name, env)
        self.assertTrue(vp.two_block_enabled(env) and vp.block_epochs_enabled(env) and vp.window_validate_enabled(env))
        self.assertTrue(vp.entry_diet_enabled(env) and vp.full_audit_enabled(env))
        self.assertTrue(tp4_sampdraft.drafter_audit_on(env))

    def test_the_timed_arm_audits_nothing(self):
        env = self.env(W1)
        self.assertEqual(tile_collective_tp.unit_major_settings(env), (True, 0))
        self.assertFalse(vp.full_audit_enabled(env))
        self.assertFalse(tp4_sampdraft.drafter_audit_on(env))
        control = self.env(CONTROL)
        self.assertEqual({key: env[key] for key in env if key.endswith('_AUDIT')}, {key: control[key] for key in control if key.endswith('_AUDIT')})

    def test_the_drafter_gathers_run_ring_on_the_stack_arms_and_linear_on_the_control_and_the_nod1_twins(self):
        # D1 is in: QWEN_FAST_CCL_TOPOLOGY=ring is read only by the drafter-side collectives (fast_ccl_topology); the engine fabric is the control's
        class Operations:
            class Topology:
                Linear, Ring = 'linear', 'ring'
        for name, wanted in ((CONTROL, 'linear'), (W1, 'ring'), (W1_AUDIT, 'ring'), (W1_LITE, 'ring'), (W1_NOD1, 'linear'), (W1_AUDIT_NOD1, 'linear')):
            self.assertEqual(mesh_link_policy.fast_ccl_topology(Operations, self.env(name)), wanted, name)

    def test_only_the_drafter_side_modules_read_the_topology_flag(self):
        readers = []
        for path in sorted(HERE.glob('*.py')):
            text_found = path.read_text(encoding='utf-8')
            if (not path.name.startswith('test_') and path.name != 'mesh_link_policy.py'
                    and ('fast_ccl_topology' in text_found or mesh_link_policy.TOPOLOGY_SWITCH in text_found)):
                readers.append(path.name)
        self.assertEqual(readers, ['dflash_device.py', 'feature_collective_tp.py', 'quad_draft.py'])

    def test_without_the_in_trace_sampler_the_request_width_warm_the_control_carries_is_still_on(self):
        for name in (CONTROL, W1, W1_AUDIT):
            self.assertEqual(self.env(name)['QWEN_FAST_M3_REQUEST_WARM'], '1', name)

    def test_the_audited_arm_keeps_the_shard_audit_off_so_nosamp_is_what_the_attach_audits(self):
        import packed_verifier

        for name in (W1, W1_AUDIT):
            env = self.env(name)
            self.assertFalse(packed_verifier.sampler_arm_enabled(SAMPLER, env), name)
        # the in-trace sampler is also disengaged under the shard audit, so an arm with the verify T1 audit on could not exercise nosamp
        self.assertEqual(self.env(W1_AUDIT)['QWEN_FAST_VERIFY_T1_AUDIT'], '0')


ENGAGED_U1 = '[PINDIAG] tp4 u1 engaged rows=64 calls=128 unit_major=128 fallbacks=0 audited=0'
AUDIT_U1 = '[PINDIAG] tp4 u1 audit shape=64x5120 owner=%s round=1 calls=32 chips=4 elements=1 exact=True'
D2_LINES = ['%s' % tp4_sampdraft.CONV_ENGAGED, tp4_sampdraft.HEADS_ENGAGED,
            tp4_sampdraft.CONV_AUDIT + ' calls=3 pairs=40 exact=True', tp4_sampdraft.HEADS_AUDIT + ' calls=3 pairs=9 exact=True']


def stack_log(*, hostgap=True, u1=True, d2=True, u1_owners=('capture3', 'capture4')):
    lines = []
    if hostgap:
        lines += hostgap_lines()
    if u1:
        lines += [ENGAGED_U1] + [AUDIT_U1 % owner for owner in u1_owners]
    if d2:
        lines += D2_LINES
    return '\n'.join(lines)


def hostgap_lines():
    return (hostgap.engaged() + hostgap.prestage_lines(40) + hostgap.verify_lines(40) + hostgap.audit_lines()
            + ['[PACKED-PRESTAGE-SHADOW] round=1 retained_bindings=ok'])


class SmokeUnionTests(unittest.TestCase):
    """The smoke rules of the three branches judge the stack together: each reads its own flags from the profile's env, a clean log
    of all three passes every rule, and the loss of any one lever's lines is caught by that lever's rule."""

    def env(self, name=W1_AUDIT):
        env = dict(load()['profiles'][name]['env'])
        env.setdefault('QWEN_FAST_TP', '4')
        return env

    def judge(self, env, log):
        problems = list(c2_smoke_check.hostgap_problems(env, log, True)[0])
        problems += c2_smoke_check.u1_problems(env, log)
        problems += c2_smoke_check.sampdraft_problems(log, env)
        return problems

    def test_a_clean_log_of_all_three_levers_passes_the_audited_arm(self):
        self.assertEqual(self.judge(self.env(), stack_log()), [])

    def test_each_lever_missing_from_the_log_is_caught_by_its_own_rule(self):
        env = self.env()
        for what, log in (('hostgap', stack_log(hostgap=False)), ('u1', stack_log(u1=False)), ('d2', stack_log(d2=False))):
            with self.subTest(missing=what):
                self.assertTrue(self.judge(env, log), what)

    def test_the_u1_audit_needs_a_replay_line_for_each_of_the_two_blocks(self):
        found = self.judge(self.env(), stack_log(u1_owners=('capture3',)))
        self.assertTrue(any('block owner' in text for text in found), found)

    def test_the_timed_arm_is_judged_on_the_engaged_lines_alone(self):
        env = self.env(W1)
        timed = '\n'.join(hostgap.engaged() + hostgap.prestage_lines(40) + hostgap.verify_lines(40) + [ENGAGED_U1] + D2_LINES[:2])
        self.assertEqual(self.judge(env, timed), [])
        self.assertTrue(self.judge(env, '\n'.join(hostgap.engaged() + hostgap.prestage_lines(40) + hostgap.verify_lines(40) + D2_LINES[:2])))

    def test_the_control_logs_none_of_the_lever_lines(self):
        env = self.env(CONTROL)
        self.assertEqual(self.judge(env, 'nothing'), [])
        self.assertTrue(self.judge(env, stack_log()))


EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'), 'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'S0c-control-attach-smoke': ('reset smoke', CONTROL, 'stop'),
    'U1a-u1-alone-audited-attach': ('reset smoke', U1_ALONE, 'soft'),
    'A1-audited-attach-smoke': ('reset smoke', W1_AUDIT, 'stop'),
    'H1-hang-shapes-w1': ('reset smoke', W1, 'stop'), 'H2-hang-shapes-w1': ('reset smoke', W1, 'stop'),
    'H3-hang-shapes-w1': ('reset smoke', W1, 'stop'), 'H4-hang-shapes-w1': ('reset smoke', W1, 'stop'),
    'H5-hang-shapes-w1': ('reset smoke', W1, 'stop'), 'H6-stall8-cold262k-w1': ('reset smoke', W1, 'soft'),
    'T1-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T2-timed-B-w1': ('reset smoke', W1, 'soft'),
    'T3-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T4-timed-B-w1': ('reset smoke', W1, 'soft'),
    'T5-timed-A-control': ('reset smoke', CONTROL, 'soft'), 'T6-timed-B-w1': ('reset smoke', W1, 'soft'),
    'T7-timed-C-w1-lite': ('reset smoke', W1_LITE, 'soft'),
    'P1-w1-8-user-profile': ('status reset gate', W1, 'soft'), 'P1c-control-8-user-profile': ('status reset gate', CONTROL, 'soft'),
    'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def pack_text(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def pack_job(name):
    return job.read_job(job.parse_env(pack_text(name)), sorted(load()['profiles']), root=ROOT)


def order_lines():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]


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
                self.assertEqual((result['actions'], result['cards'], result['tag']), (actions, 'quad', IMAGE))
                if profile:
                    self.assertEqual(result['profile'], profile)

    def test_the_first_quad_job_is_status_rescan_reset_and_no_job_touches_the_agent_or_production(self):
        self.assertEqual(pack_job('X0-status-rescan-reset')['actions'], 'status rescan reset')
        self.assertEqual(order_lines()[0][0], 'X0-status-rescan-reset')
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
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

    def test_the_control_and_the_audited_attaches_run_the_same_tests_and_both_32k_and_equal_hashes_are_recorded(self):
        self.assertEqual(tests_of('S0c-control-attach-smoke'), tests_of('A1-audited-attach-smoke'))
        self.assertEqual(tests_of('S0c-control-attach-smoke'), tests_of('U1a-u1-alone-audited-attach'))
        self.assertTrue({'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_equal'} <= set(tests_of('A1-audited-attach-smoke')))

    def test_five_hang_shape_runs_on_the_audits_off_stack_carry_the_eight_seat_shapes(self):
        names = [name for name in EXPECTED if name.startswith('H') and not name.startswith('H6')]
        self.assertEqual(len(names), 5)
        self.assertEqual(len({pack_job(name)['tests'] for name in names}), 1)
        for shape in ('concurrent8_steady', 'steady_resend', 'replay_concurrent8', 'replay_concurrent4', 'concurrent8_code_equal',
                      'concurrent8_drain'):
            self.assertIn(shape, tests_of(names[0]))

    def test_the_timing_jobs_alternate_abab_on_the_same_tests_at_32k_and_128k(self):
        timed = [name for name in EXPECTED if name.startswith('T')]
        self.assertEqual([pack_job(name)['profile'] for name in timed], [CONTROL, W1, CONTROL, W1, CONTROL, W1, W1_LITE])
        self.assertEqual(len({pack_job(name)['tests'] for name in timed}), 1)
        self.assertTrue({'warmup', 'coding', 'concurrent8_steady', 'concurrent8_code_32k', 'concurrent8_code_128k'} <= set(tests_of(timed[0])))

    def test_the_stall_shape_runs_the_steady_mix_first_and_the_control_device_profile_is_the_stacks_twin(self):
        self.assertEqual(tests_of('H6-stall8-cold262k-w1'), ['warmup', 'concurrent8_steady', 'stall8_cold262k'])
        mine, control = job.parse_env(pack_text('P1-w1-8-user-profile')), job.parse_env(pack_text('P1c-control-8-user-profile'))
        self.assertEqual(control['C2_PROFILE'], CONTROL)
        for key in ('C2_CARDS', 'C2_ACTIONS', 'C2_GATE_PLAN', 'C2_GATE_JIT', 'C2_IMAGE_TAG'):
            self.assertEqual(mine[key], control[key], key)

    def test_the_read_rules_carry_the_host_gap_precondition_the_lite_rule_the_kill_line_and_the_nosamp_read(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for text_found in ('P0. HOST-GAP PRECONDITION', 'HOST-GAP LITE RULE', 'KILL LINE', 'packed sampler arm', 'P1 against P1c', 'ABABAB'):
            self.assertIn(text_found, order)

    def test_the_device_profile_copies_the_best_packs_conventions_on_the_stack_profile(self):
        best = job.parse_env((HERE / 'references' / 'tp4-262k8-best-jobs' / 'P1-best-8-user-profile.env').read_text(encoding='utf-8'))
        mine = job.parse_env(pack_text('P1-w1-8-user-profile'))
        for key in ('C2_CARDS', 'C2_ACTIONS', 'C2_GATE_PLAN', 'C2_GATE_JIT'):
            self.assertEqual(mine[key], best[key], key)
        self.assertEqual(mine['C2_PROFILE'], W1)

    def test_the_dependencies_and_the_read_rules(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for line in ('# NEEDS A1 <- S0c', '# NEEDS H1 H2 H3 H4 H5 H6 <- A1', '# NEEDS T1 T2 T3 T4 T5 T6 T7 <- A1 H1 H2 H3 H4 H5', '# NEEDS P1 P1c <- A1 H1'):
            self.assertIn(line, order)
        for text_found in ('five consecutive completions', 'PAIRED', 'ZERO audit mismatches', '182 ms against 222 ms', 'P1', 'S1', 'D1'):
            self.assertIn(text_found, order)

    def test_no_hostname_address_registry_or_digest_and_lf_endings(self):
        for path in FOLDER.iterdir():
            text_found = path.read_text(encoding='utf-8')
            self.assertIsNone(BANNED.search(text_found), path.name)
            self.assertNotIn('\r', text_found, path.name)


class ShippingTests(unittest.TestCase):
    RUNTIME = ('verify_prestage.py', 'packed_verifier.py', 'serving_packed_step.py', 'serving_packed_bridge.py', 'serving_page_binding.py',
               'quad_draft_tp.py', 'dflash_packed_proposal_coordinator.py', 'dflash_proposal_trace.py', 'tile_collective_tp.py',
               'tp4_draft_conv.py', 'tp4_draft_heads.py', 'tp4_sampdraft.py')

    def test_every_module_the_levers_touch_is_in_both_image_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker, context = dockerfile_modules(dockerfile_text()), context_modules()
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_the_overlay_manifest_names_them_all_so_the_image_runs_this_commits_copies(self):
        listed = {line.split()[0] for line in (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
                  if line.strip() and not line.startswith('#')}
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn('scripts/ci/' + name, listed)

    def test_the_suite_runs_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_tp4_w1\b')


if __name__ == '__main__':
    unittest.main()
