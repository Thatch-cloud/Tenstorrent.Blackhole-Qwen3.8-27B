"""The four-card S2 SERVING window: its job templates (scripts/ci/references/tp4-s2-serve-jobs), their order and what each asks of
the tree (serving/tp4-s2).

Three stages: the pre-evidence gate jobs (image tp4-stackfix-3, profile c2-packed-tp4-gate), the evidence jobs on card M alone, and the
post-evidence jobs (image tp4-serve-1, the traffic profile c2-packed-tp4: build, smoke, the mixed gate again, the four-card replay).
The templates are public, so they name no rig, card, address, registry or digest; the values the driver fills in are @...@ placeholders."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-s2-serve-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
PLACEHOLDERS = {'@K64J_GRAFT_DIR@': '/graft', '@K64J_TTNNCPP_SHA256@': '0' * 64, '@SERVED_IMAGE@': 'served-image-ref',
                '@THIN_LAYER_IMAGE@': 'thin-layer-image-ref'}
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
PRE_IMAGE, POST_IMAGE = 'tp4-stackfix-3', 'tp4-serve-1'
GATES = ('G-L1-lifecycle', 'G-L2-lifecycle-arrival', 'G-M1-mixed', 'G-M2-mixed-long', 'G-S-short', 'G-B-boundaries', 'G-MEM-memory',
         'G-ST-staggered', 'G-C-churn')
RESET = ('EV-R0-quad-reset',)
EVIDENCE = ('EV-W1-cb1-cb2a-watcher', 'EV-F1-cb1', 'EV-F2-cb2a', 'EV-W2-cb2b-watcher', 'EV-F3-cb2b')
# The reader twin's bytes as tp4/stack-fix 8e0113e3 holds them: the image the pre-evidence gates serve (tp4-stackfix-3) runs these,
# CB2b qualifies the checkout's copy, and the image built after the evidence serves it; one set of bytes throughout.
READER_TP_STACK_FIX = '89761985a45b5fe4038f9ff556e497b7c9eab84808144c3cd62748db62b10152'
POST = ('SB-build', 'SS-smoke', 'SM-mixed-traffic', 'SR-quad-replay')


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        rows = [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]
    return rows


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name, fill=True):
    text = text_of(name)
    if fill:
        for placeholder, value in PLACEHOLDERS.items():
            text = text.replace(placeholder, value)
    values = job.parse_env(text)
    return values, job.read_job(values, NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))
        self.assertEqual(ordered, list(GATES + RESET + EVIDENCE + POST))

    def test_the_modes_minutes_and_images_are_the_ones_each_job_uses(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(mode, 'stop')
                self.assertTrue(minutes.isdigit() and 3 <= int(minutes) <= 60, minutes)
                values, outputs = parsed(name)
                self.assertEqual(outputs['tag'], image)
                self.assertEqual(image, POST_IMAGE if name in POST else PRE_IMAGE)

    def test_the_stages_run_gates_then_evidence_on_one_card_then_the_post_evidence_jobs(self):
        names = [row[0] for row in read_order()]
        self.assertLess(max(names.index(name) for name in GATES), min(names.index(name) for name in EVIDENCE))
        # the four-card jobs leave the ring fabric on the ethernet cores: card M alone opens only after an all-four reset (v140)
        self.assertEqual(names.index('EV-R0-quad-reset') + 1, min(names.index(name) for name in EVIDENCE))
        self.assertGreater(names.index('EV-R0-quad-reset'), max(names.index(name) for name in GATES))
        _, reset = parsed('EV-R0-quad-reset')
        self.assertEqual((reset['cards'], reset['actions']), ('quad', 'reset'))
        self.assertLess(max(names.index(name) for name in EVIDENCE), min(names.index(name) for name in POST))
        self.assertLess(names.index('EV-W1-cb1-cb2a-watcher'), names.index('EV-F1-cb1'))
        self.assertLess(names.index('EV-W1-cb1-cb2a-watcher'), names.index('EV-F2-cb2a'))
        self.assertLess(names.index('EV-W2-cb2b-watcher'), names.index('EV-F3-cb2b'))
        self.assertEqual(names[-4:], list(POST), 'build, smoke, the traffic gate, then the replay last')

    def test_the_total_is_the_plans_window_about_six_to_seven_hours_of_runner_time(self):
        total = sum(int(row[3]) for row in read_order())
        self.assertTrue(330 <= total <= 480, total)
        gates = sum(int(row[3]) for row in read_order() if row[0] in GATES)
        self.assertTrue(150 <= gates <= 220, 'the pre-evidence gates are about 2.6-3 h: %s min' % gates)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_with_and_without_the_placeholders_filled(self):
        for name, *_ in read_order():
            with self.subTest(template=name):
                parsed(name)
                values = job.parse_env(text_of(name))
                self.assertTrue(values['C2_IMAGE_TAG'])

    def test_the_templates_are_lf_and_name_no_card_host_address_registry_or_digest(self):
        for name, *_ in read_order():
            with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                self.assertNotIn(b'\r', handle.read(), name)
            self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_the_only_placeholders_are_the_ones_the_driver_fills_and_only_their_jobs_carry_them(self):
        wanted = {'EV-W1-cb1-cb2a-watcher': {'@K64J_GRAFT_DIR@', '@K64J_TTNNCPP_SHA256@'},
                  'EV-F1-cb1': {'@K64J_GRAFT_DIR@', '@K64J_TTNNCPP_SHA256@'},
                  'EV-F2-cb2a': {'@K64J_GRAFT_DIR@', '@K64J_TTNNCPP_SHA256@'},
                  'EV-W2-cb2b-watcher': {'@K64J_GRAFT_DIR@', '@K64J_TTNNCPP_SHA256@', '@SERVED_IMAGE@'},
                  'EV-F3-cb2b': {'@K64J_GRAFT_DIR@', '@K64J_TTNNCPP_SHA256@', '@SERVED_IMAGE@'},
                  'SR-quad-replay': {'@THIN_LAYER_IMAGE@'}}
        for name, *_ in read_order():
            body = '\n'.join(line for line in text_of(name).splitlines() if not line.startswith('#'))
            self.assertEqual(set(re.findall(r'@[A-Z0-9_]+@', body)), wanted.get(name, set()), name)

    def test_a_job_opens_the_cards_in_one_step_a_quad_job_resets_first_and_cardm_is_pair_only(self):
        for name, *_ in read_order():
            values, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                if outputs['cards'] == 'quad':
                    self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')
                    self.assertNotIn('cardm', actions)
                if 'cardm' in actions:
                    self.assertEqual((outputs['cards'], actions), ('pair', ['cardm']), 'cardm alone, no pair-only reset')

    def test_the_gate_jobs_serve_the_gate_profile_on_the_validated_image_and_record_kernel_cache_growth(self):
        for name in GATES:
            values, outputs = parsed(name)
            self.assertEqual((outputs['cards'], outputs['actions'], outputs['profile']), ('quad', 'reset gate', 'c2-packed-tp4-gate'))
            self.assertEqual(outputs['gate_jit'], 'record', 'no four-card warm plan exists: growth is recorded, not judged')
            self.assertEqual(outputs['gate_audits'], '', 'the default audits: the gate itself forces the rest on the plans that need them')
        self.assertIs(PROFILES['profiles']['c2-packed-tp4-gate']['gate_only'], True)
        self.assertNotIn('gate_only', PROFILES['profiles']['c2-packed-tp4'])

    def test_the_gate_plans_and_lengths_are_the_plans_and_fit_the_step(self):
        plans = {'G-L1-lifecycle': 'lifecycle', 'G-L2-lifecycle-arrival': 'lifecycle-arrival', 'G-M1-mixed': 'mixed',
                 'G-M2-mixed-long': 'mixed', 'G-S-short': 'short', 'G-B-boundaries': 'boundaries', 'G-MEM-memory': 'memory',
                 'G-ST-staggered': 'staggered', 'G-C-churn': 'churn', 'SM-mixed-traffic': 'mixed'}
        lengths = {'G-M1-mixed': '1536,20000,60000,120000', 'SM-mixed-traffic': '1536,20000,60000,120000',
                   'G-M2-mixed-long': '5000,40000,90000,123136'}
        tokens = {'G-B-boundaries': 8192, 'G-ST-staggered': 8192, 'G-M1-mixed': 4096, 'G-M2-mixed-long': 4096,
                  'SM-mixed-traffic': 4096}
        for name, plan in plans.items():
            values, outputs = parsed(name)
            self.assertEqual(outputs['gate_plan'], plan, name)
            self.assertEqual(outputs['gate_lengths'], lengths.get(name, ''), name)
            if name in tokens:
                self.assertEqual(int(outputs['gate_max_tokens']), tokens[name], name)
            given = [int(part) for part in outputs['gate_lengths'].split(',')] if outputs['gate_lengths'] else None
            arms = gate.plan_arms(plan, outputs['profile'], PROFILES, given, int(outputs['gate_max_tokens']),
                                  int(outputs['gate_memory_prompt']) if outputs['gate_memory_prompt'] else None, [], {})
            self.assertTrue(arms, name)
            # the workflow's gate step gives the plans 380 min less 10: every arm to its limit must fit once
            self.assertLessEqual(sum(arm[2] + gate.ARM_OVERHEAD_SECONDS for arm in arms), 380 * 60 - 600, name)
        self.assertEqual(parsed('G-MEM-memory')[1]['gate_memory_prompt'], '123136')
        top = max(int(part) for name in ('G-M2-mixed-long',) for part in parsed(name)[1]['gate_lengths'].split(','))
        self.assertEqual(top, PROFILES['profiles']['c2-packed-tp4']['max_prompt_tokens']
                         if 'max_prompt_tokens' in PROFILES['profiles']['c2-packed-tp4'] else 123136)

    def test_a_gate_only_profile_is_booted_with_the_gate_switch_and_the_traffic_profile_is_not(self):
        for name in GATES + ('SM-mixed-traffic',):
            values, outputs = parsed(name)
            lines = []
            code = gate.main(['--image', 'img', '--profile', outputs['profile'], '--plan', outputs['gate_plan'], '--cards', 'quad',
                              '--dry-run', '--results', os.path.join(HERE, 'no-results'), '--profiles',
                              os.path.join(HERE, 'qwen_c2_profiles.json'), '--max-tokens', outputs['gate_max_tokens'],
                              '--jit', outputs['gate_jit'] or 'auto']
                             + (['--lengths', outputs['gate_lengths']] if outputs['gate_lengths'] else [])
                             + (['--memory-prompt', outputs['gate_memory_prompt']] if outputs['gate_memory_prompt'] else []),
                             log=lines.append)
            self.assertEqual(code, 0, (name, lines))
            arms = [json.loads(line)['docker'] for line in lines[1:]]
            self.assertTrue(arms, name)
            for argv in arms:
                self.assertEqual('QWEN_C2_GATE=1' in argv, name != 'SM-mixed-traffic', name)

    def test_churn_is_judged_on_pairs_because_every_four_card_profile_switches_the_quad_draft_off(self):
        self.assertTrue(gate.quad_draft_off(PROFILES, 'c2-packed-tp4-gate'))
        self.assertTrue(gate.quad_draft_off(PROFILES, 'c2-packed-tp4'))
        self.assertFalse(gate.quad_draft_off(PROFILES, 'c2-packed'))

    def test_the_evidence_jobs_are_one_harness_on_card_m_under_the_k64j_graft_and_the_watcher_jobs_run_the_reduced_scope(self):
        seen = {}
        for name in EVIDENCE:
            values, outputs = parsed(name)
            env = dict(pair.split('=', 1) for pair in outputs['cardm_env'].split())
            self.assertEqual(outputs['cardm_harness'], 'optimisation/ttnn-op/k64j/run_card_b.sh', name)
            self.assertEqual(env['EXPECT_TTNNCPP_SHA256'], '0' * 64, 'pinned by the placeholder the driver fills with the K64j digest')
            self.assertIn('KOPGRAFT64', env)
            seen[name] = (env['K64J_HARNESS'], env.get('WATCHER') == '1', env.get('TP4_WIDTH'), 'IMAGE' in env,
                          outputs['cardm_args'])
        self.assertEqual({name: row[:4] for name, row in seen.items()},
                         {'EV-W1-cb1-cb2a-watcher': ('card', True, None, False), 'EV-F1-cb1': ('card', False, None, False),
                          'EV-F2-cb2a': ('card', False, None, False), 'EV-W2-cb2b-watcher': ('extent_reader', True, '4', True),
                          'EV-F3-cb2b': ('extent_reader', False, '4', True)})
        for name in ('EV-W1-cb1-cb2a-watcher', 'EV-F1-cb1', 'EV-F2-cb2a'):
            self.assertIn('--kv-heads 1', seen[name][4], name)
        self.assertIn('--seeds 0,1,2,3,4', seen['EV-F1-cb1'][4])
        self.assertIn('--sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky', seen['EV-F2-cb2a'][4])
        self.assertIn('--sections R1,S,R2,R4 --seeds 0,1,2 --variants normal,peaky', seen['EV-F3-cb2b'][4])
        self.assertIn('--seeds 0', seen['EV-W1-cb1-cb2a-watcher'][4])
        self.assertIn('--extents 2304,131328', seen['EV-W1-cb1-cb2a-watcher'][4])

    def test_the_full_evidence_jobs_ask_for_the_scopes_the_admission_needs(self):
        import packed_any_admission as admission
        f1, f2, f3 = (parsed(name)[1]['cardm_args'] for name in ('EV-F1-cb1', 'EV-F2-cb2a', 'EV-F3-cb2b'))
        self.assertEqual(set(admission.SEEDS), set(int(seed) for seed in re.search(r'--seeds (\S+)', f1).group(1).split(',')))
        self.assertEqual(set(admission.SEEDS), set(int(seed) for seed in re.search(r'--seeds (\S+)', f2).group(1).split(',')))
        self.assertEqual(set(admission.CB2B_SEEDS), set(int(seed) for seed in re.search(r'--seeds (\S+)', f3).group(1).split(',')))
        self.assertNotIn('--extents', f1 + f2 + f3, 'the full runs take the harness defaults: the six K1 extents and C = 131328')

    def test_the_post_evidence_jobs_serve_the_traffic_profile_on_the_new_image(self):
        for name in ('SS-smoke', 'SM-mixed-traffic'):
            values, outputs = parsed(name)
            self.assertEqual((outputs['cards'], outputs['profile'], outputs['tag']), ('quad', 'c2-packed-tp4', POST_IMAGE), name)
        _, build = parsed('SB-build')
        self.assertEqual((build['cards'], build['actions'], build['tag']), ('quad', 'status reset build push', POST_IMAGE))
        self.assertEqual(parsed('SM-mixed-traffic')[1]['gate_jit'], 'judge', 'the pre-evidence arms warmed the kernel cache')

    def test_the_smoke_names_tests_the_smoke_knows_and_includes_both_streamed_ones(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        tests = parsed('SS-smoke')[1]['tests'].split(',')
        for test in tests:
            self.assertIn("'%s'" % test, smoke, test)
        for wanted in ('warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_steady', 'steady_resend', 'tool_call',
                       'stream_tool_call', 'stream_reasoning', 'refused_n2', 'alive_after_refusal', 'stream_dropped',
                       'alive_after_drop'):
            self.assertIn(wanted, tests)
        self.assertNotIn('agreement', tests, 'the fast path has no device sampler and no log-probabilities to compare')

    def test_the_replay_job_replays_the_thin_layer_under_the_four_card_traffic_profile(self):
        values, outputs = parsed('SR-quad-replay')
        self.assertEqual((outputs['cards'], outputs['actions'], outputs['replay_profile']), ('quad', 'reset replay', 'c2-packed-tp4'))
        self.assertEqual(PROFILES['profiles'][outputs['replay_profile']]['mesh_device'], 'P150x4')
        self.assertEqual(outputs['replay_served_model'], '', 'the default Qwen/Qwen3.8-27B:tt applies')
        self.assertTrue(outputs['platform_image'])
        # a pair profile is refused under quad, and a quad replay without a profile is refused
        with self.assertRaises(job.JobError):
            job.read_job(job.parse_env(text_of('SR-quad-replay').replace('C2_REPLAY_PROFILE=c2-packed-tp4',
                                                                         'C2_REPLAY_PROFILE=c2-packed')
                                       .replace('@THIN_LAYER_IMAGE@', 'thin-layer-image-ref')), NAMES)
        with self.assertRaises(job.JobError):
            job.read_job(job.parse_env(text_of('SR-quad-replay').replace('C2_REPLAY_PROFILE=c2-packed-tp4\n', '')
                                       .replace('@THIN_LAYER_IMAGE@', 'thin-layer-image-ref')), NAMES)


class ReaderBytesTests(unittest.TestCase):
    def test_the_reader_twin_stays_at_the_bytes_the_gates_ran_and_the_record_names(self):
        import hashlib
        import packed_any_admission as admission
        with open(os.path.join(HERE, 'extent_attention_replay_tp.py'), 'rb') as handle:
            live = hashlib.sha256(handle.read().replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(live, READER_TP_STACK_FIX, 'extent_attention_replay_tp.py changed on the serving branch: the '
                         'pre-evidence gates on tp4-stackfix-3 no longer ran the served reader, and CB2b must be re-run on it')
        recorded = json.loads(admission.EVIDENCE_TP4.read_text(encoding='utf-8')).get('sources') or {}
        if 'extent_attention_replay_tp.py' in recorded:
            self.assertEqual(recorded['extent_attention_replay_tp.py'], live)


class ProfileTests(unittest.TestCase):
    """The traffic profile is what the image serves by default, and its gate twin differs from it only by the documented switches."""

    def test_the_image_default_is_the_four_card_traffic_profile_and_it_is_not_gate_only(self):
        self.assertEqual(PROFILES['default'], 'c2-packed-tp4')
        body = PROFILES['profiles']['c2-packed-tp4']
        self.assertNotIn('gate_only', body)
        self.assertNotIn('QWEN_C2_GATE_PROFILE', body['env'])
        self.assertEqual(body['mesh_device'], 'P150x4')

    def test_the_description_says_what_serves_and_no_longer_calls_it_unverified_or_unplaced(self):
        text = PROFILES['profiles']['c2-packed-tp4']['description']
        for stale in ('UNVERIFIED', 'NOT QUALIFIED', 'records no section', 'not placed until', 'slide OFF'):
            self.assertNotIn(stale, text)
        self.assertIn('TRAFFIC profile', text)

    def test_the_traffic_profile_is_the_gate_profile_less_the_gate_switches_and_the_audits(self):
        traffic, gated = PROFILES['profiles']['c2-packed-tp4'], PROFILES['profiles']['c2-packed-tp4-gate']
        extra = set(gated['env']) - set(traffic['env'])
        self.assertTrue('QWEN_C2_GATE_PROFILE' in extra)
        differing = {key for key in set(traffic['env']) & set(gated['env']) if traffic['env'][key] != gated['env'][key]}
        self.assertEqual(differing, set(), 'a flag both set must be set alike')
        self.assertEqual(traffic['engine'], gated['engine'])


if __name__ == '__main__':
    unittest.main()
