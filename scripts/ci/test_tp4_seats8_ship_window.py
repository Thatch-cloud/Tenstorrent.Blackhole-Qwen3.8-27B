"""The tp4-seats8 SHIP window: its job templates (scripts/ci/references/tp4-seats8-ship-jobs), their order, the GO rule for moving :latest, and the
build-time bake of the eight-seat serving default.

The window qualifies eight seats on two 64-row M3 blocks on ONE image, tp4-serve-8, built by SB8 from this branch with c2-packed-tp4-8 baked in as the
image's serving default (ENV QWEN_C2_PROFILE) and THATCH_SERVING_SESSION_CAP baked as its seat count: the node agent forwards neither variable, so the
image is what a platform launch serves and what a rollback moves. A build that does not name C2_BAKE_DEFAULT_PROFILE bakes nothing, and production's
four-seat default (qwen_c2_profiles.json: c2-packed-tp4) is untouched. The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_image_provenance as provenance  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-seats8-ship-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
DOCKERFILE = os.path.join(ROOT, 'docker', 'qwen-c2-serving.Dockerfile')
BUILD_SCRIPT = os.path.join(HERE, 'build-c2-serving-image.sh')
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
PLACEHOLDERS = {'@THIN_LAYER_IMAGE@': 'thin-layer-image-ref'}
IMAGE = 'tp4-serve-8'
PRODUCTION = 'c2-packed-tp4'
EIGHT, EIGHT_GATE, EIGHT_TIME, EIGHT_DIAG = ('c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate',
                                             'c2-packed-tp4-8-diag-strace')
FOUR_TIME = 'c2-packed-tp4-speed-strace'
HANG = tuple('S8-3%s-hang8-strace' % letter for letter in 'abcde')
ORDERED = ('SB8-build-base', 'A0-agent-stop', 'S8-0-flag-off-smoke', 'S8-1-seats8-attach-audited', 'S8-6-memory8') + HANG + (
    'S8-2a-matrix8-gate', 'S8-2b-matrix8-traffic', 'S8-4-staggered8', 'S8-5-churn16', 'S8-7c-seats4-timed', 'S8-7d-seats8-timed',
    'SR8-platform-replay', 'H8-handback-reset', 'A9-agent-start')
OPTIONAL = ('S8-7c-seats4-timed', 'S8-7d-seats8-timed')
PROFILE_OF = {'S8-0-flag-off-smoke': PRODUCTION, 'S8-1-seats8-attach-audited': EIGHT_GATE, 'S8-6-memory8': EIGHT,
              'S8-2a-matrix8-gate': EIGHT_GATE, 'S8-2b-matrix8-traffic': EIGHT, 'S8-4-staggered8': EIGHT_GATE,
              'S8-5-churn16': EIGHT_GATE, 'S8-7c-seats4-timed': FOUR_TIME, 'S8-7d-seats8-timed': EIGHT_TIME,
              **{name: EIGHT_DIAG for name in HANG}}
ACTIONS = {'SB8-build-base': 'status build', 'A0-agent-stop': 'agentstop unserve', 'S8-0-flag-off-smoke': 'status unserve reset smoke',
           'S8-1-seats8-attach-audited': 'reset smoke', 'S8-6-memory8': 'reset gate', 'S8-2a-matrix8-gate': 'reset gate',
           'S8-2b-matrix8-traffic': 'reset gate', 'S8-4-staggered8': 'reset gate', 'S8-5-churn16': 'reset gate',
           'S8-7c-seats4-timed': 'reset smoke', 'S8-7d-seats8-timed': 'reset smoke', 'SR8-platform-replay': 'reset replay',
           'H8-handback-reset': 'status reset', 'A9-agent-start': 'agentstart', **{name: 'reset smoke' for name in HANG}}
HANG_TESTS = ['warmup', 'concurrent4_steady', 'concurrent8_steady', 'steady_resend', 'replay_concurrent4', 'replay_concurrent8',
              'concurrent5_split', 'concurrent8_drain']
MATRIX_LENGTHS = '4096,8192,16384,24576,32768,49152,60000,120000'


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name, root=ROOT):
    text = text_of(name)
    for placeholder, value in PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return job.read_job(job.parse_env(text), sorted(profiles()['profiles']), root=root)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_in_the_ship_order(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(row[0] for row in rows), on_disk)
        self.assertEqual([row[0] for row in rows], list(ORDERED))

    def test_the_modes_the_one_image_and_the_minutes(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 5 <= int(minutes) <= 180, minutes)
                want = 'pre' if name == 'SB8-build-base' else 'optional' if name in OPTIONAL else 'stop'
                self.assertEqual(mode, want)

    def test_the_build_is_the_first_job_and_the_only_one(self):
        self.assertEqual(read_order()[0][0], 'SB8-build-base')
        for name in ORDERED:
            self.assertEqual('build' in parsed(name)['actions'].split(), name == 'SB8-build-base', name)

    def test_the_agent_stops_first_after_the_build_and_starts_last_after_the_hand_back(self):
        self.assertEqual([row[0] for row in read_order()[1:2]], ['A0-agent-stop'])
        self.assertEqual([row[0] for row in read_order()[-2:]], ['H8-handback-reset', 'A9-agent-start'])
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            self.assertEqual('agentstop' in actions, name == 'A0-agent-stop', name)
            self.assertEqual('agentstart' in actions, name == 'A9-agent-start', name)

    def test_the_hang_runs_are_the_five_in_a_row_the_cheapest_kill_signals_come_first_and_the_timing_follows_the_stop_jobs(self):
        names = [row[0] for row in read_order()]
        for earlier, later in (('S8-0-flag-off-smoke', 'S8-1-seats8-attach-audited'), ('S8-1-seats8-attach-audited', 'S8-6-memory8'),
                               ('S8-6-memory8', HANG[0]), (HANG[-1], 'S8-2a-matrix8-gate'), ('S8-5-churn16', 'S8-7c-seats4-timed'),
                               ('S8-7d-seats8-timed', 'SR8-platform-replay'), ('SR8-platform-replay', 'H8-handback-reset')):
            self.assertLess(names.index(earlier), names.index(later), (earlier, later))
        self.assertEqual(names[names.index(HANG[0]):names.index(HANG[0]) + 5], list(HANG))

    def test_the_go_rule_names_every_gate_and_every_stop_rule(self):
        text = order_text()
        for words in ('THE GO RULE FOR MOVING :latest', 'every stop job above passed', 'five CONSECUTIVE completions', 'ZERO audit mismatches',
                      'NO stall anywhere', 'hash equality', 'at least 3 GB per chip', 'strict', '0 aborts, 0 refusals',
                      '2 x P(4) + 10 ms', '0.85', 'explicit, recorded sign-off', 'PLATFORM_REPLAY passed=True',
                      'QWEN_C2_PROFILE=c2-packed-tp4-8', 'THATCH_SERVING_SESSION_CAP=8', 'the owner says ship', 'NO-GO',
                      "TODAY'S PRODUCTION", 'ROLLBACK', 'MANIFEST-LIST digest', 'A9'):
            self.assertIn(words, text)

    def test_the_hand_back_is_last_and_places_the_eight_seat_image_only_on_go(self):
        text = text_of('H8-handback-reset')
        self.assertEqual(parsed('H8-handback-reset')['actions'], 'status reset')
        for words in ('runs even when a stop job', 'the GO rule', 'NEVER a gate arm', 'fabric re-measure', 'A9', "TODAY'S PRODUCTION RECIPE",
                      'by an environment variable'):
            self.assertIn(words, text)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_is_lf_and_names_no_card_host_address_registry_or_digest(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_only_the_replay_names_a_placeholder_and_it_is_the_thin_layer_image(self):
        for name in ORDERED:
            found = set(re.findall(r'@[A-Z0-9_]+@', text_of(name)))
            self.assertEqual(found, {'@THIN_LAYER_IMAGE@'} if name == 'SR8-platform-replay' else set(), name)
        self.assertTrue(parsed('SR8-platform-replay')['platform_image'])

    def test_the_actions_are_the_plans(self):
        for name, actions in ACTIONS.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['actions'], actions)

    def test_every_card_job_resets_first_and_opens_the_cards_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(('smoke', 'gate', 'prefix', 'fabric', 'replay'))), 1)
                if set(actions) & set(('smoke', 'gate', 'replay')):
                    self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')

    def test_each_serving_job_names_its_profile_explicitly_and_every_eight_seat_arm_carries_the_request_warm(self):
        known = profiles()['profiles']
        for name, profile in PROFILE_OF.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['profile'], profile)
                env = known[profile]['env']
                if profile in (PRODUCTION, FOUR_TIME):
                    self.assertNotIn('QWEN_FAST_M3_BLOCKS', env)
                    self.assertNotIn('QWEN_FAST_M3_REQUEST_WARM', env)
                else:
                    self.assertEqual((env['QWEN_FAST_M3_BLOCKS'], env['QWEN_FAST_M3_REQUEST_WARM']), ('2', '1'))
                    self.assertEqual(known[profile]['engine']['max-num-seqs'], 8)

    def test_the_hang_runs_are_identical_and_on_the_diag_twin_with_the_eight_user_shapes(self):
        first = text_of(HANG[0])
        for name in HANG:
            self.assertEqual(parsed(name)['tests'].split(','), HANG_TESTS)
            self.assertEqual(parsed(name)['profile'], EIGHT_DIAG)
        body = [line for line in first.splitlines() if not line.startswith('#')]
        for name in HANG[1:]:
            self.assertEqual([line for line in text_of(name).splitlines() if not line.startswith('#')], body, name)

    def test_the_matrices_staggered_and_churn_are_the_qualification_shapes(self):
        for name, plan in (('S8-2a-matrix8-gate', 'matrix'), ('S8-2b-matrix8-traffic', 'matrix'), ('S8-4-staggered8', 'staggered'),
                           ('S8-5-churn16', 'churn'), ('S8-6-memory8', 'memory')):
            self.assertEqual(parsed(name)['gate_plan'], plan, name)
        for name in ('S8-2a-matrix8-gate', 'S8-2b-matrix8-traffic', 'S8-4-staggered8'):
            self.assertEqual(parsed(name)['gate_lengths'], MATRIX_LENGTHS, name)
        self.assertEqual(len(parsed('S8-5-churn16')['gate_lengths'].split(',')), 16)

    def test_the_timing_pair_is_four_seats_against_eight_on_the_same_tests(self):
        control, eight = parsed('S8-7c-seats4-timed'), parsed('S8-7d-seats8-timed')
        self.assertEqual(control['tests'], eight['tests'])
        self.assertEqual((control['profile'], eight['profile']), (FOUR_TIME, EIGHT_TIME))

    def test_the_replay_boots_the_baked_default_and_no_other_template_bakes(self):
        self.assertEqual(parsed('SR8-platform-replay')['replay_profile'], EIGHT)
        self.assertEqual(parsed('SB8-build-base')['bake_default_profile'], EIGHT)
        self.assertEqual(parsed('SB8-build-base')['profile'], EIGHT)
        for name in ORDERED[1:]:
            self.assertEqual(parsed(name)['bake_default_profile'], '', name)


class BakeTests(unittest.TestCase):
    def test_the_file_default_and_the_production_profile_are_untouched(self):
        known = profiles()
        self.assertEqual(known['default'], PRODUCTION)
        self.assertNotIn('gate_only', known['profiles'][PRODUCTION])
        self.assertNotIn('QWEN_FAST_M3_BLOCKS', known['profiles'][PRODUCTION]['env'])
        self.assertEqual(known['profiles'][PRODUCTION]['engine']['max-num-seqs'], 4)
        self.assertNotIn('gate_only', known['profiles'][EIGHT])

    def test_the_eight_seat_traffic_profile_differs_from_production_in_exactly_the_four_deltas(self):
        known = profiles()['profiles']
        four, eight = known[PRODUCTION], known[EIGHT]
        env = {key: value for key, value in eight['env'].items() if four['env'].get(key) != value}
        self.assertEqual(env, {'QWEN_FAST_M3_BLOCKS': '2', 'QWEN_FAST_M3_REQUEST_WARM': '1'})
        engine = {key for key in set(four['engine']) | set(eight['engine']) if four['engine'].get(key) != eight['engine'].get(key)}
        self.assertEqual(engine, {'max-num-seqs', 'num-gpu-blocks-override', 'additional-config'})
        self.assertEqual({key: four[key] for key in ('max_prompt_tokens', 'min_answer_tokens', 'default_max_tokens')},
                         {key: eight[key] for key in ('max_prompt_tokens', 'min_answer_tokens', 'default_max_tokens')})

    def test_the_job_key_is_refused_unless_it_is_a_four_card_serving_profile_built_in_a_build_job(self):
        known = sorted(profiles()['profiles'])
        base = {'C2_CARDS': 'quad', 'C2_ACTIONS': 'build', 'C2_IMAGE_TAG': IMAGE, 'C2_BAKE_DEFAULT_PROFILE': EIGHT}
        self.assertEqual(job.read_job(dict(base), known)['bake_default_profile'], EIGHT)
        self.assertEqual(job.read_job(dict(base, C2_BAKE_DEFAULT_PROFILE=''), known)['bake_default_profile'], '')
        for change, words in ((dict(C2_ACTIONS='status reset'), 'no build'), (dict(C2_BAKE_DEFAULT_PROFILE=EIGHT_GATE), 'not a four-card serving'),
                              (dict(C2_BAKE_DEFAULT_PROFILE='general'), 'not a four-card serving'),
                              (dict(C2_BAKE_DEFAULT_PROFILE='no-such-profile'), 'not a profile'),
                              (dict(C2_BAKE_DEFAULT_PROFILE='Bad Name'), 'must match')):
            with self.subTest(change=change):
                with self.assertRaises(job.JobError) as caught:
                    job.read_job(dict(base, **change), known)
                self.assertIn(words, str(caught.exception))
        with self.assertRaises(job.JobError):
            job.read_job(dict(base, C2_CARDS='pair'), known)

    def test_the_dockerfile_bakes_only_what_the_build_arg_gives_and_empty_otherwise(self):
        with open(DOCKERFILE, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('ARG C2_BAKE_PROFILE=\n', text)
        self.assertIn('ARG C2_BAKE_SESSION_CAP=\n', text)
        self.assertIn('ENV QWEN_C2_PROFILE=${C2_BAKE_PROFILE} THATCH_SERVING_SESSION_CAP=${C2_BAKE_SESSION_CAP}\n', text)
        self.assertEqual(text.count('QWEN_C2_PROFILE='), 1)
        self.assertLess(text.index('ENV QWEN_C2_PROFILE='), text.index('ARG SOURCE_REVISION'), 'baked before the provenance stamp')
        self.assertLess(text.index('QWEN_C2_SERVING=1'), text.index('ENV QWEN_C2_PROFILE='))

    def test_the_build_script_computes_the_cap_from_the_profile_and_reads_the_image_back(self):
        with open(BUILD_SCRIPT, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('C2_BAKE_DEFAULT_PROFILE:-', text)
        self.assertIn("entry['engine']['max-num-seqs']", text)
        self.assertIn("entry.get('gate_only') is True or entry.get('mesh_device') != 'P150x4'", text)
        build = text.index('docker build -f')
        self.assertIn('"${bake[@]}"', text[build:build + 400])
        self.assertIn('--build-arg "C2_BAKE_PROFILE="', text, 'the default build bakes nothing')
        self.assertLess(text.index("'^(QWEN_C2_PROFILE|THATCH_SERVING_SESSION_CAP)='"), text.index('docker tag "$built" "$image"'),
                        'the image is read back before it is tagged')

    def test_the_workflow_hands_the_job_key_to_the_build_script(self):
        with open(WORKFLOW, encoding='utf-8') as handle:
            text = handle.read()
        step = text[text.index('      - name: Build\n'):text.index('      - name: Prefix-reuse P0a probe')]
        self.assertIn('C2_BAKE_DEFAULT_PROFILE: ${{ steps.job.outputs.bake_default_profile }}', step)

    def test_provenance_holds_the_baked_pair_to_each_other(self):
        known = profiles()
        check = provenance.check_baked_serving
        self.assertEqual(check({}, known), [])
        self.assertEqual(check({'QWEN_C2_PROFILE': '', 'THATCH_SERVING_SESSION_CAP': ''}, known), [])
        self.assertEqual(check({'QWEN_C2_PROFILE': EIGHT, 'THATCH_SERVING_SESSION_CAP': '8'}, known), [])
        self.assertTrue(check({'QWEN_C2_PROFILE': EIGHT, 'THATCH_SERVING_SESSION_CAP': '4'}, known))
        self.assertTrue(check({'QWEN_C2_PROFILE': EIGHT, 'THATCH_SERVING_SESSION_CAP': ''}, known))
        self.assertTrue(check({'QWEN_C2_PROFILE': EIGHT_GATE, 'THATCH_SERVING_SESSION_CAP': '8'}, known))
        self.assertTrue(check({'QWEN_C2_PROFILE': '', 'THATCH_SERVING_SESSION_CAP': '8'}, known))
        self.assertEqual(check({'QWEN_C2_PROFILE': PRODUCTION, 'THATCH_SERVING_SESSION_CAP': '4'}, known), [])

    def test_the_contract_reads_an_empty_baked_profile_as_unset_and_the_baked_one_before_the_file_default(self):
        import serving_c2_contract as contract
        saved = os.environ.get('QWEN_C2_PROFILE')
        try:
            os.environ['QWEN_C2_PROFILE'] = ''
            self.assertEqual(contract.load_profile(PROFILES_PATH)['name'], PRODUCTION)
            os.environ['QWEN_C2_PROFILE'] = EIGHT
            self.assertEqual(contract.load_profile(PROFILES_PATH)['name'], EIGHT)
            self.assertEqual(contract.load_profile(PROFILES_PATH, PRODUCTION)['name'], PRODUCTION)
        finally:
            if saved is None:
                os.environ.pop('QWEN_C2_PROFILE', None)
            else:
                os.environ['QWEN_C2_PROFILE'] = saved


if __name__ == '__main__':
    unittest.main()
