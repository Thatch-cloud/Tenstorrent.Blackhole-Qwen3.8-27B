"""The four-card exactness window: its job templates (scripts/ci/references/tp4-exact-jobs), their order, and what the image
must carry for the fix to be present.

The templates are public, so they name no rig, card, address, registry or digest. Every template parses with c2_serving_job, opens
the cards in ONE step, resets first, and the gate plans they name are ones the gate accepts on the profile they name. X2 is the S3a
job of the previous window with only the image changed."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
from test_tp2_pins import image_list_membership  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-exact-jobs')
PREVIOUS = os.path.join(HERE, 'references', 'tp4-s2-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [tuple(line.split()) for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name, folder=FOLDER):
    with open(os.path.join(folder, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.parse_env(text_of(name)), job.read_job(job.parse_env(text_of(name)), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_and_the_order_names_no_other(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, _ in read_order()]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))

    def test_the_stages_are_build_spike_matrix_timings_then_the_optional_arm(self):
        order = read_order()
        self.assertEqual([name.split('-')[0] for name, _ in order], ['X0', 'X1', 'X2', 'X3', 'X4'])
        self.assertEqual([mode for _, mode in order], ['stop', 'stop', 'stop', 'soft', 'optional'])

    def test_the_build_runs_no_card_and_no_device_step_precedes_it(self):
        _, outputs = parsed('X0-build')
        self.assertEqual(outputs['actions'].split(), ['status', 'reset', 'build'])
        self.assertEqual(outputs['cards'], 'quad')
        self.assertEqual(set(outputs['actions'].split()) & set(DEVICE_STEPS), set())


class TemplateTests(unittest.TestCase):
    def test_every_template_parses(self):
        for name, _ in read_order():
            with self.subTest(template=name):
                parsed(name)

    def test_the_templates_name_no_card_host_address_registry_digest_or_placeholder(self):
        for name, _ in read_order():
            self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_one_image_tag_for_the_whole_window_and_it_is_not_the_previous_windows(self):
        tags = set(parsed(name)[1]['tag'] for name, _ in read_order())
        self.assertEqual(tags, {'tp4-exact-1'})
        self.assertNotIn('tp4-exact-1', ''.join(text_of(name, PREVIOUS) for name in
                                               (n[:-4] for n in os.listdir(PREVIOUS) if n.endswith('.env'))))

    def test_all_jobs_are_four_card_and_a_device_job_resets_first_and_opens_the_cards_once(self):
        for name, _ in read_order():
            values, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(outputs['cards'], 'quad')
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0] if 'status' not in actions else actions[1], 'reset')
                self.assertNotIn('cardm', actions)
                self.assertEqual(PROFILES['profiles'][outputs['profile']]['mesh_device'], 'P150x4')

    def test_the_spike_is_the_reduction_order_probe_alone_under_the_serving_fabric(self):
        values, outputs = parsed('X1-rs-tile-spike')
        self.assertEqual(outputs['actions'].split(), ['reset', 'fabric'])
        self.assertEqual(outputs['fabric_probe'], 'rs-tile')
        self.assertEqual(outputs['fabric'], 'FABRIC_1D', 'the fabric config the serving plugin sets')
        for name, _ in read_order():
            if name != 'X1-rs-tile-spike':
                self.assertEqual(parsed(name)[1]['fabric_probe'], 'fabric', name)
        self.assertTrue(os.path.isfile(os.path.join(HERE, 'tp4_rs_tile_spike.py')))

    def test_the_matrix_is_the_s3a_job_with_only_the_image_changed(self):
        now, _ = parsed('X2-s3a-matrix')
        before = job.parse_env(text_of('S3a-s2-matrix', PREVIOUS))
        self.assertEqual({key: value for key, value in now.items() if key != 'C2_IMAGE_TAG'},
                         {key: value for key, value in before.items() if key != 'C2_IMAGE_TAG'})
        self.assertNotEqual(now['C2_IMAGE_TAG'], before['C2_IMAGE_TAG'])

    def test_the_gate_jobs_name_plans_the_gate_accepts_on_their_profile_and_fit_the_step(self):
        for name, plan in (('X2-s3a-matrix', 'matrix'), ('X4-permuted-matrix', 'permuted')):
            values, outputs = parsed(name)
            self.assertEqual(outputs['gate_plan'], plan, name)
            lengths = [int(part) for part in outputs['gate_lengths'].split(',')]
            self.assertEqual(lengths, [4096, 16384, 32768, 60000], name)
            arms = gate.plan_arms(plan, outputs['profile'], PROFILES, lengths, int(outputs['gate_max_tokens']), None, [], {})
            self.assertEqual([arm[0] for arm in arms],
                             ['matrix-concurrent', 'matrix-solo'] if plan == 'matrix'
                             else ['permuted-forward', 'permuted-reverse'], name)
            worst = 2 * sum(arm[2] + gate.ARM_OVERHEAD_SECONDS for arm in arms)
            self.assertLessEqual(worst, 380 * 60 - 600, name)
            self.assertEqual(outputs['gate_jit'], 'record', name)

    def test_the_smoke_names_tests_the_smoke_knows_and_the_bench_fits_the_profile(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        values, outputs = parsed('X3-quad-smoke-timings')
        for test in outputs['tests'].split(','):
            self.assertIn("'%s'" % test, smoke, test)
        self.assertNotIn('agreement', outputs['tests'])
        engine = PROFILES['profiles'][outputs['profile']]['engine']
        for shape in outputs['bench_shapes'].split(','):
            streams, prompt = (int(part) for part in shape.split('x'))
            self.assertEqual(streams, 4, 'the packed block has four users: the timings of the split are at four streams')
            self.assertLessEqual(streams, engine['max-num-seqs'], shape)
            self.assertLess(prompt + 256, engine['max-model-len'] + 1, shape)


class TheImageCarriesTheFix(unittest.TestCase):
    """A file in none or one of the image copy lists ships the bundle's version: tp_addresses.install imports the wrapper at attach,
    so the module must be in the overlay, the fast-serving Dockerfile and the image workflow's loop alike."""

    def test_the_wrapper_is_in_the_overlay_the_dockerfile_and_the_workflow_loop(self):
        self.assertEqual(image_list_membership().get('tile_collective_tp.py'), {'D', 'W', 'O'})

    def test_every_module_the_install_and_the_scope_import_ships_with_it(self):
        lists = image_list_membership()
        for name in ('tp_addresses.py', 'model_batch.py', 'tp_shapes.py', 'tile_collective_tp.py'):
            self.assertLessEqual({'D', 'W'}, lists.get(name, set()), name)


if __name__ == '__main__':
    unittest.main()
