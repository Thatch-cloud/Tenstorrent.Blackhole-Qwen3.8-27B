"""The 2026-10-10 ship candidate (docs/tp4-ship-ln-w2-er.md): Lever N + W2 + engine reuse on one eight-seat 262k TRAFFIC profile, served only under
its owner traffic waiver (serving_c2_contract.TRAFFIC_WAIVER), and the ship pack that bakes and gates it (references/tp4-ship-ln-w2-er-jobs).

Holds: the profile is the Lever N traffic profile plus exactly the W2 and engine-reuse keys (the env of the gate arms -levern-w2 and -levern-parked
together) with the KV pool and trace region the engine-reuse memory margins were derived under; the contract serves it only through a well-formed
waiver that names each lever at its value (a missing, partial, stale or malformed waiver refuses, a gate instrument is never waivable); the boot logs the
waiver; no build bakes it while the owner's decision is PENDING (the job reader and the build script agree with the contract); the pack's jobs parse.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_parked_profiles as twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES = str(HERE / 'qwen_c2_profiles.json')
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic'
W2 = {'QWEN_FAST_TP4_SDPA': 'multi', 'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1'}
ER = {'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_PARKED_DRAFTS': '1', 'QWEN_FAST_LEVERN_BUILD_MS': 'learned'}
PACK = HERE / 'references' / 'tp4-ship-ln-w2-er-jobs'
APPROVED = 'APPROVED by the owner (test fixture): serve W2 and engine reuse on traffic.'


def raw():
    return json.loads(Path(PROFILES).read_text(encoding='utf-8'))


def load(name=SHIP):
    return contract.load_profile(PROFILES, name)


def flat(profile):
    body = copy.deepcopy(profile)
    body.pop('description', None)
    body.pop(contract.TRAFFIC_WAIVER, None)
    return body


class ProfileTests(unittest.TestCase):
    def test_it_is_the_levern_traffic_profile_plus_exactly_the_w2_and_engine_reuse_keys(self):
        found = raw()['profiles']
        want = flat(found[PARENT])
        want['env'].update(W2)
        want['env'].update(ER)
        self.assertEqual(flat(found[SHIP]), want)
        self.assertNotIn('gate_only', found[SHIP])
        for gate_arm, keys in ((PARENT.replace('-traffic', '-w2'), W2), (PARENT.replace('-traffic', '-parked'), ER)):
            with self.subTest(gate_arm=gate_arm):
                for flag, value in keys.items():
                    self.assertEqual(found[gate_arm]['env'].get(flag), value, 'the card-gated arm runs the same value')

    def test_the_engine_reuse_margins_hold_eight_seats_262k_and_the_file_default_is_unchanged(self):
        profile = load()
        engine = profile['engine']
        self.assertEqual(engine['num-gpu-blocks-override'], twins.KV_BLOCKS)
        self.assertEqual(engine['additional-config']['tt']['trace_region_size'], twins.TRACE_REGION)
        self.assertEqual(engine['max-num-seqs'], twins.SEATS)
        self.assertEqual(engine['max-model-len'], 262144)
        self.assertEqual(profile['mesh_device'], 'P150x4')
        self.assertEqual(profile['env']['QWEN_FAST_LEVERN_MAX_DECODE_GAP_S'], '8')
        self.assertEqual(profile['env']['QWEN_FAST_LEVERN_TTFT_TARGET_S'], '180')
        self.assertEqual(profile['env']['QWEN_PREFIX_REUSE'], '1')
        self.assertEqual(raw()['default'], 'c2-packed-tp4', 'only a build that names the profile bakes it')

    def test_no_gate_instrument_and_no_evidence_waiver_on_the_traffic_profile(self):
        env = load()['env']
        for name in contract.PARKED_GATE_ONLY + (contract.MULTI_AUDIT, contract.F1_AUDIT, contract.LEVERN_AUDIT, 'QWEN_FAST_LEVERN_FAULT',
                                                 contract.EVIDENCE_WAIVER, contract.GATE_PROFILE_MARKER):
            with self.subTest(name=name):
                self.assertNotIn(name, env)

    def test_it_is_the_only_profile_that_carries_an_owner_traffic_waiver(self):
        self.assertEqual(sorted(name for name, body in raw()['profiles'].items() if contract.TRAFFIC_WAIVER in body), [SHIP])

    def test_the_waiver_names_exactly_the_four_levers_cites_the_evidence_and_is_pending(self):
        waiver = load()[contract.TRAFFIC_WAIVER]
        self.assertEqual(waiver['levers'], contract.WAIVABLE_LEVERS)
        self.assertEqual(sorted(waiver), sorted(contract.TRAFFIC_WAIVER_FIELDS))
        self.assertTrue(any('37878277301' in item for item in waiver['evidence']), 'the SDPA multi exactness run')
        self.assertTrue(any('37918498075' in item for item in waiver['evidence']), 'the engine-reuse hang shapes')
        self.assertTrue(waiver['decision'].startswith('PENDING'), 'the owner decides; this commit only requests')


class ContractTests(unittest.TestCase):
    def test_the_ship_profile_passes_every_contract_check(self):
        profile = load()
        for check in (contract.traffic_waiver_problems, contract.multi_problems, contract.parked_problems, contract.levern_problems,
                      contract.prefix_reuse_problems, contract.mesh_problems):
            with self.subTest(check=check.__name__):
                self.assertEqual(check(profile), [])
        self.assertEqual(contract.waived_levers(profile), contract.WAIVABLE_LEVERS)

    def test_without_the_waiver_every_lever_is_refused_on_traffic(self):
        profile = load()
        profile.pop(contract.TRAFFIC_WAIVER)
        multi = contract.multi_problems(profile)
        self.assertEqual(len(multi), 2, multi)
        self.assertIn('QWEN_FAST_TP4_SDPA=multi needs a gate-only profile', multi[0])
        self.assertIn('QWEN_FAST_TP4_CONV_GATES_SPREAD=1 needs a gate-only profile', multi[1])
        parked = contract.parked_problems(profile)
        self.assertEqual([text.split('=')[0] for text in parked], ['QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS'], parked)

    def test_a_lever_the_waiver_does_not_name_is_still_refused(self):
        for flag in contract.WAIVABLE_LEVERS:
            with self.subTest(flag=flag):
                profile = load()
                del profile[contract.TRAFFIC_WAIVER]['levers'][flag]
                self.assertEqual(contract.traffic_waiver_problems(profile), [])
                refused = contract.multi_problems(profile) + contract.parked_problems(profile)
                self.assertEqual(len(refused), 1, refused)
                self.assertTrue(refused[0].startswith(flag + '='), refused)

    def test_a_malformed_or_misplaced_waiver_waives_nothing(self):
        cases = {
            'audit': lambda w, p: w['levers'].update({contract.MULTI_AUDIT: '1'}),
            'parked audit': lambda w, p: w['levers'].update({'QWEN_FAST_PARKED_AUDIT': '1'}),
            'value': lambda w, p: w['levers'].update({'QWEN_FAST_TP4_SDPA': 'grid8x4'}),
            'stale': lambda w, p: p['env'].pop('QWEN_FAST_TP4_CONV_GATES_SPREAD'),
            'unknown field': lambda w, p: w.update({'expires': 'never'}),
            'missing evidence': lambda w, p: w.pop('evidence'),
            'empty evidence': lambda w, p: w.update({'evidence': []}),
            'empty levers': lambda w, p: w.update({'levers': {}}),
            'bad id': lambda w, p: w.update({'id': 'Not An Id'}),
            'decision word': lambda w, p: w.update({'decision': 'maybe later'}),
            'gate only': lambda w, p: p.update({'gate_only': True}),
        }
        for case, change in cases.items():
            with self.subTest(case=case):
                profile = load()
                profile[contract.TRAFFIC_WAIVER] = copy.deepcopy(profile[contract.TRAFFIC_WAIVER])
                change(profile[contract.TRAFFIC_WAIVER], profile)
                self.assertTrue(contract.traffic_waiver_problems(profile), case)
                self.assertEqual(contract.waived_levers(profile), {})
                self.assertTrue(contract.traffic_waiver_pending(profile))
        not_object = dict(load(), **{contract.TRAFFIC_WAIVER: 'yes'})
        self.assertTrue(contract.traffic_waiver_problems(not_object))
        self.assertEqual(contract.waived_levers(not_object), {})

    def test_the_decision_word_sets_pending(self):
        profile = load()
        self.assertTrue(contract.traffic_waiver_pending(profile))
        profile[contract.TRAFFIC_WAIVER] = dict(profile[contract.TRAFFIC_WAIVER], decision=APPROVED)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        self.assertFalse(contract.traffic_waiver_pending(profile))
        self.assertFalse(contract.traffic_waiver_pending(load(PARENT)), 'no waiver, nothing pending')

    def test_the_production_and_levern_traffic_profiles_still_refuse_the_levers(self):
        for name in (twins.PRODUCTION, PARENT):
            with self.subTest(name=name):
                profile = load(name)
                profile['env'] = dict(profile['env'], **W2, **ER)
                self.assertEqual(len(contract.multi_problems(profile)), 2)
                self.assertEqual(len(contract.parked_problems(profile)), 2)


def boot(name, extra=None):
    """contract.boot under profile `name` as the .pth hook runs it for TRAFFIC (no gate switch): (the profile or the ValueError, the logged lines)."""
    environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': PROFILES}
    environ.update(extra or {})
    lines = []
    saved = list(sys.argv), list(sys.meta_path), list(sys.path)
    try:
        with mock.patch.object(contract, 'install_prefix_metrics', lambda api_server: None), \
                mock.patch.object(contract, 'install_teardown_skip', lambda: None), \
                mock.patch.object(contract, 'install_levern_platform', lambda: None), \
                mock.patch.object(contract, 'install_ring_check', lambda environ: None), \
                mock.patch.object(contract, 'read_salt_key', lambda environ: (b'k' * 32, 'test')), \
                mock.patch.object(contract, 'resolve_snapshot', lambda profile: profile['snapshots'][0]), \
                mock.patch.object(contract, 'log', lambda message, *values: lines.append(message % values if values else message)), \
                mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': name}):
            sys.argv[:] = ['-m', '--port', '8001']
            try:
                return contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER]), lines
            except ValueError as error:
                return error, lines
    finally:
        sys.argv[:], sys.meta_path[:], sys.path[:] = saved


class BootTests(unittest.TestCase):
    def test_the_ship_profile_boots_for_traffic_and_logs_its_waiver(self):
        result, lines = boot(SHIP)
        self.assertIsInstance(result, dict, result)
        self.assertEqual(result['name'], SHIP)
        waived = [line for line in lines if contract.TRAFFIC_WAIVER_MARKER in line]
        self.assertEqual(len(waived), 1, lines)
        for flag, value in contract.WAIVABLE_LEVERS.items():
            self.assertIn('%s=%s' % (flag, value), waived[0])
        self.assertIn('[decision: PENDING', waived[0])
        self.assertTrue(waived[0].startswith('profile %s: ' % SHIP))

    def test_a_profile_without_a_waiver_logs_none(self):
        result, lines = boot(PARENT)
        self.assertIsInstance(result, dict, result)
        self.assertFalse([line for line in lines if contract.TRAFFIC_WAIVER_MARKER in line])

    def test_a_malformed_waiver_refuses_the_boot(self):
        data = raw()
        data['profiles'][SHIP][contract.TRAFFIC_WAIVER]['levers'][contract.MULTI_AUDIT] = '1'
        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(data, handle)
            result, _lines = boot(SHIP, {'QWEN_C2_PROFILES': path})
        finally:
            shutil.rmtree(directory)
        self.assertIsInstance(result, ValueError)
        self.assertIn('is not waivable', str(result))

    def test_a_gate_instrument_in_the_process_environment_is_still_refused(self):
        result, _lines = boot(SHIP, {'QWEN_FAST_PARKED_AUDIT': '1'})
        self.assertIsInstance(result, ValueError)
        self.assertIn('outside a gate profile', str(result))


class BakeTests(unittest.TestCase):
    def root_with(self, decision):
        """A checkout root whose profiles file gives the ship waiver `decision`."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        os.makedirs(os.path.join(root, 'scripts', 'ci'))
        data = raw()
        data['profiles'][SHIP][contract.TRAFFIC_WAIVER]['decision'] = decision
        with open(os.path.join(root, 'scripts', 'ci', 'qwen_c2_profiles.json'), 'w', encoding='utf-8') as handle:
            json.dump(data, handle)
        return root

    def test_the_job_reader_refuses_to_bake_it_while_pending_and_bakes_it_once_approved(self):
        values = {'C2_BAKE_DEFAULT_PROFILE': SHIP}
        with self.assertRaises(job.JobError) as refused:
            job.read_bake(values, ['build'], 'quad')
        self.assertIn('not APPROVED', str(refused.exception))
        self.assertEqual(job.read_bake(values, ['build'], 'quad', root=self.root_with(APPROVED)), SHIP)
        with self.assertRaises(job.JobError):
            job.read_bake(values, ['build'], 'quad', root=self.root_with('APPROVEDISH but not really'))
        self.assertEqual(job.read_bake({'C2_BAKE_DEFAULT_PROFILE': PARENT}, ['build'], 'quad'), PARENT, 'a profile without a waiver bakes as before')

    def test_the_job_reader_and_the_contract_agree_on_pending(self):
        self.assertEqual(job.BAKE_WAIVER_FIELD, contract.TRAFFIC_WAIVER)
        for decision in ('PENDING OWNER DECISION', APPROVED, 'APPROVED: yes', 'approved', '', 'APPROVEDISH'):
            profile = load()
            profile[contract.TRAFFIC_WAIVER] = dict(profile[contract.TRAFFIC_WAIVER], decision=decision)
            with self.subTest(decision=decision):
                self.assertEqual(job.bake_waiver_pending(profile), contract.traffic_waiver_pending(profile))
        self.assertFalse(job.bake_waiver_pending(load(PARENT)))

    def test_the_build_script_refuses_a_pending_waiver_too(self):
        text = (HERE / 'build-c2-serving-image.sh').read_text(encoding='utf-8')
        self.assertIn("waiver = entry.get('owner_traffic_waiver')", text)
        self.assertIn(".startswith('APPROVED')", text)
        self.assertLess(text.index("entry.get('owner_traffic_waiver')"), text.index('DOCKER_BUILDKIT=1 docker build'))


class PackTests(unittest.TestCase):
    def read(self, name):
        return job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8'))

    def order(self):
        rows = [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
        return [(row[0], row[1], row[2], int(row[3])) for row in rows]

    def test_the_order_names_every_template_once_on_one_image(self):
        order = self.order()
        self.assertEqual([row[0] for row in order], ['B0-build', 'G1-ship-gate', 'G2-prefix-hit', 'SR-platform-replay'])
        self.assertEqual(sorted(path.stem for path in PACK.glob('*.env')), sorted(row[0] for row in order))
        self.assertEqual({row[2] for row in order}, {'tp4-ship-1'})
        self.assertEqual({row[1] for row in order}, {'stop'})
        for name, _rule, tag, _minutes in order:
            self.assertEqual(self.read(name)['C2_IMAGE_TAG'], tag, name)

    def test_b0_bakes_the_ship_profile_and_is_refused_until_the_owner_approves(self):
        values = self.read('B0-build')
        profiles = job.profile_names(PROFILES)
        self.assertEqual(values['C2_BAKE_DEFAULT_PROFILE'], SHIP)
        with self.assertRaises(job.JobError):
            job.read_job(values, profiles)
        approved = BakeTests('test_the_job_reader_refuses_to_bake_it_while_pending_and_bakes_it_once_approved')
        root = approved.root_with(APPROVED)
        try:
            outputs = job.read_job(values, profiles, root=root)
        finally:
            approved.doCleanups()
        self.assertEqual(outputs['bake_default_profile'], SHIP)
        self.assertEqual(outputs['actions'], 'build')

    def test_the_card_jobs_parse_and_serve_the_ship_profile_unaudited(self):
        profiles = job.profile_names(PROFILES)
        gate = job.read_job(self.read('G1-ship-gate'), profiles)
        self.assertEqual(gate['profile'], SHIP)
        tests = gate['tests'].split(',')
        for needed in ('warmup', 'coding', 'concurrent8_steady', 'concurrent8_skew', 'levern_decoder_finishes', 'levern_all_decoders_finish',
                       'levern_cancel_mid_prefill', 'levern_arrival_during_prefill', 'levern_seed_stops', 'parked_abort_reuse', 'parked_turns'):
            self.assertIn(needed, tests)
        self.assertEqual(job.read_job(self.read('G2-prefix-hit'), profiles)['prefix_profile'], SHIP)
        replay = job.read_job(self.read('SR-platform-replay'), profiles)
        self.assertEqual((replay['replay_profile'], replay['platform_image']), (SHIP, 'local:thin-layer'))
        for name in ('G1-ship-gate', 'G2-prefix-hit', 'SR-platform-replay'):
            text = (PACK / (name + '.env')).read_text(encoding='utf-8')
            self.assertNotIn('AUDIT', ''.join(line for line in text.splitlines() if not line.startswith('#')), name)

    def test_the_pack_names_no_registry_host_or_digest(self):
        for path in sorted(PACK.iterdir()):
            text = path.read_text(encoding='utf-8')
            with self.subTest(path=path.name):
                for word in ('thatch.local', 'sha256:', '192.168.', '/home/'):
                    self.assertNotIn(word, text)


if __name__ == '__main__':
    unittest.main()
