"""tp4/ship-4: the combined ship candidate (a traffic profile generated from fusion-wp/S4.json by make_fusion_profiles.py) and its two gate-only twins (make_ship4_profiles).

  -fx-traffic-s4    the approved fusion ship profile + the prefill SDPA with one q chunk per core + the 13x10 matmul grids + octo-T8 and its three levers (non-alternating) + the tile-aligned drafter
                    window + the host KV tier with the preconverted checkpoints + the third block's memory plan, under ONE owner traffic waiver (PENDING);
  -fx-s4-audit      its audited twin (the octo-3 audited twin of the combined prefill arm + the tier's instruments);
  -fx-s4-nogrid13   the switchable control: the candidate without the 13x10 grids, gate-only.

The approved ship profile and the earlier waivers stay byte-identical; nothing here is hand-edited."""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grid13_smoke  # noqa: E402
import make_fusion_jobs as jobs_gen  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_octo3_profiles as octo3  # noqa: E402
import make_prefix_tier_profiles as tier  # noqa: E402
import make_ship4_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
PROFILES = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))['profiles']
NS = fusion.NAMESPACE
SHIP, AUDIT, NOGRID13 = twins.SHIP, twins.AUDIT, twins.NOGRID13
FUSION_SHIP = NS + 'traffic'
O3_ONEQ = NS + 'traffic-o3-oneq'
TIER_SHIP = tier.SHIP
PLAN = fusion.normalise(fusion.read_manifests())
SPEC = next(item for item in PLAN['ship'] if item['name'] == twins.SHIP_NAME)
OCTO_FLAGS = {'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_OCTO_MIN_LIVE': '6', 'QWEN_FAST_OCTO_DRAFT': '1', 'QWEN_FAST_OCTO_ATTN_BUNDLE': '4', 'QWEN_FAST_OCTO_GLUE8': '1'}
FOUR = {'QWEN_FAST_TP4_SDPA': 'multi', 'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1', 'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_PARKED_DRAFTS': '1'}


def boot_problems(profile):
    from test_octo3_contract import boot_problems as every_check

    return every_check(profile) + contract.prefix_policy_problems(profile)


class CandidateTests(unittest.TestCase):
    def test_the_candidate_is_a_generated_traffic_profile_and_no_gate_twin(self):
        self.assertEqual(SHIP, NS + 'traffic-s4')
        self.assertIn(SHIP, fusion.ship_names())
        self.assertNotIn(SHIP, fusion.twin_names())
        self.assertNotIn(SHIP, profile_twins.twin_names())
        self.assertIn(SHIP, profile_twins.generated_names())
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        self.assertEqual(regenerated[SHIP], PROFILES[SHIP])
        self.assertNotIn('gate_only', PROFILES[SHIP])

    def test_the_names_fit_the_64_characters_a_baked_default_may_have(self):
        self.assertLessEqual(len(SHIP), 64)

    def test_it_is_the_octo_oneq_candidate_plus_exactly_the_grids_and_the_tier(self):
        base, ship = PROFILES[O3_ONEQ], PROFILES[SHIP]
        added = {key: value for key, value in ship['env'].items() if base['env'].get(key) != value}
        self.assertEqual(added, dict(tier.SHIP_ENV, QWEN_FAST_GRID13='prefill,drafter', **{contract.OCTO_WAIVER_ENV: SPEC['waiver']['id']}))
        self.assertEqual(sorted(set(base['env']) - set(ship['env'])), [])
        self.assertEqual(ship['engine'], base['engine'], 'the third block\'s memory plan is the octo candidates\'')
        self.assertEqual((ship['engine']['num-gpu-blocks-override'], ship['engine']['additional-config']['tt']['trace_region_size']), (16500, 640 << 20))
        for key in ship:
            if key not in ('env', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(ship[key], base[key], key)

    def test_it_is_the_fusion_ship_plus_the_prefix_tier_candidate_s_env_and_everything_in_between(self):
        base, ship = PROFILES[FUSION_SHIP], PROFILES[SHIP]
        for key, value in tier.SHIP_ENV.items():
            self.assertEqual(ship['env'][key], value, key)
            self.assertNotIn(key, base['env'])
        self.assertEqual({key: value for key, value in PROFILES[TIER_SHIP]['env'].items() if key not in base['env']}, tier.SHIP_ENV, 'the tier candidate is the fusion ship plus the tier')
        added = {key for key, value in ship['env'].items() if base['env'].get(key) != value}
        self.assertEqual(added, set(OCTO_FLAGS) | set(tier.SHIP_ENV) | {contract.OCTO_WAIVER_ENV, 'QWEN_FAST_TAP_WINDOW_ALIGNED', 'QWEN_FAST_SDPA_PF_ONEQ', 'QWEN_FAST_GRID13', 'QWEN36_MAX_TOKENS_ALL_USERS'})

    def test_the_approved_ship_profile_and_every_earlier_waiver_are_untouched(self):
        for name, ident in ((FUSION_SHIP, 'w2-er-fx-262k-2026-10-11'), (NS + 'traffic-oneq', 'oneq-fx-262k-2026-10-11'), (fusion.PARENT, 'w2-er-262k-2026-10-10')):
            self.assertEqual(PROFILES[name][contract.TRAFFIC_WAIVER]['id'], ident)
        self.assertTrue(PROFILES[FUSION_SHIP][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual(set(PROFILES[FUSION_SHIP][contract.TRAFFIC_WAIVER]['levers']), set(FOUR))
        self.assertNotIn('QWEN_FAST_OCTO', PROFILES[FUSION_SHIP]['env'])
        self.assertNotIn('QWEN_PREFIX_HOST_TIER_GIB', PROFILES[FUSION_SHIP]['env'])
        self.assertEqual(PROFILES[FUSION_SHIP]['engine']['num-gpu-blocks-override'], 19968)

    def test_exactly_one_pending_waiver_naming_the_four_the_octo_levers_and_the_tier(self):
        waiver = PROFILES[SHIP][contract.TRAFFIC_WAIVER]
        profile = dict(PROFILES[SHIP], name=SHIP)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        self.assertEqual(waiver['levers'], dict(FOUR, **dict(OCTO_FLAGS, **tier.TIER_LEVERS)))
        self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
        self.assertEqual(contract.traffic_waiver_pending(profile), waiver['decision'].startswith('PENDING'))
        self.assertEqual(waiver['id'], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])
        self.assertEqual(sorted(waiver), ['decision', 'evidence', 'id', 'levers', 'waives'])
        self.assertEqual(sum(1 for name in PROFILES if name.startswith(NS) and name.endswith('-s4')), 1)
        ids = [body[contract.TRAFFIC_WAIVER]['id'] for body in PROFILES.values() if contract.TRAFFIC_WAIVER in body]
        self.assertEqual(len(ids), len(set(ids)), 'every waiver id is its own')
        self.assertGreaterEqual(len(waiver['evidence']), 6)

    def test_the_waiver_names_each_thing_the_owner_is_deciding_and_what_has_not_run(self):
        text = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives']
        for phrase in ('w2-er-fx-262k-2026-10-11', 'QWEN_FAST_OCTO_WAIVER', 'QWEN_FAST_SDPA_PF_ONEQ', 'K64j-OQ', 'QWEN_FAST_GRID13', 'switchable', 'QWEN_FAST_TAP_WINDOW_ALIGNED', 'tool-call whitespace',
                       'QWEN_PREFIX_HOST_TIER_GIB=32', 'QWEN_PREFIX_CKPT_PRECONVERTED=1', 'QWEN_PREFIX_TENANT_QUOTA', 'Nothing in this list has run on a card TOGETHER', 'packed_any_evidence_tp4.json',
                       'G8B1', 'quarter-tile', 'tau'):
            self.assertIn(phrase, text)
        evidence = ' '.join(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'])
        for phrase in ('+46.6%', '24 of 24', '38082194508', '38082317009', '38082865675', '38083229148', '38084230415', 'K64j-OQ', 'RQ0-RQ2', 'what has NOT run'):
            self.assertIn(phrase, evidence)

    def test_no_gate_instrument_is_in_the_candidate(self):
        env = PROFILES[SHIP]['env']
        self.assertNotIn('QWEN_C2_GATE_PROFILE', env)
        for key in contract.OCTO_GATE_ONLY:
            self.assertNotIn(key, env, key)
        for key, value in env.items():
            self.assertFalse(key.endswith('_AUDIT') and value == '1' and key not in PROFILES[fusion.PARENT]['env'], key)
        self.assertEqual((env['QWEN_PREFIX_HOST_TIER_AUDIT'], env['QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'], env['QWEN_PREFIX_HOST_TIER_VERIFY']), ('0', '0', 'sample'))
        self.assertNotIn('QWEN_PREFIX_HOST_TIER_OFF_PATH', env)
        self.assertFalse([key for key in env if key.startswith('QWEN_PREFIX_TENANT_QUOTA')], 'the quotas are gate-only: the contract refuses them on traffic')

    def test_the_text_of_the_profile_is_public_safe(self):
        text = json.dumps([PROFILES[SHIP]['description'], PROFILES[SHIP][contract.TRAFFIC_WAIVER]])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)
        self.assertTrue(PROFILES[SHIP]['description'].startswith('A SHIP CANDIDATE'))

    def test_the_candidate_passes_every_boot_check_and_the_prefix_policy(self):
        self.assertEqual(boot_problems(dict(PROFILES[SHIP], name=SHIP)), [])

    def test_the_candidate_s_bake_graft_image_and_pack_are_the_entry_s(self):
        self.assertEqual((SPEC['bake_graft'], SPEC['image'], SPEC['memory_plan'], SPEC['control'], SPEC['pack']), ('K64j-OQ', 'tp4-ship-4', 'octo', 'traffic', 'ship4'))
        self.assertEqual(jobs_gen.ship_folder(SPEC), 'ship4')
        self.assertEqual(jobs_gen.ship_image(SPEC), 'tp4-ship-4')
        wanted = jobs_gen.generate(PLAN)
        self.assertFalse([name for name in wanted if name.startswith(('ship-traffic-s4/', 'ship4/'))], 'another generator owns the ship4 folder whole')


class CaptureTests(unittest.TestCase):
    def test_the_octo_blocks_warm_and_captured_writers_take_the_same_path_under_the_candidates_own_env(self):
        """Card run 38117538847 (GX1 of this candidate on tp4-ship-4): engine start died capturing the octo block's verify trace, 'Cannot load new binaries during trace capture' at
        packed_ordered_cache's to_memory_config. The K/V page writer (QWEN_FAST_KV_PAGE_WRITER=1, in the candidate's env) was attached to the octo block's warm forward and declined in the
        captured one, so the capture launched the chained writer's programs that no eager forward had run. The warm forward and the capture of the octo block are built the way the model
        builds them, under the candidate's and the audited twin's env, behind the capture rules (test_fusion_capture_rules.octo_kv_lifecycle)."""
        from test_fusion_capture_rules import octo_kv_lifecycle
        import kv_page_writer_tp4 as kvpw

        for name in (SHIP, AUDIT, NOGRID13):
            for source in ('dram', 'l1'):
                with self.subTest(profile=name, source=source):
                    found = octo_kv_lifecycle(name, source=source)
                    self.assertEqual(found.clean(), [])
                    self.assertIsNone(found.escaped)
                    self.assertIsNone(found.warm.page)
                    self.assertIsNone(found.captured.page)
                    self.assertEqual(PROFILES[name]['env']['QWEN_FAST_KV_PAGE_WRITER'], '1')
                    self.assertEqual(len([line for line in found.lines if kvpw.OCTO_SERVED in line]), 1)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_exempt(self):
        self.assertEqual(twins.twin_names(), (AUDIT, NOGRID13))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        self.assertEqual(twins.generate(data), data)

    def test_the_audit_twin_is_the_octo3_audited_prefill_twin_plus_exactly_the_tier_instruments(self):
        parent, audit = PROFILES[twins.AUDIT_PARENT], PROFILES[AUDIT]
        added = {key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}
        self.assertEqual(added, tier.ENV)
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        self.assertEqual(audit['engine'], parent['engine'])
        self.assertEqual(twins.AUDIT_PARENT, octo3.FX_ALL_PREFILL + '-octo3-audit')
        self.assertEqual((audit['engine']['num-gpu-blocks-override'], audit['engine']['additional-config']['tt']['trace_region_size']), (16500, 640 << 20))

    def test_the_audit_twin_is_the_candidate_with_its_instruments(self):
        ship, audit = PROFILES[SHIP]['env'], PROFILES[AUDIT]['env']
        self.assertEqual([key for key in ship if key not in audit], [contract.OCTO_WAIVER_ENV])
        different = {key for key in ship if key in audit and ship[key] != audit[key]}
        self.assertEqual(different, {'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_PREFIX_HOST_TIER_AUDIT', 'QWEN_PREFIX_HOST_TIER_VERIFY', 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT',
                                     'QWEN_FAST_GRID13'})
        # the grids: the candidate carries the two sub-switches proven exact on cards, the audited twin all three, so that its attach exercises and audits the decode verify sub-switch
        # (site verify.out) that run 38116769673 died on before it compared anything
        self.assertEqual((ship['QWEN_FAST_GRID13'], audit['QWEN_FAST_GRID13'], audit['QWEN_FAST_GRID13_AUDIT']), ('prefill,drafter', '1', '1'))
        self.assertEqual(sorted(grid13_smoke.subswitches(ship)), ['drafter', 'prefill'])
        self.assertEqual(sorted(grid13_smoke.subswitches(audit)), ['drafter', 'prefill', 'verify'])
        self.assertNotIn('verify.out', [label for label, _names in grid13_smoke.required_sites(ship)])
        self.assertIn('verify.out', [label for label, _names in grid13_smoke.required_sites(audit)])
        extra = set(audit) - set(ship)
        self.assertTrue(all(key.endswith('_AUDIT') or key in ('QWEN_FAST_AUDIT_FIRST', 'QWEN_FAST_AUDIT_EVERY', 'QWEN_C2_GATE_PROFILE', 'QWEN_PREFIX_HOST_TIER_OFF_PATH') for key in extra), sorted(extra))

    def test_the_control_twin_is_the_candidate_without_the_grids_and_the_waiver_id_but_gate_only(self):
        ship, control = PROFILES[SHIP], PROFILES[NOGRID13]
        changed = {key: (ship['env'].get(key), control['env'].get(key)) for key in set(ship['env']) | set(control['env']) if ship['env'].get(key) != control['env'].get(key)}
        self.assertEqual(changed, {'QWEN_FAST_GRID13': ('prefill,drafter', None), contract.OCTO_WAIVER_ENV: (SPEC['waiver']['id'], None), 'QWEN_C2_GATE_PROFILE': (None, '1')})
        self.assertEqual(control['engine'], ship['engine'])
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(control[key], ship[key], key)

    def test_the_twins_sit_right_after_the_octo3_twins(self):
        names = list(PROFILES)
        last = max(names.index(name) for name in octo3.twin_names())
        self.assertEqual(names[last + 1:last + 3], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        data = copy.deepcopy(json.loads(PROFILES_PATH.read_text(encoding='utf-8')))
        data['profiles'][twins.AUDIT_PARENT]['env']['QWEN_PREFIX_HOST_TIER_GIB'] = '32'
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(json.loads(PROFILES_PATH.read_text(encoding='utf-8')))
        del data['profiles'][SHIP]['env']['QWEN_FAST_GRID13']
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(json.loads(PROFILES_PATH.read_text(encoding='utf-8')))
        data['profiles'][SHIP]['env']['QWEN_C2_GATE_PROFILE'] = '1'
        with self.assertRaises(ValueError):
            twins.generate(data)


class ApprovalTests(unittest.TestCase):
    def test_approving_the_candidate_changes_its_decision_sentence_and_nothing_else(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner 2026-10-12 (remote control): the combined candidate', directory, SPEC['name'])
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['S4.json'])
        old, new = json.loads(before['S4.json']), json.loads(after['S4.json'])
        new['ship'][0]['waiver']['decision'] = old['ship'][0]['waiver']['decision']
        self.assertEqual(new, old)
        plan = fusion.normalise(fusion.read_manifests(directory))
        generated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles']
        self.assertFalse(contract.traffic_waiver_pending(dict(generated[SHIP], name=SHIP)))
        self.assertEqual({key: value for key, value in generated[SHIP].items() if key != contract.TRAFFIC_WAIVER}, {key: value for key, value in PROFILES[SHIP].items() if key != contract.TRAFFIC_WAIVER})
        for name in (FUSION_SHIP, O3_ONEQ, NS + 'traffic-o3', fusion.PARENT):
            self.assertEqual(generated[name], PROFILES[name], name)
        for name in fusion.ship_names():
            if name != SHIP:
                self.assertEqual(contract.traffic_waiver_pending(dict(generated[name], name=name)), contract.traffic_waiver_pending(dict(PROFILES[name], name=name)), name)


class ManifestTests(unittest.TestCase):
    def test_a_pack_key_names_a_plain_folder_and_never_ship(self):
        raw = json.loads((fusion.MANIFESTS / 'S4.json').read_text(encoding='utf-8'))['ship'][0]
        for bad in ('ship', 'SHIP', '../x', 'ship-4', '5s', 7):
            with self.assertRaises(fusion.ManifestError):
                fusion.ship_of('S4', dict(raw, pack=bad), 'S4')
        self.assertEqual(fusion.ship_of('S4', dict(raw, pack='ship4'), 'S4')['pack'], 'ship4')
        self.assertEqual(fusion.ship_of('S5', dict(raw, pack='S5'), 'S5')['pack'], 'S5', 'tp4/next-1: the pack folder of the ship-5 candidate is S5, as NP1 and NP9 are NP1 and NP9')


if __name__ == '__main__':
    unittest.main()
