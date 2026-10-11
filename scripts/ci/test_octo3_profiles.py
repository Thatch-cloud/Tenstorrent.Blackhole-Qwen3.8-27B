"""tp4/octo-3: the gate-only twins of octo-T8 on the fused stack (make_octo3_profiles): each is its parent (the combined arm, its audited twin, the combined prefill arm) plus exactly the documented
env and the third block's memory plan; gate only; the generators agree on the one profiles file; the memory plan fits; the fused stack's flags are admitted beside the octo shape over the image's ENV;
the audit sampling is on the audited twins and nowhere else."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_fusion_profiles as fusion  # noqa: E402
import make_octo_profiles as octo_twins  # noqa: E402
import make_octo2_profiles as octo2_twins  # noqa: E402
import make_octo3_profiles as twins  # noqa: E402
import octo_judge  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_octo as octo  # noqa: E402
from test_c2_packed_tp4_profiles import image_env  # noqa: E402
import test_octo as host_tests  # noqa: E402

PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
PROFILES = json.loads(PROFILES_TEXT)['profiles']
M3 = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')
NAMES = twins.twin_names()
EXPECTED = (twins.FX_ALL + '-octo3', twins.FX_ALL + '-octo3-quaddraft', twins.FX_ALL + '-octo3-audit', twins.FX_ALL_PREFILL + '-octo3', twins.FX_ALL_PREFILL + '-octo3-audit')


def container_env(name):
    environ = dict(image_env())
    contract.apply_environment(dict(PROFILES[name], name=name), environ)
    environ['QWEN_C2_GATE'] = '1'
    return environ


class GeneratorTests(unittest.TestCase):
    def test_the_twins_are_the_documented_five_and_all_exist(self):
        self.assertEqual(NAMES, EXPECTED)
        for name in NAMES:
            self.assertIn(name, PROFILES)
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, profile_twins.generated_names())

    def test_the_checked_in_twins_are_what_the_generator_makes(self):
        self.assertEqual(twins.render(twins.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_every_generator_agrees_on_the_one_file(self):
        for module in ('make_fusion_profiles', 'make_octo_profiles', 'make_octo2_profiles', 'make_octo3_profiles', 'make_parked_profiles', 'make_kvread_profiles',
                       'make_round_host_profiles', 'make_upload_profiles', 'make_w2_kill_profiles'):
            result = subprocess.run([sys.executable, '-B', '-s', str(HERE / (module + '.py')), '--check'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(HERE.parent.parent))
            self.assertEqual(result.returncode, 0, '%s: %s' % (module, result.stderr.decode('utf-8', 'replace')))

    def test_the_twins_follow_the_octo2_block_and_nothing_else_moved(self):
        names = list(PROFILES)
        block = [name for name in names if name in NAMES]
        start = names.index(block[0])
        self.assertEqual(names[start:start + len(block)], block)
        self.assertEqual(names[start - 1], octo2_twins.twin_names()[-1])

    def test_the_generator_refuses_a_moved_parent_and_a_parent_that_already_names_a_lever(self):
        data = json.loads(PROFILES_TEXT)
        moved = copy.deepcopy(data)
        moved['profiles'][twins.FX_ALL]['engine']['num-gpu-blocks-override'] = 20000
        with self.assertRaisesRegex(ValueError, 'memory plan was derived under'):
            twins.generate(moved)
        named = copy.deepcopy(data)
        named['profiles'][twins.FX_ALL_AUDIT]['env']['QWEN_FAST_OCTO_GLUE8'] = '1'
        with self.assertRaisesRegex(ValueError, 'never carries the lever'):
            twins.generate(named)
        no_w2 = copy.deepcopy(data)
        del no_w2['profiles'][twins.FX_ALL]['env']['QWEN_FAST_TP4_SDPA']
        with self.assertRaisesRegex(ValueError, 'not a W2 profile'):
            twins.generate(no_w2)
        missing = copy.deepcopy(data)
        del missing['profiles'][twins.FX_ALL]
        with self.assertRaisesRegex(ValueError, 'not defined'):
            twins.generate(missing)


class TwinTests(unittest.TestCase):
    def differing(self, name, parent):
        return {key: value for key, value in PROFILES[name]['env'].items() if PROFILES[parent]['env'].get(key) != value}

    def test_each_is_its_parent_plus_exactly_the_documented_env_and_the_memory_plan(self):
        octo = {'QWEN_C2_GATE_PROFILE': '1', 'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_OCTO_MIN_LIVE': '6', 'QWEN36_MAX_TOKENS_ALL_USERS': '1055488'}
        levers = {'QWEN_FAST_OCTO_DRAFT': '1', 'QWEN_FAST_OCTO_ATTN_BUNDLE': '4', 'QWEN_FAST_OCTO_GLUE8': '1'}
        audits = {'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1', 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT': '1',
                  'QWEN_FAST_TAP_WINDOW_ALIGNED': '1', 'QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT': '1', 'QWEN_FAST_AUDIT_FIRST': '1', 'QWEN_FAST_AUDIT_EVERY': '16'}
        expected = {
            twins.FX_ALL + '-octo3': (twins.FX_ALL, dict(octo, **levers)),
            twins.FX_ALL + '-octo3-quaddraft': (twins.FX_ALL, dict(octo, QWEN_FAST_OCTO_ATTN_BUNDLE='4', QWEN_FAST_OCTO_GLUE8='1')),
            twins.FX_ALL + '-octo3-audit': (twins.FX_ALL_AUDIT, dict(octo, **dict(levers, **audits))),
            twins.FX_ALL_PREFILL + '-octo3': (twins.FX_ALL_PREFILL, dict(octo, **levers)),
            twins.FX_ALL_PREFILL + '-octo3-audit': (twins.FX_ALL_PREFILL_AUDIT, dict(octo, **dict(levers, **audits))),
        }
        for name, (parent, env) in expected.items():
            with self.subTest(profile=name):
                self.assertEqual(self.differing(name, parent), env)
                for key in PROFILES[parent]['env']:
                    self.assertIn(key, PROFILES[name]['env'])
                body, base = copy.deepcopy(PROFILES[name]), copy.deepcopy(PROFILES[parent])
                for item in (body, base):
                    item.pop('env'), item.pop('description')
                    item['engine'].pop('num-gpu-blocks-override')
                    item['engine']['additional-config']['tt'].pop('trace_region_size')
                self.assertEqual(body, base, 'every field but the env, the description, the pool and the trace region is the parent\'s')
                self.assertEqual(PROFILES[name]['engine']['num-gpu-blocks-override'], 16500)
                self.assertEqual(PROFILES[name]['engine']['additional-config']['tt']['trace_region_size'], 640 << 20)
                self.assertIs(PROFILES[name]['gate_only'], True)

    def test_the_mode_is_live_never_alternate_so_the_timing_arms_are_not_alternating(self):
        for name in NAMES:
            self.assertEqual(octo_judge.mode(PROFILES[name]['env']), 'live', name)

    def test_the_audit_sampling_is_on_the_audited_twins_alone(self):
        import make_ship4_profiles

        for name, profile in PROFILES.items():
            has = 'QWEN_FAST_AUDIT_EVERY' in profile['env']
            self.assertEqual(has, name in (twins.FX_ALL + '-octo3-audit', twins.FX_ALL_PREFILL + '-octo3-audit', make_ship4_profiles.AUDIT), name)      # (and tp4/ship-4's audited twin of the combined candidate)
        import audit_sampling

        for name in (twins.FX_ALL + '-octo3-audit', twins.FX_ALL_PREFILL + '-octo3-audit'):
            self.assertEqual(audit_sampling.settings(PROFILES[name]['env']), (1, 16))

    def test_the_audited_twin_is_not_the_lever_n_audited_parent(self):
        """Its 8.4-minutes-a-request instruments (prefix digests, the singles audit, the Lever N audit) would make the audited attach as slow as the octo-2 one."""
        env = PROFILES[twins.FX_ALL + '-octo3-audit']['env']
        for name in ('QWEN_PREFIX_DIGESTS', 'QWEN_FAST_DRAFT_SINGLES_AUDIT', 'QWEN_FAST_LEVERN_AUDIT', 'QWEN_FAST_VERIFY_T1_AUDIT'):
            self.assertNotIn(env.get(name, '0'), ('1', 'all'), name)

    def test_the_draft_ab_twin_is_the_timing_arm_without_the_draft(self):
        timed, quad = PROFILES[twins.FX_ALL + '-octo3']['env'], PROFILES[twins.FX_ALL + '-octo3-quaddraft']['env']
        self.assertEqual({key: value for key, value in timed.items() if quad.get(key) != value}, {'QWEN_FAST_OCTO_DRAFT': '1'})
        self.assertEqual({key for key in quad if key not in timed}, set())

    def test_the_fused_stack_is_kept_whole(self):
        """W2, engine reuse, the parked drafts and every fusion lever of the combined arm are still on beside the octo shape."""
        for name in NAMES:
            env = PROFILES[name]['env']
            for flag in ('QWEN_FAST_TP4_SDPA', 'QWEN_FAST_TP4_CONV_GATES_SPREAD', 'QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS', 'QWEN_FAST_KV_PAGE_WRITER', 'QWEN_FAST_TP4_SHARD_ARGMAX',
                         'QWEN_FAST_CCL_OPTIONS', 'QWEN_FAST_MLP_CFG', 'QWEN_FAST_DRAFT_PERMUTE', 'QWEN_FAST_BATCHED_READS'):
                self.assertIn(flag, env, '%s %s' % (name, flag))

    def test_no_other_profile_names_an_octo3_only_name(self):
        import make_ship4_profiles

        owned = set(twins.owned_names()) | set(make_ship4_profiles.owned_names())      # (tp4/ship-4's twins and combined candidate carry the octo env too: test_ship4_profiles)
        for name, profile in PROFILES.items():
            if name in owned or name in octo_twins.twin_names() or name in octo2_twins.twin_names():
                continue
            self.assertNotIn('QWEN_FAST_AUDIT_EVERY', profile['env'], name)
            self.assertNotIn('QWEN_FAST_OCTO_WAIVER', profile['env'], name)


class MemoryPlanTests(unittest.TestCase):
    def test_the_figures_are_the_octo_plans_and_fit(self):
        plan = octo_twins.memory_plan()
        self.assertGreaterEqual(plan['margin_bytes'], 0)
        self.assertGreaterEqual(plan['windows'], 4)
        self.assertEqual(plan['pool_tokens'], 1055488)

    def test_eight_120k_prompts_fit_the_pool_of_the_deep_test(self):
        """concurrent8_code_128k(_long): 8 x about 120,800 tokens reserve 15,112 of the 16,492 usable blocks, so every seat is live at once."""
        usable = octo_twins.MEMORY['blocks'] - 8
        need = 8 * -(-(120000 + 800) // 64)
        self.assertLessEqual(need, 15112)            # (the smoke's own comment counts 15,112 blocks with its page rounding)
        self.assertLess(15112 + 8, usable)

    def test_the_pool_tokens_follow_the_contracts_request_rule(self):
        for name in NAMES:
            profile = dict(PROFILES[name], name=name)
            self.assertEqual(contract.kv_pool_problem(profile) if hasattr(contract, 'kv_pool_problem') else None, None)

    def test_the_apply_function_is_what_a_ship_entry_uses(self):
        profile = copy.deepcopy(PROFILES[fusion.NAMESPACE + 'traffic'])
        twins.apply_memory_plan(profile)
        self.assertEqual(profile['engine']['num-gpu-blocks-override'], 16500)
        with self.assertRaises(ValueError):
            twins.apply_memory_plan(profile)             # a second application meets the lowered pool: refused


class AdmissionOverTheImageTests(unittest.TestCase):
    """Each twin over the C2 image's own ENV: the fused stack's flags (W2's multi SDPA, engine reuse and its drafts, the op-fusion set) are admitted beside the octo shape."""

    def test_every_twin_is_admitted_with_the_device_pieces_built(self):
        self.assertEqual(octo.device_gaps(), [])
        for name in NAMES:
            with self.subTest(profile=name):
                log = host_tests.Lines()
                record = octo.octo_admission(M3, container_env(name), log=log)
                self.assertEqual((record['mode'], record['min_live'], record['rows'], record['users']), ('live', 6, 8, 8))
                self.assertEqual(bool(record.get('draft')), 'QWEN_FAST_OCTO_DRAFT' in PROFILES[name]['env'])
                self.assertFalse(any(line.startswith(octo.REFUSED_MARKER) for line in log.lines))
                self.assertTrue(any(line.endswith('(gate only)') for line in log.lines if line.startswith(octo.ADMITTED_MARKER)))

    def test_w2_multi_alone_no_longer_refuses_the_octo_shape(self):
        name = NAMES[0]
        self.assertEqual(PROFILES[name]['env']['QWEN_FAST_TP4_SDPA'], 'multi')
        environ = container_env(name)
        octo.octo_admission(M3, environ, log=host_tests.Lines())
        environ['QWEN_FAST_TP4_SDPA'] = 'l1'
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_TP4_SDPA=l1'):
            octo.octo_admission(M3, environ, log=host_tests.Lines())

    def test_the_octo_draft_is_admitted_beside_engine_reuse_and_the_parked_drafts(self):
        import octo_draft_tp

        environ = container_env(NAMES[0])
        self.assertEqual(environ['QWEN_FAST_PARKED_ENGINES'], '1')
        self.assertEqual(environ['QWEN_FAST_PARKED_DRAFTS'], '1')
        self.assertEqual(octo_draft_tp.admission_problems(environ), [])
        names = [name for name, why in octo_draft_tp.EXCLUDED_FLAGS]
        self.assertNotIn('QWEN_FAST_PARKED_ENGINES', names)
        self.assertNotIn('QWEN_FAST_PARKED_DRAFTS', names)

    def test_the_judge_reads_the_modes_these_profiles_carry(self):
        for name in NAMES:
            env = PROFILES[name]['env']
            self.assertEqual(octo_judge.min_live(env), 6)
            self.assertEqual(octo_judge.factory_entries(env), (4,) if 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT' not in env else (1, 4), name)


if __name__ == '__main__':
    unittest.main()
