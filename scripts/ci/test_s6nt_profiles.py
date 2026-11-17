"""tp4/s6-notier: the ship-6 profile without the host KV tier (a traffic profile generated from fusion-wp/S6NT.json by make_fusion_profiles.py), the contract's text-only
boot and the fast worker's one malloc_trim after the warm (docs/tp4-s6-notier.md).

  -fx-traffic-s6-nt   the ship-6 profile that serves production (-fx-traffic-s6) less exactly its six host-tier lines (QWEN_PREFIX_HOST_TIER_GIB, _AUDIT, _VERIFY,
                      _SPILL_MODE, QWEN_PREFIX_TIER_RESTORE_SLICE, _MS) plus QWEN_C2_TEXT_ONLY_BOOT=1, under ONE owner traffic waiver of its own (decision PENDING or
                      APPROVED: these tests read neither);
  ship-traffic-s6-nt  its generated ship pack (B0 G1 GX0 GX1 G2 SR); the exactness control is the ship-6 profile.

The ship-6 profile, its waiver and every profile that existed on tp4/s6 stay byte-identical (checked against the commit this branch was cut from)."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_fusion_jobs as jobs_gen  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_s6_profiles as s6  # noqa: E402
import profile_twins  # noqa: E402
import qwen_prefix_registry as registry  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_startup  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, NOTIER = s6.SHIP, s6.NOTIER
PLAN = fusion.normalise(fusion.read_manifests())
SPEC = next(item for item in PLAN['ship'] if item['name'] == 'traffic-s6-nt')
SPEC6 = next(item for item in PLAN['ship'] if item['name'] == 'traffic-s6')
PARENT_COMMIT = '43c60a52'                      # tp4/s6: the ship-6 waiver approved, the tree this branch was cut from
WAIVER_ID = 'ship6-notier-2026-11-17'
TIER_LINES = ('QWEN_PREFIX_HOST_TIER_GIB', 'QWEN_PREFIX_HOST_TIER_AUDIT', 'QWEN_PREFIX_HOST_TIER_VERIFY', 'QWEN_PREFIX_HOST_TIER_SPILL_MODE',
              'QWEN_PREFIX_TIER_RESTORE_SLICE', 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS')
TIER_LEVERS = ('QWEN_PREFIX_HOST_TIER_GIB', 'QWEN_PREFIX_HOST_TIER_SPILL_MODE')
PACK = HERE / 'references' / 'fusion-jobs' / 'ship-traffic-s6-nt'
READER = HERE / 'c2_serving_job.py'
PACK_JOBS = ('B0-ship-build', 'G1-ship-gate', 'GX0-exact-control', 'GX1-exact-ship', 'G2-prefix-hit', 'SR-platform-replay')


def boot_problems(profile):
    from test_octo3_contract import boot_problems as every_check

    return (every_check(profile) + contract.prefix_policy_problems(profile) + contract.exact_resume_problems(profile) + contract.sched_problems(profile)
            + contract.text_only_problems(profile))


def parent_profiles():
    try:
        result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT_COMMIT], capture_output=True, cwd=str(HERE), timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))['profiles']


def pending():
    return contract.traffic_waiver_pending(dict(PROFILES[NOTIER], name=NOTIER))


class ProfileTests(unittest.TestCase):
    def test_it_is_a_generated_traffic_profile_and_no_gate_twin(self):
        self.assertEqual(NOTIER, fusion.NAMESPACE + 'traffic-s6-nt')
        self.assertIn(NOTIER, fusion.ship_names())
        self.assertNotIn(NOTIER, fusion.twin_names())
        self.assertNotIn(NOTIER, profile_twins.twin_names())
        self.assertIn(NOTIER, s6.owned_names())
        self.assertNotIn(NOTIER, s6.twin_names())
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        self.assertEqual(regenerated[NOTIER], PROFILES[NOTIER])
        self.assertNotIn('gate_only', PROFILES[NOTIER])
        self.assertEqual((SPEC['pack'], SPEC['image'], SPEC['control'], SPEC['bake_graft'], SPEC['memory_plan']), (None, 'tp4-s6nt-1', 'traffic-s6', 'K64j-OQ', 'octo'))
        self.assertEqual(DATA['default'], 'c2-packed-tp4', 'the file default is untouched: a build bakes its own (C2_BAKE_DEFAULT_PROFILE)')

    def test_the_name_fits_what_a_baked_default_may_be(self):
        self.assertLessEqual(len(NOTIER), 64)
        self.assertTrue(job.PROFILE_NAME.fullmatch(NOTIER))

    def test_it_is_the_ship_6_profile_less_exactly_the_tier_lines_plus_the_text_only_boot(self):
        old, new = PROFILES[SHIP], PROFILES[NOTIER]
        removed = sorted(key for key in old['env'] if key not in new['env'])
        self.assertEqual(removed, sorted(TIER_LINES))
        changed = {key: new['env'][key] for key in new['env'] if old['env'].get(key) != new['env'][key]}
        self.assertEqual(changed, {contract.TEXT_ONLY_FLAG: '1', contract.OCTO_WAIVER_ENV: WAIVER_ID})
        for key in set(old) | set(new):
            if key not in ('env', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(new.get(key), old.get(key), key)
        self.assertEqual(new['env']['QWEN_PREFIX_STORE_GIB'], '16', 'the checkpoint store stays at the ship-6 budget')
        self.assertEqual((new['env']['QWEN_PREFIX_CKPT_PRECONVERTED'], new['env']['QWEN_PREFIX_WINDOW_CARRY'], new['env']['QWEN_PREFIX_STICKY_UNDROP']), ('1', '1', '1'))
        self.assertEqual(new['engine']['limit-mm-per-prompt'], {'image': 0, 'video': 0})
        self.assertNotIn(contract.TEXT_ONLY_FLAG, old['env'], 'the ship-6 profile is the rollback and keeps its tower')

    def test_the_registry_sees_no_tier_and_the_store_at_16_gib(self):
        env = {key: str(value) for key, value in PROFILES[NOTIER]['env'].items()}
        store = 16 * (1 << 30)
        self.assertIsNone(registry.tier_config(env, store), 'no tier: no slab is mapped')
        self.assertIsNotNone(registry.tier_config({key: str(value) for key, value in PROFILES[SHIP]['env'].items()}, store), 'the ship-6 profile keeps its tier')

    def test_every_profile_on_tp4_s6_is_untouched(self):
        parent = parent_profiles()
        if parent is None:
            self.skipTest('no git history for %s' % PARENT_COMMIT)
        for name, body in parent.items():
            self.assertEqual(PROFILES[name], body, name)
        self.assertEqual([name for name in PROFILES if name not in parent], [NOTIER])

    def test_the_profile_passes_every_boot_check(self):
        self.assertEqual(boot_problems(dict(PROFILES[NOTIER], name=NOTIER)), [])
        self.assertEqual(contract.prefix_policy_problems(dict(PROFILES[NOTIER], name=NOTIER)), [])

    def test_a_tier_line_put_back_without_the_waiver_is_refused(self):
        for flag, value in (('QWEN_PREFIX_HOST_TIER_GIB', '48'), ('QWEN_PREFIX_HOST_TIER_SPILL_MODE', 'complete'), ('QWEN_PREFIX_TIER_RESTORE_SLICE', '512')):
            profile = copy.deepcopy(dict(PROFILES[NOTIER], name=NOTIER))
            profile['env'][flag] = value
            problems = contract.prefix_policy_problems(profile)
            self.assertTrue([problem for problem in problems if flag in problem], (flag, problems))

    def test_the_text_of_the_profile_is_public_safe(self):
        text = json.dumps([PROFILES[NOTIER]['description'], PROFILES[NOTIER][contract.TRAFFIC_WAIVER]])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)
        self.assertTrue(PROFILES[NOTIER]['description'].startswith('A SHIP CANDIDATE'))


class WaiverTests(unittest.TestCase):
    def test_one_waiver_of_its_own_naming_the_ship_6_levers_less_the_tier(self):
        waiver = PROFILES[NOTIER][contract.TRAFFIC_WAIVER]
        profile = dict(PROFILES[NOTIER], name=NOTIER)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        old = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['levers']
        self.assertEqual(waiver['levers'], {flag: value for flag, value in old.items() if flag not in TIER_LEVERS})
        self.assertEqual(waiver['id'], WAIVER_ID)
        self.assertEqual(waiver['id'], PROFILES[NOTIER]['env'][contract.OCTO_WAIVER_ENV])
        self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])
        self.assertEqual(contract.traffic_waiver_pending(profile), waiver['decision'].startswith('PENDING'))
        self.assertEqual(sorted(waiver), ['decision', 'evidence', 'id', 'levers', 'waives'])
        ids = [body[contract.TRAFFIC_WAIVER]['id'] for body in PROFILES.values() if contract.TRAFFIC_WAIVER in body]
        self.assertEqual(len(ids), len(set(ids)), 'every waiver id is its own')

    def test_the_waiver_carries_the_ship_6_text_and_evidence_verbatim_and_says_what_it_removes(self):
        text, old = PROFILES[NOTIER][contract.TRAFFIC_WAIVER]['waives'], PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives']
        self.assertTrue(text.startswith(old), 'the ship-6 waiver text is carried verbatim')
        added = text[len(old):]
        for phrase in ('ship6-fx-262k-2026-10-21', '(S) REMOVED', '(T) The text-only boot', 'QWEN_C2_TEXT_ONLY_BOOT=1', 'tier_restore_requests 0', 'NEVER run on a card',
                       'malloc_trim', 'QWEN_PREFIX_STORE_GIB=16'):
            self.assertIn(phrase, added)
        for flag in TIER_LINES:
            self.assertIn(flag, added, flag)
        evidence = PROFILES[NOTIER][contract.TRAFFIC_WAIVER]['evidence']
        old_evidence = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence']
        self.assertEqual(evidence[:len(old_evidence)], old_evidence)
        self.assertIn('tier_restore_requests 0 and tier_restore_blocks 0', ' '.join(evidence[len(old_evidence):]))
        self.assertIn('what has NOT run', ' '.join(evidence[len(old_evidence):]).replace('What has NOT run', 'what has NOT run'))

    def test_the_ship_6_entry_and_its_waiver_are_untouched(self):
        self.assertEqual(SPEC6['waiver']['id'], 'ship6-fx-262k-2026-10-21')
        self.assertTrue(SPEC6['waiver']['decision'].startswith('APPROVED'))
        self.assertIn('QWEN_PREFIX_HOST_TIER_GIB', PROFILES[SHIP]['env'])


class ApprovalTests(unittest.TestCase):
    def test_approving_it_changes_its_decision_sentence_and_nothing_else(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner: the ship-6 profile without the host KV tier', directory, SPEC['name'])
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['S6NT.json'])
        plan = fusion.normalise(fusion.read_manifests(directory))
        generated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles']
        self.assertFalse(contract.traffic_waiver_pending(dict(generated[NOTIER], name=NOTIER)))
        self.assertEqual({key: value for key, value in generated[NOTIER].items() if key != contract.TRAFFIC_WAIVER},
                         {key: value for key, value in PROFILES[NOTIER].items() if key != contract.TRAFFIC_WAIVER})
        self.assertEqual(boot_problems(dict(generated[NOTIER], name=NOTIER)), [])
        for name in fusion.ship_names():
            if name != NOTIER:
                self.assertEqual(generated[name], PROFILES[name], name)


class ManifestTests(unittest.TestCase):
    def test_the_manifest_is_canonical_and_lists_its_test(self):
        text = (fusion.MANIFESTS / 'S6NT.json').read_text(encoding='utf-8')
        self.assertEqual(json.dumps(json.loads(text), indent=1) + '\n', text)
        raw = json.loads(text)
        self.assertEqual((raw['wp'], raw['branch'], len(raw['ship'])), ('S6NT', 'tp4/s6-notier', 1))
        self.assertIn('test_s6nt_profiles', raw['tests'])

    def test_the_ship_entry_is_the_ship_6_entry_less_the_tier(self):
        old = json.loads((fusion.MANIFESTS / 'S6.json').read_text(encoding='utf-8'))['ship'][0]
        new = json.loads((fusion.MANIFESTS / 'S6NT.json').read_text(encoding='utf-8'))['ship'][0]
        self.assertEqual(sorted(key for key in old['env'] if key not in new['env']), sorted(TIER_LINES))
        self.assertEqual({key: value for key, value in new['env'].items() if old['env'].get(key) != value},
                         {'QWEN_FAST_OCTO_WAIVER': WAIVER_ID, contract.TEXT_ONLY_FLAG: '1'})
        for key in ('from', 'memory_plan', 'bake_graft'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['control'], 'traffic-s6')


class TextOnlyBootTests(unittest.TestCase):
    """serving_c2_contract item 10: the switch, its refusal beside images, the inherited value dropped, and the hook on the model class."""

    def profile(self, value=None, limits=None):
        profile = copy.deepcopy(dict(PROFILES[NOTIER], name=NOTIER))
        if value is None:
            profile['env'].pop(contract.TEXT_ONLY_FLAG, None)
        else:
            profile['env'][contract.TEXT_ONLY_FLAG] = value
        if limits is not None:
            profile['engine']['limit-mm-per-prompt'] = limits
        return profile

    def test_the_switch_is_admitted_only_beside_an_engine_that_takes_no_image_and_no_video(self):
        self.assertEqual(contract.text_only_problems(self.profile('1')), [])
        self.assertEqual(contract.text_only_problems(self.profile('0')), [])
        self.assertEqual(contract.text_only_problems(self.profile(None)), [])
        self.assertEqual(contract.text_only_problems(self.profile(None, {'image': 1, 'video': 0})), [], 'without the switch the limits are not its business')
        for limits in ({'image': 1, 'video': 0}, {'image': 0, 'video': 1}, {'image': 0}, None, {'image': False, 'video': 0}):
            problems = contract.text_only_problems(self.profile('1', limits) if limits is not None else dict(self.profile('1'), engine={}))
            self.assertTrue([problem for problem in problems if contract.TEXT_ONLY_FLAG in problem], (limits, problems))
        self.assertTrue(contract.text_only_problems(self.profile('yes')))

    def test_no_other_profile_sets_the_switch(self):
        self.assertEqual(sorted(name for name, body in PROFILES.items() if contract.TEXT_ONLY_FLAG in body.get('env', {})), [NOTIER])

    def test_an_inherited_switch_is_dropped_under_a_profile_that_does_not_name_it(self):
        environ = contract.apply_environment(dict(copy.deepcopy(PROFILES[SHIP]), name=SHIP), {contract.TEXT_ONLY_FLAG: '1', 'KEEP': 'me'})
        self.assertNotIn(contract.TEXT_ONLY_FLAG, environ)
        self.assertEqual(environ['KEEP'], 'me')
        environ = contract.apply_environment(dict(copy.deepcopy(PROFILES[NOTIER]), name=NOTIER), {})
        self.assertEqual(environ[contract.TEXT_ONLY_FLAG], '1')
        for flag in TIER_LINES:
            self.assertNotIn(flag, contract.apply_environment(dict(copy.deepcopy(PROFILES[NOTIER]), name=NOTIER), {flag: '1'}), flag)

    def test_the_hook_is_armed_only_by_the_switch(self):
        seen = []
        with patch.object(contract, 'log'):
            self.assertFalse(contract.install_text_only_boot(NOTIER, {}, on_import=lambda name, callback: seen.append((name, callback))))
            self.assertEqual(seen, [])
            self.assertTrue(contract.install_text_only_boot(NOTIER, {contract.TEXT_ONLY_FLAG: '1'}, on_import=lambda name, callback: seen.append((name, callback))))
        self.assertEqual(seen, [(contract.VISION_MODULE, contract.skip_vision_init)])

    @staticmethod
    def model_module():
        calls = []

        class Qwen36Model(object):
            def __init__(self):
                self.vision_model = None

            def init_vision_model(self, reference_visual=None, vision_args=None):
                calls.append((reference_visual, vision_args))
                self.vision_model = 'tower'
                return self.vision_model

        module = ModuleType(contract.VISION_MODULE)
        module.Qwen36Model = Qwen36Model
        return module, calls

    def test_the_wrappers_default_call_builds_no_tower_and_every_other_call_runs_the_original(self):
        module, calls = self.model_module()
        with patch.object(contract, 'log_pindiag') as said:
            installed = contract.skip_vision_init(module)
            model = module.Qwen36Model()
            self.assertIsNone(model.init_vision_model())
            self.assertIsNone(model.vision_model)
            self.assertEqual(calls, [])
            self.assertEqual(said.call_count, 1)
            self.assertTrue(said.call_args[0][0].startswith(contract.TEXT_ONLY_MARKER))
            self.assertEqual(model.init_vision_model('visual'), 'tower', 'an explicit tower is built as before')
            self.assertEqual(calls, [('visual', None)])
            self.assertEqual(model.init_vision_model(), 'tower', 'a model that has a tower keeps the original behaviour')
        self.assertIs(contract.skip_vision_init(module), installed, 'idempotent')
        self.assertIs(module.Qwen36Model.init_vision_model, installed)

    def test_a_model_tree_without_the_method_is_refused(self):
        module = ModuleType(contract.VISION_MODULE)
        with self.assertRaises(ValueError):
            contract.skip_vision_init(module)
        module.Qwen36Model = type('Qwen36Model', (object,), {})
        with self.assertRaises(ValueError):
            contract.skip_vision_init(module)


class HeapTrimTests(unittest.TestCase):
    """serving_startup.trim_host_heap: once after the warm, glibc only, never fatal."""

    def run_trim(self, libc=None, glibc='glibc 2.35', rss=(1000, 400)):
        lines = []
        readings = iter(rss)
        result = serving_startup.trim_host_heap(libc=libc, say=lines.append, rss=lambda: next(readings, None), glibc=glibc)
        return result, lines

    def test_a_trim_that_returns_pages_is_logged_once_with_what_it_freed(self):
        calls = []
        libc = SimpleNamespace(malloc_trim=lambda pad: calls.append(pad) or 1)
        result, lines = self.run_trim(libc)
        self.assertEqual(result, 'trimmed')
        self.assertEqual(calls, [0])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('[PINDIAG] host heap trimmed after warmup: malloc_trim(0)=1 rss_kib=1000->400 freed_kib=600'), lines)

    def test_a_trim_with_nothing_to_return_says_so(self):
        result, _lines = self.run_trim(SimpleNamespace(malloc_trim=lambda pad: 0))
        self.assertEqual(result, 'nothing')

    def test_a_failing_trim_never_raises(self):
        def boom(pad):
            raise OSError('no')

        result, lines = self.run_trim(SimpleNamespace(malloc_trim=boom))
        self.assertEqual(result, 'failed')
        self.assertIn('ignored', lines[-1])
        self.assertEqual(serving_startup.trim_host_heap(libc=SimpleNamespace(malloc_trim=boom), say=raising_say, rss=lambda: None), 'failed')
        self.assertEqual(serving_startup.trim_host_heap(libc=SimpleNamespace(malloc_trim=lambda pad: 1), say=raising_say, rss=lambda: None), 'failed',
                         'a log that fails after the trim is swallowed too')

    def test_a_c_library_without_malloc_trim_or_not_glibc_is_left_alone(self):
        result, _lines = self.run_trim(SimpleNamespace())
        self.assertEqual(result, 'unavailable')
        for version in ('', None, 'musl 1.2'):
            with patch('ctypes.CDLL', side_effect=AssertionError('no libc is loaded off glibc')):
                lines = []
                self.assertEqual(serving_startup.trim_host_heap(say=lines.append, glibc=version or '', rss=lambda: None), 'unavailable', version)
                self.assertTrue(lines[0].startswith('[PINDIAG] host heap trim skipped: not glibc'), lines)

    def test_warmup_trims_once_after_the_attach(self):
        order = []
        worker_base = ModuleType('vllm.v1.worker.worker_base')
        worker_base.CompilationTimes = lambda **times: SimpleNamespace(**times)
        with patch.dict(sys.modules, {'vllm.v1.worker.worker_base': worker_base}), \
                patch.object(serving_startup, 'prefix_warm', side_effect=lambda worker: order.append('warm')), \
                patch.object(serving_startup, 'start', side_effect=lambda worker: order.append('start')), \
                patch.object(serving_startup, 'trim_host_heap', side_effect=lambda: order.append('trim') or 'trimmed'):
            times = serving_startup.warmup(SimpleNamespace())
        self.assertEqual(order, ['warm', 'start', 'trim'])
        self.assertEqual(times.encoder, 0.0)
        self.assertGreaterEqual(times.language_model, 0.0)

    def test_a_failed_attach_is_not_trimmed_over(self):
        worker_base = ModuleType('vllm.v1.worker.worker_base')
        worker_base.CompilationTimes = lambda **times: SimpleNamespace(**times)
        with patch.dict(sys.modules, {'vllm.v1.worker.worker_base': worker_base}), \
                patch.object(serving_startup, 'prefix_warm'), \
                patch.object(serving_startup, 'start', side_effect=ValueError('attach refused')), \
                patch.object(serving_startup, 'trim_host_heap') as trim:
            with self.assertRaises(ValueError):
                serving_startup.warmup(SimpleNamespace())
        trim.assert_not_called()


def raising_say(line):
    raise RuntimeError('the log is gone too')


class PackTests(unittest.TestCase):
    """The ship pack make_fusion_jobs.py generates for the entry: B0 bakes this profile, GX0 is the ship-6 control, SR replays this profile by the image default."""

    def values(self, name):
        return job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8'))

    def test_the_pack_is_what_the_generator_writes(self):
        wanted = jobs_gen.generate(PLAN, PACK.parent)
        mine = {name: text for name, text in wanted.items() if name.startswith(PACK.name + '/')}
        self.assertEqual(sorted(mine), sorted(['%s/%s.env' % (PACK.name, name) for name in PACK_JOBS] + ['%s/ORDER.txt' % PACK.name]))
        for name, text in mine.items():
            self.assertEqual((PACK.parent / name).read_text(encoding='utf-8'), text, name)

    def test_the_jobs_name_this_profile_its_control_and_the_one_image(self):
        for name in PACK_JOBS:
            self.assertEqual(self.values(name)['C2_IMAGE_TAG'], 'tp4-s6nt-1', name)
        self.assertEqual(self.values('B0-ship-build')['C2_BAKE_DEFAULT_PROFILE'], NOTIER)
        self.assertEqual(self.values('B0-ship-build')['C2_BAKE_GRAFT'], 'K64j-OQ')
        self.assertEqual(self.values('GX0-exact-control')['C2_PROFILE'], SHIP)
        self.assertEqual(self.values('GX1-exact-ship')['C2_PROFILE'], NOTIER)
        self.assertEqual(self.values('SR-platform-replay')['C2_REPLAY_PROFILE'], NOTIER)

    def test_every_template_exits_zero_through_the_reader_and_the_build_waits_for_the_approval(self):
        for name in PACK_JOBS:
            with self.subTest(name):
                result = subprocess.run([sys.executable, '-s', str(READER), str(PACK / (name + '.env')), str(PROFILES_PATH)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        cwd=str(HERE.parent.parent))
                if name == 'B0-ship-build' and pending():
                    self.assertNotEqual(result.returncode, 0, 'a PENDING decision is never baked')
                else:
                    self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace'))


if __name__ == '__main__':
    unittest.main()
