"""tp4/s7c: the ship-7 candidate (a traffic profile generated from fusion-wp/S7.json by make_fusion_profiles.py), its gate-only audited twin (make_s7_profiles) and the contract's admission of the
tp4/sched-2 switch the candidate carries (the ledger trim).

  -fx-traffic-s7   the ship-6 profile + the host tier's threaded upload (QWEN_PREFIX_TIER_IO_THREADS, QWEN_PREFIX_TIER_VERIFY_THREADS) + the memory-ledger trim (QWEN_FAST_ADMISSION_LEDGER_TRIM) + prompt
                   lookup (QWEN_FAST_LOOKUP_DRAFT), under ONE owner traffic waiver (decision PENDING or APPROVED: nothing here reads it);
  -fx-s7-audit     its audited twin: the ship-6 audited twin + every difference of the candidate except the ledger trim.

THE TRACED MASKED TAILS (QWEN_FAST_PREFILL_TAIL_TRACE, tp4/prefill-3) ARE OUT of both: the tp4/s7 window's audited attaches died at engine start on a read inside the trace capture (the oneq SDPA audit; docs/tp4-s7.md),
and the second audited twin (-fx-s7-audit-mid, the other two traced buckets) is retired with them (TraceTests).

ADMISSION PIPELINING (QWEN_FAST_ADMISSION_SEED_FIRST, tp4/sched-2) IS OUT of both too (tp4/s7c, SeedFirstTests): its first audited attach died at engine start on a schedule wrapper that hid Lever N's mark (fixed
in admission_pipeline.py), and a review then found that its design cannot work with vLLM's engine loop (post_step() asks for the draft tickets before the next schedule(), where the deferred bind completes).

The ship-6 profile, its waiver and every profile that existed before this branch stay byte-identical; nothing here is hand-edited. The control of every timed pair is the ship-6 profile itself. The tests are
decision-agnostic: the owner's approval changes one sentence of the manifest and these tests still pass (ApprovalTests)."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import admission_pipeline  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_pf3_profiles as pf3  # noqa: E402
import make_s5_profiles as s5  # noqa: E402
import make_s6_profiles as s6  # noqa: E402
import make_s7_profiles as twins  # noqa: E402
import prefill_tail_trace  # noqa: E402
import profile_twins  # noqa: E402
import prompt_lookup  # noqa: E402
import qwen_tier_upload  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
NS = fusion.NAMESPACE
SHIP6, SHIP, AUDIT6, AUDIT = s6.SHIP, twins.SHIP, s6.AUDIT, twins.AUDIT
PLAN = fusion.normalise(fusion.read_manifests())
SPEC = next(item for item in PLAN['ship'] if item['name'] == 'traffic-s7')
PARENT_COMMIT = '43c60a52'                      # tp4/s6 at its approval: the tree this branch was cut from
# what the candidate sets that ship 6 does not, the waiver id aside
CHANGES = {'QWEN_PREFIX_TIER_IO_THREADS': '4', 'QWEN_PREFIX_TIER_VERIFY_THREADS': '4', 'QWEN_FAST_ADMISSION_LEDGER_TRIM': '1', 'QWEN_FAST_LOOKUP_DRAFT': 'n3m12'}
SEED_FIRST = 'QWEN_FAST_ADMISSION_SEED_FIRST'
# what the four branches carry that stays OUT of the candidate (default off, a gate's own or another waiver's)
LEFT_OUT = ('QWEN_FAST_ADMISSION_HOLD_REUSE', 'QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_AGE_MS', 'QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_REUSES', 'QWEN_FAST_ADMISSION_HOLD_REUSE_DRIFT_MB',
            'QWEN_FAST_SDPA_PF_KSPLIT', 'QWEN_FAST_SDPA_PF_KSPLIT_AUDIT', 'QWEN_FAST_LOOKUP_SHADOW', 'QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS',
            'QWEN_FAST_PREFILL_TAIL_TRACE', 'QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT', 'QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS', 'QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB', SEED_FIRST)


def boot_problems(profile):
    from test_octo3_contract import boot_problems as every_check

    return (every_check(profile) + contract.prefix_policy_problems(profile) + contract.exact_resume_problems(profile) + contract.sched_problems(profile) + contract.sched2_problems(profile)
            + contract.ksplit_problems(profile))


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT_COMMIT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))['profiles']


class CandidateTests(unittest.TestCase):
    def test_the_candidate_is_a_generated_traffic_profile_and_no_gate_twin(self):
        self.assertEqual(SHIP, NS + 'traffic-s7')
        self.assertIn(SHIP, fusion.ship_names())
        self.assertNotIn(SHIP, fusion.twin_names())
        self.assertNotIn(SHIP, profile_twins.twin_names())
        self.assertIn(SHIP, profile_twins.generated_names())
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        self.assertEqual(regenerated[SHIP], PROFILES[SHIP])
        self.assertNotIn('gate_only', PROFILES[SHIP])
        self.assertEqual((SPEC['pack'], SPEC['image'], SPEC['control'], SPEC['bake_graft'], SPEC['memory_plan']), ('S7', 'tp4-s7-1', 'traffic-s6', 'K64j-OQ', 'octo'))

    def test_the_candidate_is_the_ship_6_profile_plus_exactly_the_three_groups_of_levers(self):
        old, new = PROFILES[SHIP6], PROFILES[SHIP]
        changed = {key: new['env'].get(key) for key in set(old['env']) | set(new['env']) if old['env'].get(key) != new['env'].get(key)}
        self.assertEqual(changed, dict(CHANGES, **{contract.OCTO_WAIVER_ENV: PROFILES[SHIP][contract.TRAFFIC_WAIVER]['id']}))
        for key in old:
            if key not in ('env', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(new[key], old[key], key)
        self.assertEqual((new['engine']['num-gpu-blocks-override'], new['env']['QWEN36_MAX_TOKENS_ALL_USERS']), (16500, '1055488'), 'the memory plan is the parent\'s')
        self.assertEqual([key for key in sorted(old['env']) if key != contract.OCTO_WAIVER_ENV and new['env'].get(key) != old['env'][key]], [], 'no other line of ship 6 moved')

    def test_what_stays_out_is_out(self):
        env = PROFILES[SHIP]['env']
        for flag in LEFT_OUT:
            self.assertNotIn(flag, env, flag)
        self.assertNotIn('QWEN_FAST_ADMISSION_HOLD_REUSE', contract.NEXT_WAIVABLE_LEVERS, 'the hold reuse stays a gate\'s own until a ballast-boundary run has been read')
        self.assertEqual(contract.SCHED2_WAIVABLE_LEVERS, {'QWEN_FAST_ADMISSION_SEED_FIRST': '1', 'QWEN_FAST_ADMISSION_LEDGER_TRIM': '1'})
        self.assertEqual(sorted(set(contract.SCHED2_SWITCHES) - set(contract.SCHED2_WAIVABLE_LEVERS)), ['QWEN_FAST_ADMISSION_HOLD_REUSE'])

    def test_the_lever_values_are_the_modules_own(self):
        env = PROFILES[SHIP]['env']
        self.assertEqual(qwen_tier_upload.flag_problems(env), [])
        self.assertEqual(admission_pipeline.problems(env) if hasattr(admission_pipeline, 'problems') else [], [])
        self.assertFalse(admission_pipeline.seed_first_enabled(env), 'admission pipelining is out of the candidate')
        self.assertTrue(admission_pipeline.ledger_trim_enabled(env))
        self.assertIsNotNone(prompt_lookup.parse_policy(env['QWEN_FAST_LOOKUP_DRAFT']))
        for flag in prefill_tail_trace.FLAGS:
            self.assertNotIn(flag, env, 'the traced masked tails are out of the candidate: %s' % flag)
        self.assertEqual(pf3.parent_problems(SHIP, PROFILES[SHIP], False), [], 'the tail twin generator has no complaint about a candidate without the flag')

    def test_the_ship_6_profile_and_every_earlier_profile_are_untouched(self):
        parent = parent_profiles()
        if parent is None:
            self.skipTest('no git history for %s' % PARENT_COMMIT)
        for name, body in parent.items():
            self.assertEqual(PROFILES[name], body, name)
        added = [name for name in PROFILES if name not in parent]
        added = [name for name in added if name not in set(profile_twins.generated_names()) - set(twins.owned_names())]      # (the other branches' generated twins merged beside)
        self.assertEqual(sorted(added), sorted([SHIP, AUDIT]))
        self.assertTrue(PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual(PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['id'], 'ship6-fx-262k-2026-10-21')
        for flag in CHANGES:
            self.assertNotIn(flag, PROFILES[SHIP6]['env'], flag)

    def test_one_waiver_of_its_own_naming_the_ledger_trim_beside_ship_6s_levers(self):
        waiver = PROFILES[SHIP][contract.TRAFFIC_WAIVER]
        profile = dict(PROFILES[SHIP], name=SHIP)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        old = PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['levers']
        self.assertEqual(waiver['levers'], dict(old, QWEN_FAST_ADMISSION_LEDGER_TRIM='1'))
        self.assertNotIn(SEED_FIRST, waiver['levers'])
        self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
        self.assertEqual(contract.traffic_waiver_pending(profile), waiver['decision'].startswith('PENDING'))
        self.assertEqual(waiver['id'], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])
        self.assertEqual(waiver['id'], 'ship7-fx-262k-2026-10-21')
        self.assertNotEqual(waiver['id'], PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['id'])
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])
        self.assertEqual(sorted(waiver), ['decision', 'evidence', 'id', 'levers', 'waives'])
        ids = [body[contract.TRAFFIC_WAIVER]['id'] for body in PROFILES.values() if contract.TRAFFIC_WAIVER in body]
        self.assertEqual(len(ids), len(set(ids)), 'every waiver id is its own')

    def test_the_waiver_carries_the_ship_6_text_verbatim_and_says_what_it_adds_and_what_has_not_run(self):
        text, old = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives'], PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['waives']
        self.assertIn(old, text, 'the ship-6 waiver text is carried verbatim')
        added = text[text.index(old) + len(old):]
        for phrase in ('(S) The host tier\'s threaded upload', 'QWEN_PREFIX_TIER_IO_THREADS=4', '(T) Admission pipelining, QWEN_FAST_ADMISSION_SEED_FIRST (docs/tp4-sched2.md', 'is NOT IN THIS PROFILE and NOT WAIVED', '(U) The memory-ledger trim',
                       'QWEN_FAST_ADMISSION_LEDGER_TRIM=1', '(V) Prompt lookup', 'QWEN_FAST_LOOKUP_DRAFT=n3m12', 'ESTIMATE',
                       'Never run on a card', 'Switchable line', 'NOT IN THIS PROFILE', 'QWEN_FAST_ADMISSION_HOLD_REUSE', 'QWEN_FAST_SDPA_PF_KSPLIT', 'tree verify'):
            self.assertIn(phrase, added)
        self.assertNotIn('(W) ', added, 'items (S), (U) and (V) only (and the note (T)): the traced masked tails are not waived')
        self.assertNotIn('QWEN_FAST_PREFILL_TAIL_TRACE', added)
        self.assertNotIn('QWEN_FAST_ADMISSION_SEED_FIRST=1', added, 'item (T) is a note that the lever is out, not a waived setting')
        self.assertNotIn('Switchable line: QWEN_FAST_ADMISSION_SEED_FIRST', added)
        self.assertIn('(S), (U) and (V)', text)
        self.assertNotIn('(S) to (V)', text)
        evidence = ' '.join(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'])
        for phrase in ('tp4/s6 (the ship-6 candidate', 'tp4/upload-1 (CPU)', 'tp4/sched-2 (CPU)', 'tp4/stream-1 (CPU)', 'what has NOT run'):
            self.assertIn(phrase, evidence)
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'][:len(PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['evidence'])], PROFILES[SHIP6][contract.TRAFFIC_WAIVER]['evidence'])

    def test_no_gate_instrument_is_in_the_candidate(self):
        env = PROFILES[SHIP]['env']
        self.assertNotIn('QWEN_C2_GATE_PROFILE', env)
        for key in contract.OCTO_GATE_ONLY:
            if key != 'QWEN_FAST_SOLO_PACKED':          # the one gate name the waiver admits (NEXT_WAIVABLE_LEVERS)
                self.assertNotIn(key, env, key)
        for key, value in env.items():
            self.assertFalse(key.endswith('_AUDIT') and value == '1' and key not in PROFILES[fusion.PARENT]['env'], key)
        self.assertEqual((env['QWEN_PREFIX_HOST_TIER_AUDIT'], env['QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'], env['QWEN_PREFIX_HOST_TIER_VERIFY']), ('0', '0', 'sample'))

    def test_the_text_of_the_profile_is_public_safe(self):
        text = json.dumps([PROFILES[SHIP]['description'], PROFILES[SHIP][contract.TRAFFIC_WAIVER], PROFILES[AUDIT]['description']])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)
        self.assertTrue(PROFILES[SHIP]['description'].startswith('A SHIP CANDIDATE'))

    def test_the_candidate_passes_every_boot_check(self):
        self.assertEqual(boot_problems(dict(PROFILES[SHIP], name=SHIP)), [])


class ContractTests(unittest.TestCase):
    """What the contract admits on a traffic profile now that a waiver may name the ledger trim (and, in the contract, seed first: the lever's code stays in the tree, but the candidate does not carry it)."""

    def profile(self, **edits):
        profile = copy.deepcopy(dict(PROFILES[SHIP], name=SHIP))
        for key, value in edits.items():
            if value is None:
                profile['env'].pop(key, None)
            else:
                profile['env'][key] = value
        return profile

    def test_the_two_switches_are_waivable_at_one_and_nowhere_else(self):
        for flag in ('QWEN_FAST_ADMISSION_SEED_FIRST', 'QWEN_FAST_ADMISSION_LEDGER_TRIM'):
            self.assertEqual(contract.NEXT_WAIVABLE_LEVERS[flag], '1')
            profile = self.profile()
            profile[contract.TRAFFIC_WAIVER]['levers'][flag] = '0'
            self.assertTrue(any(flag in problem for problem in contract.traffic_waiver_problems(profile)))

    def test_a_switch_the_waiver_does_not_name_is_refused_on_a_traffic_profile(self):
        for flag in ('QWEN_FAST_ADMISSION_SEED_FIRST', 'QWEN_FAST_ADMISSION_LEDGER_TRIM'):
            profile = self.profile(**{flag: '1'})
            profile[contract.TRAFFIC_WAIVER]['levers'].pop(flag, None)
            problems = boot_problems(profile)
            self.assertTrue(any(flag in problem for problem in problems), (flag, problems))
            ship6 = copy.deepcopy(dict(PROFILES[SHIP6], name=SHIP6))
            ship6['env'][flag] = '1'
            self.assertTrue(any(flag in problem for problem in contract.sched2_problems(ship6)), 'the ship-6 waiver does not name it')

    def test_the_hold_reuse_is_refused_on_a_traffic_profile_even_beside_the_waiver(self):
        problems = boot_problems(self.profile(QWEN_FAST_ADMISSION_HOLD_REUSE='1', QWEN_FAST_EXTENT_REPLAY='1'))
        self.assertTrue(any('QWEN_FAST_ADMISSION_HOLD_REUSE' in problem for problem in problems), problems)
        self.assertTrue(boot_problems(self.profile(QWEN_FAST_ADMISSION_HOLD_REUSE_DRIFT_MB='256')))

    def test_seed_first_needs_engine_reuse_the_lean_admission_and_lever_n(self):
        for flag in ('QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_ADMISSION_LEAN', 'QWEN_FAST_LEVER_N'):
            edited = self.profile(**{'QWEN_FAST_ADMISSION_SEED_FIRST': '1', flag: None})
            self.assertTrue(any('QWEN_FAST_ADMISSION_SEED_FIRST' in problem for problem in contract.sched2_problems(edited)), flag)
        self.assertEqual([problem for problem in contract.sched2_problems(self.profile()) if 'SEED_FIRST' in problem], [], 'the candidate does not set it, so nothing is asked of it')

    def test_the_upload_levers_are_refused_without_the_tier_and_the_call_size_stays_a_gates_own(self):
        self.assertTrue(boot_problems(self.profile(QWEN_PREFIX_HOST_TIER_GIB=None)))
        self.assertTrue(boot_problems(self.profile(QWEN_PREFIX_TIER_IO_THREADS='0', QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS='128')))
        self.assertTrue(boot_problems(self.profile(QWEN_PREFIX_TIER_IO_THREADS='99')))

    def test_the_k_split_is_refused_on_the_candidate_unless_a_waiver_names_it(self):
        problems = boot_problems(self.profile(QWEN_FAST_SDPA_PF_KSPLIT='1'))
        self.assertTrue(any('QWEN_FAST_SDPA_PF_KSPLIT' in problem for problem in problems), problems)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in (AUDIT,):
            self.assertIs(PROFILES[name]['gate_only'], True, name)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertNotIn(contract.OCTO_WAIVER_ENV, PROFILES[name]['env'])
            self.assertEqual(PROFILES[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
            self.assertIn(name, profile_twins.twin_names())
            self.assertLessEqual(len(name), twins.MAX_NAME)
        self.assertEqual(twins.twin_names(), (AUDIT,))
        self.assertEqual(twins.owned_names(), (AUDIT, SHIP))
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_twin_is_the_ship_6_audited_twin_plus_the_candidates_levers_but_not_the_ledger_trim(self):
        parent = PROFILES[AUDIT6]
        audit = PROFILES[AUDIT]
        changed = {key: audit['env'].get(key) for key in set(parent['env']) | set(audit['env']) if parent['env'].get(key) != audit['env'].get(key)}
        want = dict((flag, value) for flag, value in CHANGES.items() if flag != 'QWEN_FAST_ADMISSION_LEDGER_TRIM')
        self.assertEqual(changed, want)
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['engine'], PROFILES[SHIP]['engine'], 'the audited twin has the candidate\'s memory plan')

    def test_the_twins_need_no_owner_waiver_and_there_is_no_timed_twin(self):
        names = [name for name in PROFILES if name.startswith(NS) and 's7' in name]
        self.assertEqual(sorted(names), sorted([SHIP, AUDIT]), 'the timed arm is the candidate itself, its control the ship-6 profile')

    def test_a_twin_is_refused_when_its_parent_moves(self):
        for mutate in (lambda data: data['profiles'][AUDIT6]['env'].__setitem__('QWEN_FAST_PREFILL_TAIL_TRACE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_FAST_PREFILL_TAIL_TRACE', '1'),
                       lambda data: data['profiles'][AUDIT6]['env'].__setitem__(SEED_FIRST, '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__(SEED_FIRST, '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop(contract.OCTO_WAIVER_ENV),
                       lambda data: data['profiles'][AUDIT6].pop('gate_only'),
                       lambda data: data['profiles'].pop(AUDIT6)):
            data = copy.deepcopy(DATA)
            mutate(data)
            with self.assertRaises(ValueError):
                twins.generate(data)

    def test_the_generators_of_the_chain_commute(self):
        import make_pf3k_profiles
        import make_sched2_profiles

        data = copy.deepcopy(DATA)
        for module in (twins, make_pf3k_profiles, make_sched2_profiles, s6, twins):
            data = module.generate(data)
        self.assertEqual(list(data['profiles']), list(DATA['profiles']), 'every generator places its twins by the chain, so the order they run in does not matter')


class TraceTests(unittest.TestCase):
    """The traced masked tails (tp4/prefill-3) are out of the s7 chain, and the second audited twin with them."""

    def test_neither_the_candidate_nor_its_twin_nor_the_manifest_names_a_traced_tails_knob(self):
        raw = json.loads((fusion.MANIFESTS / 'S7.json').read_text(encoding='utf-8'))['ship'][0]
        for flag in prefill_tail_trace.FLAGS:
            for name in (SHIP, AUDIT):
                self.assertNotIn(flag, PROFILES[name]['env'], (name, flag))
            self.assertNotIn(flag, raw['env'], flag)
            self.assertNotIn(flag, json.dumps(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives']), 'the waiver does not waive the lever')

    def test_the_second_audited_twin_is_retired_and_the_generator_drops_a_stale_copy(self):
        retired = NS + 's7-audit-mid'
        self.assertEqual(twins.RETIRED, (retired,))
        self.assertNotIn(retired, PROFILES)
        stale = copy.deepcopy(DATA)
        stale['profiles'][retired] = copy.deepcopy(PROFILES[AUDIT])
        self.assertEqual(twins.generate(stale), DATA, 'a previous generation\'s second twin is dropped')

    def test_the_decision_sentence_names_no_second_audited_twin(self):
        decision = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision']
        self.assertNotIn('-s7-audit-mid', decision)


class SeedFirstTests(unittest.TestCase):
    """Admission pipelining (QWEN_FAST_ADMISSION_SEED_FIRST, tp4/sched-2) is out of the s7 chain: neither profile, the manifest nor the waiver's levers carry it, and the twin generator refuses a parent that does."""

    def test_neither_the_candidate_nor_its_twin_nor_the_manifest_nor_the_waiver_names_it(self):
        raw = json.loads((fusion.MANIFESTS / 'S7.json').read_text(encoding='utf-8'))['ship'][0]
        for name in (SHIP, AUDIT):
            self.assertNotIn(SEED_FIRST, PROFILES[name]['env'], name)
        self.assertNotIn(SEED_FIRST, raw['env'])
        self.assertNotIn(SEED_FIRST, raw['waiver']['levers'])
        self.assertNotIn(SEED_FIRST, PROFILES[SHIP][contract.TRAFFIC_WAIVER]['levers'])
        self.assertNotIn(SEED_FIRST, PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'])
        self.assertNotIn(SEED_FIRST, twins.candidate_changes(PROFILES))
        for name in (SHIP, AUDIT):
            self.assertEqual([problem for problem in boot_problems(dict(PROFILES[name], name=name)) if 'SEED_FIRST' in problem], [], name)

    def test_the_ledger_trim_stays_in_the_candidate_and_out_of_the_twin(self):
        self.assertEqual(PROFILES[SHIP]['env']['QWEN_FAST_ADMISSION_LEDGER_TRIM'], '1')
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['levers']['QWEN_FAST_ADMISSION_LEDGER_TRIM'], '1')
        self.assertNotIn('QWEN_FAST_ADMISSION_LEDGER_TRIM', PROFILES[AUDIT]['env'])
        self.assertEqual(contract.sched2_problems(dict(PROFILES[SHIP], name=SHIP)), [], 'the trim rides on no other sched-2 switch')

    def test_the_waiver_says_why_it_is_out_and_it_is_the_notes_only_mention(self):
        text = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives']
        note = text[text.index('(T) Admission pipelining'):text.index('(U) The memory-ledger trim')]
        for phrase in ('NOT IN THIS PROFILE and NOT WAIVED', 'take_draft_token_ids', 'Live prepared request ticket required before scheduler admission', 'functools.update_wrapper', '39649281798'):
            self.assertIn(phrase, note)
        self.assertEqual(text.count('QWEN_FAST_ADMISSION_SEED_FIRST'), 1)

    def test_the_twin_generator_refuses_a_parent_that_names_it(self):
        for holder in (AUDIT6, SHIP):
            data = copy.deepcopy(DATA)
            data['profiles'][holder]['env'][SEED_FIRST] = '1'
            self.assertTrue([problem for problem in twins.parent_problems(data['profiles']) if SEED_FIRST in problem and holder in problem], holder)
            with self.assertRaisesRegex(ValueError, SEED_FIRST):
                twins.generate(data)
        self.assertEqual(twins.parent_problems(PROFILES), [])

    def test_the_stock_files_are_the_generators_without_it(self):
        self.assertEqual(twins.generate(DATA), DATA)
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        self.assertEqual(regenerated[SHIP], PROFILES[SHIP])


class ApprovalTests(unittest.TestCase):
    def test_approving_the_candidate_changes_its_decision_sentence_and_nothing_else(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner (remote control): the ship-7 candidate', directory, SPEC['name'])
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['S7.json'])
        old, new = json.loads(before['S7.json']), json.loads(after['S7.json'])
        new['ship'][0]['waiver']['decision'] = old['ship'][0]['waiver']['decision']
        self.assertEqual(new, old)
        plan = fusion.normalise(fusion.read_manifests(directory))
        generated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles']
        self.assertFalse(contract.traffic_waiver_pending(dict(generated[SHIP], name=SHIP)))
        self.assertEqual({key: value for key, value in generated[SHIP].items() if key != contract.TRAFFIC_WAIVER}, {key: value for key, value in PROFILES[SHIP].items() if key != contract.TRAFFIC_WAIVER})
        self.assertEqual(boot_problems(dict(generated[SHIP], name=SHIP)), [])
        for name in fusion.ship_names():
            if name != SHIP:
                self.assertEqual(generated[name], PROFILES[name], name)
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'] = generated
        self.assertEqual(twins.generate(data)['profiles'][AUDIT], PROFILES[AUDIT])


class ManifestTests(unittest.TestCase):
    def test_the_manifest_keeps_the_pack_folder_and_lists_its_tests(self):
        raw = json.loads((fusion.MANIFESTS / 'S7.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['wp'], 'S7')
        self.assertEqual(raw['branch'], 'tp4/s7')
        self.assertEqual(len(raw['ship']), 1)
        self.assertEqual(set(raw) - {'wp', 'branch', 'description', 'tests', 'ship'}, set())
        for name in ('test_s7_profiles', 'test_s7_pack', 'test_sched2_pipeline', 'test_sched2_model', 'test_sched2_profiles', 'test_sched2_pack'):
            self.assertIn(name, raw['tests'])

    def test_the_manifest_is_in_the_canonical_form_approve_ship_rewrites(self):
        text = (fusion.MANIFESTS / 'S7.json').read_text(encoding='utf-8')
        self.assertEqual(json.dumps(json.loads(text), indent=1) + '\n', text)

    def test_the_ship_entry_is_the_ship_6_entry_plus_the_lever_lines(self):
        old = json.loads((fusion.MANIFESTS / 'S6.json').read_text(encoding='utf-8'))['ship'][0]
        new = json.loads((fusion.MANIFESTS / 'S7.json').read_text(encoding='utf-8'))['ship'][0]
        self.assertEqual({key: value for key, value in new['env'].items() if old['env'].get(key) != value}, dict(CHANGES, QWEN_FAST_OCTO_WAIVER='ship7-fx-262k-2026-10-21'))
        self.assertEqual([key for key in old['env'] if key not in new['env']], [])
        for key in ('from', 'memory_plan', 'bake_graft'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['control'], 'traffic-s6')


if __name__ == '__main__':
    unittest.main()
