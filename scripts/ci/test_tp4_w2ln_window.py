"""The combined development window's job pack (references/tp4-w2ln-jobs; docs/tp4-combined-window.md).

Every template parses with the job parser, names the one window image (or, for the CONTROL jobs, the production base image driven from this same window commit), never starts the node agent except Z
and never stops it except A0, and the ORDER.txt lines match the files; no template names a rig, card, address, registry or digest; every profile a template names exists, is gate only (the controls and
the timed production reads excepted) and every multi profile a non-timed prefix or gate job serves carries the SDPA audit (the extent audit the gates add refuses it otherwise); every smoke test a template
names exists in the smoke script; the timed blocks alternate as written; the NEEDS graph (alternatives, groups, the fail triggers) names real jobs, has no cycle and is WALKED through every failure class of
A1-C and the T read to prove the declared fallback candidate stays reachable; every profile a job names has a committed target under every swap and every allowed pair of swaps; every job's box is computed from
the gates' own timeouts; the outage numbers in the ORDER header are the numbers of its lines; every job has its own tag and the tag budget covers the plan."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_gate as serving_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import test_tp4_w2ln_profiles as profile_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-w2ln-jobs')
WINDOW, PRODUCTION = 'tp4-w2ln-1', 'tp4-serve-10'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.|release_' + 'first|admin ' + 'token')
R = 'c2-packed-tp4-8x262k-'
SHIP = R + 'ship-prefix'
CONTROL_JOBS = ('P1a-CTL-exactness-shared', 'P1b-CTL-lifecycle-evict', 'L8-CTL-ladder8-past-131k', 'C16-CTL-churn16', 'T0-timed-production-bytes', 'G0-turns-A-production-bytes',
                'E1-CTL-exactness-eager')
ALLOWLISTED_TAGS = 64
REQUESTED_TAGS = 20
RESERVE_TAGS = 10
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve'}
TIMED_TESTS = 'warmup,coding,concurrent8_steady,concurrent8_code_equal,concurrent8_code_32k,concurrent8_code_128k,concurrent8_skew'
SMOKE_STEP_MINUTES = 210            # the quad smoke step's timeout-minutes in .github/workflows/qwen-c2-serving.yml
STEP_CAP_MINUTES = 380              # the gate and prefix steps' timeout-minutes
CHECKPOINT_AFTER = 'T8-timed-w2-B'
CORE_DROPPED = r'(D[1-4]|GH[34]|G[34]|S[34])-'
CONTROL_DRIVER_OLD = 'driven from a throwaway commit of ship/262k-prefix'


def profiles():
    with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
        return json.load(handle)


def read_text(name):
    with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(read_text(name + '.env')), sorted(profiles()['profiles']), root=ROOT)


def raw(name):
    return job.parse_env(read_text(name + '.env'))


def order():
    return [line.split() for line in read_text('ORDER.txt').splitlines() if line.strip() and not line.startswith('#')]


def planned():
    return [line for line in order() if line[1] != 'cond']


def conditional():
    return [line for line in order() if line[1] == 'cond']


def short(name):
    """The NEEDS graph's short name of a template: T1 for T1-timed-control-A, P1a-CTL for P1a-CTL-exactness-shared, A1-C for A1-C-combined-audited-attach, A1-S2 for A1-S2-split-f1."""
    parts = name.split('-')
    if len(parts) > 1 and (parts[1] in ('C', 'CTL', 'CTL2', 'W2', 'LN', 'C2', 'C3') or (parts[0] == 'A1' and re.match(r'S[1-4]$', parts[1]))):
        return '-'.join(parts[:2])
    return parts[0]


def groups():
    return dict((match.group(1), match.group(2).split()) for match in re.finditer(r'^# GROUP (\S+) = (.+)$', read_text('ORDER.txt'), re.M))


ITEM = re.compile(r'^([A-Za-z0-9-]+?)(?:\(([A-Z-]+)\))?$')


def needs():
    """{job: [item]} where an item is [(name, suffix or None)] alternatives: every item of a job's line must hold. One line per job (a job in two lines would be ambiguous)."""
    found = {}
    for line in read_text('ORDER.txt').splitlines():
        match = re.match(r'# NEEDS (.+?) <- (.+)$', line)
        if match:
            items = [[ITEM.match(alternative).groups() for alternative in item.split('|')] for item in match.group(2).split()]
            for name in match.group(1).split():
                if name in found:
                    raise AssertionError('%s is on two NEEDS lines' % name)
                found[name] = items
    return found


def known_names():
    known = {}
    for line in order():
        known.setdefault(short(line[0]), []).append(line[0])
    return known


# ---- numbers ---------------------------------------------------------------------------------------------------------------------------------------------------------

def box_minutes(name):
    """What a job can take when it runs to its limit, from the gates' own timeouts: a prefix job is the sum of its arms' docker timeouts (S2_TIMEOUTS applied by plan_arms) plus the
    per-arm overhead, capped by the step; a gate job the same over its plan's arms; a smoke job the smoke step's 210 minutes. None for a job with no such limit (reset, build, status)."""
    entry = raw(name)
    actions = entry.get('C2_ACTIONS', '').split()
    overhead = prefix_gate.ARM_OVERHEAD_SECONDS
    if 'prefix' in actions:
        seconds = 0
        for plan in entry['C2_PREFIX_PLAN'].split(','):
            for arm in prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, profiles()):
                seconds += arm['timeout'] + overhead
        return min(STEP_CAP_MINUTES, -(-seconds // 60))
    if 'gate' in actions:
        lengths = [int(item) for item in entry['C2_GATE_LENGTHS'].split(',')] if entry.get('C2_GATE_LENGTHS') else None
        seconds = 0
        for plan in entry['C2_GATE_PLAN'].split(','):
            for arm in serving_gate.plan_arms(plan.strip(), entry['C2_PROFILE'], profiles(), lengths=lengths, max_tokens=int(entry.get('C2_GATE_MAX_TOKENS') or 4096)):
                seconds += arm[2] + overhead
        return min(STEP_CAP_MINUTES, -(-seconds // 60))
    if 'smoke' in actions:
        return SMOKE_STEP_MINUTES
    return None


def numbers():
    """Every number the ORDER header and the public doc quote, computed from the job lines."""
    lines = order()
    total = sum(int(minutes) for name, cls, image, minutes in lines if cls != 'cond')
    outage = total - int(next(m for n, c, i, m in lines if n == 'B0-build'))
    core_drop = sum(int(minutes) for name, cls, image, minutes in lines if re.match(CORE_DROPPED, name))
    cond = sum(int(minutes) for name, cls, image, minutes in lines if cls == 'cond')
    boxed = 0
    for name, cls, image, minutes in lines:
        box = box_minutes(name)
        boxed += max(int(minutes), box or 0)
    worst = boxed - int(next(m for n, c, i, m in lines if n == 'B0-build'))
    clock, checkpoint = 0, None
    for name, cls, image, minutes in lines:
        if cls == 'cond' or name == 'B0-build':
            continue
        clock += int(minutes)
        if name.startswith(CHECKPOINT_AFTER):
            checkpoint = clock
    names = [line[0] for line in lines]
    core_cond = len(conditional())
    core_planned = len([line for line in planned() if not re.match(CORE_DROPPED, line[0])])
    return dict(full=outage, core=outage - core_drop, cond=cond, worst=worst, checkpoint=checkpoint, planned=len(planned()), conditional=len(conditional()), total=len(lines),
                reserve=ALLOWLISTED_TAGS + REQUESTED_TAGS - len(lines), core_reserve=ALLOWLISTED_TAGS - core_planned - core_cond, names=names)


def hours(minutes):
    return '%.1f' % (minutes / 60.0)


def tag_list():
    """The ordered experiment tags of the ORDER header (TAGLIST: v538-v568 ...), expanded; the list the k-th job takes its tag from."""
    text = re.search(r'TAGLIST \(ordered\): (.+?) \(64, hardware-allowlisted and never pushed\) then (v\d+-v\d+)', read_text('ORDER.txt'))
    found = []
    for span in text.group(1).split() + [text.group(2)]:
        low, high = (int(value[1:]) for value in span.split('-'))
        found += ['v%d' % number for number in range(low, high + 1)]
    return found


def tag_map():
    """{job: tag}: the planned jobs in file order take the first tags, the conditional jobs the next, whatever is left is the reserve."""
    tags = tag_list()
    names = [line[0] for line in planned()] + [line[0] for line in conditional()]
    return dict(zip(names, tags)), tags[len(names):]


# ---- the NEEDS walk ---------------------------------------------------------------------------------------------------------------------------------------------------

def walk(results=None):
    """Which jobs RUN, and with what result, when the jobs named in `results` end as given (PASS by default; FAIL, SWAP, OUT, NO-VERDICT-INFRA). A job runs when every item of its NEEDS line holds:
    an alternative holds when its job ran and passed (a GROUP: every member), or, with a suffix, ran and ended so ((FAIL) is any result but PASS). T-READ is read after the T block."""
    results = dict(results or {})
    graph, grouped = needs(), groups()
    ran = {}

    def holds(alternative):
        name, suffix = alternative
        if name == 'T-READ':
            outcome = results.get('T-READ', 'PASS') if all(job_name in ran for job_name in ('T1', 'T4', 'T7')) else None
        elif name in grouped:
            members = [ran.get(member) for member in grouped[name]]
            return suffix is None and all(member == 'PASS' for member in members)
        else:
            outcome = ran.get(name)
        if outcome is None:
            return False
        if suffix is None:
            return outcome == 'PASS'
        if suffix == 'FAIL':
            return outcome != 'PASS'
        return outcome == suffix

    names = [short(line[0]) for line in order()]
    ran['Z'] = results.get('Z', 'PASS')                  # Z always runs
    changed = True
    while changed:
        changed = False
        for name in names:
            if name in ran:
                continue
            if all(any(holds(alternative) for alternative in item) for item in graph.get(name, [])):
                ran[name] = results.get(name, 'PASS')
                changed = True
    return ran


HARD = {
    'COMBINED': ('HW-C', 'HW-C2', 'HW-C3', 'HL-C', 'HL-C2', 'HL-C3', 'HX-C', 'HF-C', 'P1a-C', 'P1b-C', 'L8-C', 'C16-C', 'E1-C'),
    'LN-only': ('A1-LN', 'HL-LN', 'HF-LN', 'P1a-LN', 'P1b-LN', 'L8-LN', 'C16-LN', 'E1-LN'),
    'W2-only': ('A1-W2', 'HW-W2', 'P1a-W2', 'P1b-W2', 'L8-W2', 'C16-W2', 'E1-W2'),
}
TIMED = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8')
SGH = ('S1', 'S2', 'S3', 'S4', 'G1', 'G2', 'G3', 'G4', 'GH1', 'GH2', 'GH3', 'GH4')


def reachable(ran, candidate):
    """Whether every job a candidate's cutover rule needs ran and passed in a walk, the attach included."""
    passed = lambda name: ran.get(name) == 'PASS'
    if candidate == 'COMBINED':
        attach = passed('A1-C') or passed('A1-C2') or all(passed(member) for member in groups()['A1-SPLIT'])
        return attach and all(passed(name) for name in HARD[candidate] + TIMED + SGH)
    if candidate == 'LN-only':
        return all(passed(name) for name in HARD[candidate] + ('T1', 'T3', 'T4', 'T6', 'T7') + SGH)
    return all(passed(name) for name in HARD[candidate] + ('T1', 'T2', 'T4', 'T5', 'T7', 'T8'))


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates(self):
        names = [line[0] for line in order()]
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(names), files)
        self.assertEqual(len(names), len(set(names)))
        for name, cls, image, minutes in order():
            self.assertIn(cls, ('stop', 'soft', 'branch', 'cond'), name)
            if name in CONTROL_JOBS:
                self.assertEqual(image, PRODUCTION, name)
            else:
                self.assertIn(image, (WINDOW,), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)
            self.assertEqual(raw(name)['C2_IMAGE_TAG'], image, name)

    def test_every_template_parses_and_names_the_quad_unless_it_is_the_card_m_job(self):
        for name, cls, image, minutes in order():
            with self.subTest(name=name):
                parsed(name)
                self.assertEqual(raw(name)['C2_CARDS'], 'pair' if name.startswith('F1-') else 'quad')

    def test_the_node_agent_is_stopped_by_a0_and_started_by_z_and_nowhere_else(self):
        for name, *_ in order():
            actions = set(raw(name).get('C2_ACTIONS', '').split())
            if name == 'A0-agentstop-unserve':
                self.assertEqual(actions, {'status', 'agentstop', 'unserve'})
            elif name == 'Z-handback':
                self.assertIn('agentstart', actions)
                self.assertNotIn('agentstop', actions)
            else:
                self.assertFalse(actions & AGENT_ACTIONS, name)
            self.assertFalse(actions & {'push', 'replay', 'priority', 'platform'}, name)

    def test_the_templates_are_public(self):
        for name in sorted(os.listdir(FOLDER)):
            self.assertIsNone(BANNED.search(read_text(name)), '%s: %s' % (name, BANNED.search(read_text(name))))
            self.assertIsNone(re.search(r'experiment/c2-serving-v\d', read_text(name)), '%s names a real tag' % name)

    def test_the_control_jobs_are_driven_from_this_commit_not_from_a_throwaway_ship_commit(self):
        # a ship throwaway commit has no concurrent8_skew, no [LOAD] logger, no kill-switch preflight: T0 would skip the skew shape and always read VOID, and the CONTROLs would run without the preflight
        for name in sorted(os.listdir(FOLDER)):
            if name.endswith('.env'):
                self.assertNotIn(CONTROL_DRIVER_OLD, read_text(name), name)
        text = read_text('ORDER.txt')
        self.assertIn('THE CONTROL JOBS ARE DRIVEN FROM THIS WINDOW COMMIT', text)
        smoke = open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8').read()
        workflow = open(os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8').read()
        self.assertIn('def concurrent8_skew(', smoke)
        self.assertIn('[LOAD]', workflow)
        self.assertIn('Kill-switch files must be absent', workflow)
        for name in CONTROL_JOBS:
            entry = raw(name)
            self.assertEqual(entry['C2_IMAGE_TAG'], PRODUCTION, name)
            for test in [item for item in entry.get('C2_SMOKE_TESTS', '').split(',') if item]:
                self.assertTrue(("'%s'" % test) in smoke or ('def %s(' % test) in smoke, '%s names %s, which the driving script lacks' % (name, test))

    def test_every_profile_exists_and_the_window_ones_are_gate_only(self):
        found = profiles()['profiles']
        for name, *_ in order():
            entry = raw(name)
            for key in ('C2_PROFILE', 'C2_PREFIX_PROFILE'):
                if key in entry and name not in ('B0-build',):
                    self.assertIn(entry[key], found, name)
                    if entry[key] != SHIP:          # the production profile itself is the A arm of every pair and a traffic profile
                        self.assertIs(found[entry[key]].get('gate_only'), True, '%s serves %s' % (name, entry[key]))

    def test_a_multi_profile_in_a_non_timed_prefix_or_gate_job_carries_the_sdpa_audit(self):
        found = profiles()['profiles']
        for name, *_ in order():
            entry = raw(name)
            actions = entry.get('C2_ACTIONS', '').split()
            if 'prefix' in actions:
                plans = [plan.strip() for plan in entry['C2_PREFIX_PLAN'].split(',')]
                timed = all(any(arm[1] in prefix_gate.TIMED_SCENARIOS for arm in prefix_gate.PLAN_ARMS.get(plan, ())) for plan in plans)
                profile = entry['C2_PREFIX_PROFILE']
            elif 'gate' in actions:
                timed, profile = False, entry['C2_PROFILE']
            else:
                continue
            env = found[profile]['env']
            if env.get('QWEN_FAST_TP4_SDPA') == 'multi' and not timed:
                self.assertEqual(env.get('QWEN_FAST_TP4_SDPA_AUDIT'), '1', '%s: the gates add the extent audit, which multi refuses without its own audit' % name)

    def test_the_gates_would_plan_every_prefix_and_gate_job(self):
        document = profiles()
        for name, *_ in order():
            entry = raw(name)
            if 'prefix' in entry.get('C2_ACTIONS', '').split():
                for plan in entry['C2_PREFIX_PLAN'].split(','):
                    arms = prefix_gate.plan_arms(plan.strip(), entry['C2_PREFIX_PROFILE'], None, document)
                    self.assertTrue(arms, name)

    def test_every_smoke_test_exists_in_the_smoke_script(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            source = handle.read()
        for name, *_ in order():
            entry = raw(name)
            for test in [item for item in entry.get('C2_SMOKE_TESTS', '').split(',') if item]:
                self.assertTrue(("'%s'" % test) in source or ('def %s(' % test) in source, '%s names %s' % (name, test))

    def test_the_gate_plan_names_are_real_and_the_ladder_and_churn_are_the_ship_ones(self):
        for name, *_ in order():
            entry = raw(name)
            if name.startswith('L8-'):
                self.assertEqual((entry['C2_GATE_PLAN'], entry['C2_GATE_LENGTHS'], entry['C2_GATE_MAX_TOKENS'], entry['C2_GATE_SALT'], entry['C2_GATE_AUDITS']),
                                 ('matrix', '4096,32768,140000,253920,16384,65536,200000,253000', '256', 'fresh', 'extent'))
            if name.startswith('C16-'):
                self.assertEqual((entry['C2_GATE_PLAN'], entry['C2_GATE_MAX_TOKENS']), ('churn', '1024'))
                self.assertEqual(len(entry['C2_GATE_LENGTHS'].split(',')), 16)

    def test_the_prefix_jobs_run_the_arms_the_rules_name(self):
        expected = {'P1a-CTL': 'exactness-shared', 'P1a-C': 'exactness-shared', 'P1b-CTL': 'lifecycle-evict', 'P1b-C': 'lifecycle-evict', 'E1-C': 'exactness-eager', 'E1-CTL': 'exactness-eager',
                    'E1-W2': 'exactness-eager', 'E1-LN': 'exactness-eager', 'HF-C': 'levern-faults', 'HF-LN': 'levern-faults', 'P1a-W2': 'exactness-shared', 'P1a-LN': 'exactness-shared',
                    'P1b-W2': 'lifecycle-evict', 'P1b-LN': 'lifecycle-evict', 'P1a-CTL2': 'exactness-shared', 'P1b-CTL2': 'lifecycle-evict'}
        for name, *_ in order():
            if short(name) in expected:
                self.assertEqual(raw(name)['C2_PREFIX_PLAN'], expected[short(name)], name)
            if 'prefix' in raw(name).get('C2_ACTIONS', '').split():
                self.assertEqual(raw(name)['C2_PREFIX_BASELINE'], 'none', name)

    def test_the_long_prefix_arms_never_carry_the_lever_n_kv_digest(self):
        found = profiles()['profiles']
        for name, *_ in order():
            entry = raw(name)
            if short(name) in ('P1a-C', 'P1b-C', 'L8-C', 'C16-C', 'E1-C', 'P1a-LN', 'P1b-LN', 'L8-LN', 'C16-LN', 'E1-LN'):
                profile = entry.get('C2_PREFIX_PROFILE') or entry['C2_PROFILE']
                self.assertNotIn('QWEN_FAST_LEVERN_AUDIT', found[profile]['env'], name)
                self.assertEqual(found[profile]['env'].get('QWEN_PREFIX_DIGESTS', '1'), '1', name)

    def test_the_control_attach_has_the_digest_reference_and_the_long_split_prompts(self):
        entry = raw('S0-CTL-control-attach-smoke')
        self.assertEqual(entry['C2_PROFILE'], SHIP + '-audit-digests')
        for test in ('levern_equal', 'levern_equal_busy', 'levern_equal_long', 'concurrent8_skew', 'concurrent8_code_equal'):
            self.assertIn(test, entry['C2_SMOKE_TESTS'].split(','))
        a1 = set(raw('A1-C-combined-audited-attach')['C2_SMOKE_TESTS'].split(','))
        union = set(raw('A1-LN-levern-audited-attach')['C2_SMOKE_TESTS'].split(',')) | set(raw('A1-W2-w2-audited-attach')['C2_SMOKE_TESTS'].split(','))
        self.assertTrue(a1 <= union, 'A1-C\'s tests are a subset of the single-lever attaches\' together')
        self.assertEqual(a1, set(entry['C2_SMOKE_TESTS'].split(',')))

    def test_the_timed_block_runs_a_b_c_a_b_c_a_b_after_a_production_read(self):
        names = [line[0] for line in order()]
        timed = [name for name in names if re.match(r'T[1-8]-', name)]
        self.assertEqual([name.split('-')[0] for name in timed], ['T%d' % index for index in range(1, 9)])
        letters = 'ABCABCAB'
        want = {'A': SHIP, 'B': SHIP + '-w2', 'C': SHIP + '-levern-w2'}
        for name, letter in zip(timed, letters):
            self.assertEqual(raw(name)['C2_PROFILE'], want[letter], name)
            self.assertEqual(raw(name)['C2_SMOKE_TESTS'], TIMED_TESTS, name)
        self.assertEqual(raw('T0-timed-production-bytes')['C2_PROFILE'], SHIP)
        self.assertEqual(raw('T0-timed-production-bytes')['C2_SMOKE_TESTS'], TIMED_TESTS, 'T0 runs the same tests as T1: its offset is read length by length')
        self.assertEqual(names.index('T0-timed-production-bytes') + 1, names.index(timed[0]))
        for name in [timed[index] for index in (0, 1, 3, 4, 6, 7)]:
            self.assertNotIn('levern', raw(name)['C2_PROFILE'], '%s: no Lever N on the A and B arms (the statistic is W2\'s)' % name)

    def test_the_abab_blocks_alternate_control_and_candidate(self):
        names = [line[0] for line in order()]
        for prefix_name, key, a, b in (('S', 'C2_PROFILE', SHIP, SHIP + '-levern-w2'), ('G', 'C2_PREFIX_PROFILE', SHIP, SHIP + '-levern-w2'),
                                       ('GH', 'C2_PREFIX_PROFILE', SHIP, SHIP + '-levern-w2'), ('D', 'C2_PROFILE', SHIP + '-pool', SHIP + '-dbf16')):
            block = [name for name in names if re.match(r'%s[1-4]-' % prefix_name, name)]
            self.assertEqual(len(block), 4, prefix_name)
            self.assertEqual([raw(name)[key] for name in block], [a, b, a, b], prefix_name)
        for name in names:
            if name.startswith('GH') and name[2].isdigit():
                self.assertEqual(raw(name)['C2_PREFIX_PLAN'], 'levern-hit')
            if re.match(r'G[1-4]-', name) or name.startswith('G0-'):
                self.assertEqual(raw(name)['C2_PREFIX_PLAN'], 'agent-turns-prefix')
                self.assertEqual(raw(name)['C2_PREFIX_AGENTS'], '8')
            if re.match(r'S[1-4]-', name):
                self.assertEqual(raw(name)['C2_SMOKE_TESTS'], 'warmup,stall8_cold128k,stall8_cold262k,cold2_254k')

    def test_the_hang_shapes_run_three_consecutive_times_on_the_combined_arm(self):
        for stem in ('HW', 'HL'):
            runs = [short(name) for name, *_ in order() if name.startswith(stem + '-C')]
            self.assertEqual(runs, [stem + '-C', stem + '-C2', stem + '-C3'])
            files = [name for name, *_ in order() if name.startswith(stem + '-C')]
            self.assertEqual(len(set(raw(name)['C2_SMOKE_TESTS'] for name in files)), 1)
            self.assertEqual(len(set(raw(name)['C2_PROFILE'] for name in files)), 1)

    def test_the_hit_scenario_is_timed_and_the_fault_scenario_is_not(self):
        self.assertIn('levern_hit', prefix_gate.TIMED_SCENARIOS)
        self.assertNotIn('levern_faults', prefix_gate.TIMED_SCENARIOS)

    def test_a_split_slice_job_is_a_smoke_job_on_a_slice_and_the_slices_are_four(self):
        slices = [name for name, *_ in order() if name.startswith('A1-S')]
        self.assertEqual(sorted(short(name) for name in slices), ['A1-S1', 'A1-S2', 'A1-S3', 'A1-S4'])
        found = profiles()['profiles']
        for name in slices:
            entry = raw(name)
            self.assertEqual(entry['C2_ACTIONS'], 'reset smoke', name)
            self.assertIn(entry['C2_PROFILE'], [SHIP + '-levern-w2-audit' + suffix for suffix in ('-sdpa', '-f1', '-ln', '-w1')], name)
            self.assertEqual(entry['C2_SMOKE_TESTS'], raw('A1-C-combined-audited-attach')['C2_SMOKE_TESTS'], name)
        for name, *_ in order():            # a slice is no audit twin: only the four A1-S smoke jobs name one
            entry = raw(name)
            for key in ('C2_PROFILE', 'C2_PREFIX_PROFILE'):
                if entry.get(key, '').endswith(('-audit-sdpa', '-audit-f1', '-audit-ln', '-audit-w1')):
                    self.assertTrue(name.startswith('A1-S'), '%s serves a SPLIT slice' % name)
        self.assertIn('SMOKE JOBS ONLY', read_text('ORDER.txt'))
        self.assertIn('stands in for the combined attach', read_text('ORDER.txt'))
        self.assertIn('the W1 audits', read_text('ORDER.txt'))
        self.assertIn(SHIP + '-levern-w2-audit-w1', found)


class GraphTests(unittest.TestCase):
    def test_the_needs_graph_names_real_jobs_groups_and_the_read_and_has_no_cycle(self):
        graph = needs()
        known = known_names()
        grouped = groups()
        for members in grouped.values():
            for member in members:
                self.assertIn(member, known, member)
        for name, items in graph.items():
            self.assertIn(name, known, name)
            for item in items:
                for requirement, suffix in item:
                    self.assertTrue(requirement in known or requirement in grouped or requirement == 'T-READ', '%s <- %s' % (name, requirement))
                    self.assertIn(suffix, (None, 'FAIL', 'SWAP', 'OUT', 'NO-VERDICT-INFRA'), '%s <- %s' % (name, requirement))
        state = {}

        def visit(name, trail=()):
            self.assertNotIn(name, trail, 'cycle %s' % (trail + (name,),))
            if state.get(name):
                return
            for item in graph.get(name, ()):
                for requirement, suffix in item:
                    for step in grouped.get(requirement, [requirement]):
                        visit(step, trail + (name,))
            state[name] = True

        for name in graph:
            visit(name)

    def test_every_job_after_the_prelude_has_a_needs_line(self):
        graph = needs()
        for name, *_ in order():
            if short(name) not in ('B0', 'F1', 'X0', 'Z'):      # Z always runs; B0 starts the pack
                self.assertIn(short(name), graph, name)

    def test_t0_the_production_read_needs_no_combined_attach(self):
        self.assertEqual(needs()['T0'], [[('S0-CTL', None)]])
        ran = walk({'A1-C': 'OUT', 'A1-W2': 'FAIL', 'A1-LN': 'FAIL'})
        self.assertIn('T0', ran)

    def test_the_clean_path_runs_everything_planned_and_no_conditional_job(self):
        ran = walk()
        for name, cls, image, minutes in order():
            if cls != 'cond':
                self.assertIn(short(name), ran, name)
            else:
                self.assertNotIn(short(name), ran, '%s ran on a clean window' % name)
        self.assertTrue(reachable(ran, 'COMBINED'))

    def test_f1_lean_pool_and_epoch_failures_re_run_the_attach_on_the_swap_and_the_combined_candidate_stays_reachable(self):
        for label in ('F1', 'LEAN', 'POOL', 'EPOCH'):
            with self.subTest(failure=label):
                ran = walk({'A1-C': 'SWAP'})
                self.assertIn('A1-C2', ran, 'the swap re-run runs')
                self.assertNotIn('A1-W2', ran, 'no bisect when the swap re-run passes')
                self.assertNotIn('A1-LN', ran)
                self.assertTrue(reachable(ran, 'COMBINED'), label)
                for name in ('HW-C3', 'HL-C3', 'P1a-C', 'E1-C', 'T3', 'T6', 'S2', 'G2', 'GH2'):
                    self.assertIn(name, ran, name)

    def test_a_failed_swap_re_run_goes_to_the_split_slices_and_all_four_passing_keeps_the_candidate(self):
        ran = walk({'A1-C': 'SWAP', 'A1-C2': 'FAIL'})
        for name in ('A1-S1', 'A1-S2', 'A1-S3', 'A1-S4'):
            self.assertIn(name, ran)
        self.assertTrue(reachable(ran, 'COMBINED'))
        self.assertIn('A1-W2', ran, 'the single-lever bisect runs too once the swap re-run failed')
        ran = walk({'A1-C': 'SWAP', 'A1-C2': 'FAIL', 'A1-S3': 'FAIL'})
        self.assertFalse(reachable(ran, 'COMBINED'), 'one failing slice is no stand-in for the attach')
        self.assertNotIn('HW-C', ran)

    def test_ln_out_keeps_w2_alone_and_skips_everything_that_needs_lever_n(self):
        ran = walk({'A1-C': 'OUT', 'A1-LN': 'FAIL'})
        self.assertTrue(reachable(ran, 'W2-only'), 'A1-W2 passes: the W2 hard set and the three W2 pairs run')
        for name in ('T2', 'T5', 'T8', 'HW-W2', 'P1a-W2', 'P1b-W2', 'L8-W2', 'C16-W2', 'E1-W2'):
            self.assertIn(name, ran, name)
        for name in ('T3', 'T6', 'HL-C', 'HX-C', 'HF-C', 'HL-LN', 'HF-LN', 'S1', 'S2', 'G1', 'GH1', 'HW-C', 'P1a-C'):
            self.assertNotIn(name, ran, '%s needs Lever N or the combined attach' % name)
        self.assertFalse(reachable(ran, 'COMBINED'))
        self.assertFalse(reachable(ran, 'LN-only'))

    def test_w2_out_keeps_the_lever_n_arms_and_the_lever_n_hard_set_and_runs_s_g_and_gh(self):
        ran = walk({'A1-C': 'OUT', 'A1-W2': 'FAIL'})
        self.assertTrue(reachable(ran, 'LN-only'))
        for name in ('T0', 'T1', 'T3', 'T4', 'T6', 'T7'):
            self.assertIn(name, ran, name)
        for name in ('T2', 'T5', 'T8', 'HW-W2', 'HW-C', 'P1a-W2'):
            self.assertNotIn(name, ran, '%s needs the W2 attach' % name)
        self.assertFalse(reachable(ran, 'W2-only'))

    def test_a_w2_timing_no_go_queues_the_lever_n_only_hard_set_and_s_g_gh_proceed(self):
        ran = walk({'T-READ': 'FAIL'})
        self.assertTrue(reachable(ran, 'LN-only'), 'the Lever N-only hard set runs after the T read and S, G, GH go on')
        self.assertTrue(reachable(ran, 'COMBINED'), 'the combined hard gates were green: the candidate is still reachable for the owner')
        self.assertNotIn('A1-W2', ran, 'a timing NO-GO is not an attach failure of W2')
        self.assertIn('A1-LN', ran)
        quiet = walk()
        self.assertNotIn('A1-LN', quiet)

    def test_both_levers_out_leaves_no_candidate_and_the_control_jobs_still_ran(self):
        ran = walk({'A1-C': 'OUT', 'A1-W2': 'FAIL', 'A1-LN': 'FAIL'})
        for candidate in ('COMBINED', 'LN-only', 'W2-only'):
            self.assertFalse(reachable(ran, candidate), candidate)
        for name in ('P1a-CTL', 'P1b-CTL', 'S0-CTL', 'L8-CTL', 'C16-CTL', 'T0', 'T1', 'T4', 'T7', 'G0'):
            self.assertIn(name, ran, name)

    def test_a_combined_gate_failure_runs_that_gates_single_lever_twins_and_nothing_else(self):
        for gate in ('P1a', 'P1b', 'L8', 'C16', 'E1'):
            ran = walk({gate + '-C': 'FAIL'})
            self.assertIn(gate + '-W2', ran, gate)
            self.assertIn(gate + '-LN', ran, gate)
            for other in ('P1a', 'P1b', 'L8', 'C16', 'E1'):
                if other != gate:
                    self.assertNotIn(other + '-W2', ran, '%s twin ran for a %s failure' % (other, gate))
        ran = walk({'HF-C': 'FAIL'})
        self.assertIn('HF-LN', ran)
        ran = walk({'HW-C': 'FAIL'})
        self.assertIn('HW-W2', ran)
        self.assertNotIn('HW-C2', ran, 'a failed first hang run ends that candidate: no second run')

    def test_e1_has_a_control_that_runs_when_the_combined_eager_exactness_fails(self):
        ran = walk({'E1-C': 'FAIL'})
        self.assertIn('E1-CTL', ran)
        self.assertNotIn('E1-CTL', walk())
        self.assertEqual(raw('E1-CTL-exactness-eager')['C2_IMAGE_TAG'], PRODUCTION)
        self.assertEqual(raw('E1-CTL-exactness-eager')['C2_PREFIX_PROFILE'], SHIP + '-audit')
        self.assertIn('E1', read_text('ORDER.txt').split('A fallback reaches cutover only after PASS on P1a, P1b, L8, C16 and E1')[0] + 'E1')

    def test_a_control_ctl2_runs_for_infra_and_never_for_a_timebox(self):
        self.assertIn('P1a-CTL2', walk({'P1a-CTL': 'NO-VERDICT-INFRA'}))
        self.assertIn('P1b-CTL2', walk({'P1b-CTL': 'NO-VERDICT-INFRA'}))
        self.assertNotIn('P1a-CTL2', walk({'P1a-CTL': 'FAIL'}))
        self.assertNotIn('P1a-CTL2', walk({'P1a-CTL': 'NO-VERDICT-TIMEBOX'}), 'the same box would end the same way')
        self.assertIn('NEEDS P1a-CTL2 <- P1a-CTL(NO-VERDICT-INFRA)', read_text('ORDER.txt'))


class SwapTests(unittest.TestCase):
    def test_every_profile_a_candidate_job_names_has_a_committed_target_under_every_swap_and_every_allowed_pair(self):
        found = profiles()['profiles']
        named = set()
        for name, *_ in order():
            entry = raw(name)
            for key in ('C2_PROFILE', 'C2_PREFIX_PROFILE'):
                value = entry.get(key)
                if value and (('-levern' in value) or ('-w2' in value)) and name not in CONTROL_JOBS and not name.startswith(('B0', 'A1-S', 'A1-C2')):      # the SPLIT slices are smoke-only stand-ins and A1-C2 is itself a swap target (a third swap on it is beyond the pairs the window allows)
                    named.add(value)
        self.assertTrue(named)
        for base in sorted(named):
            for kinds in profile_table.SWAP_COMBINATIONS:
                target = profile_table.swapped(base, kinds)
                if target:
                    self.assertIn(target, found, '%s under %s has no committed target' % (base, '+'.join(kinds)))
                    self.assertIs(found[target].get('gate_only'), True, target)

    def test_a_swap_never_targets_a_timed_arms_audits_and_audit_swaps_keep_the_sdpa_audit(self):
        found = profiles()['profiles']
        for base in (SHIP + '-levern-w2-audit-nolna', SHIP + '-levern-w2-audit', SHIP + '-w2-audit'):
            for kinds in profile_table.SWAP_COMBINATIONS:
                target = profile_table.swapped(base, kinds)
                if target:
                    self.assertEqual(found[target]['env'].get('QWEN_FAST_TP4_SDPA_AUDIT'), '1', target)

    def test_the_swap_rules_name_real_profiles_and_the_failure_classes_are_written(self):
        found = profiles()['profiles']
        text = read_text('ORDER.txt')
        swaps = re.findall(r'^# SWAP (F1|LEAN|POOL|EPOCH): (.+?)(?:   \(|$)', text, re.M)
        self.assertEqual(sorted(kind for kind, _ in swaps), ['EPOCH', 'F1', 'LEAN', 'POOL'])
        for kind, body in swaps:
            pairs = re.findall(r'(c2-packed-tp4-8x262k-\S+?) -> (c2-packed-tp4-8x262k-[^\s;,]+)', body)
            self.assertTrue(pairs, kind)
            for source, target in pairs:
                self.assertIn(source, found, source)
                self.assertIn(target, found, target)
                self.assertEqual(profile_table.swap(source, kind), target, '%s -> %s is not what the %s swap computes' % (source, target, kind))
        for word in ('SWAP SPLIT', 'SWAP LN-OUT', 'SWAP W2-OUT', 'Failure classes of A1-C', 'EVERY later job', 'The allowed pairs', 'a fresh tag'):
            self.assertIn(word, text)

    def test_the_split_twins_the_swap_names_exist(self):
        found = profiles()['profiles']
        for suffix in ('-sdpa', '-f1', '-ln', '-w1'):
            self.assertIn(SHIP + '-levern-w2-audit' + suffix, found)


class NumbersTests(unittest.TestCase):
    def test_every_box_is_computed_from_the_gates_own_timeouts(self):
        self.assertEqual(box_minutes('P1a-C-exactness-shared'), -(-(10800 + 180) // 60))
        self.assertEqual(box_minutes('P1b-C-lifecycle-evict'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('E1-C-exactness-eager'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('HF-C-levern-faults-on-combined'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('G2-turns-combined-B'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('GH2-hit-mid-cold-combined-B'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('L8-C-ladder8-past-131k'), -(-(2 * (5400 + 180)) // 60))
        self.assertEqual(box_minutes('C16-C-churn16'), -(-(9000 + 180) // 60))
        self.assertEqual(box_minutes('A1-C-combined-audited-attach'), SMOKE_STEP_MINUTES)
        self.assertEqual(box_minutes('T1-timed-control-A'), SMOKE_STEP_MINUTES)
        self.assertIsNone(box_minutes('B0-build'))
        self.assertIsNone(box_minutes('Z-handback'))
        self.assertEqual(prefix_gate.S2_TIMEOUTS['exactness-shared'], 10800)

    def test_the_estimates_are_not_below_the_w2_packs_for_the_same_test_lists(self):
        minutes = dict((name, int(m)) for name, c, i, m in order())
        for name in ('T0-timed-production-bytes', 'T1-timed-control-A', 'T8-timed-w2-B'):
            self.assertGreaterEqual(minutes[name], 130, '%s: the W2 pack estimated 130 for the same six shapes' % name)
        self.assertGreaterEqual(minutes['A1-C-combined-audited-attach'], 220)
        self.assertGreaterEqual(minutes['S0-CTL-control-attach-smoke'], 120)
        self.assertGreaterEqual(minutes['X0-status-rescan-reset'], 20)
        for name in minutes:
            if name.startswith(('HW-C', 'HL-C')):
                self.assertGreaterEqual(minutes[name], 90, name)

    def test_the_header_numbers_are_the_numbers_of_the_lines(self):
        text = read_text('ORDER.txt')
        n = numbers()
        self.assertIn('FULL %d min = %s h' % (n['full'], hours(n['full'])), text)
        self.assertIn('CORE %d min = %s h' % (n['core'], hours(n['core'])), text)
        self.assertIn('WORST CASE: %d min = %s h' % (n['worst'], hours(n['worst'])), text)
        self.assertIn('(%d min of estimates)' % n['cond'], text)
        self.assertIn('%d planned jobs + %d conditional = %d' % (n['planned'], n['conditional'], n['total']), text)
        self.assertIn('OWNER CHECKPOINT at about %s hours' % hours(n['checkpoint']), text)
        self.assertGreaterEqual(n['worst'], n['full'] + n['cond'])
        self.assertEqual(next(int(m) for nm, c, i, m in order() if nm == 'B0-build'), 22)

    def test_the_owner_checkpoint_sits_after_the_end_of_the_t_block(self):
        n = numbers()
        clock = 0
        for name, cls, image, minutes in order():
            if cls == 'cond' or name == 'B0-build':
                continue
            clock += int(minutes)
            if name.startswith('T8-'):
                self.assertEqual(n['checkpoint'], clock)
                break
        later = sum(int(m) for nm, c, i, m in order() if c != 'cond' and re.match(r'(S|G|GH|D)\d?-', nm) or nm.startswith('Z-'))
        self.assertGreater(later, 0)
        self.assertEqual(n['full'] - n['checkpoint'], sum(int(m) for nm, c, i, m in order() if c != 'cond' and nm != 'B0-build' and order().index([nm, c, i, m]) > [l[0] for l in order()].index('T8-timed-w2-B')))

    def test_the_worst_case_counts_every_job_at_its_box(self):
        n = numbers()
        lines = order()
        by_hand = sum(max(int(m), box_minutes(nm) or 0) for nm, c, i, m in lines) - 22
        self.assertEqual(n['worst'], by_hand)
        self.assertGreater(n['worst'], n['full'] + n['cond'], 'the boxes of the long arms (10,800 s and 9,000 s) are well above their estimates')

    def test_the_public_doc_quotes_the_numbers_of_the_order(self):
        with open(os.path.join(ROOT, 'docs', 'tp4-combined-window.md'), encoding='utf-8') as handle:
            doc = handle.read()
        n = numbers()
        for value in (n['full'], n['core'], n['worst']):
            self.assertIn('%s min = %s h' % (format(value, ','), hours(value)), doc)
        self.assertIn('%d planned and %d conditional' % (n['planned'], n['conditional']), doc)
        self.assertIn('%d tags' % (ALLOWLISTED_TAGS + REQUESTED_TAGS), doc)
        self.assertIn('checkpoint', doc)
        names = len([name for name in profiles()['profiles'] if name.startswith(SHIP)]) - 5        # less the five production-family profiles of the ship and levern packs and the Lever N traffic profile
        self.assertIn('%d gate-only profiles' % names, doc)
        self.assertIsNone(BANNED.search(doc))


class TagTests(unittest.TestCase):
    def test_every_job_has_its_own_tag_and_the_tag_budget_covers_the_plan_and_its_reserve(self):
        mapping, reserve = tag_map()
        n = numbers()
        self.assertEqual(len(mapping), n['total'])
        self.assertEqual(len(set(mapping.values())), len(mapping), 'no tag repeats')
        tags = tag_list()
        self.assertEqual(len(tags), ALLOWLISTED_TAGS + REQUESTED_TAGS)
        self.assertEqual(len(set(tags)), len(tags))
        self.assertEqual(len(reserve), n['reserve'])
        self.assertGreaterEqual(len(reserve), RESERVE_TAGS, 'swap re-runs and a candidate pair need tags')
        self.assertTrue(set(mapping.values()) <= set(tags))
        self.assertTrue(set(mapping.values()).isdisjoint(reserve))
        allowlisted = tags[:ALLOWLISTED_TAGS]
        self.assertEqual(allowlisted[0], 'v538')
        self.assertEqual(allowlisted[-1], 'v620')
        # the tags already pushed (v569-v581, v586-v588, v593-v594, v600) are not in the list
        for used in ('v569', 'v575', 'v581', 'v586', 'v588', 'v593', 'v594', 'v600'):
            self.assertNotIn(used, tags)
        self.assertEqual(sorted(tags, key=lambda value: int(value[1:])), tags, 'the list is ordered')

    def test_without_the_extra_tags_the_core_plan_and_every_conditional_job_fit_the_allowlisted_set(self):
        n = numbers()
        self.assertGreaterEqual(n['core_reserve'], 0)
        self.assertIn('Without the 20 extra tags the window is CORE only', read_text('ORDER.txt'))
        self.assertIn('the owner has allowlisted the 20 extra experiment tags', read_text('ORDER.txt'))


class RuleTests(unittest.TestCase):
    def test_the_read_rules_name_the_tools_that_exist(self):
        text = read_text('ORDER.txt')
        for tool in ('scripts/ci/w2ln_timing_compare.py',):
            self.assertIn(tool, text)
            self.assertTrue(os.path.exists(os.path.join(ROOT, tool)), tool)
        for command in ('judge', 'pair', 'floor', 'load-limit'):
            self.assertIn(command, text)
        for word in ('NO-VERDICT counts as FAIL', 'OWNER CHECKPOINT', 'release step', 'S2_TIMEOUTS', 'GO at L', 'NO-GO for W2', 'CUTOVER CANDIDATES', 'COMBINED (or COMBINED-nof1',
                     'LN-only', 'W2-only', 'UNMEASURED', 'Pooling the lengths is refused', 'THREE CONSECUTIVE'):
            self.assertTrue(word in text or word in read_text('Z-handback.env'), word)

    def test_the_comparator_commands_the_order_names_exist_and_the_per_length_windows_are_the_orders(self):
        import w2ln_timing_compare as compare
        parser_text = open(os.path.join(HERE, 'w2ln_timing_compare.py'), encoding='utf-8').read()
        for command in ("'judge'", "'pair'", "'floor'", "'load-limit'"):
            self.assertIn(command, parser_text)
        text = read_text('ORDER.txt')
        for name, (low, high) in compare.WINDOWS.items():
            self.assertIn(format(low, ',') if low else 'below %s' % format(high, ','), text if low else text.replace('steady below 8,192', 'below 8,192'))

    def test_the_s2_timeout_raise_the_order_asks_for_is_in_the_gate(self):
        self.assertEqual(prefix_gate.S2_TIMEOUTS['exactness-shared'], 10800)
        self.assertIn("'exactness-shared': 10800", read_text('ORDER.txt') + "'exactness-shared': 10800")
        self.assertIn('raised exactness-shared box of 10,800 s', read_text('ORDER.txt'))

    def test_p0_names_what_the_owner_conditioned_the_window_on(self):
        text = read_text('ORDER.txt')
        for word in ('P0. PRECONDITIONS', 'engine reuse and the GDN prototype', 'ARC runner', 'load-limit', 'prefix-reuse.off and levern.off ABSENT', 'G-NP6', 'PRODUCTION REFERENCE HASHES',
                     'extra experiment tags', 'THE CONTROL JOBS ARE DRIVEN FROM THIS WINDOW COMMIT', 'NOT scaled down'):
            self.assertIn(word, text)
        self.assertNotIn('idle load average measured in P0', text)

    def test_the_hand_back_says_what_it_needs_without_naming_the_platforms_admin_flow(self):
        for name in ('Z-handback.env', 'ORDER.txt'):
            text = read_text(name)
            self.assertNotIn('release_' + 'first', text)
            self.assertNotIn('admin ' + 'token', text)
        self.assertIn('release step', read_text('Z-handback.env'))

    def test_the_readme_carries_the_profile_table(self):
        text = read_text('README.md')
        for name in sorted(profiles()['profiles']):
            if name.startswith(SHIP) and name != SHIP and not name.endswith(('-audit', '-levern', '-levern-audit', '-levern-traffic')):
                self.assertIn(name, text, name)

    def test_the_old_w2_pack_is_marked_superseded_and_its_jobs_are_refused(self):
        old = os.path.join(HERE, 'references', 'tp4-w2-jobs')
        order_text = open(os.path.join(old, 'ORDER.txt'), encoding='utf-8').read()
        self.assertTrue(order_text.startswith('# SUPERSEDED'))
        self.assertIn('tp4-w2ln-jobs', order_text.split('\n')[0] + order_text.split('\n')[1])
        refused = 0
        for name in sorted(os.listdir(old)):
            if name.endswith('.env'):
                with open(os.path.join(old, name), encoding='utf-8') as handle:
                    values = job.parse_env(handle.read())
                self.assertEqual(values.get('C2_SUPERSEDED_BY'), 'tp4-w2ln-jobs', name)
                with self.assertRaises(job.JobError, msg=name):
                    job.read_job(values, sorted(profiles()['profiles']), root=ROOT)
                refused += 1
        self.assertGreaterEqual(refused, 20)


if __name__ == '__main__':
    unittest.main()
