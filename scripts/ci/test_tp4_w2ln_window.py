"""The combined development window's job pack (references/tp4-w2ln-jobs; docs/tp4-combined-window.md).

Every template parses with the job parser, names the one window image (or, for the CONTROL jobs, the production base image), never starts the node agent except Z and never stops it except A0,
and the ORDER.txt lines match the files; no template names a rig, card, address, registry or digest; every profile a template names exists, is gate only (the controls and the timed
production reads excepted) and every multi profile a non-timed prefix or gate job serves carries the SDPA audit (the extent audit the gates add refuses it otherwise); every smoke test a
template names exists in the smoke script; the timed blocks alternate as written; the NEEDS graph names real jobs and has no cycle; the swap rules name real profiles; the outage
numbers in the ORDER header are the numbers of its lines; the tag count fits the allowlist."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-w2ln-jobs')
WINDOW, PRODUCTION = 'tp4-w2ln-1', 'tp4-serve-10'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
R = 'c2-packed-tp4-8x262k-'
SHIP = R + 'ship-prefix'
CONTROL_JOBS = ('P1a-CTL-exactness-shared', 'P1b-CTL-lifecycle-evict', 'L8-CTL-ladder8-past-131k', 'C16-CTL-churn16', 'T0-timed-production-bytes', 'G0-turns-A-production-bytes')
ALLOWLISTED_TAGS = 64
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve'}


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


def needs():
    found = {}
    for line in read_text('ORDER.txt').splitlines():
        match = re.match(r'# NEEDS (.+?) <- (.+)$', line)
        if match:
            for name in match.group(1).split():
                found.setdefault(name, set()).update(item.split('(')[0] for item in match.group(2).split())
    return found


def short(name):
    """The NEEDS graph's short name of a template: T1 for T1-timed-control-A, P1a-CTL for P1a-CTL-exactness-shared, A1-C for A1-C-combined-audited-attach."""
    parts = name.split('-')
    return '-'.join(parts[:2]) if len(parts) > 1 and parts[1] in ('C', 'CTL', 'CTL2', 'W2', 'LN') else parts[0]


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates(self):
        names = [line[0] for line in order()]
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(names), files)
        self.assertEqual(len(names), len(set(names)))
        for name, cls, image, minutes in order():
            self.assertIn(cls, ('stop', 'soft', 'branch', 'cond'), name)
            self.assertEqual(image, PRODUCTION if name in CONTROL_JOBS else WINDOW, name)
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
            self.assertIsNone(BANNED.search(read_text(name)), name)
            self.assertIsNone(re.search(r'experiment/c2-serving-v\d', read_text(name)), '%s names a real tag' % name)

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
        expected = {'P1a-CTL': 'exactness-shared', 'P1a-C': 'exactness-shared', 'P1b-CTL': 'lifecycle-evict', 'P1b-C': 'lifecycle-evict', 'E1-C': 'exactness-eager', 'HF-C': 'levern-faults',
                    'HF-LN': 'levern-faults', 'P1a-W2': 'exactness-shared', 'P1a-LN': 'exactness-shared', 'P1b-W2': 'lifecycle-evict', 'P1b-LN': 'lifecycle-evict',
                    'P1a-CTL2': 'exactness-shared', 'P1b-CTL2': 'lifecycle-evict'}
        for name, *_ in order():
            if short(name) in expected:
                self.assertEqual(raw(name)['C2_PREFIX_PLAN'], expected[short(name)], name)
            if 'prefix' in raw(name).get('C2_ACTIONS', '').split():
                self.assertEqual(raw(name)['C2_PREFIX_BASELINE'], 'none', name)

    def test_the_long_prefix_arms_never_carry_the_lever_n_kv_digest(self):
        found = profiles()['profiles']
        for name, *_ in order():
            entry = raw(name)
            if short(name) in ('P1a-C', 'P1b-C', 'L8-C', 'C16-C', 'E1-C', 'P1a-LN', 'P1b-LN', 'L8-LN', 'C16-LN'):
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
            self.assertEqual(raw(name)['C2_SMOKE_TESTS'], 'warmup,coding,concurrent8_steady,concurrent8_code_32k,concurrent8_code_128k,concurrent8_skew', name)
        self.assertEqual(raw('T0-timed-production-bytes')['C2_PROFILE'], SHIP)
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

    def test_the_hit_scenario_is_timed_and_the_fault_scenario_is_not(self):
        self.assertIn('levern_hit', prefix_gate.TIMED_SCENARIOS)
        self.assertNotIn('levern_faults', prefix_gate.TIMED_SCENARIOS)

    def test_the_needs_graph_names_real_jobs_and_has_no_cycle(self):
        graph = needs()
        known = {}
        for line in order():
            known.setdefault(short(line[0]), []).append(line[0])
        for name, requirements in graph.items():
            self.assertIn(name, known, name)
            for requirement in requirements:
                self.assertIn(requirement, known, '%s <- %s' % (name, requirement))
        state = {}

        def visit(name, trail=()):
            self.assertNotIn(name, trail, 'cycle %s' % (trail + (name,),))
            if state.get(name):
                return
            for requirement in graph.get(name, ()):
                visit(requirement, trail + (name,))
            state[name] = True

        for name in graph:
            visit(name)

    def test_every_job_after_the_prelude_has_a_needs_line(self):
        graph = needs()
        for name, *_ in order():
            if short(name) not in ('B0', 'F1', 'X0', 'Z'):      # Z always runs; B0 starts the pack
                self.assertIn(short(name), graph, name)

    def test_the_swap_rules_name_real_profiles_and_the_failure_classes_are_written(self):
        found = profiles()['profiles']
        text = read_text('ORDER.txt')
        swaps = re.findall(r'^# SWAP (F1|LEAN|POOL|EPOCH): (.+?)(?:   \(|$)', text, re.M)
        self.assertEqual(sorted(kind for kind, _ in swaps), ['EPOCH', 'F1', 'LEAN', 'POOL'])
        for kind, body in swaps:
            for source, target in re.findall(r'(c2-packed-tp4-8x262k-\S+?) -> (c2-packed-tp4-8x262k-[^\s;,]+)', body):
                self.assertIn(source, found, source)
                self.assertIn(target, found, target)
        for word in ('SWAP SPLIT', 'SWAP LN-OUT', 'SWAP W2-OUT', 'Failure classes of A1-C'):
            self.assertIn(word, text)

    def test_the_split_twins_the_swap_names_exist(self):
        found = profiles()['profiles']
        for suffix in ('-sdpa', '-f1', '-ln'):
            self.assertIn(SHIP + '-levern-w2-audit' + suffix, found)

    def test_the_header_numbers_are_the_numbers_of_the_lines(self):
        text = read_text('ORDER.txt')
        lines = order()
        total = sum(int(minutes) for name, cls, image, minutes in lines if cls != 'cond')
        outage = total - 22
        core_drop = sum(int(minutes) for name, cls, image, minutes in lines
                        if re.match(r'(D[1-4]|GH[34]|G[34]|S[34])-', name))
        cond = sum(int(minutes) for name, cls, image, minutes in lines if cls == 'cond')
        self.assertIn('FULL %d min' % outage, text)
        self.assertIn('CORE %d min' % (outage - core_drop), text)
        self.assertIn('(%d min)' % cond, text)
        self.assertIn('%d planned jobs + %d conditional = %d' % (len(planned()), len(conditional()), len(lines)), text)
        self.assertLessEqual(len(lines), ALLOWLISTED_TAGS)
        self.assertEqual(next(int(m) for n, c, i, m in lines if n == 'B0-build'), 22)

    def test_the_public_doc_quotes_the_numbers_of_the_order(self):
        with open(os.path.join(ROOT, 'docs', 'tp4-combined-window.md'), encoding='utf-8') as handle:
            doc = handle.read()
        lines = order()
        total = sum(int(minutes) for name, cls, image, minutes in lines if cls != 'cond')
        outage = total - 22
        core = outage - sum(int(minutes) for name, cls, image, minutes in lines if re.match(r'(D[1-4]|GH[34]|G[34]|S[34])-', name))
        cond = sum(int(minutes) for name, cls, image, minutes in lines if cls == 'cond')
        worst = outage + 25 + 85 + 50 + 20 + 80 + cond
        for value in (outage, core, worst):
            self.assertIn('%s min = %.1f h' % (format(value, ','), value / 60.0), doc)
        self.assertIn('%d of the %d allowlisted' % (len(lines), ALLOWLISTED_TAGS), doc)
        self.assertIn('%d planned and %d conditional' % (len(planned()), len(conditional())), doc)
        names = len([name for name in profiles()['profiles'] if name.startswith(SHIP)]) - 4        # less the four production-family profiles of the ship and levern packs
        self.assertIn('%d gate-only profiles' % names, doc)
        self.assertIsNone(BANNED.search(doc))

    def test_the_read_rules_name_the_tools_that_exist(self):
        text = read_text('ORDER.txt')
        for tool in ('scripts/ci/w2ln_timing_compare.py',):
            self.assertIn(tool, text)
            self.assertTrue(os.path.exists(os.path.join(ROOT, tool)), tool)
        for word in ('NO-VERDICT counts as FAIL', 'OWNER CHECKPOINT', 'release_first', 'S2_TIMEOUTS', 'GO at L', 'NO-GO for W2', 'CUTOVER CANDIDATES', 'COMBINED (or COMBINED-nof1)', 'LN-only', 'W2-only'):
            self.assertTrue(word in text or word in read_text('Z-handback.env'), word)

    def test_the_s2_timeout_raise_the_order_asks_for_is_in_the_gate(self):
        self.assertEqual(prefix_gate.S2_TIMEOUTS['exactness-shared'], 10800)
        self.assertIn("'exactness-shared': 10800", read_text('ORDER.txt'))

    def test_p0_names_what_the_owner_conditioned_the_window_on(self):
        text = read_text('ORDER.txt')
        for word in ('P0. PRECONDITIONS', 'engine reuse and the GDN prototype', 'ARC runner', 'idle load average', 'prefix-reuse.off and levern.off ABSENT'):
            self.assertIn(word, text)

    def test_the_readme_carries_the_profile_table(self):
        text = read_text('README.md')
        for name in sorted(profiles()['profiles']):
            if name.startswith(SHIP) and name != SHIP and not name.endswith(('-audit', '-levern', '-levern-audit')):
                self.assertIn(name, text, name)


if __name__ == '__main__':
    unittest.main()
