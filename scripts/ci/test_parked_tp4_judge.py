"""Engine reuse, the gate side: parked_markers (the log's lines, parsed), parked_judge (what a parked arm's log must show), parked_compare (the arms against
each other, and the negative controls against their own judges) and the smoke tests' own checks (c2_smoke_check.parked_smoke_problems).

The lines the markers parse are the producers' real ones: the parked set, the factory and the coordinator run on the four-chip census world and their log
is what the judge reads. The negatives are synthetic logs that break exactly one rule each."""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
import parked_compare  # noqa: E402
import parked_judge as judge  # noqa: E402
import parked_markers as markers  # noqa: E402
import serving_parked_engines as parked  # noqa: E402
from test_parked_tp4_census import PAGE_WIDTH, World  # noqa: E402
from test_parked_tp4_set import make_set  # noqa: E402
from test_parked_tp4_wiring import PARKED, admit, run_to_end  # noqa: E402

ENV = {'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_PUBLISH_PREWARM': '1'}
CAPACITY = PAGE_WIDTH * 64


def real_log(audit=False, requests=((300, 24), (40, 3), (2049, 16), (20, 2)), fault=None, drafts=False):
    """The log of a parked arm on the census world: the set's attach lines, then one rebind per request (and the fault's unpark and re-park)."""
    environ = dict(PARKED)
    if audit:
        environ[parked.AUDIT_FLAG] = '1'
    if fault:
        environ[parked.FAULT_FLAG] = fault
    if drafts:
        environ[parked.DRAFTS_FLAG] = '1'
    with World(environment=PARKED) as world:
        engines = make_set(world, environ={name: value for name, value in environ.items() if name != 'QWEN_FAST_PARKED_ENGINES'})
        engines.build()
        for index, (prompt, budget) in enumerate(requests):
            request = admit(world, engines, 'req-%d' % index, prompt, budget)
            run_to_end(request)
            request.close('req-%d' % index)
            if fault and index == 0:
                engines.repark_idle()
        lines = list(world.lines)
    return '\n'.join(lines) + '\n'


class MarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = real_log()
        cls.audited = real_log(audit=True)

    def test_every_parked_line_the_set_logs_is_parsed(self):
        facts = markers.scan(self.plain.splitlines())
        self.assertEqual([entry['k'] for entry in facts['built']], [4])
        self.assertEqual(facts['built'][0]['slots'], 4)
        self.assertEqual([entry['rows'] for entry in facts['warm']], [2048])
        self.assertTrue(facts['prewarm'] and facts['prewarm'][0] > 0)
        self.assertEqual(len(facts['rebinds']), 4)
        first = facts['rebinds'][0]
        self.assertEqual((first['request'], first['slot'], first['prompt'], first['budget'], first['capacity']), ('req-0', 0, 300, 24, CAPACITY))
        self.assertEqual(first['widths'], (1, 2, 4))
        self.assertEqual([entry['widths'] for entry in facts['rebinds']], [(1, 2, 4), (1, 2), (1, 2, 4), (1,)])
        self.assertGreater(facts['markers'], 5)

    def test_the_audit_lines_are_parsed(self):
        facts = markers.scan(self.audited.splitlines())
        self.assertEqual(len(facts['digests']), 4)
        self.assertTrue(all(entry['equal'] and entry['slot0'] and entry['zeroed'] for entry in facts['digests']))
        self.assertTrue(facts['peaks'])

    def test_a_flag_off_log_has_no_parked_marker(self):
        with World() as world:
            request = admit(world, None, 'cold', 300, 16)
            run_to_end(request)
            request.close('cold')
            facts = markers.scan(world.lines)
        self.assertEqual([key for key in markers.KEYS if key not in ('prewarm', 'prewarm_skipped') and facts[key]], [], 'no parked marker')

    def test_the_markers_the_producers_log_start_with_the_constants_the_parser_names(self):
        for constant in (parked.BUILT_MARKER, parked.WARM_MARKER, parked.REBIND_MARKER, parked.REFUSED_MARKER, parked.PEAK_MARKER, parked.DIGEST_MARKER,
                         parked.PROGRAMS_MARKER, parked.LADDER_MARKER.split('{')[0], parked.INSTANCE_MARKER.split('{')[0], parked.OFF_MARKER.split('{')[0],
                         parked.NEGATIVE_MARKER):
            self.assertTrue(any(constant.startswith(marker) or marker.startswith(constant) for marker in markers.LEAK_MARKERS), constant)
        import dflash_packed_proposal_coordinator as coordinator
        import serving_prefill_admission as admission
        import levern_policy
        import serving_request_factory

        self.assertTrue(coordinator.PARKED_RELEASED_LINE.startswith(markers.RELEASED.replace('slot=', '')) or markers.RELEASED.startswith('[PACKED-PROPOSE] released'))
        self.assertTrue(coordinator.BOOK_QUAD_LINE.startswith(markers.BOOK_QUAD))
        self.assertTrue(admission.DRAM_AGE_LINE.startswith(markers.HOLD_AGE))
        self.assertTrue(levern_policy.COST_LINE.startswith(markers.COST))
        self.assertTrue(serving_request_factory.REBIND_FAILED_MARKER.startswith(markers.FAILED))

    def test_the_other_producers_lines_parse(self):
        import dflash_packed_proposal_coordinator as coordinator
        import levern_policy
        import serving_prefill_admission as admission

        text = '\n'.join([
            coordinator.PARKED_RELEASED_LINE.format(slot=2, quad=1, pairs=[[0, 1]]),
            coordinator.BOOK_QUAD_LINE.format(slots='0,1,2,3', round=41),
            admission.DRAM_AGE_LINE.format('req-9', 1234.4, 6),
            levern_policy.COST_LINE.format('rebind', 311.0, 391.5),
            parked.OFF_MARKER.format(3),
            parked.LADDER_MARKER.format(2, 206300000, 1),
            parked.INSTANCE_MARKER.format(3, 'device instance attributes shadow methods: commit'),
            parked.UNPARKED_MARKER.format(4, 'engine phase failed'),
            parked.REPARKED_MARKER.format(4, 2210.5),
            '[PINDIAG] sticky engine built req=r1 ms=330.5 frontier=0 prompt=4097 kind=rebind',
        ])
        facts = markers.scan(text.splitlines())
        self.assertEqual(facts['released'], [dict(slot='2', quad=True, pairs='[[0, 1]]')])
        self.assertEqual(facts['book_quads'], [dict(slots='0,1,2,3', round=41)])
        self.assertEqual(facts['hold_ages'], [dict(request='req-9', ms=1234, decodes=6)])
        self.assertEqual(facts['costs'], [dict(kind='rebind', ms=311.0, ewma=391.5)])
        self.assertEqual(facts['off'], [dict(unparked=3)])
        self.assertEqual(facts['ladder'], [dict(rung=2, freed=206300000, slot=1)])
        self.assertEqual(facts['instance'][0]['slot'], 3)
        self.assertEqual(facts['unparked'], [dict(slot=4, reason='engine phase failed')])
        self.assertEqual(facts['reparked'], [dict(slot=4, ms=2210.5)])
        self.assertEqual(facts['sticky'][0]['kind'], 'rebind')


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = real_log()
        cls.audited = real_log(audit=True)
        cls.park_fault = real_log(fault='park')
        cls.rebind_fault = real_log(fault='rebind')

    def verdict(self, text, **env):
        return judge.judge(dict(ENV, **dict({'QWEN_FAST_M3_BLOCKS': '1'}, **env)), text)[0]

    def test_a_clean_parked_arm_passes_and_so_do_the_audited_one_and_the_faulted_ones(self):
        self.assertEqual(self.verdict(self.plain), [])
        self.assertEqual(self.verdict(self.audited, QWEN_FAST_PARKED_AUDIT='1'), [])
        self.assertEqual(self.verdict(self.park_fault, QWEN_FAST_PARKED_FAULT='park'), [])
        self.assertEqual(self.verdict(self.rebind_fault, QWEN_FAST_PARKED_FAULT='rebind'), [])

    def test_the_kill_switch_trigger_latches_once_after_n_rebinds_and_nothing_rebinds_after(self):
        off = '[PINDIAG] parked engines off (kill switch): unparked=4\n'
        rebinds = len(markers.scan(self.plain.splitlines())['rebinds'])
        self.assertEqual(rebinds, 4)
        sticky = ''.join('[PINDIAG] sticky engine built req=req-%d ms=300.0 frontier=0 prompt=300 kind=rebind' % index + chr(10) for index in range(4))
        sticky += '[PINDIAG] sticky engine built req=cold-9 ms=2400.0 frontier=0 prompt=300 kind=build' + chr(10)
        self.assertEqual(self.verdict(self.plain + sticky + off, QWEN_FAST_PARKED_OFF_AFTER='4'), [])
        found = self.verdict(self.plain + sticky, QWEN_FAST_PARKED_OFF_AFTER='4')
        self.assertTrue(any('0 kill-switch lines' in text for text in found), found)
        found = self.verdict(self.plain + sticky + off + off, QWEN_FAST_PARKED_OFF_AFTER='4')
        self.assertTrue(any('2 kill-switch lines' in text for text in found), found)
        found = self.verdict(self.plain + sticky + off, QWEN_FAST_PARKED_OFF_AFTER='3')
        self.assertTrue(any('4 rebinds under QWEN_FAST_PARKED_OFF_AFTER=3' in text for text in found), found)

    def test_a_fault_the_profile_did_not_inject_is_a_finding(self):
        found = self.verdict(self.park_fault)
        self.assertTrue(any('slots unparked' in text for text in found), found)
        found = self.verdict(self.rebind_fault)
        self.assertTrue(any('rebind refusals' in text for text in found), found)

    def test_a_flag_off_profile_whose_log_carries_a_parked_marker_is_a_finding(self):
        found, facts = judge.judge({'QWEN_FAST_M3_BLOCKS': '2'}, self.plain)
        self.assertTrue(any('leaves QWEN_FAST_PARKED_ENGINES off' in text for text in found), found)
        self.assertEqual(judge.judge({}, '[PINDIAG] pool slot 0 acquired for x\n')[0], [])
        found, _ = judge.judge({}, '[PINDIAG] sticky engine built req=r ms=1.0 frontier=0 prompt=9 kind=build\n')
        self.assertTrue(any('kind=' in text for text in found))

    def test_the_attach_rules(self):
        short = self.plain.replace('built k=4 of 4', 'built k=3 of 4')
        self.assertTrue(any('built 3 of 4' in text for text in self.verdict(short)))
        self.assertTrue(any('no "[PINDIAG] parked engines built' in text for text in self.verdict('')))
        stopped = self.plain + '[PINDIAG] parked engines stopped at k=2 of 4: short of free (free=1 largest_free=2 need=3)\n'
        self.assertTrue(any('stopped at k=2' in text for text in self.verdict(stopped)))
        cold = self.plain.replace('P=2048 ms', 'P=1 ms')
        self.assertTrue(any('not 2048' in text for text in self.verdict(cold)))
        unwarmed = self.plain.replace('count=7', 'count=0').replace('prewarm_pairs=7', 'prewarm_pairs=0')
        self.assertTrue(any('no publish prewarm' in text for text in self.verdict(unwarmed)))
        self.assertEqual(self.verdict(self.plain.replace('built k=4 of 4', 'built k=8 of 8'), QWEN_FAST_M3_BLOCKS='2'), [])

    def test_a_rebind_must_carry_a_cold_engines_widths(self):
        lines = self.plain.splitlines()
        wide = [line.replace('widths=1,2 ', 'widths=1,2,4 ') for line in lines]
        found = self.verdict('\n'.join(wide))
        self.assertTrue(any('R4' in text and 'not the cold engine' in text for text in found), found)
        # the negative control's own arm is exempt (it is judged the other way round)
        self.assertEqual(self.verdict('\n'.join(wide), QWEN_FAST_PARKED_NEGATIVE='widths'), [])
        zero = [re.sub(r' gen=[0-9]+ ', ' gen=0 ', line) for line in lines]
        self.assertTrue(any('generation 0' in text for text in self.verdict('\n'.join(zero))))

    def test_a_rebind_that_compiled_a_program_is_the_s8_class(self):
        programs = '\n'.join(['[PINDIAG] parked programs rebind slot=0 programs=849->849', '[PINDIAG] parked programs rebind slot=1 programs=849->852'])
        found = self.verdict(self.plain + programs + '\n')
        self.assertTrue(any('compiled 3 program' in text for text in found), found)
        clean = '\n'.join(['[PINDIAG] parked programs rebind slot=0 programs=849->900', '[PINDIAG] parked programs rebind slot=1 programs=900->900'])
        self.assertEqual(self.verdict(self.plain + clean + '\n'), [], 'the first rebind may compile (the warm of its shapes); later ones may not')

    def test_a_first_rebind_of_a_short_prompt_length_may_compile_but_a_long_or_repeated_one_may_not(self):
        line = '[PINDIAG] parked programs rebind slot=%d programs=%d->%d P=%d'
        first = line % (0, 849, 849, 2048)
        short = '\n'.join([first, line % (1, 849, 855, 63), line % (0, 855, 855, 63), line % (1, 855, 861, 129)])
        self.assertEqual(self.verdict(self.plain + short + '\n'), [], 'the first rebind of each short length compiles what a cold build compiles')
        again = '\n'.join([first, line % (1, 849, 855, 63), line % (0, 855, 858, 63)])
        self.assertTrue(any('compiled 3 program' in text for text in self.verdict(self.plain + again + '\n')))
        long = '\n'.join([first, line % (1, 849, 855, 2048)])
        self.assertTrue(any('compiled 6 program' in text for text in self.verdict(self.plain + long + '\n')), 'from 2048 on the attach warm covers every program')
        self.assertEqual(markers.scan(['[PINDIAG] parked programs rebind slot=0 programs=1->2'])['programs'][0]['prompt'], None)

    def test_the_memory_churn_rules_the_log_can_show(self):
        def before(chip, floor, trace):
            return ('[MEMLEDGER] before op=engine point=req=r chip%d largest_free=900.0MB free=4.000GB estimate=500.0MB margin=900.0MB floor=%.1fMB '
                    'contiguous=300.0MB trace_used=%.1fMB trace_largest_free=80.0MB' % (chip, floor, trace))
        book = '[PINDIAG] parked draft book quad slots=0,1,2,3 captured_at_round=9'
        good = chr(10).join([before(0, 600.0, 400.0), book, before(0, 480.0, 440.0), before(1, 520.0, 440.2)]) + chr(10)
        self.assertEqual(judge.memory_problems(good), [])
        low = chr(10).join([before(0, 600.0, 400.0), before(0, 120.0, 400.0)]) + chr(10)
        self.assertTrue(any('floor fell to 120.0 MB' in text for text in judge.memory_problems(low)))
        moved = chr(10).join([before(0, 600.0, 400.0), book, before(0, 600.0, 440.0), before(0, 600.0, 447.5)]) + chr(10)
        self.assertTrue(any('trace region moved 7.5 MB' in text for text in judge.memory_problems(moved)))
        hold = '[PINDIAG] dram hold prompt=253920 largest_free=900.0MB need=1300.0MB request=r1 decodes=7 free=4000.0MB trace_largest_free=80.0MB short=%s'
        self.assertEqual(judge.memory_problems(hold % 'free+contiguous'), [])
        self.assertTrue(any('names short=none' in text for text in judge.memory_problems(hold % 'none')))
        self.assertEqual(judge.memory_problems(''), [])

    def test_failures_and_the_audit(self):
        failed = self.plain + '[PINDIAG] parked rebind failed req=r slot=1 reason=RuntimeError: x fallback=build\n'
        self.assertTrue(any('failed after their first device write' in text for text in self.verdict(failed)))
        missing = '\n'.join(line for line in self.audited.splitlines() if 'rebind digest' not in line)
        self.assertTrue(any('went unaudited' in text for text in self.verdict(missing, QWEN_FAST_PARKED_AUDIT='1')))
        unequal = self.audited.replace('equal=1', 'equal=0')
        self.assertTrue(any('not clean' in text for text in self.verdict(unequal, QWEN_FAST_PARKED_AUDIT='1')))
        self.assertTrue(any('without QWEN_FAST_PARKED_AUDIT' in text for text in self.verdict(self.audited)))

    def test_instance_state_holds_and_the_book(self):
        instance = self.plain + '[PINDIAG] parked instance state slot=2 problem=device gained instance attributes: x\n'
        self.assertTrue(any('kept per-request state' in text for text in self.verdict(instance)))
        slow = self.plain + '[PINDIAG] dram hold age request=r ms=%d decodes=3\n' % (judge.HOLD_AGE_LIMIT_MS + 1)
        self.assertTrue(any('did not clear without a drain' in text for text in self.verdict(slow)))
        twice = self.plain + ('[PINDIAG] parked draft book quad slots=0,1,2,3 captured_at_round=%d\n' * 2) % (10, 90)
        self.assertTrue(any('captured 2 times' in text for text in self.verdict(twice, QWEN_FAST_PARKED_DRAFTS='1')))
        released = self.plain + '[PACKED-PROPOSE] released parked slot=1 quad=0 pairs=[[0, 1]]\n'
        self.assertTrue(any('traces are the slots' in text for text in self.verdict(released, QWEN_FAST_PARKED_DRAFTS='1')))
        self.assertEqual(self.verdict(released), [], 'E1 releases at a park')
        self.assertTrue(any('without QWEN_FAST_PARKED_DRAFTS' in text for text in self.verdict(twice)))

    def test_the_sticky_line_says_which_it_was(self):
        rebind = self.plain + '[PINDIAG] sticky engine built req=req-0 ms=300.0 frontier=0 prompt=300 kind=rebind\n'
        self.assertEqual(self.verdict(rebind), [])
        wrong = self.plain + '[PINDIAG] sticky engine built req=req-0 ms=300.0 frontier=0 prompt=300 kind=build\n'
        self.assertTrue(any('says kind=build but its rebind line is present' in text for text in self.verdict(wrong)))
        bare = self.plain + '[PINDIAG] sticky engine built req=req-0 ms=300.0 frontier=0 prompt=300\n'
        self.assertTrue(any('no kind=' in text for text in self.verdict(bare)))

    def test_er0s_numeric_rules(self):
        def ledger(phase, chip, allocated, free):
            return '[MEMLEDGER] phase=%s point=x chip%d allocated=%.3fGB free=%.3fGB largest_free=%.1fMB total=33.0GB' % (phase, chip, allocated, free, free * 900)
        good = '\n'.join(ledger('P7', chip, 24.78, 8.87) for chip in range(4)) + '\n' + '\n'.join(
            ledger('P7p', chip, 24.78 + 8 * 0.48, 8.87 - 8 * 0.48) for chip in range(4)) + '\n'
        facts = markers.scan(good.splitlines())
        self.assertEqual(judge.ledger_problems(facts, 8, False), [])
        heavy = good.replace('28.620', '29.900')
        self.assertTrue(any('took' in text for text in judge.ledger_problems(markers.scan(heavy.splitlines()), 8, False)))
        thin = '\n'.join(ledger('P7', chip, 26.04, 7.61) for chip in range(4)) + '\n' + '\n'.join(
            ledger('P7p', chip, 26.04 + 8 * 0.48, 3.0) for chip in range(4)) + '\n'
        self.assertTrue(any('floor' in text for text in judge.ledger_problems(markers.scan(thin.splitlines()), 8, True)))
        self.assertEqual(judge.ledger_problems(markers.scan([]), 8, True), [])


def smoke_log(results):
    return 'SMOKE_JSON ' + json.dumps(results) + '\n'


def user(content, tokens=16, finish='length', **extra):
    return dict(content_sha256=content, reasoning_sha256='r', tokens=tokens, finish=finish, ttft=1.0, **extra)


def row(content, length, tokens=16, finish='length'):
    return dict(content_sha256=content, tokens=tokens, finish=finish, prompt_tokens=length, prompt_tokens_sent=length, status=200)


def rounds(sequences):
    """Container-log round lines: one request per sequence of (prefix, emitted)."""
    lines = []
    for index, sequence in enumerate(sequences):
        for prefix, emitted in sequence:
            lines.append('[SEQ-PUBLISH] request=cmpl-%d rows=4 prefix=%d step_ms=3.20 emitted=%d' % (index, prefix, emitted))
    return '\n'.join(lines) + '\n'


class CompareTests(unittest.TestCase):
    def arms(self, change=None):
        control = {'parked_equal': dict(prompts={'64': row('a', 64), '2048': row('b', 2048)}),
                   'parked_churn': dict(users=[user('c'), user('d')])}
        parked_arm = json.loads(json.dumps(control))
        if change:
            change(parked_arm)
        return smoke_log(control), smoke_log(parked_arm)

    def test_identical_arms_pass_and_a_changed_hash_is_a_mismatch(self):
        control, other = self.arms()
        problems, compared, lines = parked_compare.compare(control, other)
        self.assertEqual((problems, compared), ([], 4))

        def change(arm):
            arm['parked_equal']['prompts']['2048']['content_sha256'] = 'x'
        control, other = self.arms(change)
        problems, compared, lines = parked_compare.compare(control, other)
        self.assertTrue(any('content hash differs' in text for text in problems), problems)

    def test_the_container_logs_add_the_judge_and_the_equivalence(self):
        control, other = self.arms()
        sequence = rounds([[(3, 4), (2, 3)], [(4, 5)]])
        problems, compared, lines = parked_compare.compare(control, other, sequence, sequence, env=dict(ENV, QWEN_FAST_M3_BLOCKS='1'))
        self.assertTrue(any('parked arm: ' in text for text in problems), 'the sequence log has no parked attach: the judge says so')
        different = rounds([[(3, 4), (1, 2)], [(4, 5)]])
        solo, shortfalls, facts = judge.equivalence(sequence, different)
        self.assertEqual(len(solo), 1)
        self.assertIn('round 1', solo[0])
        self.assertEqual(judge.equivalence(sequence, sequence)[0], [])
        self.assertTrue(judge.equivalence('', sequence)[1])
        live, _, detail = judge.equivalence(rounds([[(3, 4)] * 30]), rounds([[(3, 4)] * 20 + [(1, 1)] * 10]), solo=False)
        self.assertTrue(live and 'concurrent acceptance' in live[0], live)

    def test_exact_equivalence_binds_the_solo_requests_and_concurrent_ones_only_by_acceptance(self):
        def lines(entries):
            return '\n'.join('[%s] request=%s prefix=%d emitted=%d' % entry for entry in entries) + '\n'
        solo = [('SEQUENTIAL', 'solo-a', 5, 3), ('SEQUENTIAL', 'solo-a', 8, 2)]
        busy_one = [('PACKED', 'x', 1, 4), ('SEQUENTIAL', 'y', 1, 4), ('PACKED', 'x', 5, 4), ('SEQUENTIAL', 'y', 5, 4)]
        busy_two = [('SEQUENTIAL', 'x', 1, 4), ('PACKED', 'y', 1, 4), ('SEQUENTIAL', 'x', 5, 4), ('PACKED', 'y', 5, 4)]
        control, other = lines(solo + busy_one), lines(solo + busy_two)
        self.assertEqual([name for name, _ in judge.accepted_sequences(control, 'solo')], ['solo-a'])
        self.assertEqual([name for name, _ in judge.accepted_sequences(control, 'concurrent')], ['x', 'y'])
        self.assertTrue(judge.equivalence(control, other)[0], 'over the whole log the path letters differ')
        self.assertEqual(judge.equivalence(control, other, only='solo')[0], [])
        self.assertEqual(judge.equivalence(control, other, solo=False, only='concurrent')[0], [])
        a, b = parked_compare.compare(*self.arms(), control, other)[:2]
        self.assertEqual([text for text in a if 'drafter equivalence' in text or 'concurrent drafting' in text], [])
        changed = lines([('SEQUENTIAL', 'solo-a', 5, 3), ('SEQUENTIAL', 'solo-a', 8, 1)] + busy_two)
        self.assertTrue(any('drafter equivalence' in text for text in parked_compare.compare(*self.arms(), control, changed)[0]))
        # the drafter negative control is judged on the solo requests: identical solo sequences fail it however the busy ones differ
        problems, _ = judge.negative_verdict('drafter', [], control, other)
        self.assertTrue(any('drafted exactly like the flag-off arm' in text for text in problems), problems)
        problems, _ = judge.negative_verdict('drafter', [], control, changed)
        self.assertEqual(problems, [])
        with self.assertRaises(ValueError):
            judge.accepted_sequences(control, 'both')

    def test_a_comparison_with_nothing_in_common_is_not_a_pass(self):
        self.assertEqual(parked_compare.main(['--control', '/nonexistent-a', '--parked', '/nonexistent-b']), 2)

    def test_the_negative_controls_must_fail_their_own_judges(self):
        control, other = self.arms()
        for kind in ('carry', 'pages'):
            problems, compared, lines = parked_compare.compare(control, other, mode=kind)
            self.assertTrue(any('kept every token' in text for text in problems), (kind, problems))

            def change(arm):
                arm['parked_equal']['prompts']['64']['content_sha256'] = 'diverged'
            control, diverged = self.arms(change)
            problems, compared, lines = parked_compare.compare(control, diverged, mode=kind)
            self.assertEqual(problems, [], kind)
        # drafter: every token kept AND the equivalence judge fails
        control, same = self.arms()
        sequence, other_sequence = rounds([[(3, 4), (2, 3)]]), rounds([[(3, 4), (0, 1)]])
        problems, compared, lines = parked_compare.compare(control, same, sequence, other_sequence, mode='drafter')
        self.assertEqual(problems, [])
        problems, compared, lines = parked_compare.compare(control, same, sequence, sequence, mode='drafter')
        self.assertTrue(any('drafted exactly like the flag-off arm' in text for text in problems), problems)
        # widths: the ticket-width line is its judge
        wide = self.mk_widths_log(bad=True)
        problems, facts = judge.negative_verdict('widths', [], '', wide)
        self.assertEqual(problems, [])
        problems, facts = judge.negative_verdict('widths', [], '', self.mk_widths_log(bad=False))
        self.assertTrue(any('R4' in text for text in problems))
        with self.assertRaises(ValueError):
            judge.negative_verdict('everything', [], '', '')

    def mk_widths_log(self, bad):
        return ('[PINDIAG] parked rebind req=r slot=0 gen=1 P=300 budget=3 ms=10.0 single_rebuilt=0 window=256 widths=%s capacity=%d\n'
                % ('1,2,4' if bad else '1,2', CAPACITY))

    def test_er5_pairs_the_turns_by_index_and_never_decides(self):
        def turns(wall, gap):
            return dict(users=[dict(started_at=0.0, ended_at=wall + index, longest_gap_s=gap, ttft=1.0) for index in range(4)])
        control = smoke_log({'parked_turns': dict(turns(30.0, 2.5), makespan_s=200.0, tokens_per_s=40.0, completion_tokens=8000)})
        other = smoke_log({'parked_turns': dict(turns(20.0, 0.5), makespan_s=150.0, tokens_per_s=55.0, completion_tokens=8000)})
        problems, compared, lines = parked_compare.compare(control, other)
        text = '\n'.join(lines)
        self.assertIn('"turns": 4', text)
        self.assertIn('"parked_faster": 4', text)
        self.assertIn('makespan_s=150.0', text)
        rows, summary = judge.paired_turns([], [])
        self.assertEqual(summary['turns'], 0)


class SmokeChecksTests(unittest.TestCase):
    def test_the_row_checks(self):
        results = {'parked_equal': dict(prompts={'64': row('a', 64)}), 'parked_budgets': dict(prompts={'1': row('a', 4097, tokens=1), '4': row('b', 4097, tokens=4)})}
        self.assertEqual(check.parked_smoke_problems(results), [])
        broken = json.loads(json.dumps(results))
        broken['parked_equal']['prompts']['64']['prompt_tokens'] = 65
        broken['parked_budgets']['prompts']['1']['tokens'] = 2
        broken['parked_budgets']['prompts']['4']['tokens'] = 9
        found = check.parked_smoke_problems(broken)
        self.assertTrue(any('the server counted 65 tokens of the 64' in text for text in found), found)
        self.assertTrue(any('past the budget' in text for text in found), found)
        self.assertTrue(any('a one-token budget produced' in text or 'past the budget' in text for text in found))

    def test_the_abort_and_reuse_round_ends_a_one_token_request_at_its_seed(self):
        good = {'parked_abort_reuse': dict(users=[user('a'), user('b', tokens=1, finish='length'), user('c')] * 2)}
        self.assertEqual([text for text in check.parked_smoke_problems(good) if 'one-token' in text], [])
        bad = json.loads(json.dumps(good))
        bad['parked_abort_reuse']['users'][1]['tokens'] = 7
        self.assertTrue(any('the one-token request produced 7' in text for text in check.parked_smoke_problems(bad)))

    def test_an_errored_test_is_a_problem(self):
        self.assertEqual(check.parked_smoke_problems({'parked_churn': dict(error='boom')}), ['parked_churn: boom'])

    def test_the_container_hook_passes_a_non_parked_profile_through_the_flag_off_rule(self):
        problems, facts = check.parked_container_problems({'QWEN_FAST_M3_BLOCKS': '2'}, '')
        self.assertEqual((problems, facts.get('markers')), ([], 0))
        self.assertEqual(check.parked_container_problems(None, 'anything'), ([], {}))


if __name__ == '__main__':
    unittest.main()
