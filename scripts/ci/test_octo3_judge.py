"""tp4/octo-3: the judge and smoke rules the octo shape needs on the fused stack, and the three defects of the octo-2 card runs.

  1. '[OCTO-ATTN-BUNDLE] engaged' logged twice a boot (the warm fixture and the captured fixture each build a bundle): the attach says it once (test_octo_attn_bundle holds the latch).
  2. The bundle path logs no '[QWEN-SDPA] runtime-extent entries=1 kv_share=false' line: its K64j program is the B-entry one, so the factory logs entries=4; the judge reads the bundle's launch sizes.
  3. The glue8 audit never finished: the audits sample (test_audit_sampling).
Also: the K/V page writer's octo served-path line, the round-host keyed share of an octo step, the pre-stage rules at the octo block, the permutation smoke at shape=octo, and the lever status table."""

import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as check  # noqa: E402
import draft_permute_smoke as permute_smoke  # noqa: E402
import kv_page_writer_tp4 as page_writer  # noqa: E402
import kv_page_writer_tp4_smoke as page_smoke  # noqa: E402
import octo3_levers as levers  # noqa: E402
import octo_judge  # noqa: E402
from test_octo_judge import Boot, env_of  # noqa: E402

PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern'
FACTORY_4 = '[QWEN-SDPA] runtime-extent entries=4 kv_share=false q_slice=false writer=writer_decode_qwen_slice.cpp cur_pos_stick_bytes=64'
FACTORY_1 = Boot.FACTORY


class FactoryEntriesTests(unittest.TestCase):
    def boot(self, *factory, mode='live'):
        boot = Boot(mode, programs=lambda: 5000, programs_ran=False)
        boot.lines.extend([Boot.ENGAGED, *factory])
        for _ in range(20):
            boot.round('octo')
        return boot

    def judged(self, env, boot):
        return [problem for problem in octo_judge.octo_problems(env, env['QWEN_FAST_OCTO'], octo_judge.markers.scan(boot.text()), {}, boot.text()) if 'factory' in problem]

    def test_the_wanted_launch_sizes_follow_the_bundle_flag(self):
        self.assertEqual(octo_judge.factory_entries({}), (1,))
        self.assertEqual(octo_judge.factory_entries({'QWEN_FAST_OCTO_ATTN_BUNDLE': '0'}), (1,))
        self.assertEqual(octo_judge.factory_entries({'QWEN_FAST_OCTO_ATTN_BUNDLE': '4'}), (4,))
        self.assertEqual(octo_judge.factory_entries({'QWEN_FAST_OCTO_ATTN_BUNDLE': '5'}), (4,))
        self.assertEqual(octo_judge.factory_entries({'QWEN_FAST_OCTO_ATTN_BUNDLE': '6'}), (4,))
        self.assertEqual(octo_judge.factory_entries({'QWEN_FAST_OCTO_ATTN_BUNDLE': '2'}), (2,))
        self.assertEqual(octo_judge.factory_entries({'QWEN_FAST_OCTO_ATTN_BUNDLE': '4', 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT': '1'}), (1, 4), 'the audit runs the eight single launches beside the bundled ones')

    def test_the_single_launch_log_is_the_octo_rule_as_before(self):
        env = env_of('live')
        self.assertEqual(self.judged(env, self.boot(FACTORY_1)), [])
        self.assertTrue(self.judged(env, self.boot()), 'no factory line at all')
        self.assertTrue(self.judged(env, self.boot(FACTORY_4)), 'a four-entry program is not the single launch')

    def test_the_bundle_arm_logs_entries_four_and_that_is_what_the_judge_wants(self):
        """The defect: the bundle arm's log held entries=4 (the B-entry program) and no entries=1 line, and the octo rule demanded entries=1."""
        env = env_of('live', QWEN_FAST_OCTO_ATTN_BUNDLE='4')
        self.assertEqual(self.judged(env, self.boot(FACTORY_4)), [])
        problems = self.judged(env, self.boot(FACTORY_1))
        self.assertTrue(problems and 'entries=4' in problems[0], problems)
        self.assertIn('not the single launch', problems[0])

    def test_the_audited_bundle_wants_both_sizes(self):
        env = env_of('live', QWEN_FAST_OCTO_ATTN_BUNDLE='4', QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT='1')
        self.assertEqual(self.judged(env, self.boot(FACTORY_1, FACTORY_4)), [])
        self.assertEqual(len(self.judged(env, self.boot(FACTORY_4))), 1)
        self.assertEqual(len(self.judged(env, self.boot(FACTORY_1))), 1)

    def test_the_facts_carry_the_entries_the_log_held(self):
        boot = self.boot(FACTORY_1, FACTORY_4)
        problems, facts = octo_judge.judge(env_of('live', QWEN_FAST_OCTO_ATTN_BUNDLE='4'), boot.text())
        self.assertEqual(facts['octo_factory_entries'], [1, 4])

    def test_the_k64j_factory_line_is_what_the_judge_parses(self):
        self.assertTrue(octo_judge.FACTORY_ENTRIES.search(FACTORY_4))
        self.assertEqual(octo_judge.FACTORY_ENTRIES.search(FACTORY_4).group(1), '4')
        self.assertIsNone(octo_judge.FACTORY_ENTRIES.search('[QWEN-SDPA] runtime-extent entries=2 kv_share=true q_slice=false'))


class PageWriterTests(unittest.TestCase):
    ON = {'QWEN_FAST_KV_PAGE_WRITER': '1', 'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_TP': '4'}
    ENGAGED = '[PINDIAG] tp4 kv page writer engaged cores=64 wt=2 source=l1 ordered=0 width=4096 audit=0 design=0123456789abcdef'

    def setUp(self):
        page_writer._NOTED.clear()
        self.addCleanup(page_writer._NOTED.clear)

    def test_the_octo_block_logs_its_served_path_once_and_never_a_fell_back_line(self):
        lines = []
        from unittest.mock import patch

        with patch.object(page_writer, 'log_line', side_effect=lines.append), patch.object(page_writer, 'enabled', return_value=True):
            for layer in range(16):
                self.assertIsNone(page_writer.attach(object(), object(), {}, positions=object(), pages=object(), spans=page_writer.OCTO_SPANS, launch_rows=64))
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(page_writer.OCTO_SERVED))
        self.assertNotIn('fell back', lines[0])
        self.assertEqual(page_smoke.OCTO_SERVED, page_writer.OCTO_SERVED)
        self.assertEqual(page_writer.OCTO_SPANS, tuple((8 * user, 8 * user + 8) for user in range(8)))

    def test_the_decision_reads_the_blocks_layout_so_the_warm_forward_takes_the_same_path(self):
        """The warm fixture's spans are one chain over every row; the octo block is recognised by its users' layout, in the warm forward as in the captured one (card run 38117538847)."""
        lines = []
        from unittest.mock import patch

        warm_spans = ((0, 64),)
        with patch.object(page_writer, 'log_line', side_effect=lines.append), patch.object(page_writer, 'enabled', return_value=True):
            self.assertIsNone(page_writer.attach(object(), object(), {}, positions=object(), pages=object(), spans=warm_spans, launch_rows=64, layout_spans=page_writer.OCTO_SPANS))
            self.assertIsNone(page_writer.attach(object(), object(), {}, positions=object(), pages=object(), spans=page_writer.OCTO_SPANS, launch_rows=64,
                                                 layout_spans=page_writer.OCTO_SPANS))
            self.assertEqual(len(lines), 1)
            # a four-user block is not the octo block whatever its warm chain looks like: the attach goes on to build the writer
            lines[:] = []
            m3 = ((0, 16), (16, 32), (32, 48), (48, 64))
            with patch.object(page_writer, 'KVPageWriter', side_effect=lambda *args, **kwargs: ('writer', kwargs['spans'])):
                self.assertEqual(page_writer.attach(object(), object(), {}, positions=object(), pages=object(), spans=warm_spans, launch_rows=64, layout_spans=m3),
                                 ('writer', warm_spans))
                self.assertEqual(page_writer.attach(object(), object(), {}, positions=object(), pages=object(), spans=m3, launch_rows=64)[0], 'writer')
            self.assertFalse([line for line in lines if page_writer.OCTO_SERVED in line])

    def test_an_octo_profile_must_log_the_line_and_a_plain_one_must_not(self):
        text = self.ENGAGED + '\n' + page_writer.OCTO_SERVED + ' reason=x\n'
        self.assertEqual(page_smoke.problems(self.ON, text), [])
        self.assertTrue([problem for problem in page_smoke.problems(self.ON, self.ENGAGED + '\n') if 'octo 8-row served path' in problem])
        plain = dict(self.ON, QWEN_FAST_OCTO='off')
        self.assertTrue([problem for problem in page_smoke.problems(plain, text) if 'without the octo block' in problem])
        self.assertEqual(page_smoke.problems(plain, self.ENGAGED + '\n'), [])
        twice = text + page_writer.OCTO_SERVED + ' reason=y\n'
        self.assertTrue([problem for problem in page_smoke.problems(self.ON, twice) if 'one per process' in problem])

    def test_a_real_fell_back_line_still_fails_the_arm_beside_the_octo_line(self):
        text = self.ENGAGED + '\n' + page_writer.OCTO_SERVED + ' reason=x\n' + page_writer.FELL_BACK + ' reason=spans\n'
        self.assertTrue([problem for problem in page_smoke.problems(self.ON, text) if 'fell back' in problem])
        found = check.fusion_problems(text, dict(PROFILES[BASE + '-w2-er-fx-all-octo3']['env']))
        self.assertTrue([problem for problem in found if 'K/V page writer fell back' in problem])

    def test_the_octo_line_alone_is_not_a_fusion_fall_back(self):
        text = page_writer.OCTO_SERVED + ' reason=x\n'
        found = check.fusion_problems(text, {'QWEN_FAST_KV_PAGE_WRITER': '1'})
        self.assertFalse([problem for problem in found if 'fell back' in problem])


class RoundHostKeyedShareTests(unittest.TestCase):
    """An eight-live M3 step verifies twice (v0, v1); an octo step once on the one block (v1 is '-')."""

    FIELDS = ('gap', 'entry', 'v0', 'v0_in', 'v0_tr', 'v0_sy', 'v0_rb', 'c0', 'v1', 'v1_in', 'v1_tr', 'v1_sy', 'v1_rb', 'c1', 'ed', 'ed_pre', 'launch', 'window', 'fence', 'collect', 'flush',
              'select', 'tail', 'step')

    def line(self, number, octo, keyed):
        values = {name: '1.00' for name in self.FIELDS}
        if octo:
            for name in ('v1', 'v1_in', 'v1_tr', 'v1_sy', 'v1_rb', 'c1'):
                values[name] = '-'
        fields = ' '.join('%s=%s' % (name, values[name]) for name in self.FIELDS)
        return '[PACKED-ROUND-HOST] round=%d live=8 pos=4000 %s sel=1 read=1 keyed=%d guard=1' % (number, fields, keyed)

    def judge(self, octo, keyed):
        env = {'QWEN_FAST_TP4_ROUND_HOST_KEYED': '1', 'QWEN_FAST_TP4_ROUND_HOST_LEAN': '1'}
        text = '\n'.join(self.line(number, octo, keyed) for number in range(1, 80))
        return [problem for problem in check.round_host_problems(env, text + '\n', True)[0] if 'keyed write' in problem]

    def test_an_octo_step_with_its_one_verify_keyed_is_a_full_share(self):
        self.assertEqual(self.judge(octo=True, keyed=1), [], 'the old count of two verifies per eight-live step capped the share at 0.5 and failed this')

    def test_an_m3_step_with_one_keyed_verify_of_two_is_the_half_share_as_before(self):
        self.assertEqual(self.judge(octo=False, keyed=1), [])
        self.assertTrue(self.judge(octo=False, keyed=0))
        self.assertTrue(self.judge(octo=True, keyed=0))


class HostGapOctoTests(unittest.TestCase):
    def test_the_diff_path_may_be_shown_at_eight_live_on_an_octo_profile(self):
        facts = check.hostgap_facts('[PACKED-PRESTAGE] round=3 path=diff buffers=150 reason=- live=8\n')
        self.assertEqual((facts['hostgap_live4_diff'], facts['hostgap_live8_diff']), (0, 1))

    def test_the_audit_wants_the_octo_block_label_on_an_octo_profile(self):
        source = (HERE / 'c2_smoke_check.py').read_text(encoding='utf-8')
        self.assertIn("wanted = ['C']", source)
        self.assertIn("octo_blocks(env) and facts['hostgap_live8_diff']", source)


class PermuteSmokeOctoTests(unittest.TestCase):
    ON = {'QWEN_FAST_DRAFT_PERMUTE': '1', 'QWEN_FAST_QUAD_DRAFT': '1', 'QWEN_FAST_OCTO_DRAFT': '1'}

    def log(self, shapes, audited=False):
        out = []
        for shape in shapes:
            for site in permute_smoke.SITES:
                out.append('[PINDIAG] tp4 draft permute engaged site=%s shape=%s halves=2 users=4' % (site, shape))
                if audited:
                    out.append('[PINDIAG] tp4 draft permute audit exact=True site=%s shape=%s tensors=4' % (site, shape))
        return '\n'.join(out) + '\n'

    def test_the_octo_fold_and_unfold_are_demanded_now_that_they_are_hooked(self):
        self.assertEqual(permute_smoke.problems(self.ON, self.log(['quad', 'octo'])), [])
        found = permute_smoke.problems(self.ON, self.log(['quad']))
        self.assertEqual(len(found), 3)
        self.assertTrue(all('shape=octo' in problem for problem in found))

    def test_the_audit_must_have_compared_at_the_octo_pass_too(self):
        env = dict(self.ON, QWEN_FAST_DRAFT_PERMUTE_AUDIT='1')
        self.assertEqual(permute_smoke.problems(env, self.log(['quad', 'octo'], audited=True)), [])
        text = self.log(['quad'], audited=True) + self.log(['octo'])
        self.assertEqual(len(permute_smoke.problems(env, text)), 3)


class LeverTableTests(unittest.TestCase):
    def combined_over_production(self):
        production = PROFILES[BASE + '-w2-er-traffic']['env']
        combined = PROFILES[BASE + '-w2-er-fx-all-oneq-grid13']['env']
        flags = {key for key, value in combined.items() if production.get(key) != value}
        return flags | {key for key, value in production.items() if key in ('QWEN_FAST_TP4_SDPA', 'QWEN_FAST_TP4_CONV_GATES_SPREAD', 'QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS',
                                                                         'QWEN_FAST_LEVERN_BUILD_MS')}

    def test_every_lever_the_combined_arm_adds_has_an_octo_status(self):
        flags = {flag for flag in self.combined_over_production() if not flag.endswith('_AUDIT')}
        missing = sorted(flag for flag in flags if flag not in levers.LEVERS)
        self.assertEqual(missing, [], 'a lever added to the combined arm needs a row of octo3_levers.LEVERS (native, octo-served, m3-only, drafter, host or prefill)')
        for flag, (status, note) in levers.LEVERS.items():
            self.assertIn(status, levers.STATUSES, flag)
            self.assertGreater(len(note), 20, flag)

    def test_the_octo_served_lever_is_the_page_writer_alone_and_its_line_is_the_judged_one(self):
        self.assertEqual([flag for flag, (status, note) in levers.LEVERS.items() if status == levers.SERVED], ['QWEN_FAST_KV_PAGE_WRITER'])
        self.assertIn(page_writer.OCTO_SERVED, levers.LEVERS['QWEN_FAST_KV_PAGE_WRITER'][1])

    def test_the_doc_lists_every_lever_and_its_status(self):
        text = (HERE.parent.parent / 'docs' / 'tp4-octo3.md').read_text(encoding='utf-8')
        for flag, status, note in levers.table():
            self.assertIn('`%s`' % flag, text, flag)
        self.assertEqual(len(re.findall(r'(?m)^\| `QWEN_', text)) >= len(levers.LEVERS), True)



class OctoModeJudgingTests(unittest.TestCase):
    """Card run 38129663805 (the audited attach of the combined candidate, tp4/ship-4) read red on rules that assume the M3 quads: under QWEN_FAST_OCTO=live with the octo draft the
    eight-seat rounds are octo rounds drafted by ONE pass over eight slots, the second quad is never built, and the keyed round-host write is off while the T2 audit asks for its lines."""

    OCTO = {'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2', 'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_OCTO_DRAFT': '1'}
    FACTS = dict(quad_markers=1, quad_marker_slots=['0,1,2,3'], quads_rounds=0, octo_pass_rounds=147, quad_fallbacks=0, quad_disabled=0,
                 pooled_mask_slots=['0,1,2,3'], pooled_output_slots=['0,1,2,3'], pooled_refused=0, singles_groups_equal=['0,1,2,3'])

    def problems(self, env, **changes):
        return check.blocks_problems(dict(self.FACTS, **changes), env, True)

    def test_the_log_of_the_card_run_is_clean_under_the_octo_draft(self):
        self.assertEqual(self.problems(self.OCTO), [])
        self.assertEqual(self.problems(dict(self.OCTO, QWEN_FAST_DRAFT_SINGLES_AUDIT='1')), [])

    def test_the_first_quad_is_still_required_and_the_second_at_most_once(self):
        self.assertTrue(any('quad marker' in p for p in self.problems(self.OCTO, quad_markers=0, quad_marker_slots=[])))
        self.assertTrue(any('quad marker' in p for p in self.problems(self.OCTO, quad_markers=1, quad_marker_slots=['4,5,6,7'])))
        self.assertEqual(self.problems(self.OCTO, quad_markers=2, quad_marker_slots=['0,1,2,3', '4,5,6,7'], pooled_mask_slots=['0,1,2,3', '4,5,6,7'],
                                       pooled_output_slots=['0,1,2,3', '4,5,6,7']), [])
        self.assertTrue(any('quad marker' in p for p in self.problems(self.OCTO, quad_markers=2, quad_marker_slots=['0,1,2,3', '0,1,2,3'])))

    def test_a_round_must_have_been_drafted_by_the_octo_pass_or_by_both_quads(self):
        self.assertTrue(any('no round was drafted by the octo pass' in p for p in self.problems(self.OCTO, octo_pass_rounds=0)))
        self.assertEqual(self.problems(self.OCTO, octo_pass_rounds=0, quads_rounds=3), [])

    def test_a_fallback_or_a_disable_still_fails_under_the_octo_draft(self):
        self.assertTrue(any('fell back 2 times and was disabled 1' in p for p in self.problems(self.OCTO, quad_fallbacks=2, quad_disabled=1)))

    def test_the_m3_rules_are_what_they_were_without_the_octo_draft_and_off_octo(self):
        for env in (dict(self.OCTO, QWEN_FAST_OCTO_DRAFT='0'), {'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2'}, dict(self.OCTO, QWEN_FAST_OCTO='alternate')):
            with self.subTest(env=sorted(env.items())):
                found = self.problems(env)
                self.assertTrue(any('quad marker' in p and 'once for each of 0,1,2,3 and 4,5,6,7' in p for p in found), found)
                self.assertTrue(any('no round was served by both quads' in p for p in found), found)
                self.assertTrue(any('pooled' in p and '4,5,6,7' in p for p in found), found)
        both = dict(self.FACTS, quad_markers=2, quad_marker_slots=['0,1,2,3', '4,5,6,7'], quads_rounds=9, octo_pass_rounds=0, pooled_mask_slots=['0,1,2,3', '4,5,6,7'],
                    pooled_output_slots=['0,1,2,3', '4,5,6,7'])
        self.assertEqual(check.blocks_problems(both, dict(self.OCTO, QWEN_FAST_OCTO_DRAFT='0'), True), [])

    def test_the_facts_count_the_octo_pass_selection(self):
        text = '[PACKED-SELECT] round=3 pairs=[[0, 1, 2, 3, 4, 5, 6, 7]] users=8 calls=1 collect_ms=4.4 select_ms=2.8\n[PACKED-SELECT] round=4 pairs=[[0, 1, 2, 3], [4, 5]] users=6 calls=1\n'
        facts = check.draft_facts(text)
        self.assertEqual((facts['octo_pass_rounds'], facts['quads_rounds']), (1, 0))

    def test_the_candidates_profiles_are_judged_as_octo_pass_arms(self):
        for name in sorted(PROFILES):
            if name.startswith(BASE + '-w2-er-fx-') and 'QWEN_FAST_OCTO_DRAFT' in PROFILES[name]['env'] and PROFILES[name]['env'].get('QWEN_FAST_OCTO') == 'live':
                self.assertTrue(check.octo_pass_mode(PROFILES[name]['env']), name)
        self.assertFalse(check.octo_pass_mode({}))


class KeyedUnderTheT2AuditTests(unittest.TestCase):
    """verify_prestage.key_matches returns False while QWEN_FAST_VERIFY_T2_AUDIT=1 (an audited verify takes the diff that writes the audit's lines): the audited arm takes no keyed write, M3 or octo."""

    FLAGS = {'QWEN_FAST_TP4_ROUND_HOST_LOG': '1', 'QWEN_FAST_TP4_ROUND_HOST_SELECT': '1', 'QWEN_FAST_TP4_ROUND_HOST_READ': '1', 'QWEN_FAST_TP4_ROUND_HOST_KEYED': '1',
             'QWEN_FAST_TP4_ROUND_HOST_LEAN': '1', 'QWEN_FAST_TP4_ROUND_HOST_AUDIT': '1'}

    def text(self, keyed=0):
        import round_host

        engaged = '[PINDIAG] round host engaged ' + ' '.join('%s=%d' % (name.rsplit('_', 1)[-1].lower(), int(self.FLAGS.get(name) == '1')) for name in round_host.FLAGS)
        lines = [engaged]
        for number in range(120):
            values = {name: 1.0 for name in round_host.Ledger.FIELDS}
            lines.append(round_host.format_line(number + 1, 8, values, dict(select=1, read=4, keyed=keyed, guard=0), {}, 1000 + number))
        lines += ['[ROUND-HOST-AUDIT] kind=select equal=1', '[ROUND-HOST-AUDIT] kind=merge equal=1']
        return '\n'.join(lines) + '\n'

    def test_the_audited_octo_twin_is_not_held_to_a_keyed_share_or_a_keyed_audit_line(self):
        env = dict(self.FLAGS, QWEN_FAST_VERIFY_T2_AUDIT='1')
        problems, facts = check.round_host_problems(env, self.text(keyed=0), True)
        self.assertEqual(problems, [])
        self.assertEqual(facts.get('round_host_keyed_exempt'), 'verify_t2_audit')
        self.assertNotIn('round_host_keyed_share', facts)

    def test_without_the_t2_audit_both_rules_stand(self):
        for value in ('0', None):
            env = dict(self.FLAGS, **({} if value is None else {'QWEN_FAST_VERIFY_T2_AUDIT': value}))
            problems, facts = check.round_host_problems(env, self.text(keyed=0), True)
            self.assertTrue(any('took the keyed write' in p for p in problems), problems)
            self.assertTrue(any('kind=keyed' in p for p in problems), problems)
        problems, facts = check.round_host_problems(dict(self.FLAGS), self.text(keyed=1) + '[ROUND-HOST-AUDIT] kind=keyed equal=1 checked=149 changed=0\n', True)
        self.assertEqual(problems, [])

    def test_the_keyed_share_is_reported_in_the_facts_the_smoke_check_prints(self):
        """G1's SMOKE_CHECK line read keyed_share: null although the rule had computed it: found.update(facts) ran before the share was set."""
        problems, facts = check.round_host_problems(dict(self.FLAGS), self.text(keyed=2) + '[ROUND-HOST-AUDIT] kind=keyed equal=1 checked=149 changed=0\n', True)
        self.assertEqual(problems, [])
        self.assertEqual(facts['round_host_keyed_share'], 1.0)           # (an eight-live M3 step verifies twice: two keyed writes of two possible)

    def test_the_candidates_own_profiles_say_which_arm_is_exempt(self):
        for name in sorted(PROFILES):
            env = PROFILES[name]['env']
            if name.startswith(BASE + '-w2-er-fx-') and env.get('QWEN_FAST_TP4_ROUND_HOST_KEYED') == '1' and env.get('QWEN_FAST_VERIFY_T2_AUDIT') == '1':
                self.assertEqual(env.get('QWEN_FAST_TP4_ROUND_HOST_AUDIT'), '1', name)
                self.assertTrue(PROFILES[name].get('gate_only') is True, name)


class OctoDramFloorTests(unittest.TestCase):
    """The free DRAM after the eight parked engines: 2.186 GB a chip measured on the audited octo arm (card run 38129663805), the four-user stack's floors (3.6 audited, 4.7 audits-off) are not its."""

    def ledger(self, free):
        facts = '\n'.join('[MEMLEDGER] phase=P7 point=x chip%d allocated=%.3fGB free=%.3fGB largest_free=%.1fMB total=33.508GB' % (chip, 33.508 - free - 3.346, free + 3.346, (free + 3.3) * 900)
                         for chip in range(4)) + '\n'
        facts += '\n'.join('[MEMLEDGER] phase=P7p point=parked k=8 chip%d allocated=%.3fGB free=%.3fGB largest_free=%.1fMB total=33.5GB' % (chip, 33.508 - free, free, free * 900)
                          for chip in range(4)) + '\n'
        import parked_markers as markers

        return markers.scan(facts.splitlines())

    def test_the_measured_audited_reading_passes_the_octo_floor_and_fails_the_four_user_one(self):
        import parked_judge as judge

        facts = self.ledger(2.186)
        band = judge.ENGINE_GB_LEVERS                    # (the fusion levers' band: 0.418 GB an engine, measured on this arm too)
        self.assertEqual(judge.ledger_problems(facts, 8, True, band, octo=True), [])
        self.assertTrue(any('3.6 GB floor of the audited arm' in p for p in judge.ledger_problems(facts, 8, True, band)))
        self.assertTrue(any('2.1 GB floor of the octo audited arm' in p for p in judge.ledger_problems(self.ledger(2.0), 8, True, band, octo=True)))

    def test_the_audits_off_octo_floor_is_the_four_user_one_and_the_measured_traffic_reading_passes_it(self):
        """G1 of the combined candidate (card run 38131783025), the traffic profile: 4.907 GB free a chip after the parked engines; the audits hold 2.72 GB of the octo stack's DRAM."""
        import parked_judge as judge

        band = judge.ENGINE_GB_LEVERS
        self.assertEqual(judge.ledger_problems(self.ledger(4.907), 8, False, band, octo=True), [])
        self.assertEqual(judge.LEDGER_FREE_FLOOR_OCTO_GB['plain'], judge.LEDGER_FREE_FLOOR_GB['plain'])
        self.assertTrue(any('4.7 GB floor of the octo audits-off arm' in p for p in judge.ledger_problems(self.ledger(4.0), 8, False, band, octo=True)))
        self.assertTrue(any('4.7 GB floor of the audits-off arm' in p for p in judge.ledger_problems(self.ledger(4.0), 8, False, band)))

    def test_the_octo_floors_are_set_by_the_flag_the_profile_sets(self):
        import parked_judge as judge

        self.assertTrue(judge.octo_on({'QWEN_FAST_OCTO': 'live'}) and judge.octo_on({'QWEN_FAST_OCTO': 'alternate'}))
        self.assertFalse(judge.octo_on({}) or judge.octo_on({'QWEN_FAST_OCTO': 'off'}))
        self.assertLess(judge.LEDGER_FREE_FLOOR_OCTO_GB['audited'], 2.186)
        self.assertLess(judge.LEDGER_FREE_FLOOR_OCTO_GB['plain'], 4.907)              # below the measured traffic reading
        self.assertGreater(judge.LEDGER_FREE_FLOOR_OCTO_GB['audited'], 1.47 + 0.25 - 1.0)   # (the arrival's need is met at steady state by the lower consumers' arithmetic in parked_judge)

if __name__ == '__main__':
    unittest.main()
