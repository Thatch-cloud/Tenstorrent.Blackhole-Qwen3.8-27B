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


if __name__ == '__main__':
    unittest.main()
