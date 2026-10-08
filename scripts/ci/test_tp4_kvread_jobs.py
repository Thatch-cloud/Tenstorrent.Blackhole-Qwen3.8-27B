"""The region-read pack (references/tp4-kvread-jobs) on the CPU: every template parses with the job parser, the ORDER rows match the files, the tags are W-1's reserve, and the pack's rules are the ones the code enforces.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_kvread_jobs` from scripts/ci."""

import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

PACK = HERE / 'references' / 'tp4-kvread-jobs'
W1 = HERE / 'references' / 'tp4-w1-levern-jobs'


def profiles():
    return sorted(json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles'])


def parsed(name):
    return job.read_job(job.parse_env((PACK / (name + '.env')).read_text(encoding='utf-8')), profiles(), root=ROOT)


def order():
    return [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]


class PackTests(unittest.TestCase):
    def test_every_template_has_an_order_row_and_parses(self):
        names = sorted(path.name[:-4] for path in PACK.glob('*.env'))
        self.assertEqual(names, sorted(row[0] for row in order()))
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(parsed(name)['cards'], 'quad')

    def test_q1_is_the_kvread_fabric_probe_on_the_window_image_alone(self):
        outputs = parsed('Q1-kvread-qualify')
        self.assertEqual((outputs['actions'], outputs['fabric_probe'], outputs['tag']), ('fabric', 'kvread', 'tp4-serve-11'))
        self.assertEqual(outputs['box_minutes'], '', 'the fabric step has no box of its own')

    def test_the_control_re_run_mounts_the_graft_on_the_production_base(self):
        outputs = parsed('P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict')
        self.assertEqual((outputs['tag'], outputs['prefix_kvread_mount'], outputs['prefix_plan']), ('tp4-serve-10', '1', 'exactness-shared,lifecycle-evict'))
        self.assertEqual(outputs['prefix_profile'], 'c2-packed-tp4-8x262k-ship-prefix-audit', "W-0's production audited profile")
        for name in ('P1b-CTLR-lifecycle-evict-prod-bytes', 'P1b-LN-lifecycle-only'):
            self.assertEqual(parsed(name)['prefix_kvread_mount'], '', name)
            self.assertEqual(parsed(name)['prefix_plan'], 'lifecycle-evict', 'the fallbacks read no KV')

    def test_the_boxes_are_inside_the_gates_own_worst_case(self):
        row = {line[0]: line for line in order()}
        both = row['P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict']
        self.assertEqual((both[3], both[5]), ('125', '240'), "P1ab-LN's box for the same two arms")
        self.assertEqual(parsed('P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict')['box_minutes'], '240')
        arms = {arm[0]: arm[4] for plan in ('exactness-shared', 'lifecycle-evict') for arm in gate.PLAN_ARMS[plan]}
        worst = (arms['exactness-shared'] + 180 + arms['lifecycle-evict'] + 180) / 60.0
        self.assertLessEqual(240, worst)
        self.assertLessEqual(int(row['P1b-CTLR-lifecycle-evict-prod-bytes'][5]), (arms['lifecycle-evict'] + 180) / 60.0)
        self.assertEqual(row['P1b-CTLR-lifecycle-evict-prod-bytes'][5], row['P1b-LN-lifecycle-only'][5])

    def test_the_tags_come_from_w1s_reserve_and_the_alternatives_share_them(self):
        w1_order = (W1 / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('RESERVE is v566, v567, v568, v582', w1_order)
        row = {line[0]: line for line in order()}
        self.assertEqual(row['Q1-kvread-qualify'][4], 'v566')
        self.assertEqual(row['P1-CTLR-prod-bytes-exactness-shared-lifecycle-evict'][4], 'v567')
        self.assertEqual(row['P1b-CTLR-lifecycle-evict-prod-bytes'][4], 'v567', 'exclusive with P1-CTLR: Q1 failed')
        p1ab = next(line.split() for line in w1_order.splitlines() if line.startswith('P1ab-LN-exactness-shared'))
        self.assertEqual(row['P1b-LN-lifecycle-only'][4], p1ab[4], "takes P1ab-LN's own tag: P1ab-LN is not run then")

    def test_the_exactness_shared_arm_asks_for_the_cross_check_and_lifecycle_takes_no_audit(self):
        self.assertEqual(gate.PLAN_ARMS['exactness-shared'][0][3], 'auditcross')
        self.assertEqual(gate.PLAN_ARMS['lifecycle-evict'][0][3], 'dev', 'lifecycle-evict reads no KV and takes no audit')
        self.assertEqual(gate.DERIVED['auditcross']['env']['QWEN_PREFIX_AUDIT_READ'], 'cross')

    def test_the_pack_says_how_the_graft_is_built_and_what_a_fail_means(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for phrase in ('5b2ad8d72bf134f1ef6413994d2b75511779660555cf0251d2b132773dcbdd8a', 'QUALIFY RULE', 'NOT QUALIFIED', 'C2_PREFIX_KVREAD_MOUNT=1', 'stage_prod_audit.py',
                       'NEEDS Q1 <- A0X0'):
            self.assertIn(phrase, text)


if __name__ == '__main__':
    unittest.main()
