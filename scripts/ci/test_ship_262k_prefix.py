"""ship/262k-prefix: the both-levers eight-seat 262k traffic profile and its audited twin, and the real 262k evidence records.

c2-packed-tp4-8x262k-ship-prefix is c2-packed-tp4-8x262k-ship plus exactly the W1 levers (tp4/w1) plus exactly the prefix delta
(tp4/packed-prefix), with the in-trace reference sampler key removed and the request-width warm kept. It is a traffic profile: not
gate-only, no waiver marker, no audits, no host-gap log. Its audited twin, c2-packed-tp4-8x262k-ship-prefix-audit, differs from it
by the audits (the W1 lever audits and the two verify audits) and gate_only, and nothing else.

The two 262k evidence records (ordered_writer_evidence_tp4.json, packed_any_evidence_tp4_262144.json) are the recorded one-card window
on the tp4/w1 image: with them the 262,144 window is admitted WITHOUT the waiver, in a traffic process."""

import copy
import json
import os
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import page_width_tp4 as pw  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_packed_any_admission as pair_tests  # noqa: E402
import test_packed_any_admission_tp4 as quad_tests  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
SHIP = 'c2-packed-tp4-8x262k-ship'
PREFIX = 'c2-packed-tp4-8x262k-ship-prefix'
AUDIT = 'c2-packed-tp4-8x262k-ship-prefix-audit'
W1_ADDED = {'QWEN_FAST_CCL_TOPOLOGY': 'ring', 'QWEN_FAST_TP4_DRAFT_CONV': '1', 'QWEN_FAST_TP4_DRAFT_HEADS': '1',
            'QWEN_FAST_TP4_ENTRY_DIET': '1', 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS': '1', 'QWEN_FAST_TP4_RS_UNIT_MAJOR': '1',
            'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE': '1', 'QWEN_FAST_TP4_WINDOW_VALIDATE': '1'}
PREFIX_ADDED = {'QWEN_FAST_STICKY_SESSIONS': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_PREFIX_STORE_GIB': '8'}
AUDITS = {'QWEN_FAST_TP4_DRAFT_CONV_AUDIT': '1', 'QWEN_FAST_TP4_DRAFT_HEADS_AUDIT': '1', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT': '1',
          'QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT': '1', 'QWEN_FAST_FUSED_COMMIT_AUDIT': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1',
          'QWEN_FAST_DRAFT_SINGLES_AUDIT': 'all', 'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'}
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{64,}|/dev/tenstorrent|home/|zot\.')


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


class ProfileTests(unittest.TestCase):
    def test_ship_prefix_is_ship_plus_the_w1_levers_plus_the_prefix_delta_and_nothing_else(self):
        ship, mine = profiles()[SHIP], profiles()[PREFIX]
        env = dict(ship['env'])
        env.pop('QWEN_FAST_PACKED_SAMPLER_IN_TRACE')
        env.update(W1_ADDED)
        env.update(PREFIX_ADDED)
        self.assertEqual(mine['env'], env)
        engine = dict(ship['engine'])
        self.assertIs(engine.pop('no-enable-prefix-caching'), True)
        self.assertIs(engine.pop('no-enable-chunked-prefill'), True)
        engine.update({'enable-prefix-caching': True, 'enable-chunked-prefill': True, 'prefix-caching-hash-algo': 'sha256'})
        self.assertEqual(mine['engine'], engine)
        for key in set(mine) | set(ship):
            if key not in ('env', 'engine', 'description'):
                self.assertEqual(mine.get(key), ship.get(key), key)

    def test_ship_prefix_is_traffic_clean(self):
        mine = profiles()[PREFIX]
        self.assertNotIn('gate_only', mine)
        for key in ('QWEN_FAST_262K_EVIDENCE_WAIVER', 'QWEN_C2_GATE_PROFILE', 'QWEN_FAST_TP4_HOSTGAP_LOG',
                    'QWEN_FAST_PACKED_SAMPLER_IN_TRACE', *[k for k in AUDITS if 'VERIFY_T' not in k]):
            self.assertNotIn(key, mine['env'], key)
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(mine['env'][key], '0', key)
        self.assertEqual(mine['env']['QWEN_FAST_M3_REQUEST_WARM'], '1')
        self.assertEqual(mine['engine']['max-model-len'], 262144)
        self.assertEqual(mine['engine']['max-num-seqs'], 8)
        self.assertEqual(contract.gate_problems(mine, {}), [])
        self.assertEqual(contract.prefix_reuse_problems(mine), [])
        self.assertTrue(contract.prefix_reuse(mine) and contract.sticky_sessions(mine))

    def test_the_audit_twin_differs_from_ship_prefix_by_the_audits_and_gate_only(self):
        mine, audit = profiles()[PREFIX], profiles()[AUDIT]
        self.assertEqual(audit['env'], dict(mine['env'], **AUDITS))
        self.assertEqual(audit['engine'], mine['engine'])
        self.assertIs(audit.get('gate_only'), True)
        self.assertNotIn('QWEN_FAST_262K_EVIDENCE_WAIVER', audit['env'])
        self.assertNotIn('QWEN_C2_GATE_PROFILE', audit['env'])
        for key in set(mine) | set(audit):
            if key not in ('env', 'engine', 'description', 'gate_only'):
                self.assertEqual(mine.get(key), audit.get(key), key)

    def test_the_audits_are_the_w1_audit_and_prefix_gate_additions(self):
        every = profiles()
        w1 = {k: v for k, v in every['c2-packed-tp4-8x262k-w1-audit']['env'].items() if every['c2-packed-tp4-8x262k-w1']['env'].get(k) != v}
        prefix = {k: v for k, v in every['c2-packed-tp4-8x262k-prefix-gate']['env'].items()
                  if every['c2-packed-tp4-8x262k-prefix-time-gate']['env'].get(k) != v}
        self.assertEqual(dict(w1, **prefix), AUDITS)

    def test_the_new_profiles_hold_nothing_a_public_repo_must_not(self):
        for name in (PREFIX, AUDIT):
            self.assertIsNone(BANNED.search(json.dumps(profiles()[name])), name)


class RealRecordTests(unittest.TestCase):
    """The recorded one-card window, read at its pins."""

    def setUp(self):
        patcher = mock.patch.dict(admission._STATE, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.lines = []
        self.runtime = dict(binaries={'build_Release/lib/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256,
                                      'build_Release/ttnn/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256})
        self.environ = dict(quad_tests.GOOD_ENV, QWEN_FAST_MAX_POSITION='262144')       # no gate switch, no marker, no waiver

    def log(self, template, *values):
        self.lines.append(template.format(*values))

    def test_the_ordered_writer_record_passes_at_its_pin(self):
        ok, problems = pw.evidence_state()
        self.assertEqual(problems, [])
        self.assertTrue(ok)

    def test_width_4096_is_admitted_without_the_waiver(self):
        with mock.patch.object(pw, '_log') as log:
            self.assertTrue(pw.admitted(4096, self.environ))
        log.assert_not_called()

    def test_the_262144_admission_passes_qualified_in_a_traffic_process(self):
        with mock.patch.object(admission, 'check_runtime', side_effect=lambda root, binaries: self.runtime), \
                mock.patch.dict(os.environ, self.environ, clear=True):
            record = admission.admit('/opt/tt-metal', m3=pair_tests.M3, environ=dict(self.environ), log=self.log)
        self.assertEqual(record['capacity'], 262144)
        self.assertIsNotNone(record['evidence'])
        self.assertFalse(record.get('waived'))
        self.assertTrue(admission.admitted())
        self.assertFalse([line for line in self.lines if 'WAIVED' in line or admission.UNQUALIFIED_MARKER in line], self.lines)

    def test_the_records_hold_no_board_host_digest_or_address(self):
        for name in ('ordered_writer_evidence_tp4.json', 'packed_any_evidence_tp4_262144.json'):
            text = (HERE / name).read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|zot\.|sha256:[0-9a-f]{16}|@sha256|/dev/tenstorrent|'
                                        r'\b\d{1,3}(?:\.\d{1,3}){3}\b|[A-Za-z]:\\\\', text), name)


if __name__ == '__main__':
    unittest.main()
