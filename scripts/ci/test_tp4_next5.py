"""tp4/next-5: the combined-recipe profiles (tp4/next-4's best-ship plus the tp4/tpub and tp4/gluefix flags).

c2-packed-tp4-best-ship-tpub is c2-packed-tp4-best-ship plus QWEN_FAST_TP4_TRACED_PUBLISH=1; c2-packed-tp4-best-ship-glue is c2-packed-tp4-best-ship plus
QWEN_FAST_GDN_PAIR_SLICE=1 (idle beside QWEN_FAST_TP4_GDN_GLUE, which the recipe has on). The gate twins of the glue profile are c2-packed-tp4-best-strace-glue (timed)
and c2-packed-tp4-best-gate-glue (audited); the tpub twins came with tp4/tpub. The file default stays c2-packed-tp4."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
PRODUCTION = 'c2-packed-tp4'
SHIP, SHIP_TPUB, SHIP_GLUE = 'c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-tpub', 'c2-packed-tp4-best-ship-glue'
STRACE, STRACE_TPUB, STRACE_GLUE = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-best-strace-glue'
GATE, GATE_TPUB, GATE_GLUE = 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-tpub', 'c2-packed-tp4-best-gate-glue'
TPUB, SLICE, GLUE_V2, MARKER = 'QWEN_FAST_TP4_TRACED_PUBLISH', 'QWEN_FAST_GDN_PAIR_SLICE', 'QWEN_FAST_TP4_GDN_GLUE', 'QWEN_C2_GATE_PROFILE'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def only_env_delta(mine, base, delta):
    """mine is base plus exactly delta in env, and nothing else differs (description aside)."""
    assert mine['env'] == dict(base['env'], **delta), (set(mine['env'].items()) ^ set(dict(base['env'], **delta).items()))
    return sorted(key for key in set(mine) | set(base) if key not in ('env', 'description') and mine.get(key) != base.get(key))


class ProfileTests(unittest.TestCase):
    def test_the_file_default_is_still_production(self):
        self.assertEqual(profiles()['default'], PRODUCTION)

    def test_each_new_profile_is_its_base_plus_exactly_one_flag(self):
        found = profiles()['profiles']
        for name, base, delta in ((SHIP_TPUB, SHIP, {TPUB: '1'}), (SHIP_GLUE, SHIP, {SLICE: '1'}),
                                  (STRACE_GLUE, STRACE, {SLICE: '1'}), (GATE_GLUE, GATE, {SLICE: '1'})):
            with self.subTest(profile=name):
                self.assertEqual(only_env_delta(found[name], found[base], delta), [])

    def test_the_traffic_candidates_are_the_gate_twins_less_the_marker_and_gate_limits(self):
        found = profiles()['profiles']
        for ship, strace in ((SHIP_TPUB, STRACE_TPUB), (SHIP_GLUE, STRACE_GLUE)):
            with self.subTest(profile=ship):
                self.assertEqual(found[strace]['env'], dict(found[ship]['env'], **{MARKER: '1'}))
                for key in set(found[ship]) | set(found[strace]):
                    if key not in ('description', 'env', 'gate_only', 'min_answer_tokens', 'max_prompt_tokens'):
                        self.assertEqual(found[ship].get(key), found[strace].get(key), key)
                self.assertNotIn('gate_only', found[ship])
                self.assertNotIn(MARKER, found[ship]['env'])
                self.assertEqual((found[ship]['max_prompt_tokens'], found[ship]['min_answer_tokens']), (123136, 8192))
                self.assertIs(found[strace]['gate_only'], True)
                self.assertTrue(found[strace]['description'].startswith('GATE ONLY'))

    def test_the_glue_gate_twins_are_gate_only_and_the_audited_one_keeps_the_audits(self):
        found = profiles()['profiles']
        for name in (STRACE_GLUE, GATE_GLUE):
            self.assertIs(found[name]['gate_only'], True, name)
            self.assertTrue(found[name]['description'].startswith('GATE ONLY'), name)
            self.assertEqual(found[name]['env'][MARKER], '1', name)
        self.assertEqual(found[GATE_GLUE]['env']['QWEN_FAST_TP4_VGLUE_AUDIT'], '1')
        self.assertEqual((found[STRACE_GLUE]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], found[STRACE_GLUE]['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))

    def test_the_glue_profiles_say_the_flag_is_idle_beside_v2(self):
        found = profiles()['profiles']
        for name in (SHIP_GLUE, STRACE_GLUE, GATE_GLUE):
            self.assertEqual(found[name]['env'][GLUE_V2], '1', name)
            self.assertIn('IDLE' if name == SHIP_GLUE else 'idle', found[name]['description'], name)

    def test_the_flags_sit_only_where_they_are_meant_to(self):
        found = profiles()['profiles']
        self.assertEqual(sorted(name for name, body in found.items() if SLICE in body['env']),
                         sorted([GATE_GLUE, SHIP_GLUE, STRACE_GLUE, 'c2-packed-tp4-gate-pairslice', 'c2-packed-tp4-speed-strace-pairslice']))
        self.assertEqual(sorted(name for name, body in found.items() if TPUB in body['env']),
                         sorted([GATE_TPUB, SHIP_TPUB, STRACE_TPUB]))

    def test_production_and_best_ship_are_untouched(self):
        found = profiles()['profiles']
        for name in (PRODUCTION, SHIP, 'c2-packed-tp4-best-ship-warm4'):
            self.assertNotIn(SLICE, found[name]['env'], name)
            self.assertNotIn(TPUB, found[name]['env'], name)
            self.assertNotIn('gate_only', found[name], name)

    def test_the_descriptions_name_no_host_address_registry_or_digest(self):
        for name in (SHIP_TPUB, SHIP_GLUE, STRACE_GLUE, GATE_GLUE):
            self.assertIsNone(BANNED.search(profiles()['profiles'][name]['description']), name)


if __name__ == '__main__':
    unittest.main()
