"""tp4/262k8: the eight-seat 262k profiles with the best levers (c2-packed-tp4-8x262k-best, -best-audit, -best-time-gate, -ship).

Each is its 262k eight-seat twin plus exactly the levers of c2-packed-tp4-8-best-quad (the nine best levers and QWEN_FAST_QUAD_DRAFT_BLOCKS=2),
the lever audits for the audited twin, and a smaller pooled KV cache: the levers' extra DRAM (two quad captures, the fused commit's tables for eight
segments) comes out of the pool. The 262k twins, production and the 131k eight-seat family are untouched (test_c2_packed_tp4_262k_profiles holds them
by digest)."""

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dflash_packed_proposal_coordinator as coordinator  # noqa: E402
import quad_draft  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
MIB = 2 ** 20
BLOCK_BYTES = 557056                    # one KV block per chip (the seats8-262k design's figure)
OLD_BLOCKS = 21760
BLOCKS = 19968
TOKENS = (BLOCKS - 8) * 64
GATE, TIME_GATE, TRAFFIC = 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate', 'c2-packed-tp4-8x262k'
BEST, AUDIT, TIMED, SHIP = ('c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-time-gate',
                            'c2-packed-tp4-8x262k-ship')
NEW = (BEST, AUDIT, TIMED, SHIP)
BASE = {BEST: GATE, AUDIT: GATE, TIMED: TIME_GATE, SHIP: TRAFFIC}
LEVERS = {'QWEN_FAST_FUSED_COMMIT': '1', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '1', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '1',
          'QWEN_FAST_QUAD_DRAFT': '1', 'QWEN_FAST_TP4_COMMIT_LANES': '1', 'QWEN_FAST_TP4_SHARD_VALUES': '1',
          'QWEN_FAST_TP4_GDN_GLUE': '1', 'QWEN_FAST_TP4_GDN_BLOCK_CONV': '1', 'QWEN_FAST_TP4_ATTN_FOLD': '1',
          'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2'}
LEVER_AUDITS = {'QWEN_FAST_FUSED_COMMIT_AUDIT': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1', 'QWEN_FAST_DRAFT_SINGLES_AUDIT': 'all'}


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def flat(profile):
    out = copy.deepcopy(profile)
    out.pop('description')
    return out


def expected(name):
    """The base twin with the levers, the (smaller) pool and the lever audits where the arm carries them."""
    found = profiles()
    out = flat(found[BASE[name]])
    out['env'].update(LEVERS)
    if name == AUDIT:
        out['env'].update(LEVER_AUDITS)
    out['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = str(TOKENS)
    out['engine']['num-gpu-blocks-override'] = BLOCKS
    return out


class DeltaTests(unittest.TestCase):
    def test_each_arm_is_its_262k_twin_plus_exactly_the_levers_the_pool_and_the_lever_audits(self):
        found = profiles()
        for name in NEW:
            with self.subTest(profile=name):
                self.assertEqual(flat(found[name]), expected(name))

    def test_the_levers_are_the_eight_seat_quad_arms_levers_and_every_best_strace_lever_is_present(self):
        found = profiles()
        quad = found['c2-packed-tp4-8-best-quad']['env']
        self.assertEqual({key: quad[key] for key in LEVERS}, LEVERS)
        best_strace = found['c2-packed-tp4-best-strace']['env']
        for key, value in best_strace.items():
            if key.startswith(('QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_TP4_', 'QWEN_FAST_QUAD_DRAFT')) and key != 'QWEN_FAST_TP4':
                for name in (BEST, AUDIT, TIMED, SHIP):
                    with self.subTest(profile=name, lever=key):
                        self.assertEqual(found[name]['env'].get(key), value)

    def test_only_the_audited_twin_carries_the_lever_audits_and_the_singles_audit_the_quad_gate_runs(self):
        found = profiles()
        for name in (BEST, TIMED, SHIP):
            for key in LEVER_AUDITS:
                self.assertNotIn(key, found[name]['env'], (name, key))
        self.assertEqual({key: found[AUDIT]['env'][key] for key in LEVER_AUDITS}, LEVER_AUDITS)
        self.assertEqual(found['c2-packed-tp4-8-best-quad-gate']['env']['QWEN_FAST_DRAFT_SINGLES_AUDIT'], 'all')

    def test_audits_gate_only_and_limits(self):
        found = profiles()
        for name in (BEST, AUDIT):
            self.assertEqual((found[name]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], found[name]['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
        for name in (TIMED, SHIP):
            self.assertEqual((found[name]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], found[name]['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
        for name in (BEST, AUDIT, TIMED):
            self.assertTrue(found[name]['gate_only'], name)
            self.assertEqual(found[name]['min_answer_tokens'], 256)
            self.assertNotIn('max_prompt_tokens', found[name])
        for name in (BEST, AUDIT, TIMED):
            self.assertEqual(found[name]['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1', name)
        self.assertNotIn('QWEN_FAST_262K_EVIDENCE_WAIVER', found[SHIP]['env'], 'a traffic profile never waives the 262k evidence')
        ship = found[SHIP]
        self.assertNotIn('gate_only', ship)
        self.assertNotIn('QWEN_C2_GATE_PROFILE', ship['env'])
        self.assertEqual((ship['max_prompt_tokens'], ship['min_answer_tokens']), (253920, 8192))

    def test_the_262k_window_the_headroom_and_the_drafters_request_context_are_the_twins(self):
        for name, profile in profiles().items():
            if name not in NEW:
                continue
            with self.subTest(profile=name):
                self.assertEqual((profile['engine']['max-model-len'], profile['engine']['max-num-batched-tokens']), (262144, 262144))
                self.assertEqual(profile['env']['QWEN_FAST_MAX_POSITION'], '262144')
                self.assertEqual(profile['env']['QWEN_DSPARK_REQUEST_CONTEXT'], '131072')
                self.assertEqual(profile['drafter_headroom_tokens'], 32)
                self.assertEqual(profile['engine']['max-num-seqs'], 8)
                self.assertEqual(profile['env']['QWEN_FAST_KV_RESERVATION'], '1')

    def test_the_untouched_profiles_stay_as_they_were(self):
        found = profiles()
        for name in (GATE, TIME_GATE, TRAFFIC):
            self.assertEqual(found[name]['engine']['num-gpu-blocks-override'], OLD_BLOCKS)
            self.assertEqual(found[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS'], '1392128')
            self.assertNotIn('QWEN_FAST_QUAD_DRAFT_BLOCKS', found[name]['env'])
        self.assertEqual(json.loads(PROFILES.read_text(encoding='utf-8'))['default'], 'c2-packed-tp4')


class PoolTests(unittest.TestCase):
    def test_the_pool_and_its_token_variable_agree_the_way_the_contract_holds_them(self):
        found = profiles()
        for name in NEW:
            with self.subTest(profile=name):
                self.assertIsNone(contract.kv_pool_problem(found[name]))
                self.assertEqual(int(found[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS']) // 64 + 8, found[name]['engine']['num-gpu-blocks-override'])
        self.assertEqual(TOKENS, 1277440)

    def test_the_dram_arithmetic_behind_the_pool(self):
        """The seats8-262k design's high band: non-KV at eight seats 20.75 GB of 33.91 GB, a 1.0 GB margin, the 262k growth 0.07 GB. The levers add two
        quad captures (the coordinator checks one at a time, 450 MiB each) and the fused commit's tables and deltas for eight segments; the pool is the
        design's formula with the free DRAM less the levers, rounded down to a multiple of 64 blocks."""
        quads = 2 * quad_draft.QUAD_CAPTURE_BYTES_EST
        self.assertEqual(quads, 900 * MIB)
        import fused_commit
        tables = 2 * 1 * 1 * 32 * 128 * 2                       # cos and sin, bf16 tiles
        deltas = fused_commit.DRAFT_LAYERS * len(fused_commit.HEADS) * 4 * 32 * 128 * 2
        self.assertEqual((tables, deltas), (16384, 327680))
        segments = 8 * (tables + deltas)
        self.assertEqual(segments, 2752512)
        fused_bound = 16 * MIB                                   # the fused commit in all: tables, deltas and T_proj's owned temporaries (e)
        self.assertLess(segments, fused_bound)
        levers = quads + fused_bound
        self.assertAlmostEqual(levers / 1e9, 0.9605, places=4)
        total, non_kv_high, margin, growth = 33.91e9, 20.75e9, 1.0e9, 0.07e9
        free_without_levers = total - non_kv_high - 16416 * BLOCK_BYTES
        self.assertAlmostEqual(free_without_levers / 1e9, 4.0155, places=3)
        old_need = 16416 + (free_without_levers - margin - growth) // BLOCK_BYTES
        self.assertGreaterEqual(old_need, OLD_BLOCKS - 100)     # the design's 21,760 sits on the high band's edge (about 0.03 GB short)
        with_levers = 16416 + int((free_without_levers - levers - margin - growth) // BLOCK_BYTES)
        self.assertEqual(with_levers, 19979)
        self.assertLessEqual(BLOCKS, with_levers)
        self.assertEqual(BLOCKS % 64, 0)
        self.assertGreater(BLOCKS + 64, with_levers)             # the largest 64-aligned pool that fits
        self.assertEqual(OLD_BLOCKS - BLOCKS, 1792)
        self.assertEqual((OLD_BLOCKS - BLOCKS) * BLOCK_BYTES, 998244352)
        self.assertEqual((OLD_BLOCKS - BLOCKS) * 64, 114688)

    def test_the_reservation_admission_five_windows_become_four(self):
        window_blocks = 262144 // 64 + 1                         # a full window reserves 4,097 blocks
        self.assertGreaterEqual(OLD_BLOCKS - 1, 5 * window_blocks)
        self.assertLess(BLOCKS - 1, 5 * window_blocks)
        self.assertGreaterEqual(BLOCKS - 1, 4 * window_blocks)
        ladder = (4096, 32768, 131072, 261856, 16384, 65536, 200000, 253920)
        reserved = sum(-(-(length + 256 + 32) // 64) + 1 for length in ladder)
        self.assertLessEqual(reserved, BLOCKS - 1)               # the eight-seat ladder (L1) still fits with no hold

    def test_the_quad_captures_need_only_what_the_attach_time_arithmetic_names(self):
        import quad_draft_tp
        self.assertEqual(quad_draft_tp.blocks_capture_bytes(2), 900 * MIB)
        self.assertEqual(quad_draft_tp.blocks_capture_need(2, coordinator.dram_reserve_bytes()), 900 * MIB + 256 * MIB)


if __name__ == '__main__':
    unittest.main()
