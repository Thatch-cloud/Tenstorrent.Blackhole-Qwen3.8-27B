"""The DRAM admission at the 262,144-token window (tp4/seats8-262k, B5): the three env-tunable bounds as the 262k profiles set them, the optional
long-prefill tier, and the arithmetic that sizes the pooled KV cache from the free DRAM at eight live seats.

The three bounds (QWEN_FAST_DRAM_ENGINE_BUILD_MB / PREFILL_TRANSIENT_MB / LARGEST_BUFFER_MB) were made tunable for eight seats (EightSeatDramTests); the
262k profiles name conservative values and the M9a job measures a 253,920-token prefill and re-sets them without an image rebuild. The long tier
(QWEN_FAST_DRAM_PREFILL_LONG_FROM / _MB) charges a prompt past a length its own transient; with both unset every decision is what it was.
The pool formula is the plan's: blocks = 16,416 + floor((F8 - 1.0 GB - growth) / 557,056 bytes), F8 the free DRAM at eight live seats per chip."""

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import serving_prefill_admission as admission  # noqa: E402

MB = 10 ** 6
RESERVE = 256 * 2 ** 20
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
FLAGS = ('QWEN_FAST_DRAM_ENGINE_BUILD_MB', 'QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB', 'QWEN_FAST_DRAM_LARGEST_BUFFER_MB',
         admission.LONG_FROM_FLAG, admission.LONG_MB_FLAG)
BLOCK_BYTES = 557056            # one 64-token KV block per chip: 16 layers x 2 caches x 17,408 bytes (the plan's measured 8 x 2,052 - 4 x 2,052 slope)
GROWTH_262K = 67 * MB           # RoPE tables at 262k: about +67 MB per chip (serving_runtime.py:480-485)


class Clean(unittest.TestCase):
    def setUp(self):
        saved = {flag: os.environ.pop(flag, None) for flag in FLAGS}
        self.addCleanup(lambda: [os.environ.__setitem__(flag, value) for flag, value in saved.items() if value is not None])


class ProfileBoundsTests(Clean):
    def test_the_262k_profiles_name_the_three_bounds_and_the_131k_ones_none(self):
        for name, profile in PROFILES.items():
            env = profile['env']
            with self.subTest(profile=name):
                if '262k' in name:
                    self.assertEqual({flag: env[flag] for flag in FLAGS[:3]},
                                     {'QWEN_FAST_DRAM_ENGINE_BUILD_MB': '800', 'QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB': '600',
                                      'QWEN_FAST_DRAM_LARGEST_BUFFER_MB': '256'})
                    self.assertNotIn(admission.LONG_FROM_FLAG, env, 'the optional tier is M9a\'s to turn on')
                else:
                    for flag in FLAGS:
                        self.assertNotIn(flag, env)

    def test_the_profiles_bounds_parse_and_the_need_is_larger_than_the_defaults(self):
        env = PROFILES['c2-packed-tp4-8x262k']['env']
        with patch.dict(os.environ, {flag: env[flag] for flag in FLAGS[:3]}):
            self.assertEqual(admission.tuning_problems(), [])
            self.assertEqual((admission.engine_build_bytes(), admission.prefill_transient_bytes(), admission.largest_buffer_bytes()),
                             (800 * MB, 600 * MB, 256 * MB))
            self.assertEqual(admission.dram_need(253920, RESERVE), 1000 * MB + 600 * MB + RESERVE)
            self.assertEqual(admission.dram_need(100, RESERVE), 1000 * MB + RESERVE, 'a short prompt carries no transient')
            self.assertEqual(admission.contiguous_need(RESERVE), RESERVE + 256 * MB)
            self.assertEqual(admission.admission_contiguous_need(253920, RESERVE), RESERVE + 256 * MB + 600 * MB)
            self.assertEqual(admission.admission_contiguous_need(100, RESERVE), RESERVE + 256 * MB + 100 * MB)
        self.assertEqual(admission.dram_need(253920, RESERVE), 1000 * MB + 300 * MB + RESERVE, 'unset: today\'s need')

    def test_the_value_is_logged_when_the_dram_admission_registers(self):
        import serving_request_factory as factory
        from types import SimpleNamespace

        pool = SimpleNamespace(dram_statistics=lambda: [dict(chip=0, free=5_000 * MB, largest_free=4_000 * MB)])
        lines = []
        with patch.dict(os.environ, {'QWEN_FAST_DRAM_ENGINE_BUILD_MB': '800', 'QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB': '600',
                                     'QWEN_FAST_DRAM_LARGEST_BUFFER_MB': '256'}):
            unregister = factory.register_dram_admission(pool, log=lambda message, *values: lines.append(message.format(*values)))
            unregister()
        self.assertTrue(lines[0].startswith(factory.DRAM_REGISTERED))
        self.assertIn('engine 800000000', lines[0])
        self.assertIn('prefill 600000000', lines[0])
        self.assertIn('+ 256000000 (the largest buffer)', lines[0])
        self.assertEqual(len(lines), 1, 'no tier line when the tier is off')


class LongTierTests(Clean):
    def test_off_it_is_byte_identical(self):
        self.assertIsNone(admission.long_prefill_tier())
        for prompt in (None, 1, 2047, 2048, 4096, 123136, 253920, 262111, '4096'):
            self.assertEqual(admission.prefill_transient(prompt), 0 if type(prompt) is int and prompt < 2048
                             else admission.PREFILL_TRANSIENT_BYTES, prompt)
        self.assertEqual(admission.tuning_problems(), [])

    def test_on_a_prompt_past_the_length_is_charged_the_tiers_bytes_and_a_shorter_one_the_flat_ones(self):
        with patch.dict(os.environ, {admission.LONG_FROM_FLAG: '131072', admission.LONG_MB_FLAG: '900',
                                     'QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB': '600'}):
            self.assertEqual(admission.long_prefill_tier(), (131072, 900 * MB))
            self.assertEqual(admission.prefill_transient(131071), 600 * MB)
            self.assertEqual(admission.prefill_transient(131072), 900 * MB)
            self.assertEqual(admission.prefill_transient(253920), 900 * MB)
            self.assertEqual(admission.prefill_transient(2048), 600 * MB)
            self.assertEqual(admission.prefill_transient(2047), 0)
            self.assertEqual(admission.prefill_transient(None), 900 * MB, 'an unreadable length counts as the longest')
            self.assertEqual(admission.dram_need(253920, RESERVE), 1000 * MB + 900 * MB + RESERVE)
            self.assertEqual(admission.dram_need(65536, RESERVE), 1000 * MB + 600 * MB + RESERVE)
            self.assertEqual(admission.admission_contiguous_need(253920, RESERVE), RESERVE + 128 * MB + 900 * MB)
            self.assertEqual(admission.backstop_need(RESERVE), 1000 * MB + RESERVE, 'after the prefill, no transient')
            self.assertEqual(admission.tuning_problems(), [])

    def test_the_tier_may_charge_less_than_the_flat_bound_too(self):
        with patch.dict(os.environ, {admission.LONG_FROM_FLAG: '4096', admission.LONG_MB_FLAG: '0'}):
            self.assertEqual(admission.prefill_transient(8192), 0)
            self.assertEqual(admission.prefill_transient(3000), admission.PREFILL_TRANSIENT_BYTES)

    def test_one_flag_without_the_other_or_a_bad_value_is_refused_by_name(self):
        for environ in ({admission.LONG_FROM_FLAG: '131072'}, {admission.LONG_MB_FLAG: '900'}):
            with patch.dict(os.environ, environ):
                self.assertEqual(len(admission.tuning_problems()), 1)
                self.assertIn('set together or not at all', admission.tuning_problems()[0])
                with self.assertRaisesRegex(ValueError, 'set together'):
                    admission.prefill_transient(4096)
        for bad in ('', 'x', '-1', '1.5', '2047', '0'):
            with patch.dict(os.environ, {admission.LONG_FROM_FLAG: bad, admission.LONG_MB_FLAG: '900'}):
                self.assertEqual(len(admission.tuning_problems()), 1, bad)
                self.assertIn(admission.LONG_FROM_FLAG, admission.tuning_problems()[0])
        for bad in ('', 'x', '-1', '1.5'):
            with patch.dict(os.environ, {admission.LONG_FROM_FLAG: '131072', admission.LONG_MB_FLAG: bad}):
                self.assertEqual(len(admission.tuning_problems()), 1, bad)
                self.assertIn(admission.LONG_MB_FLAG, admission.tuning_problems()[0])

    def test_the_explicit_environ_is_what_is_read(self):
        self.assertEqual(admission.long_prefill_tier({admission.LONG_FROM_FLAG: '4096', admission.LONG_MB_FLAG: '7'}), (4096, 7 * MB))
        self.assertIsNone(admission.long_prefill_tier({}))
        self.assertEqual(admission.tuning_problems({admission.LONG_FROM_FLAG: '4096'}),
                         ['%s and %s are set together or not at all' % (admission.LONG_FROM_FLAG, admission.LONG_MB_FLAG)])

    def test_the_registration_logs_the_tier_and_refuses_a_half_set_one(self):
        import serving_request_factory as factory
        from types import SimpleNamespace

        pool = SimpleNamespace(dram_statistics=lambda: [dict(chip=0, free=5_000 * MB, largest_free=4_000 * MB)])
        lines = []

        def log(message, *values):
            lines.append(message.format(*values))

        with patch.dict(os.environ, {admission.LONG_FROM_FLAG: '131072', admission.LONG_MB_FLAG: '900'}):
            factory.register_dram_admission(pool, log=log)()
        self.assertEqual(len(lines), 2)
        self.assertIn('long-prefill tier: a prompt of >= 131072 tokens is charged 900000000 bytes', lines[1])
        with patch.dict(os.environ, {admission.LONG_MB_FLAG: '900'}), self.assertRaisesRegex(ValueError, 'set together'):
            factory.register_dram_admission(pool, log=log)


class PoolFormulaTests(unittest.TestCase):
    """The provisional pool: 16,416 (8 x 2,052) plus what the free DRAM at eight live seats leaves over the 1.0 GB floor and the 262k growth."""

    @staticmethod
    def blocks(free_gb):
        return 16416 + int((free_gb * 1e9 - 1.0e9 - GROWTH_262K) // BLOCK_BYTES)

    def test_the_provisional_pool_needs_about_4_05_gb_free_at_eight_live(self):
        provisional = PROFILES['c2-packed-tp4-8x262k']['engine']['num-gpu-blocks-override']
        self.assertEqual(provisional, 21760)
        self.assertGreaterEqual(self.blocks(4.05), provisional - 16)
        self.assertLess(self.blocks(4.0), provisional)
        self.assertGreaterEqual(self.blocks(4.06), provisional)

    def test_at_three_gb_the_pool_is_about_19880_blocks_which_is_four_full_windows(self):
        pool = self.blocks(3.0)
        self.assertLess(abs(pool - 19880), 20)
        self.assertEqual((pool - 1) // 4097, 4)
        self.assertEqual((21760 - 1) // 4097, 5)

    def test_the_pool_is_monotone_in_the_free_dram_and_never_below_the_131k_pool_at_the_floor(self):
        previous = 0
        for hundredths in range(150, 600, 5):
            value = self.blocks(hundredths / 100)
            self.assertGreaterEqual(value, previous)
            previous = value
        self.assertEqual(self.blocks(1.067), 16416)

    def test_eight_seats_at_the_m9b_prompt_fill_the_pool_with_no_hold(self):
        import serving_kv_reservation as kv

        eight = 8 * kv.request_blocks(157000, 16384)
        self.assertEqual(eight, 21688)
        self.assertLessEqual(eight, 21760 - 1)

    def test_the_dram_admission_fits_eight_262k_seats_on_the_m9_bounds_at_three_gb_free(self):
        # eight live engines already built; the ninth arrival's need on the profile's bounds against 3 GB free and a 1.5 GB block
        with patch.dict(os.environ, {'QWEN_FAST_DRAM_ENGINE_BUILD_MB': '800', 'QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB': '600',
                                     'QWEN_FAST_DRAM_LARGEST_BUFFER_MB': '256'}):
            need = admission.dram_need(253920, RESERVE)
            self.assertLess(need, 3_000 * MB - admission.STRANDED_BYTES)
            self.assertLessEqual(admission.admission_contiguous_need(253920, RESERVE), 1_500 * MB)
            self.assertEqual(admission.split_short(3_000 * MB, 1_500 * MB, need, RESERVE, 110 * MB,
                                                   contiguous=admission.admission_contiguous_need(253920, RESERVE)), ())
            # and is short at 1.5 GB free: the hold the memory plan's floor would read
            self.assertEqual(admission.split_short(1_500 * MB, 1_500 * MB, need, RESERVE, 110 * MB,
                                                   contiguous=admission.admission_contiguous_need(253920, RESERVE)), ('free',))


if __name__ == '__main__':
    unittest.main()
