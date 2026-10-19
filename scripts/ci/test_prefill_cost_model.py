"""The CPU cost model of long-context prefill (prefill_cost_model): the measured rows it is built from, and what it says of the shapes production sees.

  profile    the PF1 fits reproduce the per-chunk totals of the profile (run 38051000905) at its first, middle and last chunk, class by class they sum to the total, and the SDPA share grows with context
             the way the profile's does (5 percent at 4k, 47 at 128k on the ship-4 stack);
  oneq       the card-M ratio is 0.72 at 4k and 0.50 from 199k, monotone, and halves the long-context SDPA of the paired call;
  step       a 3.1k-token returning turn at 120k costs about what the production route lines say (1,094-1,161 ms), two passes of which the end-of-prompt pass is the dearer;
  tails      the 256-row masked bucket is the anomaly the production fit shows (two q chunks per head pair onto six cores), the 512-row one is not, and a tail of 129-256 tokens is priced on it;
  levers     the table prices exact resume at about half a step, the hand-off at a few percent, and keeps its classes (exact / tolerance / kernel) apart; every saving is non-negative and
             none exceeds the step it is a part of;
  hygiene    the module names no host, address, registry, digest or path."""

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prefill_cost_model as model  # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_the_fits_reproduce_the_profiles_chunk_totals(self):
        # PF1 (paired attention as profiled), median chip, kernel ms: chunk 1 275.05, chunk 32 486.77, chunk 63 706.66
        for context, measured in ((2048, 275.05), (65536, 486.77), (129024, 706.66)):
            with self.subTest(context=context):
                self.assertAlmostEqual(model.chunk_kernel_ms(context, oneq=False) / measured, 1.0, delta=0.02)

    def test_the_classes_sum_to_the_total_and_the_non_sdpa_constant_is_the_profiles(self):
        self.assertAlmostEqual(model.NON_SDPA_TOTAL_MS, 261.15, places=1)
        for context, groups, kernel, wall, share in model.chunk_table():
            self.assertAlmostEqual(sum(groups.values()), kernel, places=6)
            self.assertAlmostEqual(wall, 5.5 + 1.1286 * kernel, places=6)
            self.assertAlmostEqual(share, groups['SDPA'] / kernel, places=9)

    def test_the_sdpa_share_grows_with_context_as_the_profile_says(self):
        shares = {context: share for context, _g, _k, _w, share in model.chunk_table()}
        self.assertLess(shares[4096], 0.08)
        self.assertGreater(shares[131072], 0.40)
        self.assertLess(shares[131072], 0.55)
        self.assertEqual(sorted(shares.values()), [shares[context] for context in sorted(shares)])

    def test_the_unprofiled_twins_wall_is_5_5_ms_plus_1_1286_of_the_kernel(self):
        # the twin measured 761 ms of wall at chunk 59 (context 120,832) for 673.9 kernel ms
        paired = model.chunk_wall_ms(120832, oneq=False)
        self.assertAlmostEqual(paired / 761.0, 1.0, delta=0.05)


class OneqTests(unittest.TestCase):
    def test_the_ratio_is_the_card_m_one(self):
        self.assertAlmostEqual(model.oneq_ratio(4096), 0.72, delta=0.01)
        self.assertAlmostEqual(model.oneq_ratio(131000), 0.5065, delta=0.001)
        self.assertAlmostEqual(model.oneq_ratio(254000), 0.503, delta=0.001)
        ratios = [model.oneq_ratio(context) for context in range(4000, 260000, 8000)]
        self.assertEqual(ratios, sorted(ratios, reverse=True))
        self.assertAlmostEqual(model.oneq_ratio(1000), model.oneq_ratio(4000))        # clamped below the first card-M point
        self.assertAlmostEqual(model.oneq_ratio(400000), model.oneq_ratio(254000))

    def test_oneq_halves_the_long_context_sdpa_and_changes_nothing_else(self):
        paired, one = model.chunk_table(oneq=False), model.chunk_table(oneq=True)
        for (_c, groups_p, *_), (_c2, groups_o, *_) in zip(paired, one):
            self.assertEqual({key: value for key, value in groups_p.items() if key != 'SDPA'}, {key: value for key, value in groups_o.items() if key != 'SDPA'})
        self.assertAlmostEqual(one[3][1]['SDPA'] / paired[3][1]['SDPA'], 0.552, delta=0.01)       # card M's 0.505 x the in-situ factor the production route slope gives

    def test_the_block_floor_gives_the_k_stream_time_oneq_reaches(self):
        # at one q chunk per core the busiest core streams (context / 128 + 16) blocks of 13.51 us a layer: 128k of context -> ~14 ms a layer, ~227 ms a chunk
        blocks = 131072 // 128 + 16
        floor_ms = blocks * model.BLOCK_STEP_US / 1000.0 * model.ATTENTION_LAYERS
        self.assertAlmostEqual(floor_ms / model.sdpa_kernel_ms(131072), 1.0, delta=0.25)

    def test_the_in_situ_factor_is_what_the_production_slope_and_the_ttft_pair_say(self):
        # the production chunk pass costs 2.108 ms more per 1k tokens of context: 1.1286 x 3.3841 x r
        self.assertAlmostEqual(2.108 / (model.WALL_PER_KERNEL * model.SDPA_B_MS_PER_K), 0.552, places=3)
        self.assertAlmostEqual(model.ONEQ_IN_SITU * 0.5065, 0.552, places=3)
        # the solo ladder's A/B at 131,072 tokens: 37.0 s without the lever, 31.1 s with it; 64 chunks of paired SDPA are 16.1 s of wall at 3.3841 ms a 1k
        paired_wall = sum(model.sdpa_kernel_ms(c * model.CHUNK, oneq=False) for c in range(64)) * model.WALL_PER_KERNEL / 1000.0
        saved_wall = sum(model.sdpa_kernel_ms(c * model.CHUNK, oneq=False) - model.sdpa_kernel_ms(c * model.CHUNK) for c in range(64)) * model.WALL_PER_KERNEL / 1000.0
        self.assertAlmostEqual(paired_wall, 16.1, delta=0.5)
        self.assertGreater(saved_wall, 5.9)           # the measured 5.9 s is a floor: the model's 6.8 s is the card-M ratio softened by the production slope
        self.assertLess(saved_wall, 7.8)


class StepTests(unittest.TestCase):
    def test_a_returning_turn_at_120k_costs_what_the_route_lines_say(self):
        for tokens in (3112, 3388):
            self.assertTrue(1000 < model.route_ms(120832, tokens) < 1260, model.route_ms(120832, tokens))
        info = model.step_breakdown(120832, 3112)
        self.assertEqual(info['passes'], 2)
        self.assertAlmostEqual(info['sdpa_ms'] + info['other_device_ms'] + info['host_and_handoff_ms'], info['total_ms'], places=6)
        self.assertGreater(info['sdpa_ms'] / info['total_ms'], 0.33)
        self.assertLess(info['sdpa_ms'] / info['total_ms'], 0.47)

    def test_passes_split_as_the_model_splits_them(self):
        self.assertEqual(model.passes(3112), (1, 1064))
        self.assertEqual(model.passes(4096), (2, 0))
        self.assertEqual(model.bucket_for(1064), 2048)
        self.assertEqual([model.bucket_for(tail) for tail in (1, 128, 129, 256, 257, 512, 1024, 1025, 2047)], [128, 128, 256, 256, 512, 512, 1024, 2048, 2048])
        self.assertEqual(model.bucket_for(0), 0)

    def test_a_step_that_is_not_final_must_end_on_a_chunk_boundary(self):
        with self.assertRaises(ValueError):
            model.route_ms(0, 3000, final=False)
        self.assertGreater(model.route_ms(0, 2048, final=False), 0)

    def test_a_tail_pass_costs_a_fixed_floor_that_a_smaller_bucket_barely_lowers(self):
        small, large = model.PRODUCTION_TAIL[128], model.PRODUCTION_TAIL[2048]
        self.assertGreater(small[0], 0.75 * large[0])                    # the eager dispatch floor: ~320 ms against ~410 ms for a full bucket
        self.assertLess(small[1], large[1] + 0.5)

    def test_the_256_row_bucket_is_the_anomaly_and_the_512_row_one_is_not(self):
        at = lambda bucket, context: model.PRODUCTION_TAIL[bucket][0] + model.PRODUCTION_TAIL[bucket][1] * context / 1000.0      # noqa: E731
        self.assertGreater(at(256, 160000) - at(512, 160000), 250.0)
        self.assertGreater(at(256, 160000) - at(128, 160000), 250.0)
        self.assertLess(abs(at(512, 160000) - at(128, 160000)), 60.0)


class LeverTests(unittest.TestCase):
    def rows(self, start=120832, tokens=3545):
        return {name: (low, central, high, klass) for name, low, central, high, klass, _cost, _risk, _label in model.lever_table(start, tokens)}

    def test_exact_resume_is_about_half_a_step_and_the_hand_off_a_few_percent(self):
        base = model.route_ms(120832, 3545)
        rows = self.rows()
        p0 = next(row for name, row in rows.items() if name.startswith('P0'))
        handoff = next(row for name, row in rows.items() if name.startswith('PF2-H'))
        self.assertGreater(p0[1] / base, 0.40)
        self.assertLess(p0[1] / base, 0.55)
        self.assertGreater(handoff[1] / base, 0.005)
        self.assertLess(handoff[1] / base, 0.06)
        self.assertEqual(p0[3], 'E0')
        self.assertEqual(handoff[3], 'E0')

    def test_savings_are_ordered_non_negative_and_inside_the_step(self):
        for start, tokens in ((4096, 3000), (32768, 3100), (120832, 3112), (163840, 3400), (251904, 3800)):
            base = model.route_ms(start, tokens)
            for name, (low, central, high, _klass) in self.rows(start, tokens).items():
                with self.subTest(name=name, start=start):
                    self.assertGreaterEqual(low, 0.0)
                    self.assertLessEqual(low, central + 1e-9)
                    self.assertLessEqual(central, high + 1e-9)
                    self.assertLess(high, base)

    def test_the_tolerance_class_and_the_kernel_classes_are_not_called_exact(self):
        rows = self.rows()
        for name, (_low, _central, _high, klass) in rows.items():
            if name.startswith(('split-K', 'traced masked tails')):
                self.assertNotEqual(klass, 'E0', name)
                self.assertIn('K', klass)

    def test_the_256_bucket_lever_prices_only_a_short_tail(self):
        self.assertEqual(self.rows(120832, 3545)['tail bucket 256 -> 512 above 16k of context'][1], 0.0)
        priced = self.rows(120832, 2048 + 200)['tail bucket 256 -> 512 above 16k of context']
        self.assertGreater(priced[1], 200.0)

    def test_the_handoff_saving_is_a_fraction_of_what_the_profile_measured(self):
        profiled = model.HANDOFF_PROFILED['read_ms'] + model.HANDOFF_PROFILED['write_ms']
        self.assertAlmostEqual(profiled, 163.8, places=1)
        for level in ('low', 'central', 'high'):
            self.assertLess(model.handoff_saving_ms(level), profiled)
        self.assertLess(model.handoff_saving_ms('low'), model.handoff_saving_ms('central'))
        self.assertLess(model.handoff_saving_ms('central'), model.handoff_saving_ms('high'))

    def test_the_command_line_prints_the_tables(self):
        import contextlib
        import io

        for argv in (['chunk'], ['step', '120832', '3112'], ['levers', '140000', '3500']):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(model.main(argv), 0)
            self.assertIn('ms', out.getvalue())


class HygieneTests(unittest.TestCase):
    def test_the_module_names_no_host_address_registry_digest_or_path(self):
        text = (HERE / 'prefill_cost_model.py').read_text(encoding='utf-8')
        pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|/home/|/Users/|[A-Za-z]:[\\/]|\.local\b|\.lan\b|\bssh\b|ghcr\.io|docker\.io|sha256:[0-9a-f]{12}|spark-|thatch@')
        self.assertIsNone(pattern.search(text), pattern.search(text) and pattern.search(text).group(0))


if __name__ == '__main__':
    unittest.main()
