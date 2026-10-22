"""sampled_accept: the host half of sampled verification (docs/tp4-sampled-verify.md), pure CPU.

What is held:
  - the noise is a pure function of (seed, position): uniform, uncorrelated across positions and seeds, avalanching, and pinned by known answers (the
    reference path and the device step must reproduce these exact numbers);
  - the filter: temperature 0 is today's argmax (first occurrence on ties), top_k / top_p / temperature cut and weight the candidates as an independent
    pure-Python oracle does, chi-square over many seeds; the clamp to K_MAX is the one approximation; a row is independent of its batch;
  - the slices: the per-chip top-K merge is exactly the global top-K, ties to the lowest id;
  - the walk: at temperature 0 it is greedy_verify.select_prefix; a tree walk on a chain is the linear walk;
  - EXACTNESS on a toy model, on the real GreedySession: the speculative text equals plain one-token-at-a-time sampling with the same seed for every draft
    length, drafter, round split (narrowed, aborted, capped), chain tree (tree_verify), real tree (walk_tree) and block packing (rows of several requests and idle
    rows in one draw_batch call), including across an end-of-text token and the budget;
  - DISTRIBUTION: the committed pair/triple frequencies of a speculative run over many seeds match the model's filtered distribution (chi-square), and the test
    has power: a biased acceptance rule and noise keyed by the round-local row (not the absolute position) are both caught.
"""
import math
import random
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'speculative-decoding/harness'))

import sampled_accept as sa  # noqa: E402
import tree_verify as tree  # noqa: E402
from greedy_session import BlockTicket, GreedySession  # noqa: E402
from greedy_verify import select_prefix  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------------------------------

def chi2_critical(df, z=3.719):
    """Wilson-Hilferty upper quantile of chi-square(df) (z 3.719: p = 1e-4)."""
    return df * (1 - 2.0 / (9 * df) + z * math.sqrt(2.0 / (9 * df))) ** 3


def chi_square(counts, probabilities):
    """(statistic, degrees of freedom) of observed `counts` against `probabilities`; cells under 5 expected are pooled; an outcome of probability 0 that occurred is
    infinite."""
    total = sum(counts.values())
    stat, cells, pool_observed, pool_expected = 0.0, 0, 0.0, 0.0
    for key in counts:
        if probabilities.get(key, 0.0) <= 0.0:
            return float('inf'), 1
    for key, probability in probabilities.items():
        expected = probability * total
        if expected < 5:
            pool_observed += counts.get(key, 0)
            pool_expected += expected
            continue
        stat += (counts.get(key, 0) - expected) ** 2 / expected
        cells += 1
    if pool_expected >= 5:
        stat += (pool_observed - pool_expected) ** 2 / pool_expected
        cells += 1
    return stat, max(cells - 1, 1)


def reference_distribution(values, ids, spec):
    """An independent pure-Python oracle of the filter: {token id: probability}, spec.temperature > 0."""
    pairs = sorted(zip(values, ids), key=lambda pair: (-pair[0], pair[1]))[:spec.effective_top_k]
    top = pairs[0][0]
    weights = [math.exp((value - top) / spec.temperature) for value, _ in pairs]
    total = sum(weights)
    kept, before = [], 0.0
    for (value, token), weight in zip(pairs, weights):
        probability = weight / total
        if before < spec.top_p:
            kept.append((token, probability))
        before += probability
    norm = sum(probability for _, probability in kept)
    return dict((token, probability / norm) for token, probability in kept)


def draws_at(values, ids, spec, count, position=17, first_seed=0):
    values = np.tile(np.asarray(values, dtype=np.float64), (count, 1))
    ids = np.tile(np.asarray(ids, dtype=np.int64), (count, 1))
    return sa.draw_batch(values, ids, spec.temperature, spec.top_p, spec.top_k, range(first_seed, first_seed + count), [position] * count)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------------------------------------------------------------------------

class NoiseTest(unittest.TestCase):
    def test_known_answers_pin_the_hash(self):
        """The reference path and the device step reproduce these numbers or the keyed sampling is not the same function."""
        known = {(0, 0): 0.03121713052699171, (1, 0): 0.935379973166463, (0, 1): 0.03349493407689241, (123456789, 4096): 0.9763708287157002,
                 ((1 << 64) - 1, 262111): 0.36004100742330036}
        for (seed, position), value in known.items():
            self.assertEqual(float(sa.noise(seed, position)[0]), value, (seed, position))

    def test_the_numpy_hash_is_the_scalar_definition(self):
        """The definition, in plain integers (what a kernel or a C port implements): three SplitMix64 finalisers and the top 53 bits."""
        mask = (1 << 64) - 1

        def mix(x):
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
            return x ^ (x >> 31)

        def scalar(seed, position):
            key = mix((seed + 0x9E3779B97F4A7C15) & mask)
            counter = mix((position + 0xD1B54A32D192ED03) & mask)
            return (mix(key ^ counter) >> 11) * 2.0 ** -53

        rng = random.Random(11)
        for _ in range(500):
            seed, position = rng.getrandbits(64), rng.getrandbits(rng.choice((8, 20, 40)))
            self.assertEqual(float(sa.noise(seed, position)[0]), scalar(seed, position))

    def test_a_pure_function_of_seed_and_position(self):
        seeds, positions = [5, 5, 9, 9], [1, 2, 1, 2]
        together = sa.noise(seeds, positions)
        for index in range(4):
            self.assertEqual(together[index], sa.noise(seeds[index], positions[index])[0])
        self.assertEqual(list(sa.noise([7], [3])), list(sa.noise(7, 3)))
        self.assertTrue(np.array_equal(sa.noise(11, [4, 5, 6]), sa.noise([11, 11, 11], [4, 5, 6])))
        self.assertTrue(np.array_equal(sa.noise([11, 12, 13], 4), sa.noise([11, 12, 13], [4, 4, 4])))

    def test_uniform_in_the_unit_interval(self):
        for label, seeds, positions in (('both vary', range(200000), range(200000)),
                                        ('positions vary', [91] * 200000, range(200000)),
                                        ('seeds vary', range(200000), [4097] * 200000)):
            with self.subTest(label):
                u = sa.noise(list(seeds), list(positions))
                self.assertTrue(bool(((u >= 0.0) & (u < 1.0)).all()))
                counts = np.bincount((u * 64).astype(int), minlength=64)
                stat = float(((counts - len(u) / 64.0) ** 2 / (len(u) / 64.0)).sum())
                self.assertLess(stat, chi2_critical(63), (label, stat))
                self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.005)

    def test_no_correlation_across_adjacent_positions_or_seeds(self):
        n = 200000
        a, b = sa.noise([3] * n, range(n)), sa.noise([3] * n, range(1, n + 1))
        c = sa.noise(range(n), [50] * n)
        d = sa.noise(range(1, n + 1), [50] * n)
        for name, left, right in (('adjacent positions', a, b), ('adjacent seeds', c, d), ('seed vs position', a, c)):
            with self.subTest(name):
                self.assertLess(abs(float(np.corrcoef(left, right)[0, 1])), 0.01)

    def test_one_flipped_input_bit_changes_about_half_the_output_bits(self):
        rng = random.Random(5)
        for what in ('seed', 'position'):
            distances = []
            for _ in range(300):
                seed, position = rng.getrandbits(64), rng.getrandbits(40)
                base = int(sa.noise(seed, position)[0] * (1 << 53))
                for bit in range(0, 40 if what == 'position' else 64, 3):
                    other = (int(sa.noise(seed ^ (1 << bit), position)[0] * (1 << 53)) if what == 'seed'
                             else int(sa.noise(seed, position ^ (1 << bit))[0] * (1 << 53)))
                    distances.append(bin(base ^ other).count('1'))
            self.assertAlmostEqual(sum(distances) / len(distances), 26.5, delta=1.0, msg=what)

    def test_request_seed_is_a_64_bit_key(self):
        self.assertEqual(sa.request_seed(-1), (1 << 64) - 1)
        self.assertEqual(sa.request_seed(1 << 64), 0)
        self.assertEqual(float(sa.noise(-1, 7)[0]), float(sa.noise((1 << 64) - 1, 7)[0]))
        for bad in (True, 1.0, '5', None, np.int64(3)):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                sa.request_seed(bad)

    def test_positions_are_whole_and_not_negative(self):
        for bad in (-1, 1.5, [0, -3]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sa.noise(1, bad)
        with self.assertRaises(ValueError):
            sa.noise([1, 2], [1, 2, 3])

    def test_the_position_convention(self):
        """Row i of a ticket at `position` consumed the token at position + i and samples the token at position + i + 1."""
        self.assertEqual(sa.noise_positions(100, 4), [101, 102, 103, 104])
        self.assertEqual(sa.tree_positions(100, [-1, 0, 1, 1, 0]), [101, 102, 103, 103, 102])
        for parents in ([0], [-1, 1], [-1, 0, 5], [-1, 0, -1]):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                sa.tree_positions(0, parents)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# spec, slices
# ---------------------------------------------------------------------------------------------------------------------------------------------

class SpecTest(unittest.TestCase):
    def test_defaults_are_the_models(self):
        self.assertEqual(sa.MODEL_DEFAULTS, dict(temperature=1.0, top_p=0.95, top_k=20))
        self.assertEqual(sa.SamplingSpec(), sa.SamplingSpec.from_request())
        self.assertEqual(sa.SamplingSpec.from_request(0.0), sa.SamplingSpec(0.0, 0.95, 20))
        self.assertEqual(sa.SamplingSpec.from_request(top_k=5), sa.SamplingSpec(1.0, 0.95, 5))

    def test_greedy_and_the_clamp(self):
        self.assertTrue(sa.SamplingSpec(0.0).greedy)
        self.assertFalse(sa.SamplingSpec(0.01).greedy)
        self.assertEqual([sa.SamplingSpec(1.0, 1.0, k).effective_top_k for k in (-1, 0, 1, 20, 64, 65, 1000)], [64, 64, 1, 20, 64, 64, 64])

    def test_invalid_terms_are_refused(self):
        for kwargs in (dict(temperature=-0.1), dict(temperature=float('nan')), dict(temperature=float('inf')), dict(temperature=True),
                       dict(top_p=0.0), dict(top_p=1.01), dict(top_p=float('nan')), dict(top_k=2.5), dict(top_k=True) if False else dict(top_k='3')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sa.SamplingSpec(**kwargs)


class SliceTest(unittest.TestCase):
    def test_topk_slice_orders_by_value_then_id_and_matches_a_brute_force(self):
        rng = np.random.default_rng(1)
        logits = np.round(rng.normal(size=(6, 500)) * 8) / 8        # a coarse grid: many ties, as bf16 gives
        values, ids = sa.topk_slice(logits, 64)
        for row in range(6):
            order = sorted(range(500), key=lambda token: (-logits[row, token], token))[:64]
            self.assertEqual(ids[row].tolist(), order)
            self.assertEqual(values[row].tolist(), [logits[row, token] for token in order])

    def test_the_first_candidate_is_the_first_occurrence_argmax(self):
        rng = np.random.default_rng(2)
        logits = np.round(rng.normal(size=(40, 300)) * 2) / 2
        logits[3] = 1.0                                              # all equal: token 0
        values, ids = sa.topk_slice(logits, 8)
        self.assertEqual(ids[:, 0].tolist(), np.argmax(logits, axis=1).tolist())
        self.assertEqual(ids[3].tolist(), list(range(8)))

    def test_nan_logits_are_a_fault(self):
        logits = np.zeros((1, 100))
        logits[0, 4] = np.nan
        with self.assertRaises(ValueError):
            sa.topk_slice(logits, 8)
        with self.assertRaises(ValueError):
            sa.order_candidates([[1.0, float('nan')]], [[0, 1]])
        with self.assertRaises(ValueError):
            sa.order_candidates([[1.0, 2.0]], [[3, 3]])

    def test_the_merge_of_per_chip_slices_is_the_global_slice(self):
        """Each chip returns the top K of its own vocabulary shard; the union holds the global top K, whatever the ties."""
        rng = np.random.default_rng(3)
        width, chips, k = 1000, 4, 64
        for trial in range(6):
            logits = np.round(rng.normal(size=(8, width * chips)) * (2 if trial % 2 else 8)) / 8
            parts = [sa.topk_slice(logits[:, chip * width:(chip + 1) * width], k) for chip in range(chips)]
            merged_values, merged_ids = sa.merge_shard_topk([p[0] for p in parts], [p[1] for p in parts], width, k)
            global_values, global_ids = sa.topk_slice(logits, k)
            self.assertTrue(np.array_equal(merged_ids, global_ids), trial)
            self.assertTrue(np.array_equal(merged_values, global_values), trial)

    def test_the_merge_refuses_what_is_not_a_slice(self):
        with self.assertRaises(ValueError):
            sa.merge_shard_topk([], [], 10)
        with self.assertRaises(ValueError):
            sa.merge_shard_topk([[[1.0]]], [[[-1]]], 10)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# filter and draw
# ---------------------------------------------------------------------------------------------------------------------------------------------

TOY_VALUES = [3.0, 2.5, 2.5, 1.0, 0.5, 0.0, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
TOY_IDS = [40, 7, 3, 99, 12, 5, 61, 8, 77, 21, 30, 2]


class DrawTest(unittest.TestCase):
    def test_temperature_zero_is_argmax_with_the_lowest_id_on_ties(self):
        rng = np.random.default_rng(4)
        logits = np.round(rng.normal(size=(64, 200)) * 2) / 2
        values, ids = sa.topk_slice(logits, 32)
        out = sa.draw_batch(values, ids, 0.0, 0.95, 20, range(64), range(64))
        self.assertEqual(out.tolist(), np.argmax(logits, axis=1).tolist())
        self.assertEqual(sa.draw(TOY_VALUES, TOY_IDS, sa.GREEDY, 5, 5), 40)
        self.assertEqual(sa.draw([1.0, 1.0, 1.0], [9, 4, 6], sa.GREEDY, 5, 5), 4)

    def test_top_k_one_and_a_tiny_top_p_are_greedy_at_any_temperature(self):
        rng = np.random.default_rng(5)
        logits = rng.normal(size=(32, 200)) * 3
        values, ids = sa.topk_slice(logits, 64)
        expected = np.argmax(logits, axis=1).tolist()
        for temperature in (0.3, 1.0, 2.0):
            self.assertEqual(sa.draw_batch(values, ids, temperature, 1.0, 1, range(32), range(32)).tolist(), expected)
            self.assertEqual(sa.draw_batch(values, ids, temperature, 1e-9, 0, range(32), range(32)).tolist(), expected)

    def test_the_filtered_distribution_is_the_oracles(self):
        """Chi-square over 100000 seeds against the pure-Python oracle, for filters that cut by k, by p, by both and by neither."""
        cases = [sa.SamplingSpec(1.0, 1.0, 0), sa.SamplingSpec(0.7, 0.9, 8), sa.SamplingSpec(1.3, 1.0, 5), sa.SamplingSpec(1.0, 0.5, 0),
                 sa.SamplingSpec(0.4, 0.95, 20), sa.SamplingSpec(2.0, 0.8, 3)]
        for spec in cases:
            with self.subTest(spec=spec):
                expected = reference_distribution(TOY_VALUES, TOY_IDS, spec)
                counts = {}
                for token in draws_at(TOY_VALUES, TOY_IDS, spec, 100000).tolist():
                    counts[token] = counts.get(token, 0) + 1
                stat, df = chi_square(counts, expected)
                self.assertLess(stat, chi2_critical(df), (stat, df))
                self.assertTrue(set(counts) <= set(expected))

    def test_top_p_keeps_the_smallest_prefix_reaching_p(self):
        """Four equal logits are four exact quarters: top_p 0.5 keeps two (the cumulative BEFORE the third is 0.5, not below it), just over keeps three."""
        ids = [30, 10, 20, 40]
        for top_p, kept in ((0.25, {10}), (0.2500001, {10, 20}), (0.5, {10, 20}), (0.5000001, {10, 20, 30}), (0.75, {10, 20, 30}), (1.0, {10, 20, 30, 40})):
            with self.subTest(top_p=top_p):
                seen = set(draws_at([1.0] * 4, ids, sa.SamplingSpec(1.0, top_p, 0), 4000).tolist())
                self.assertEqual(seen, kept)

    def test_top_k_above_the_slice_width_keeps_the_whole_slice(self):
        self.assertEqual(set(draws_at([1.0] * 4, [1, 2, 3, 4], sa.SamplingSpec(1.0, 1.0, 0), 2000).tolist()), {1, 2, 3, 4})

    def test_the_k_max_clamp_is_the_one_approximation(self):
        """A slice wider than K_MAX: top_k 0, 64, 65 and 1000 are the same sampler; top_k 20 is not."""
        rng = np.random.default_rng(6)
        values = np.sort(rng.normal(size=(1, 200)) * 0.3, axis=1)[:, ::-1]
        ids = rng.permutation(5000)[:200][None, :]
        seeds = list(range(3000))
        tiled_values, tiled_ids = np.tile(values, (3000, 1)), np.tile(ids, (3000, 1))
        results = {}
        for top_k in (0, -1, 64, 65, 1000, 20):
            results[top_k] = sa.draw_batch(tiled_values, tiled_ids, 1.0, 1.0, top_k, seeds, [9] * 3000).tolist()
        self.assertEqual(results[0], results[64])
        self.assertEqual(results[-1], results[64])
        self.assertEqual(results[65], results[64])
        self.assertEqual(results[1000], results[64])
        self.assertNotEqual(results[20], results[64])
        self.assertTrue(set(results[64]) <= set(ids[0][:64].tolist()))
        self.assertTrue(set(results[20]) <= set(ids[0][:20].tolist()))

    def test_a_row_does_not_depend_on_its_batch_or_the_order_of_its_candidates(self):
        rng = np.random.default_rng(7)
        rows = 48
        logits = np.round(rng.normal(size=(rows, 300)) * 4) / 8
        values, ids = sa.topk_slice(logits, 64)
        temperature = rng.choice([0.0, 0.5, 1.0, 1.7], rows)
        top_p = rng.choice([0.5, 0.95, 1.0], rows)
        top_k = rng.choice([0, 1, 5, 20, 64, 500], rows)
        seeds = [int(x) for x in rng.integers(0, 1 << 62, rows)]
        positions = [int(x) for x in rng.integers(0, 300000, rows)]
        together = sa.draw_batch(values, ids, temperature, top_p, top_k, seeds, positions)
        for row in range(rows):
            alone = sa.draw_batch(values[row:row + 1], ids[row:row + 1], temperature[row], top_p[row], top_k[row], seeds[row:row + 1], positions[row:row + 1])
            self.assertEqual(int(alone[0]), int(together[row]), row)
            self.assertEqual(sa.draw(values[row], ids[row], sa.SamplingSpec(float(temperature[row]), float(top_p[row]), int(top_k[row])), seeds[row],
                                     positions[row]), int(together[row]))
        shuffled = rng.permutation(64)
        again = sa.draw_batch(values[:, shuffled], ids[:, shuffled], temperature, top_p, top_k, seeds, positions)
        self.assertEqual(again.tolist(), together.tolist())
        order = rng.permutation(rows)
        reordered = sa.draw_batch(values[order], ids[order], temperature[order], top_p[order], top_k[order], [seeds[i] for i in order],
                                  [positions[i] for i in order])
        self.assertEqual(reordered.tolist(), together[order].tolist())

    def test_inactive_rows_are_not_read(self):
        values = np.array([[2.0, 1.0], [float('nan'), 1.0], [5.0, 1.0]])
        ids = np.array([[1, 2], [3, 4], [5, 6]])
        out = sa.draw_batch(values, ids, 0.0, 1.0, 0, [1, 2, 3], [1, 2, 3], active=[True, False, True])
        self.assertEqual(out.tolist(), [1, -1, 5])
        with self.assertRaises(ValueError):
            sa.draw_batch(values, ids, 0.0, 1.0, 0, [1, 2, 3], [1, 2, 3])
        self.assertEqual(sa.draw_batch(values, ids, 1.0, 1.0, 0, [1, 2, 3], [1, 2, 3], active=[False, False, False]).tolist(), [-1, -1, -1])

    def test_only_the_seed_the_position_and_the_logits_matter(self):
        spec = sa.SamplingSpec()
        base = sa.draw(TOY_VALUES, TOY_IDS, spec, 31, 700)
        moved = [sa.draw(TOY_VALUES, TOY_IDS, spec, seed, position) for seed, position in ((32, 700), (31, 701), (31, 700))]
        self.assertEqual(moved[2], base)
        draws = set(sa.draw(TOY_VALUES, TOY_IDS, spec, 31, position) for position in range(400))
        self.assertGreater(len(draws), 3)                           # the position moves the draw
        self.assertEqual(sa.sample_logits(np.arange(300, dtype=np.float64)[::-1], spec, 4, 8), sa.sample_logits(np.arange(300, dtype=np.float64)[::-1], spec, 4, 8))

    def test_invalid_settings_are_refused(self):
        values, ids = np.array([[2.0, 1.0]]), np.array([[1, 2]])
        for temperature, top_p in ((-1.0, 1.0), (float('nan'), 1.0), (1.0, 0.0), (1.0, 1.5)):
            with self.subTest(temperature=temperature, top_p=top_p), self.assertRaises(ValueError):
                sa.draw_batch(values, ids, temperature, top_p, 0, [1], [1])
        with self.assertRaises(ValueError):
            sa.draw_batch(values, ids, 1.0, 1.0, 0, [1, 2], [1])


# ---------------------------------------------------------------------------------------------------------------------------------------------
# the walks
# ---------------------------------------------------------------------------------------------------------------------------------------------

class WalkTest(unittest.TestCase):
    def test_at_temperature_zero_the_walk_is_greedy_select_prefix(self):
        rng = random.Random(8)
        for _ in range(500):
            k = rng.randrange(0, 16)
            targets = [rng.randrange(50) for _ in range(k + 1)]
            drafts = [t if rng.random() < 0.7 else rng.randrange(50) for t in targets[:k]]
            decision = select_prefix(drafts, targets, vocab_size=100, eos_ids=())
            accepted, emitted = sa.walk_linear(drafts, targets)
            self.assertEqual((accepted, emitted), (decision.accepted, decision.emitted))
            self.assertEqual(len(emitted), decision.state_rows)

    def test_the_walk_reads_no_draw_past_the_first_mismatch(self):
        for drafts, ys, wanted in (((5, 6, 7), (5, 6, 7, 8), (3, (5, 6, 7, 8), 4)), ((5, 9, 7), (5, 6, 7, 8), (1, (5, 6), 2)),
                                   ((4,), (5, 6), (0, (5,), 1)), ((), (3,), (0, (3,), 1))):
            calls = []

            def draw(row, ys=ys, calls=calls):
                calls.append(row)
                return ys[row]

            accepted, emitted = sa.walk_linear(drafts, draw)
            self.assertEqual((accepted, emitted, len(calls)), wanted)
            self.assertEqual(calls, list(range(wanted[2])))
            self.assertEqual(emitted, tuple(ys[:accepted + 1]))

    def test_a_tree_walk_on_a_chain_is_the_linear_walk(self):
        rng = random.Random(9)
        for _ in range(300):
            k = rng.randrange(0, 12)
            ys = [rng.randrange(6) for _ in range(k + 1)]
            drafts = [y if rng.random() < 0.7 else rng.randrange(6) for y in ys[:k]]
            rows, emitted = sa.walk_tree((99,) + tuple(drafts), [-1] + list(range(k)), ys)
            accepted, linear = sa.walk_linear(drafts, ys)
            self.assertEqual((emitted, len(rows) - 1), (linear, accepted))
            self.assertEqual(list(rows), list(range(accepted + 1)))

    def test_a_tree_descends_to_the_child_that_matches(self):
        #        0 (seed)
        #      /  |  \
        #     1   2   3        tokens 10 20 30
        #    / \      |
        #   4   5     6        tokens 11 12 31
        tokens = (0, 10, 20, 30, 11, 12, 31)
        parents = (-1, 0, 0, 0, 1, 1, 3)
        table = {0: 30, 3: 31, 6: 77}
        rows, emitted = sa.walk_tree(tokens, parents, lambda row: table[row])
        self.assertEqual((rows, emitted), ((0, 3, 6), (30, 31, 77)))
        rows, emitted = sa.walk_tree(tokens, parents, {0: 10, 1: 12, 5: 3}.__getitem__)
        self.assertEqual((rows, emitted), ((0, 1, 5), (10, 12, 3)))
        rows, emitted = sa.walk_tree(tokens, parents, {0: 99}.__getitem__)
        self.assertEqual((rows, emitted), ((0,), (99,)))
        rows, emitted = sa.walk_tree(tokens, parents, {0: 20, 2: 5}.__getitem__)
        self.assertEqual((rows, emitted), ((0, 2), (20, 5)))

    def test_a_malformed_tree_is_refused(self):
        for tokens, parents in (((1,), (0,)), ((1, 2), (-1,)), ((1, 2), (-1, 1)), ((1, 2), (-1, -1)), ((), ())):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                sa.walk_tree(tokens, parents, lambda row: 0)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# exactness on a toy model, on the real GreedySession
# ---------------------------------------------------------------------------------------------------------------------------------------------

V = 100
EOS = V - 1
SPECS = (sa.SamplingSpec(), sa.SamplingSpec(0.7, 0.9, 8), sa.SamplingSpec(1.0, 1.0, 0), sa.SamplingSpec(0.0, 1.0, 0))


def toy_logits(context):
    """A deterministic language model of the last two tokens, on a bf16-like grid (so ties occur)."""
    a, b = (context[-2], context[-1]) if len(context) > 1 else (0, context[-1])
    rng = np.random.default_rng((a * 1000003 + b * 7919 + 12345) & 0xFFFFFFFF)
    logits = rng.normal(size=V) * 2.5
    logits[rng.integers(0, V, 3)] += 4.0
    return np.round(logits * 8) / 8


def toy_sample(context, spec, seed):
    return sa.sample_logits(toy_logits(context), spec, seed, len(context))


def plain_run(prompt, n, seed, spec):
    context, out = list(prompt), []
    while len(out) < n:
        token = toy_sample(context, spec, seed)
        out.append(token)
        context.append(token)
        if token == EOS:
            break
    return out


def approx_drafter(history, count):
    """The drafter's top choice under a perturbed copy of the model: usually right, sometimes not. Deterministic in the history."""
    context, drafts = list(history), []
    for _ in range(count):
        noise = np.random.default_rng((context[-1] * 31 + len(context)) & 0xFFFF).normal(size=V) * 0.9
        token = int(np.argmax(toy_logits(context) + noise))
        drafts.append(token)
        context.append(token)
    return drafts


def copy_drafter(history, count):
    """Prompt-lookup style: the tokens that followed the last earlier occurrence of the final two."""
    key = tuple(history[-2:])
    for start in range(len(history) - 3, -1, -1):
        if tuple(history[start:start + 2]) == key:
            return list(history[start + 2:start + 2 + count])
    return []


def hash_drafter(history, count):
    rng = random.Random(len(history) * 7 + history[-1])
    return [rng.randrange(V) for _ in range(count)]


def peaked_drafter(history, count):
    """Mostly right (the plain greedy choice of the model): exercises long accepted runs."""
    context, drafts = list(history), []
    for _ in range(count):
        token = int(np.argmax(toy_logits(context)))
        drafts.append(token)
        context.append(token)
    return drafts


DRAFTERS = dict(approx=approx_drafter, copy=copy_drafter, hash=hash_drafter, peaked=peaked_drafter)


def _no_drafter(request_id, history, count):
    return ()


def new_session(prompt, n, first):
    return GreedySession('r', tuple(prompt), first, vocab_size=V, max_new_tokens=n, eos_ids=(EOS,), neural={'dflash2': _no_drafter}, lookup_enabled=False)


def stage(session, tokens, source='dflash2'):
    session.epoch += 1
    ticket = BlockTicket(session.request_id, session.epoch, session.position, tuple(tokens), source, 0)
    session.pending, session.phase = ticket, 'pending'
    return ticket


def row_draws(history, tokens, spec, seed):
    """The target's draw after each row of a chain ticket (tokens[0] is the last committed token, history ends with it)."""
    return [toy_sample(list(history) + list(tokens[1:i + 1]), spec, seed) for i in range(len(tokens))]


def spec_run(prompt, n, seed, spec, drafter, rows, rng=None, cap=None):
    """The real GreedySession driven by sampled predictions. `rng` adds round splits: a ticket narrowed before its commit or aborted and re-proposed narrower."""
    first = toy_sample(list(prompt), spec, seed)
    session = new_session(prompt, n, first)
    while not session.finished:
        history = list(prompt) + session.emitted
        width = rows
        tokens = (session.seed,) + tuple(drafter(history, width - 1))
        ticket = stage(session, tokens)
        if rng is not None and len(tokens) > 1:
            roll = rng.random()
            if roll < 0.3:
                ticket = session.narrow('r', ticket, max(w for w in (1, 2, 4, 8, 16) if w <= rng.randrange(1, len(tokens) + 1)))
            elif roll < 0.5:
                session.abort('r', ticket, lambda state_rows: None)
                width = rng.choice((1, 2, 4))
                ticket = stage(session, (session.seed,) + tuple(drafter(history, width - 1)))
        ys = row_draws(history, ticket.tokens, spec, seed)
        session.commit('r', ticket, ys, lambda state_rows: None, max_rows=None if cap is None else min(cap, len(ticket.tokens)))
    return list(session.emitted)


class ExactnessTest(unittest.TestCase):
    PROMPTS = ([5, 17, 3, 3, 40], [60, 2, 9, 9, 9, 9, 12, 60, 2, 9], [1])

    def test_every_draft_length_and_drafter_commits_the_plain_sampled_text(self):
        combinations = 0
        for prompt in self.PROMPTS[:2]:
            for seed in (77, 2 ** 63 + 5):
                for spec in SPECS:
                    plain = plain_run(prompt, 36, seed, spec)
                    for name, drafter in DRAFTERS.items():
                        for rows in (1, 2, 4, 8, 16):
                            with self.subTest(prompt=prompt[:2], seed=seed, spec=spec, drafter=name, rows=rows):
                                self.assertEqual(spec_run(prompt, 36, seed, spec, drafter, rows), plain)
                            combinations += 1
        self.assertEqual(combinations, 2 * 2 * 4 * 4 * 5)

    def test_the_coupling_is_exercised(self):
        """Drafts really are accepted (so the test is not the all-rejected 1-token-a-round case) and really are rejected."""
        prompt, seed, spec = self.PROMPTS[0], 3, sa.SamplingSpec(0.7, 0.9, 8)
        first = toy_sample(list(prompt), spec, seed)
        session = new_session(prompt, 60, first)
        rounds = []
        while not session.finished:
            history = list(prompt) + session.emitted
            ticket = stage(session, (session.seed,) + tuple(peaked_drafter(history, 7)))
            decision = session.commit('r', ticket, row_draws(history, ticket.tokens, spec, seed), lambda state_rows: None)
            rounds.append(decision.accepted)
        self.assertGreater(max(rounds), 2)
        self.assertIn(0, rounds)
        self.assertEqual(list(session.emitted), plain_run(prompt, 60, seed, spec))

    def test_round_splits_narrowing_aborts_and_commit_caps_change_nothing(self):
        for seed in range(6):
            for spec in SPECS[:3]:
                plain = plain_run(self.PROMPTS[1], 40, seed, spec)
                for name in ('approx', 'peaked'):
                    for cap in (None, 3):
                        with self.subTest(seed=seed, spec=spec, drafter=name, cap=cap):
                            got = spec_run(self.PROMPTS[1], 40, seed, spec, DRAFTERS[name], 16, rng=random.Random(seed * 13 + 1), cap=cap)
                            self.assertEqual(got, plain)

    def test_the_budget_cut_and_an_end_of_text_token(self):
        eos_runs = 0
        for seed in range(40):
            spec = sa.SamplingSpec(1.0, 1.0, 0)
            plain = plain_run(self.PROMPTS[2], 70, seed, spec)
            eos_runs += plain[-1] == EOS
            for budget in (1, 2, 5, 33):
                for rows in (2, 8, 16):
                    expected = plain_run(self.PROMPTS[2], budget, seed, spec)
                    self.assertEqual(spec_run(self.PROMPTS[2], budget, seed, spec, peaked_drafter, rows), expected, (seed, budget, rows))
        self.assertGreater(eos_runs, 0)              # a sampled end-of-text token occurs, both as a correction and inside accepted drafts

    def test_chain_trees_of_tree_verify_commit_the_plain_text(self):
        won = 0
        for seed in range(10):
            for spec in SPECS[:3]:
                prompt, n = self.PROMPTS[0], 40
                first = toy_sample(list(prompt), spec, seed)
                session = new_session(prompt, n, first)
                while not session.finished:
                    history = list(prompt) + session.emitted
                    primary = stage(session, (session.seed,) + tuple(approx_drafter(history, 7)))
                    alternatives = [('lookup', copy_drafter(history, 7)), ('peaked', peaked_drafter(history, 7)), ('hash', hash_drafter(history, 7))]
                    candidates = tree.build_candidates(primary, alternatives)
                    predictions = [row_draws(history, candidate.tokens, spec, seed) for candidate in candidates]
                    commit = tree.commit_tree(session, 'r', candidates, predictions, lambda branch: (lambda state_rows: None))
                    won += commit.copy_back
                self.assertEqual(list(session.emitted), plain_run(prompt, n, seed, spec), (seed, spec))
        self.assertGreater(won, 0)                    # a branch won rounds

    def test_real_trees_commit_the_plain_text(self):
        """Branching drafts that share prefixes: from each node the drafter's two best guesses, to depth 3 (a node with a child per guess)."""
        def build(history, depth=3, width=2):
            tokens, parents, paths = [history[-1]], [-1], [[]]
            frontier = [0]
            for _ in range(depth):
                next_frontier = []
                for node in frontier:
                    context = list(history) + paths[node]
                    noise = np.random.default_rng((context[-1] * 17 + len(context)) & 0xFFFF).normal(size=V) * 0.7
                    best = np.argsort(-(toy_logits(context) + noise), kind='stable')[:width]
                    for token in best.tolist():
                        tokens.append(token)
                        parents.append(node)
                        paths.append(paths[node] + [token])
                        next_frontier.append(len(tokens) - 1)
                frontier = next_frontier
            return tokens, parents, paths

        deepest = 0
        for seed in range(12):
            for spec in SPECS[:3]:
                prompt, n = self.PROMPTS[0], 40
                out = [toy_sample(list(prompt), spec, seed)]
                while len(out) < n and out[-1] != EOS:
                    history = list(prompt) + out
                    tokens, parents, paths = build(history)
                    positions = sa.tree_positions(len(history) - 1, parents)
                    ys = lambda row: sa.sample_logits(toy_logits(list(history) + paths[row]), spec, seed, positions[row])
                    rows, emitted = sa.walk_tree(tokens, parents, ys)
                    deepest = max(deepest, len(rows) - 1)
                    out.extend(emitted)
                    if EOS in out:
                        out = out[:out.index(EOS) + 1]
                self.assertEqual(out[:n], plain_run(prompt, n, seed, spec), (seed, spec))
        self.assertGreaterEqual(deepest, 2)

    def test_block_packing_and_admission_timing_change_nothing(self):
        """Four requests share rounds: a random subset is live, widths differ, idle rows (NaN logits) pad the block, and every row of every request is drawn
        by ONE draw_batch call. Each request's text equals its own plain run."""
        rng = random.Random(10)
        prompts = [self.PROMPTS[0], self.PROMPTS[1], self.PROMPTS[2], [8, 8, 8, 8, 1]]
        seeds = [101, 202, 303, 2 ** 64 - 9]
        specs = [SPECS[0], SPECS[1], SPECS[2], SPECS[3]]
        n = 34
        sessions = [new_session(p, n, toy_sample(list(p), s, seed)) for p, s, seed in zip(prompts, specs, seeds)]
        started = [False, False, True, True]               # two requests are admitted later
        rounds = 0
        while not all(session.finished for session in sessions):
            rounds += 1
            if rounds == 3:
                started = [True] * 4
            live = [i for i, s in enumerate(sessions) if started[i] and not s.finished and rng.random() < 0.8] or \
                   [i for i, s in enumerate(sessions) if started[i] and not s.finished]
            if not live:
                continue
            block_values, block_ids, owner, temps, top_ps, top_ks, key_seeds, positions, active = [], [], [], [], [], [], [], [], []
            plans = {}
            for i in live:
                history = list(prompts[i]) + sessions[i].emitted
                width = rng.choice((1, 2, 4, 8, 16))
                tokens = (sessions[i].seed,) + tuple(DRAFTERS[rng.choice(('approx', 'peaked', 'copy'))](history, width - 1))
                plans[i] = (history, stage(sessions[i], tokens))
                for row in range(len(tokens)):
                    logits = toy_logits(list(history) + list(tokens[1:row + 1]))
                    values, ids = sa.topk_slice(logits[None, :], sa.K_MAX)
                    block_values.append(values[0])
                    block_ids.append(ids[0])
                    owner.append((i, row))
                    temps.append(specs[i].temperature)
                    top_ps.append(specs[i].top_p)
                    top_ks.append(specs[i].top_k)
                    key_seeds.append(seeds[i])
                    positions.append(sessions[i].position + 1 + row)
                    active.append(True)
                    if rng.random() < 0.2:                        # an idle row of the padded block, junk logits
                        block_values.append(np.full(sa.K_MAX, np.nan))
                        block_ids.append(np.arange(sa.K_MAX))
                        owner.append(None)
                        temps.append(1.0)
                        top_ps.append(1.0)
                        top_ks.append(0)
                        key_seeds.append(0)
                        positions.append(0)
                        active.append(False)
            order = list(range(len(owner)))
            rng.shuffle(order)
            drawn = sa.draw_batch(np.array(block_values)[order], np.array(block_ids)[order], np.array(temps)[order], np.array(top_ps)[order],
                                  np.array(top_ks)[order], [key_seeds[j] for j in order], [positions[j] for j in order], active=[active[j] for j in order])
            per_request = dict((i, [None] * len(plans[i][1].tokens)) for i in live)
            for slot, j in enumerate(order):
                if owner[j] is not None:
                    per_request[owner[j][0]][owner[j][1]] = int(drawn[slot])
                else:
                    self.assertEqual(int(drawn[slot]), -1)
            for i in live:
                sessions[i].commit('r', plans[i][1], per_request[i], lambda state_rows: None)
        for i in range(4):
            self.assertEqual(list(sessions[i].emitted), plain_run(prompts[i], n, seeds[i], specs[i]), i)

    def test_a_different_seed_gives_a_different_text(self):
        spec = sa.SamplingSpec(1.0, 1.0, 0)
        texts = set(tuple(plain_run(self.PROMPTS[0], 12, seed, spec)) for seed in range(20))
        self.assertGreater(len(texts), 15)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# distribution of the speculative output, with controls that must fail
# ---------------------------------------------------------------------------------------------------------------------------------------------

CHAIN = np.array([[0.0, 1.0, 0.5, -1.0], [1.5, 0.0, -0.5, 0.2], [-0.3, 0.8, 0.0, 1.2], [0.4, -0.7, 1.1, 0.0]])
CHAIN_SPEC = sa.SamplingSpec(1.0, 0.9, 3)
CHAIN_PROMPT = 5                    # the prompt is five tokens long; its last token is 0, the first generated token is at position 5


def chain_filtered(last):
    return reference_distribution(CHAIN[last].tolist(), list(range(4)), CHAIN_SPEC)


def chain_theory(depth):
    """P(first token t0, then `depth` more) of the Markov toy, by enumeration."""
    out = {}

    def extend(prefix, probability):
        if len(prefix) == depth + 1:
            out[tuple(prefix)] = probability
            return
        for token, p in chain_filtered(prefix[-1] if prefix else 0).items():
            extend(prefix + [token], probability * p)
    extend([], 1.0)
    return out


def y_table(seeds, positions_of, spec=CHAIN_SPEC):
    """Y[s, p, a]: the draw for seed s at the p-th position index with the model row of last token a; `positions_of(p)` is the noise position."""
    seeds = list(seeds)
    count, slots = len(seeds), len(positions_of)
    out = np.zeros((count, slots, 4), dtype=np.int64)
    for a in range(4):
        for p, position in enumerate(positions_of):
            values = np.tile(CHAIN[a], (count, 1))
            ids = np.tile(np.arange(4), (count, 1))
            out[:, p, a] = sa.draw_batch(values, ids, spec.temperature, spec.top_p, spec.top_k, seeds, [position] * count)
    return out


def chain_drafts(last, length):
    drafts = []
    for _ in range(length):
        last = int(np.argmax(CHAIN[last]))
        drafts.append(last)
    return drafts


def simulate_chain(seeds, draft_length, keying='position', acceptance='coupled', depth=3):
    """The speculative walk over many seeds on the Markov toy; returns {(t0, e1..e_depth): count}. keying 'row' draws row i of a round with the noise of
    ROW i (a position-agnostic key); acceptance 'typical' accepts a draft whose filtered probability is at least 0.05 (a biased rule)."""
    seeds = list(seeds)
    span = depth + draft_length + 2
    table = y_table(seeds, [CHAIN_PROMPT + p for p in range(span + 1)])         # positions P .. P + span
    local = y_table(seeds, list(range(span + 1))) if keying == 'row' else None
    counts = {}
    for s in range(len(seeds)):
        cur = int(table[s, 0, 0])                    # the first token: position P, model row of the prompt's last token (0)
        first, emitted, position = cur, [], CHAIN_PROMPT
        while len(emitted) < depth:
            drafts = chain_drafts(cur, draft_length)
            tokens = [cur] + drafts

            def y(row, tokens=tokens, position=position):
                return int((local if keying == 'row' else table)[s, row if keying == 'row' else position + 1 + row - CHAIN_PROMPT, tokens[row]])

            if acceptance == 'typical':
                accepted = 0
                for row, draft in enumerate(drafts):
                    if chain_filtered(tokens[row]).get(draft, 0.0) >= 0.05:
                        accepted += 1
                    else:
                        break
                emitted_now = tuple(drafts[:accepted]) + (y(accepted),)
            else:
                accepted, emitted_now = sa.walk_linear(drafts, y)
            emitted.extend(emitted_now)
            cur = emitted_now[-1]
            position += accepted + 1
        key = (first,) + tuple(emitted[:depth])
        counts[key] = counts.get(key, 0) + 1
    return counts


class DistributionTest(unittest.TestCase):
    N = 24000

    def test_the_speculative_triple_frequencies_match_the_models(self):
        theory = chain_theory(3)
        for draft_length in (1, 3, 6):
            with self.subTest(draft_length=draft_length):
                counts = simulate_chain(range(1000, 1000 + self.N), draft_length)
                stat, df = chi_square(counts, theory)
                self.assertLess(stat, chi2_critical(df), (stat, df))

    def test_the_test_has_power_a_biased_acceptance_rule_is_caught(self):
        stat, df = chi_square(simulate_chain(range(1000, 1000 + self.N), 3, acceptance='typical'), chain_theory(3))
        self.assertGreater(stat, 3 * chi2_critical(df), (stat, df))

    def test_the_test_has_power_noise_keyed_by_the_round_local_row_is_caught(self):
        """Reusing the noise of row i for whichever position row i lands on correlates a round's first token with the previous round's last."""
        stat, df = chi_square(simulate_chain(range(1000, 1000 + self.N), 3, keying='row'), chain_theory(3))
        self.assertGreater(stat, 3 * chi2_critical(df), (stat, df))

    def test_plain_sampling_is_the_same_distribution(self):
        """Plain one-at-a-time sampling over the same seeds: the walk with no drafts at all."""
        counts = simulate_chain(range(1000, 1000 + self.N), 0)
        stat, df = chi_square(counts, chain_theory(3))
        self.assertLess(stat, chi2_critical(df), (stat, df))


if __name__ == '__main__':
    unittest.main()
