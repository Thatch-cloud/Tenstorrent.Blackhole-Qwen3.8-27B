"""ft_dataset: the example schema, the feature file round trip, the tap-major assembly (and the mutation that blind concatenation
is wrong), bf16 bits, and the storage arithmetic the design quotes."""
import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import ft_dataset as ds  # noqa: E402


def example(**kwargs):
    base = dict(example_id=7, source='swe', context_ids=[1, 2, 3], answer_ids=[4, 5, 6, 7], feature_file='x.ftf', kind='postfc',
                finish='stop', censored=False, weight=1.0)
    base.update(kwargs)
    return base


class AssemblyTests(unittest.TestCase):
    def chips(self, rows=3, taps=5, per_chip=4, chips=4):
        """Chip c's tap t slice holds the value 1000 * t + 100 * c + column, so the global order is visible in the numbers."""
        out = []
        for chip in range(chips):
            columns = [1000 * tap + 100 * chip + np.arange(per_chip) for tap in range(taps)]
            out.append(np.tile(np.concatenate(columns), (rows, 1)).astype(np.float32))
        return out

    def test_global_order_is_tap_major_over_the_chips(self):
        rows = ds.assemble_taps(self.chips(), taps=5, per_chip=4)
        self.assertEqual(rows.shape, (3, 5 * 4 * 4))
        first = rows[0]
        # tap 0 first: chips 0..3 in order, then tap 1
        expected = [1000 * tap + 100 * chip + col for tap in range(5) for chip in range(4) for col in range(4)]
        self.assertEqual(first.tolist(), expected)

    def test_blind_concatenation_is_the_wrong_order(self):
        chips = self.chips()
        blind = np.concatenate(chips, axis=1)
        self.assertFalse(np.array_equal(blind, ds.assemble_taps(chips, taps=5, per_chip=4)))

    def test_split_is_the_inverse(self):
        chips = self.chips()
        again = ds.split_taps(ds.assemble_taps(chips, 5, 4), chips=4, taps=5)
        for a, b in zip(chips, again):
            self.assertTrue(np.array_equal(a, b))

    def test_a_ragged_chip_is_refused(self):
        chips = self.chips()
        chips[2] = chips[2][:, :-1]
        with self.assertRaises(ds.DatasetError):
            ds.assemble_taps(chips, 5, 4)
        with self.assertRaises(ds.DatasetError):
            ds.assemble_taps([], 5, 4)

    def test_the_real_widths(self):
        self.assertEqual((ds.PER_CHIP, ds.PER_CHIP * ds.CHIPS, ds.TAPS * ds.HIDDEN), (1280, 5120, 25600))


class Bf16Tests(unittest.TestCase):
    def test_known_patterns_and_round_to_nearest_even(self):
        bits = ds.bf16_bits(np.array([1.0, -2.0, 0.0, 3.140625], dtype=np.float32))
        self.assertEqual(bits.tolist(), [0x3F80, 0xC000, 0x0000, 0x4049])
        tie = np.array([1.0 + 2.0 ** -8], dtype=np.float32)            # halfway between two bf16 values: rounds to even (1.0)
        self.assertEqual(ds.bf16_bits(tie).tolist(), [0x3F80])
        self.assertEqual(ds.bf16_bits(np.array([np.nan], dtype=np.float32)).tolist(), [0x7FC0])

    def test_round_trip_is_idempotent(self):
        values = np.random.RandomState(0).randn(64).astype(np.float32)
        once = ds.bf16_values(ds.bf16_bits(values))
        self.assertTrue(np.array_equal(ds.bf16_values(ds.bf16_bits(once)), once))
        self.assertLess(float(np.abs(once - values).max()), 0.02 * float(np.abs(values).max()))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.path = os.path.join(self.root, 'a.ftf')
        self.bits = np.random.RandomState(1).randint(0, 65535, size=(9, ds.HIDDEN)).astype(np.uint16)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_round_trip_and_verify(self):
        header = ds.write_features(self.path, 'postfc', self.bits)
        self.assertEqual(header['payload_offset'] % ds.ALIGN, 0)
        features = ds.FeatureFile(self.path)
        self.assertEqual((features.kind, features.count, features.width), ('postfc', 9, ds.HIDDEN))
        self.assertTrue(np.array_equal(features.rows(2, 5), self.bits[2:5]))
        self.assertTrue(features.verify())
        features.close()

    def test_raw_kind_has_the_five_tap_width(self):
        bits = np.zeros((2, 25600), dtype=np.uint16)
        ds.write_features(self.path, 'raw', bits)
        features = ds.FeatureFile(self.path)
        self.assertEqual(features.width, 25600)
        self.assertEqual(os.path.getsize(self.path) - features.header['payload_offset'], 2 * 51200)
        features.close()

    def test_wrong_shapes_and_kinds_are_refused(self):
        with self.assertRaises(ds.DatasetError):
            ds.write_features(self.path, 'raw', self.bits)
        with self.assertRaises(ds.DatasetError):
            ds.write_features(self.path, 'bf8', self.bits)
        with self.assertRaises(ds.DatasetError):
            ds.write_features(self.path, 'postfc', self.bits.astype(np.float32))

    def test_corruption_is_caught(self):
        ds.write_features(self.path, 'postfc', self.bits)
        with open(self.path, 'r+b') as handle:                          # flip a payload byte
            handle.seek(-5, os.SEEK_END)
            byte = handle.read(1)
            handle.seek(-5, os.SEEK_END)
            handle.write(bytes([byte[0] ^ 0xFF]))
        features = ds.FeatureFile(self.path)
        with self.assertRaises(ds.DatasetError):
            features.verify()
        features.close()

    def test_truncation_a_bad_magic_and_a_bad_header_are_refused(self):
        ds.write_features(self.path, 'postfc', self.bits)
        with open(self.path, 'rb') as handle:
            data = handle.read()
        for broken in (data[:-100], b'XXXX' + data[4:], data[:4] + struct.pack('<I', 10 ** 6) + data[8:]):
            with open(self.path, 'wb') as handle:
                handle.write(broken)
            with self.assertRaises(ds.DatasetError):
                ds.FeatureFile(self.path)

    def test_out_of_range_rows_are_refused(self):
        ds.write_features(self.path, 'postfc', self.bits)
        features = ds.FeatureFile(self.path)
        with self.assertRaises(ds.DatasetError):
            features.rows(0, 10)
        features.close()

    def test_an_empty_file_round_trips(self):
        ds.write_features(self.path, 'postfc', np.zeros((0, ds.HIDDEN), dtype=np.uint16))
        features = ds.FeatureFile(self.path)
        self.assertEqual(features.count, 0)
        self.assertTrue(features.verify())
        features.close()


class SchemaTests(unittest.TestCase):
    def test_a_valid_example_passes(self):
        self.assertEqual(ds.validate_example(example())['source'], 'swe')

    def test_every_refusal(self):
        bad = [example(source='web'), example(context_ids=[]), example(answer_ids=[1, -2]), example(answer_ids=[1.5]),
               example(kind='bf8'), example(weight=0), example(finish='abort'), example(censored=True),
               example(finish='length', censored=False), dict((k, v) for k, v in example().items() if k != 'weight')]
        for item in bad:
            with self.assertRaises(ds.DatasetError, msg=str(item)):
                ds.validate_example(item)

    def test_a_censored_example_is_one_cut_by_its_budget(self):
        self.assertTrue(ds.validate_example(example(finish='length', censored=True))['censored'])

    def test_expected_rows_and_the_features_check(self):
        self.assertEqual(ds.expected_rows(35000, 260), 2048 + 259)
        self.assertEqual(ds.expected_rows(100, 5), 100 + 4)
        root = tempfile.mkdtemp()
        try:
            path = os.path.join(root, 'f.ftf')
            ds.write_features(path, 'postfc', np.zeros((3 + 3, ds.HIDDEN), dtype=np.uint16))      # 3 prompt rows + 3 answer rows
            features = ds.FeatureFile(path)
            self.assertTrue(ds.check_features(example(), features))
            with self.assertRaises(ds.DatasetError):
                ds.check_features(example(answer_ids=[4, 5]), features)
            with self.assertRaises(ds.DatasetError):
                ds.check_features(example(kind='raw'), features)
            features.close()
        finally:
            shutil.rmtree(root)


class StorageTests(unittest.TestCase):
    def test_the_designs_per_turn_sizes(self):
        raw = ds.storage_per_turn('raw', 35000, 260)
        self.assertAlmostEqual(raw / 1e6, 118.1, delta=0.5)
        self.assertAlmostEqual(ds.storage_per_turn('postfc', 35000, 260) / 1e6, 23.6, delta=0.2)
        self.assertAlmostEqual(ds.storage_per_turn('raw_bf8', 35000, 260) / 1e6, 62.7, delta=0.5)

    def test_the_eval_tier(self):
        self.assertAlmostEqual(ds.tier_bytes('raw', 1000) / 1e9, 118.1, delta=0.5)
        self.assertAlmostEqual(ds.tier_bytes('postfc', 5000) / 1e9, 118.0, delta=0.5)

    def test_short_prompts_store_their_whole_prompt(self):
        self.assertEqual(ds.storage_per_turn('postfc', 100, 11), (100 + 10) * 10240)
        with self.assertRaises(ds.DatasetError):
            ds.storage_per_turn('nope', 1, 1)


if __name__ == '__main__':
    unittest.main()
