import unittest
from mlp_register_epilogue import CAST, transform as original_transform
from mlp_rounding_policy import INSTRUCTIONS, OVERRIDE, round_word, transform, validate_header
from test_mlp_register_epilogue import fixture


class RoundingPolicyTests(unittest.TestCase):
    def test_policy_only_changes_two_constants_after_cast_init(self):
        control = original_transform(fixture())
        candidate = transform(fixture())
        self.assertEqual(candidate.replace(OVERRIDE, ''), control)
        self.assertLess(candidate.index(f'typecast_tile_init<{CAST}>();'), candidate.index(OVERRIDE))
        self.assertLess(candidate.index(OVERRIDE), candidate.index(f'typecast_tile<{CAST}>(0);'))
        self.assertLess(candidate.index(OVERRIDE), candidate.index('mul_binary_tile_init();'))

    def test_all_finite_bf16_bins_and_both_signs(self):
        for upper in range(65536):
            if upper & 0x7f80 == 0x7f80:
                continue
            for tail in (0, 0x7fff, 0x8000, 0x8001, 0xffff):
                word = (upper << 16) | tail
                nearest = round_word(word, ties_away=False)
                away = round_word(word, ties_away=True)
                if tail == 0x8000 and upper % 2 == 0:
                    self.assertEqual(away, nearest + 65536)
                else:
                    self.assertEqual(away, nearest)

    def test_unknown_or_changed_native_cast_rejected(self):
        source = 'inline void calculate_typecast_fp32_to_fp16b() {\n' + INSTRUCTIONS + '\n}\n'
        source += 'sfpi::vConstIntPrgm0 = 1;\nsfpi::vConstIntPrgm1 = 0x7fff;\nsfpi::vConstIntPrgm2 = 0xffff0000;'
        validate_header(source)
        for changed in (source + source, source.replace('LREG12', 'LREG11'),
                source.replace('0x7fff', '0x8000'), source.replace('InstrModLoadStore::FP32', 'InstrModLoadStore::DEFAULT')):
            with self.assertRaises(ValueError):
                validate_header(changed)

    def test_nonfinite_and_invalid_words_rejected(self):
        for word in (-1, 0x100000000, True, 0x7f800000, 0xff800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                round_word(word, ties_away=True)


if __name__ == '__main__':
    unittest.main()
