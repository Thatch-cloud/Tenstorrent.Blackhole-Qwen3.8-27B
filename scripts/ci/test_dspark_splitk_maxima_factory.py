import unittest

from dspark_splitk_maxima_factory import transform


class MaximaFactoryTests(unittest.TestCase):
    def fixture(self):
        return 'const bool qwen_splitk_fp32 = enabled;\n' + '\n'.join(
            f'    add_cb(CBIndex::c_{index}, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);'
            for index in (27, 28, 29, 30))

    def test_only_maxima_change_under_existing_guard(self):
        source = self.fixture()
        result = transform(source)
        self.assertEqual(result.count('if (qwen_splitk_fp32)'), 2)
        self.assertEqual(result.count('statistics_tiles * im_tile_size'), 2)
        for line in source.splitlines()[1:]:
            self.assertEqual(result.count(line), 1)
        self.assertTrue(result.endswith(source.splitlines()[-1]))

    def test_rejects_unqualified_or_repeated_transform(self):
        for source in ('', self.fixture().replace('c_27', 'c_99'), transform(self.fixture())):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
