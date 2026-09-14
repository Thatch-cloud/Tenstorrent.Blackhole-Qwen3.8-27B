import unittest
from pathlib import Path
from types import SimpleNamespace

import dspark_score_sfpu
from dspark_center_tile_fill import START, center_fill_scope, transform


class CenterFillTests(unittest.TestCase):
    def test_simulator_route_is_selected_only_in_simulator_mode(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/qwen-experiments.yml').read_text()
        step = workflow.split('      - name: CPU-only draft simulator\n', 1)[1]
        condition = step.split('        if: ', 1)[1].splitlines()[0]
        for simulator_only in (False, True):
            inputs = SimpleNamespace(simulator_only=simulator_only, suite='dspark-center-tile-fill-sim')
            self.assertEqual(eval(condition.replace('&&', ' and ').replace('||', ' or '),
                {'__builtins__': {}}, {'inputs': inputs}), simulator_only)
        self.assertIn("inputs.suite == 'dspark-center-tile-fill-sim' && 'dspark-center-tile-fill'", step)

    def test_preserves_staging_and_exact_row_maxima(self):
        original = dspark_score_sfpu.HELPER
        changed = transform(original)
        self.assertEqual(original.split(START)[0], changed.split(START)[0])
        rows = '    for (uint32_t row = 0; row < 32; ++row) {'
        self.assertEqual(original[original.index(rows):], changed[changed.index(rows):])
        self.assertNotIn('scratch[index] = 0;', changed)
        self.assertIn('fill_tile(0, 0.0f);', changed)
        with center_fill_scope():
            self.assertEqual(dspark_score_sfpu.HELPER, changed)
        self.assertEqual(dspark_score_sfpu.HELPER, original)

    def test_rejects_reapplication(self):
        with self.assertRaises(ValueError):
            transform(transform(dspark_score_sfpu.HELPER))


if __name__ == '__main__':
    unittest.main()
