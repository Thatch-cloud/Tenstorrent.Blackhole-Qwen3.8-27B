import unittest
from pathlib import Path
from types import SimpleNamespace

import dspark_ladder_normalization
from dspark_normalization_direct_stage import END, normalization_stage_scope, transform


class NormalizationStageTests(unittest.TestCase):
    def test_hardware_workflow_selection(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/qwen-experiments.yml').read_text()
        step = workflow.split('      - name: DSpark backbone, target-bound proposal or full-request integration\n', 1)[1].split('      - name:', 1)[0]
        inputs = SimpleNamespace(suite='dspark-normalization-direct-stage-hardware')
        condition = step.split('        if: ', 1)[1].splitlines()[0]
        self.assertTrue(eval(condition.replace('&&', ' and ').replace('||', ' or '),
            {'__builtins__': {}}, {'inputs': inputs}))
        expected = {'QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE': '1', 'QWEN_DSPARK_DIRECT_FP32_STAGE': '1',
            'QWEN_DSPARK_SFPU_NUMERICAL': '1', 'QWEN_DSPARK_MASK_BITS': '1',
            'QWEN_DSPARK_SFPU_REQUEST_SCREEN': '0', 'QWEN_DSPARK_SFPU_TIMED': '0'}
        for name, setting in expected.items():
            expression = step.split(name + ': ${{ ', 1)[1].split(' }}', 1)[0]
            self.assertEqual(eval(expression.replace('&&', ' and ').replace('||', ' or '),
                {'__builtins__': {}}, {'inputs': inputs}), setting, name)

    def test_simulator_workflow_selection(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/qwen-experiments.yml').read_text()
        step = workflow.split('      - name: CPU-only draft simulator\n', 1)[1]
        condition = step.split('        if: ', 1)[1].splitlines()[0]
        expression = condition.replace('&&', ' and ').replace('||', ' or ')
        for simulator_only in (False, True):
            inputs = SimpleNamespace(simulator_only=simulator_only, suite='dspark-normalization-direct-stage-sim')
            self.assertEqual(eval(expression, {'__builtins__': {}}, {'inputs': inputs}), simulator_only)
        self.assertIn("inputs.suite == 'dspark-normalization-direct-stage-sim' && 'dspark-normalization-direct-stage'", step)

    def test_request_screen_workflow_selection(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/qwen-experiments.yml').read_text()
        step = workflow.split('      - name: DSpark backbone, target-bound proposal or full-request integration\n', 1)[1].split('      - name:', 1)[0]
        inputs = SimpleNamespace(suite='dspark-normalization-request-screen')
        condition = step.split('        if: ', 1)[1].splitlines()[0]
        self.assertTrue(eval(condition.replace('&&', ' and ').replace('||', ' or '),
            {'__builtins__': {}}, {'inputs': inputs}))
        expected = {'QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE': '1', 'QWEN_DSPARK_DIRECT_FP32_STAGE': '1',
            'QWEN_DSPARK_SFPU_NUMERICAL': '0', 'QWEN_DSPARK_MASK_BITS': '1',
            'QWEN_DSPARK_SFPU_REQUEST_SCREEN': '1', 'QWEN_DSPARK_SFPU_TIMED': '0',
            'QWEN_TARGET_T16_64K_REQUEST': '1'}
        for name, setting in expected.items():
            expression = step.split(name + ': ${{ ', 1)[1].split(' }}', 1)[0]
            self.assertEqual(eval(expression.replace('&&', ' and ').replace('||', ' or '),
                {'__builtins__': {}}, {'inputs': inputs}), setting, name)

    def test_preserves_normalization_math_and_buffer_lifetime(self):
        original = dspark_ladder_normalization.HELPER
        changed = transform(original)
        self.assertEqual(changed[changed.index(END):], original[original.index(END):])
        self.assertNotIn('index < 1024', changed)
        self.assertIn('qwen_stage_score_tile(reciprocal_cb, scratch_cb, false);', changed)
        with normalization_stage_scope():
            self.assertEqual(dspark_ladder_normalization.HELPER, changed)
        self.assertEqual(dspark_ladder_normalization.HELPER, original)

    def test_rejects_changed_or_already_transformed_input(self):
        for source in ('', transform(dspark_ladder_normalization.HELPER)):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
