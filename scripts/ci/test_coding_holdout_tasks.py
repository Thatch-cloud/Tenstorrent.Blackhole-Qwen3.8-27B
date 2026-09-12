import unittest
from unittest.mock import Mock, patch

from coding_holdout_tasks import EXPECTED, TASKS, make_context_prompt, messages, task_manifest


class CodingHoldoutTaskTests(unittest.TestCase):
    def test_distinct_tasks_with_private_cases(self):
        self.assertEqual(task_manifest(), EXPECTED)
        self.assertEqual(len(task_manifest()), 3)
        self.assertEqual(len({task['function'] for task in TASKS}), 3)
        for task in TASKS:
            self.assertGreaterEqual(len(task['cases']), 4)
            self.assertEqual(messages(task['name'])[-1]['content'], task['prompt'])
            self.assertNotIn('cases', messages(task['name'])[-1])
            self.assertNotEqual(task['function'], 'merge_intervals')

    def test_unknown_task_rejected(self):
        with self.assertRaises(ValueError):
            messages('merge_intervals')

    def test_changed_task_contract_rejected(self):
        with patch('coding_holdout_tasks.task_manifest', return_value={}):
            with self.assertRaisesRegex(ValueError, 'definitions changed'):
                messages('stable_unique_v1')

    def test_ci_task_selection_retains_original_default(self):
        from pathlib import Path
        import yaml

        root = Path(__file__).resolve().parents[2]
        workflow = yaml.safe_load((root / '.github/workflows/qwen-experiments.yml').read_text())
        dispatch = workflow.get('on', workflow.get(True))['workflow_dispatch']
        selection = dispatch['inputs']['dspark_coding_task']
        self.assertEqual(selection['default'], 'merge_intervals')
        self.assertEqual(set(selection['options']), {'merge_intervals', *EXPECTED})
        steps = [step for job in workflow['jobs'].values() for step in job.get('steps', [])
            if step.get('run', '').endswith('bash scripts/ci/run-dspark-hardware.sh')]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]['env']['QWEN_DSPARK_CODING_TASK'], '${{ inputs.dspark_coding_task }}')
        runner = (root / 'scripts/ci/run-dspark-hardware.sh').read_text()
        self.assertIn('test "$mode" = request-target-attention', runner)
        self.assertIn('-e "QWEN_DSPARK_CODING_TASK=$task"', runner)

    def test_complete_task_survives_context_fitting(self):
        for task in TASKS:
            tokenizer = Mock()

            def encode(prompt, **options):
                self.assertTrue(prompt[-1]['content'].endswith(task['prompt']))
                self.assertIn('</repository_context>', prompt[-1]['content'])
                self.assertIs(options['enable_thinking'], False)
                return [ord(character) for character in prompt[-1]['content']]

            tokenizer.apply_chat_template.side_effect = encode
            tokens, metadata = make_context_prompt(tokenizer, task['name'])
            self.assertEqual(len(tokens), 4096)
            self.assertEqual(metadata['task_sha256'], EXPECTED[task['name']])
            self.assertEqual(metadata['task'], task['name'])
            self.assertFalse(metadata['functional_quality_qualified'])

    def test_reference_cases_are_internally_consistent(self):
        def stable_unique(values):
            return list(dict.fromkeys(values))

        def run_length_encode(text):
            from itertools import groupby
            return [(character, len(list(group))) for character, group in groupby(text)]

        def rotate_right(values, steps):
            if not values:
                return []
            offset = steps % len(values)
            return values[-offset:] + values[:-offset] if offset else list(values)

        references = dict(stable_unique=stable_unique, run_length_encode=run_length_encode,
            rotate_right=rotate_right)
        for task in TASKS:
            for arguments, expected in task['cases']:
                self.assertEqual(references[task['function']](*arguments), expected)
