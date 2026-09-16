import unittest

from frozen_draft_tail_stage import adapt


class DraftTailStageTests(unittest.TestCase):
    def test_only_scope_and_label_change(self):
        source = {'dspark_8k_scope.py': 'from frozen_gdn_norm_scope import runtime_scope as incremental_scope\n',
            'dspark_request_experiment.py': 'description = "Original versus prefetched norm bridge; incremental publication and shared Q/K in both arms"\n'}
        result = adapt(source)
        self.assertEqual(result['dspark_8k_scope.py'],
            'from frozen_draft_tail_scope import runtime_scope as incremental_scope\n')
        self.assertIn('norm prefetch, incremental publication and shared Q/K in both arms', result['dspark_request_experiment.py'])
        with self.assertRaises(ValueError):
            adapt(result)
