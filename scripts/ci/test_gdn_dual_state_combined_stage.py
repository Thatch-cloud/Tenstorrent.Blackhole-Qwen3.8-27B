import unittest

from gdn_dual_state_comparison import SCHEDULE
from gdn_dual_state_combined_stage import adapt


class DualStateStageTests(unittest.TestCase):
    def fixture(self):
        return {
            'dspark_request_experiment.py': "def run():\n    schedule = (('publication', True), ('publication', False), ('publication', False))\n    return schedule\n",
            'dspark-target-hardware.py': 'def main():\n    if True:\n        if True:\n            from dspark_request_experiment import run_loaded_requests\n',
            'run-dspark-hardware.sh': 'docker create \\\n    -e "QWEN_DSPARK_MODE=$mode"\n',
        }

    def test_two_audits_and_abba_keep_publication_route(self):
        sources = self.fixture()
        result = adapt(sources)
        namespace = {}
        exec(result['dspark_request_experiment.py'], namespace)
        self.assertEqual(namespace['run'](), tuple(('publication', audit) for precision, audit in SCHEDULE))
        self.assertIn('from gdn_dual_state_experiment import run_loaded_requests', result['dspark-target-hardware.py'])
        self.assertIn('QWEN_GDN_DUAL_STATE_COPY:-0', result['run-dspark-hardware.sh'])
        self.assertEqual(sources, self.fixture())

    def test_duplicate_or_changed_baseline_rejected(self):
        with self.assertRaises(ValueError):
            adapt(adapt(self.fixture()))
        sources = self.fixture()
        sources['dspark_request_experiment.py'] = 'schedule = ()'
        with self.assertRaises(ValueError):
            adapt(sources)
