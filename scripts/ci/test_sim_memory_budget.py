from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from sim_memory_budget import MEMORY_MAX, SWAP_MAX, require_clean, snapshot


class SimulatorMemoryBudgetTests(unittest.TestCase):
    def fixture(self, root):
        controller = root/'controller'
        directory = controller/'test.scope'
        directory.mkdir(parents=True)
        values = {'memory.max':MEMORY_MAX,'memory.swap.max':SWAP_MAX,'memory.current':4096,
            'memory.peak':8192,'memory.swap.current':0,'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0'}
        for name,value in values.items():
            (directory/name).write_text(str(value)+'\n')
        (root/'membership').write_text('0::/test.scope\n')
        (root/'boot').write_text('fixed-boot-id\n')
        return dict(bounded=True,membership=root/'membership',root=controller,boot=root/'boot')

    def test_effective_budget_and_clean_reclaim_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            before = snapshot(**self.fixture(Path(directory)))
            self.assertEqual(before['limits']['memory.max'],MEMORY_MAX)
            self.assertEqual(before['limits']['memory.swap.max'],SWAP_MAX)
            after = deepcopy(before)
            after['events']['max'] = 9
            after['limits']['memory.peak'] = MEMORY_MAX
            require_clean(before,after)

    def test_unbounded_controller_cannot_claim_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.fixture(Path(directory))
            (options['root']/'test.scope/memory.max').write_text('max\n')
            with self.assertRaises(ValueError):
                snapshot(**options)
            self.assertIsNone(snapshot(**dict(options,bounded=False))['limits']['memory.max'])

    def test_escape_and_missing_membership_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            options = self.fixture(Path(directory))
            for value in ('0::/../../outside\n','1:memory:/test.scope\n'):
                options['membership'].write_text(value)
                with self.assertRaises(ValueError):
                    snapshot(**options)

    def test_boot_changes_and_oom_are_not_clean_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            before = snapshot(**self.fixture(Path(directory)))
            for field in ('boot_id','oom','oom_kill','memory.max'):
                after = deepcopy(before)
                if field=='boot_id':
                    after[field] = 'another-boot'
                elif field=='memory.max':
                    after['limits'][field] += 4096
                else:
                    after['events'][field] += 1
                with self.assertRaises(ValueError):
                    require_clean(before,after)


if __name__ == '__main__':
    unittest.main()
