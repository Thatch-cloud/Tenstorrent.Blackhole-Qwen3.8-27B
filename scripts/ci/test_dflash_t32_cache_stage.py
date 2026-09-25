from copy import deepcopy
from pathlib import Path
import unittest

from dflash_t32_cache_gate import qualify
from dflash_t32_cache_stage import payloads


class CacheReplayStageTests(unittest.TestCase):
    def test_staged_adapter_and_bounded_weight_free_case(self):
        directory = Path(__file__).parent
        original = (directory / 'dflash_proposal_trace.py').read_bytes()
        sources = payloads(directory)
        for name, source in sources.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')
        compile((directory / 'dflash-t32-cache-replay-probe.py').read_text(), 'probe', 'exec')
        self.assertIn('device.block_rows == 32', sources['dflash_proposal_trace.py'])
        self.assertIn('timeout -k 15 360', sources['simulator-suite.sh'])
        self.assertIn('dflash_t32_cache_gate.py', sources['simulator-suite.sh'])
        self.assertNotIn('for context in 31 2048', sources['simulator-suite.sh'])
        self.assertEqual(original, (directory / 'dflash_proposal_trace.py').read_bytes())

    def report(self):
        checks = [dict(stage='initial', chip=chip, position=2048, exact=True, finite=True, rows=32) for chip in range(2)]
        for prefix, before, after in ((1, 2048, 2049), (16, 2049, 2065), (32, 2065, 2097)):
            checks.append(dict(stage='pending-rejected', prefix=prefix, exact=True))
            for stage, position in ((f'discard-{prefix}', before), (f'commit-{prefix}', after)):
                checks.extend(dict(stage=stage, chip=chip, position=position, exact=True, finite=True, rows=32) for chip in range(2))
        return dict(passed=True, closed_cleanly=True, block_rows=32, learned_qualified=False,
            hardware_qualified=False, performance_qualified=False, sources={'probe': 'fixture'},
            native_sources={'native': 'fixture'}, native_sources_after={'native': 'fixture'},
            runtime_binaries={'binary': 'fixture'}, runtime_binaries_after={'binary': 'fixture'}, checks=checks)

    def admit(self, report):
        return qualify(report, {'probe': 'fixture'}, {'native': 'fixture'}, {'binary': 'fixture'})

    def test_each_matrix_entry_and_provenance_are_required(self):
        report = self.report()
        self.assertTrue(self.admit(report)['cache_boundary_qualified'])
        for index in range(len(report['checks'])):
            broken = deepcopy(report)
            del broken['checks'][index]
            with self.assertRaises(ValueError):
                self.admit(broken)
        for name in ('sources', 'native_sources_after', 'runtime_binaries_after'):
            with self.assertRaises(ValueError):
                self.admit({**report, name: {}})
        for name in ('learned_qualified', 'hardware_qualified', 'performance_qualified'):
            with self.assertRaises(ValueError):
                self.admit({**report, name: True})


if __name__ == '__main__':
    unittest.main()
