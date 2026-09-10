import copy
import hashlib
import json
from pathlib import Path
import unittest

from dram_mlp_gate import SOURCES, qualify, qualify_hardware
from tiny_mlp_gate import PACKER, ORIGINAL_PACKER, TP_COMMON, HARDWARE_TP_COMMON


class DramMlpGateTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent
        self.report = json.loads((root / 'dram-mlp-simulator.json').read_text())
        self.sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
        self.native = self.report['native_sources']

    def test_recorded_simulator_pass_matches_current_sources(self):
        self.assertTrue(qualify(self.report, self.sources, self.native, '0')['passed'])

    def test_incomplete_duplicate_or_failed_checks_rejected(self):
        for mutation in ('missing', 'duplicate', 'failed', 'unclean'):
            report = copy.deepcopy(self.report)
            if mutation == 'missing':
                report['replay_checks'].pop()
            elif mutation == 'duplicate':
                report['replay_checks'][1] = report['replay_checks'][0]
            elif mutation == 'failed':
                report['eager_checks'][0]['exact'] = False
            else:
                report['closed_cleanly'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(report, self.sources, self.native, '0')

    def test_nonzero_exit_and_source_drift_rejected(self):
        with self.assertRaises(ValueError):
            qualify(self.report, self.sources, self.native, '1')
        changed = dict(self.sources)
        changed['dram_mlp.py'] = '0' * 64
        with self.assertRaises(ValueError):
            qualify(self.report, changed, self.native, '0')

    def test_only_audited_hardware_differences_allowed(self):
        native = dict(self.native, **{PACKER: ORIGINAL_PACKER, TP_COMMON: HARDWARE_TP_COMMON})
        self.assertTrue(qualify(self.report, self.sources, native, '0', hardware=True)['passed'])
        factory = next(name for name in native if name.endswith('dram_sharded_program_factory.cpp'))
        native[factory] = '0' * 64
        with self.assertRaises(ValueError):
            qualify(self.report, self.sources, native, '0', hardware=True)

    def test_hardware_raw_samples_and_clean_closure_required(self):
        report = dict(passed=True, closed_cleanly=True, backend='hardware', stage='complete',
            rows=16, streams=1, layer=0, collective_links=4, seeds=[1659, 2670, 3781],
            repeats_per_sample=50, sources=self.sources, sources_after=self.sources,
            native_sources=self.native, native_sources_after=self.native,
            eager_checks=[dict(pattern=pattern, chip=chip, exact=True)
                for pattern in range(3) for chip in range(2)],
            trace_checks=[dict(pattern=pattern, arm=arm, chip=chip, exact=True)
                for pattern in range(3) for arm in range(2) for chip in range(2)],
            blocks=[dict(pattern=pattern, block=block, samples_ms=[2.0, 1.0, 1.0, 2.0],
                control_ms=2.0, candidate_ms=1.0, ratio=2.0) for pattern in range(3) for block in range(3)],
            control_ms=2.0, candidate_ms=1.0, eligible_for_full_model_gate=True)
        self.assertTrue(qualify_hardware(report)['eligible_for_full_model_gate'])
        broken = copy.deepcopy(report)
        broken['blocks'][0]['samples_ms'][0] = 3.0
        with self.assertRaises(ValueError):
            qualify_hardware(broken)
        report['closed_cleanly'] = False
        with self.assertRaises(ValueError):
            qualify_hardware(report)
