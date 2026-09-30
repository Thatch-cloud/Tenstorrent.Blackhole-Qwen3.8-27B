import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gdn_gate_exp_report import KERNEL, inspect, validate_report
from gdn_gate_exp_stage import adapt
from gdn_multitoken import HASHES, KERNEL_ROOT
from gdn_shared_qk_gate import LOCAL_SOURCES


def fixture():
    report = dict(passed=True, closed_cleanly=True, norm_unchanged=True,
        state_math_unchanged=True, shared_qk_preparation=True, gate_exp_fusion=True,
        stage='complete', backend='simulator', rows=16, sources={'source': 'digest'},
        sources_after={'source': 'digest'}, hardware_qualified=False,
        timing_qualified=False, generated_kernels=[dict(KERNEL)])
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        report[field] = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
    return report


class GateExpReportTests(unittest.TestCase):
    def test_complete_exact_matrix(self):
        validate_report(fixture())

    def test_incomplete_failed_or_unrelated_evidence_rejected(self):
        for field, value in (('passed', False), ('closed_cleanly', False),
                ('gate_exp_fusion', False), ('checks', fixture()['checks'][:-1]),
                ('immutable_checks', []), ('stage', 'eager'), ('backend', 'hardware'),
                ('sources_after', {}), ('hardware_qualified', True), ('timing_qualified', True),
                ('generated_kernels', []), ('numerical_failures', [{'max_abs': 1}])):
            with self.subTest(field=field):
                report = fixture()
                report[field] = value
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_candidate_fingerprint_cannot_be_substituted(self):
        report = fixture()
        report['generated_kernels'][0]['candidate_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_artifact_source_and_exit_validation(self):
        directory = Path(__file__).parent
        report = fixture()
        sources = {f'/opt/tt-metal/{KERNEL_ROOT}/{name}': digest
                   for name, digest in HASHES.items()}
        for name in (*LOCAL_SOURCES, 'gdn_gate_exp_fusion.py'):
            payload = (directory / name).read_bytes()
            if name == 'gdn-shared-recurrence-probe.py':
                payload = adapt(payload.decode()).encode()
            sources[f'/experiment-scripts/ci/{name}'] = hashlib.sha256(payload).hexdigest()
        report.update(sources=sources, sources_after=copy.deepcopy(sources))
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            report_path = evidence / 'gdn-shared-recurrence.json'
            report_path.write_text(json.dumps(report))
            (evidence / 'gdn-shared-recurrence.exit-status').write_text('0')
            (evidence / 'simulator-runtime.txt').write_text('9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9')
            (evidence / 'container-cleanup.json').write_text(json.dumps(dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)))
            result = inspect(evidence, directory)
            self.assertTrue(result['simulator_qualified'])
            self.assertFalse(result['hardware_qualified'])
            self.assertFalse(result['performance_qualified'])
            report['sources']['/experiment-scripts/ci/gdn_gate_exp_fusion.py'] = '0' * 64
            report['sources_after'] = copy.deepcopy(report['sources'])
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                inspect(evidence, directory)
            (evidence / 'gdn-shared-recurrence.exit-status').write_text('1')
            with self.assertRaises(ValueError):
                inspect(evidence, directory)
