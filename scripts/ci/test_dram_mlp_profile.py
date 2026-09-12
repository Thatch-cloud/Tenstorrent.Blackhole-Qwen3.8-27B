from contextlib import redirect_stdout
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dram_mlp_gate import qualify_hardware
from dram_mlp_profile_report import validate_profile
from tensix_mlp_profile import MlpTraceProfile
from tensix_mlp_profile_report import attribute
from test_tensix_mlp_profile import device_rows, fixture as t8_fixture


def fixture():
    unused, report, unused_console = t8_fixture()
    operations = SimpleNamespace(synchronize_device=Mock(), ReadDeviceProfiler=Mock(), execute_trace=Mock())
    observer = MlpTraceProfile(operations, object(), [23, 24], signpost=Mock(), rows=16)
    with redirect_stdout(io.StringIO()) as console:
        for pattern in range(3):
            for arm in range(2):
                observer.replay(arm, pattern, 'audit')
            for sample, arm in enumerate((0, 1, 1, 0)):
                observer.replay(arm, pattern, 'measurement', sample)
    report.update(rows=16, sharded_product=True, profile=observer.summary(),
        input_checks=[dict(pattern=pattern, unchanged=True, bindings_stable=True) for pattern in range(3)])
    return report, console.getvalue()


class DramMlpProfileTests(unittest.TestCase):
    def test_both_chips_and_arms_have_all_six_measurements(self):
        report, console = fixture()
        records = validate_profile(report, console)
        self.assertEqual(len(records), 18)
        devices = attribute(records, device_rows(records), full_rows=16)
        self.assertEqual(len(devices), 4)
        self.assertTrue(all(operation['samples'] == 6 for device in devices
            for operation in device['measured_operations']))
        with self.assertRaises(ValueError):
            qualify_hardware(report)

    def test_missing_changed_or_promoted_evidence_rejected(self):
        original, console = fixture()
        for failure in ('missing', 'duplicate', 'order', 'ordinal', 'timing', 'promotion', 'input', 'exact', 'rows'):
            report = deepcopy(original)
            records = report['profile']['records']
            if failure == 'missing':
                records.pop()
            elif failure == 'duplicate':
                records[1] = records[0]
            elif failure == 'order':
                records[0], records[1] = records[1], records[0]
            elif failure == 'ordinal':
                records[0]['trace_ordinal'] = False
            elif failure == 'timing':
                report['control_ms'] = 0.3
            elif failure == 'promotion':
                report['eligible_for_full_model_gate'] = True
            elif failure == 'input':
                report['input_checks'].pop()
            elif failure == 'exact':
                report['profile_checks'][0]['exact'] = False
            else:
                report['rows'] = 8
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate_profile(report, console)
        for changed in (console.replace('QWEN_MLP_PROFILE_END', 'missing', 1),
                console.replace('QWEN_MLP_PROFILE_END', 'markers were dropped\nQWEN_MLP_PROFILE_END', 1)):
            with self.assertRaises(ValueError):
                validate_profile(original, changed)


if __name__ == '__main__':
    unittest.main()
