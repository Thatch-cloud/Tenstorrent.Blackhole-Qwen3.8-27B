from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('ladder_probe_tested',
    Path(__file__).with_name('dspark-ladder-attention-probe.py'))
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class LadderProbeTests(unittest.TestCase):
    def test_context_specific_report_and_child_entrypoint(self):
        for context in PROBE.CONTEXTS:
            with self.subTest(context=context):
                fixture = PROBE.geometry(context)
                stub = SimpleNamespace(SOURCES=(), json=json)
                def run():
                    self.assertEqual(Path(stub.__file__).name, 'dspark-ladder-attention-probe.py')
                    self.assertIn('dspark-native-8k-attention-probe.py', stub.SOURCES)
                    report = json.loads(stub.json.dumps(dict(capacity=fixture['capacity'],
                        numerical_tolerances={}, native_padded_keys=8704, added_masked_poison_rows=192)))
                    self.assertEqual(report['native_padded_keys'], fixture['native_keys'])
                    self.assertEqual(report['added_masked_poison_rows'], fixture['extra_masked_keys'])
                    self.assertEqual(report['ladder_geometry']['context'], context)
                    self.assertFalse(report['stage_instrumented'])
                    self.assertFalse(report['value_diagnostics_enabled'])
                    self.assertEqual(PROBE.dspark_attention_value_diagnostics.KINDS, ())
                    self.assertFalse(report['performance_qualified'])
                    self.assertEqual(report['sum_unpack_mode'], 'native-tf32')
                    self.assertEqual(report['reciprocal_mode'], 'tr0-scalar-fp32-diagnostic')
                    self.assertIn('dspark_ladder_scalar_reciprocal.py', stub.SOURCES)
                    self.assertFalse(report['ladder_geometry']['runtime_admitted'])
                stub.main = run
                with patch.dict(os.environ, dict(QWEN_LADDER_CONTEXT=str(context),
                        QWEN_SIM_CASE='dspark-ladder-attention', QWEN_DRAFT_FP32_INTERMEDIATES='1')), \
                        patch.dict(os.environ, {'QWEN_LADDER_VALUE_DIAGNOSTICS': '0'}), \
                        patch.object(PROBE, 'fixture_probe', return_value=nullcontext(stub)):
                    PROBE.main()
                self.assertIs(stub.json, json)

    def test_diagnostics_opt_in_and_restoration(self):
        kinds = PROBE.dspark_attention_value_diagnostics.KINDS
        self.assertTrue(kinds)
        stub = SimpleNamespace(SOURCES=(), json=json)
        def run():
            self.assertEqual(PROBE.dspark_attention_value_diagnostics.KINDS, kinds)
            report = json.loads(stub.json.dumps(dict(capacity=33792, numerical_tolerances={})))
            self.assertTrue(report['value_diagnostics_enabled'])
        stub.main = run
        with patch.dict(os.environ, dict(QWEN_LADDER_CONTEXT='32768',
                QWEN_SIM_CASE='dspark-ladder-attention', QWEN_DRAFT_FP32_INTERMEDIATES='1',
                QWEN_LADDER_VALUE_DIAGNOSTICS='1')), \
                patch.object(PROBE, 'fixture_probe', return_value=nullcontext(stub)):
            PROBE.main()
        self.assertEqual(PROBE.dspark_attention_value_diagnostics.KINDS, kinds)

    def test_rejects_invalid_diagnostics_mode(self):
        with patch.dict(os.environ, {'QWEN_LADDER_VALUE_DIAGNOSTICS': 'yes'}), \
                patch.object(PROBE, 'fixture_probe') as factory:
            with self.assertRaises(ValueError):
                PROBE.main()
            factory.assert_not_called()

    def test_rejects_unsupported_context_before_fixture_creation(self):
        with patch.dict(os.environ, {'QWEN_LADDER_CONTEXT': '262144'}), \
                patch.object(PROBE, 'fixture_probe') as factory:
            with self.assertRaises(ValueError):
                PROBE.main()
            factory.assert_not_called()
