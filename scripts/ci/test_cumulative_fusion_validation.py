import copy
import unittest

from cumulative_fusion_validation import BASELINE_SHA256, REGISTER_SHA256, validate_fusion_policy


def fixture(register=False):
    checksum = REGISTER_SHA256 if register else BASELINE_SHA256
    return dict(fused_t16_mlp=dict(passed_simulator=checksum, rows=16, layers=64,
        restored=True, native_bindings_unchanged=True, extra_weight_allocations=0, hits=[2] * 64,
        weight_audit=dict(passed=True, checks=[dict(layer=layer, offset=offset, chip=chip,
            exact=True, pages=43520, mismatched_words=0)
            for layer in range(64) for offset in (0, 1) for chip in (0, 1)])),
        register_epilogue=dict(register_resident=register, report_sha256=checksum if register else None,
            constructions=64 if register else 0, calls=128 if register else 0, restored=True))


class FusionValidationTests(unittest.TestCase):
    def test_explicit_policies_and_legacy_baseline(self):
        baseline = fixture()
        self.assertEqual(validate_fusion_policy(baseline), BASELINE_SHA256)
        del baseline['register_epilogue']
        validate_fusion_policy(baseline)
        self.assertEqual(validate_fusion_policy(fixture(True), 'register'), REGISTER_SHA256)
        with self.assertRaises(ValueError):
            validate_fusion_policy(fixture(True))
        with self.assertRaises(ValueError):
            validate_fusion_policy(baseline, 'register')
        with self.assertRaises(ValueError):
            validate_fusion_policy(baseline, 'unknown')

    def test_register_preserves_full_weight_and_execution_checks(self):
        mutations = (
            lambda value: value['register_epilogue'].update(calls=127),
            lambda value: value['register_epilogue'].update(constructions=63),
            lambda value: value['register_epilogue'].update(restored=False),
            lambda value: value['register_epilogue'].update(report_sha256=BASELINE_SHA256),
            lambda value: value['fused_t16_mlp'].update(native_bindings_unchanged=False),
            lambda value: value['fused_t16_mlp'].update(extra_weight_allocations=1),
            lambda value: value['fused_t16_mlp']['weight_audit']['checks'].pop(),
            lambda value: value['fused_t16_mlp']['weight_audit']['checks'][0].update(exact=False),
            lambda value: value['fused_t16_mlp']['weight_audit']['checks'][0].update(mismatched_words=1),
        )
        for mutation in mutations:
            value = copy.deepcopy(fixture(True))
            mutation(value)
            with self.assertRaises(ValueError):
                validate_fusion_policy(value, 'register')

    def test_baseline_cannot_hide_register_activity(self):
        for field, value in (('calls', 1), ('constructions', 64), ('restored', False),
                             ('register_resident', True), ('report_sha256', REGISTER_SHA256)):
            request = fixture()
            request['register_epilogue'][field] = value
            with self.assertRaises(ValueError):
                validate_fusion_policy(request)
