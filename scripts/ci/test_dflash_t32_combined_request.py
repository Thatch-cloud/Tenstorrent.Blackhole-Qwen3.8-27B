import unittest

from dflash_t32_combined_request import validate_request, STREAM_SHA256, DOWN_SHA256, GDN_SHA256, WINDOW_SHA256


def fixture():
    return dict(selected_drafter='dflash2', dflash=dict(block_rows=32), exact=True, state_exact=True, inactive_exact=True,
        fused_t32_mlp=dict(passed_simulator=STREAM_SHA256, rows=32, layers=64, restored=True,
            native_bindings_unchanged=True, extra_weight_allocations=64, hits=[2] * 64,
            weight_audit=dict(passed=True, checks=[dict(layer=layer, offset=offset, chip=chip,
                exact=True, pages=43520, mismatched_words=0) for layer in range(64)
                for offset in (0, 1) for chip in (0, 1)])),
        block_stream=dict(report_sha256=STREAM_SHA256, rows=32, restored=True, constructions=64,
            stream_allocations=64, calls=128, constructed_layers=list(range(64)), serving_defaults_changed=False),
        mlp_down_grid=dict(report_sha256=DOWN_SHA256, rows=32, restored=True, hits=[2] * 64),
        gdn_shared_qk=dict(admission=dict(report_sha256=GDN_SHA256), restored=True, released=True,
            loads=[dict(rows=32, programs=3, retained_preparation_buffers=2) for unused in range(96)]),
        gdn_direct_window=dict(admission=dict(report_sha256=WINDOW_SHA256), rows=32, restored=True, hits=96))


class T32CombinedRouteTests(unittest.TestCase):
    def test_complete_component_coverage(self):
        validate_request(fixture())

    def test_partial_routes_and_wrong_evidence_rejected(self):
        for mutate in (
                lambda request: request['fused_t32_mlp']['hits'].__setitem__(0, 0),
                lambda request: request['mlp_down_grid']['hits'].__setitem__(0, 1),
                lambda request: request['gdn_shared_qk']['loads'].pop(),
                lambda request: request['gdn_direct_window'].update(hits=48),
                lambda request: request['block_stream'].update(calls=127),
                lambda request: request['block_stream'].update(rows=16),
                lambda request: request.update(state_exact=False),
                lambda request: request['fused_t32_mlp']['weight_audit']['checks'].pop()):
            report = fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                validate_request(report)


if __name__ == '__main__':
    unittest.main()
