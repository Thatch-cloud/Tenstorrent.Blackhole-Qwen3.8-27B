import copy
from pathlib import Path
import unittest

from mlp_progressive_input_report import digest, inspect, source_record, validate_candidate
from test_mlp_block_stream_gate import BlockStreamGateTests


class ProgressiveReportTests(unittest.TestCase):
    def fixture(self):
        directory = Path(__file__).parent
        staged = source_record(directory)
        serial = BlockStreamGateTests().fixture()
        serial.update(kernels=[dict(fused_compute_sha256='unchanged',
            reader_sha256={'fused_1d_input.cpp': staged['before']['fused_1d_input.cpp']})],
            weight_checks=[dict(chip=chip, pages=43520, mismatched_words=0, exact=True,
                workers=64, source_exact=True, projection=projection)
                for projection in ('gate', 'up') for chip in (0, 1)],
            stream_binding=dict(addresses=[[1, 2], [3, 4]], arithmetic_changed=False),
            packer_header_sha256='unchanged', packer_zero_graft=True, precision='unchanged',
            control_epilogue='unchanged', weight_check_backend='unchanged', weight_check_sources={})
        report = copy.deepcopy(serial)
        report['kernels'][0].update(progressive_input=True, input_buffer_tiles=160)
        report['kernels'][0]['reader_sha256']['fused_1d_input.cpp'] = staged['after']['fused_1d_input.cpp']
        report.update(buffer_candidate_sha256=staged['after']['fused_1d.py'],
            trace_source_sha256=digest((directory / 'fusion_trace.py').read_bytes()))
        return report, serial, staged, directory

    def test_only_admitted_source_delta_passes(self):
        report, serial, staged, directory = self.fixture()
        validate_candidate(report, serial, staged, directory)
        for mutation in ('compute', 'buffer', 'source', 'weights', 'alias', 'packer', 'replay'):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(report)
                if mutation == 'compute':
                    changed['kernels'][0]['fused_compute_sha256'] = 'different'
                elif mutation == 'buffer':
                    changed['kernels'][0]['input_buffer_tiles'] = 16
                elif mutation == 'source':
                    changed['buffer_candidate_sha256'] = 'different'
                elif mutation == 'weights':
                    changed['weight_checks'].pop()
                elif mutation == 'alias':
                    changed['stream_binding']['addresses'][0] = [1, 1]
                elif mutation == 'packer':
                    changed['packer_zero_graft'] = False
                else:
                    changed['trace_replays'][0]['checks'].pop()
                with self.assertRaises(ValueError):
                    validate_candidate(changed, serial, staged, directory)

    def test_unpinned_reference_rejected_before_candidate(self):
        with self.assertRaisesRegex(ValueError, 'Pinned serial'):
            inspect('.', Path(__file__), Path(__file__).parent)


if __name__ == '__main__':
    unittest.main()
