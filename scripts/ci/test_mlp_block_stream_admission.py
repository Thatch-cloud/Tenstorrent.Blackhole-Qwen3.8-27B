import copy
import hashlib
from pathlib import Path
import unittest

from mlp_block_stream import geometry
from mlp_block_stream_admission import SOURCES, admit_transport


class TransportAdmissionTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).parent
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
        return dict(passed=True, closed_cleanly=True, backend='simulator', sources=sources,
            sources_after=dict(sources), mlp_qualified=False, hardware_qualified=False,
            performance_qualified=False, checks=[dict(pairs=pairs, blocks=blocks, pattern=pattern,
                chip=chip, pages=geometry(pairs, blocks)['stream_pages'], source_unchanged=True,
                exact_all_words=True) for pairs, blocks in ((8, 2), (272, 1))
                for pattern in (0, 1) for chip in (0, 1)])

    def test_transport_does_not_qualify_mlp_or_performance(self):
        result = admit_transport(self.fixture(), Path(__file__).parent)
        self.assertTrue(result['transport_qualified'])
        self.assertEqual(result['checks'], 8)
        for field in ('mlp_qualified', 'hardware_qualified', 'performance_qualified'):
            self.assertFalse(result[field])

    def test_rejects_incomplete_duplicate_or_inexact_checks(self):
        report = self.fixture()
        mutations = [lambda value: value['checks'].pop(),
            lambda value: value['checks'].__setitem__(1, copy.deepcopy(value['checks'][0]))]
        for field, invalid in (('pages', 1), ('chip', False), ('pattern', 2),
                ('source_unchanged', False), ('exact_all_words', 1)):
            mutations.append(lambda value, field=field, invalid=invalid:
                value['checks'][0].__setitem__(field, invalid))
        for mutate in mutations:
            candidate = copy.deepcopy(report)
            mutate(candidate)
            with self.assertRaises(ValueError):
                admit_transport(candidate, Path(__file__).parent)

    def test_rejects_stale_sources_failed_close_and_overclaims(self):
        for field, invalid in (('sources', {}), ('sources_after', {}), ('passed', 1),
                ('closed_cleanly', False), ('backend', 'hardware'), ('error', 'failed'),
                ('mlp_qualified', True), ('performance_qualified', True), ('hardware_qualified', True)):
            candidate = self.fixture()
            candidate[field] = invalid
            with self.assertRaises(ValueError):
                admit_transport(candidate, Path(__file__).parent)


if __name__ == '__main__':
    unittest.main()
