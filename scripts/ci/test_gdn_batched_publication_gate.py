import copy
import hashlib
from pathlib import Path
import unittest

from gdn_batched_publication_gate import SOURCES, validate


class BatchedPublicationGateTests(unittest.TestCase):
    def fixture(self, layers):
        root = Path(__file__).parent
        hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
        from tensor_bit_compare_gate import qualify
        report = dict(passed=True, closed_cleanly=True, padding_audited=True, backend='simulator',
            stage='complete', rows=16, layers=layers, sources=hashes, sources_after=dict(hashes),
            checks=[], padding_checks=[], poison_checks=[], comparator=qualify(root))
        prefixes = range(17) if layers == 1 else (0, 1, 8, 16)
        executions = []
        for prefix in prefixes:
            for pattern in (0, 1):
                executions.extend([(prefix, pattern, 'native', None), (prefix, pattern, 'candidate', None)])
        for prefix in prefixes:
            executions.extend((prefix, pattern, 'replay', ordinal) for ordinal, pattern in enumerate((0, 1, 0)))
        for prefix, pattern, arm, repetition in executions:
            for layer in range(layers):
                for chip in (0, 1):
                    entry = dict(prefix=prefix, pattern=pattern, arm=arm, repetition=repetition,
                        layer=layer, chip=chip, exact=True)
                    report['checks'].append(entry)
                    for operand in range(20):
                        if operand % 5:
                            report['padding_checks'].append(dict(entry, operand=operand))
                    report['poison_checks'].append(dict(pattern=pattern, layer=layer, chip=chip, exact=True))
        return report

    def test_complete_matrices_do_not_claim_hardware_or_serving(self):
        for layers, checks in ((1, 238), (48, 2688)):
            result = validate(self.fixture(layers), Path(__file__).parent, layers, '0\n')
            self.assertEqual(result['checks'], checks)
            self.assertEqual(result['padding_checks'], checks * 16)
            self.assertFalse(result['hardware_qualified'])
            self.assertFalse(result['serving_qualified'])

    def test_incomplete_stale_failed_or_duplicate_evidence_is_rejected(self):
        original = self.fixture(1)
        for field in ('checks', 'padding_checks', 'poison_checks'):
            for change in ('missing', 'duplicate', 'false'):
                report = copy.deepcopy(original)
                if change == 'missing':
                    report[field].pop()
                elif change == 'duplicate':
                    report[field].append(report[field][0])
                else:
                    report[field][0]['exact'] = False
                with self.assertRaises(ValueError):
                    validate(report, Path(__file__).parent, 1, '0')
        for field, value in (('padding_audited', False), ('closed_cleanly', False), ('stage', 'replay_0'),
                ('sources_after', {}), ('error', 'interrupted'), ('backend', 'hardware'), ('comparator', {})):
            report = copy.deepcopy(original)
            report[field] = value
            with self.assertRaises(ValueError):
                validate(report, Path(__file__).parent, 1, '0')
        with self.assertRaises(ValueError):
            validate(original, Path(__file__).parent, 1, '124')
        with self.assertRaises(ValueError):
            validate(original, Path(__file__).parent, True, '0')
