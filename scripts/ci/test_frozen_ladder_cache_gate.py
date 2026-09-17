import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from frozen_ladder_cache_gate import HASHES, RUNTIME, SOURCES, qualify, validate


def fixture(context=65536):
    sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in SOURCES}
    checks = [dict(context=context, seed=seed, chip=chip, name=name, exact=True)
        for seed in (0, 1) for chip in (0, 1) for name in ('replay', 'input_unchanged')]
    checks.extend(dict(context=context, seed=0, chip=chip, name='eager', exact=True) for chip in (0, 1))
    return dict(passed=True, closed_cleanly=True, context=context, backend='simulator', stage='complete',
        performance_qualified=False, model_integrated=False, native_hashes=HASHES,
        generated_hashes=HASHES, sources=sources, sources_after=dict(sources), checks=checks)


class CacheGateTests(unittest.TestCase):
    def test_complete_matrix_and_partial_timeout_rejected(self):
        for context in (65536, 131072):
            self.assertEqual(validate(fixture(context), context)['context'], context)
        for key, value in (('passed', False), ('closed_cleanly', False), ('stage', 'replay'),
                ('backend', 'hardware'), ('context', 131072), ('performance_qualified', True)):
            report = fixture()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(report, 65536)

    def test_missing_duplicate_wrong_chip_seed_or_changed_source_rejected(self):
        for mode in ('missing', 'duplicate', 'chip', 'seed', 'exact', 'source'):
            report = copy.deepcopy(fixture())
            if mode == 'missing':
                report['checks'].pop()
            elif mode == 'duplicate':
                report['checks'][-1] = report['checks'][0]
            elif mode == 'source':
                report['sources_after']['ordered_cache.py'] = '0' * 64
            else:
                report['checks'][0][mode] = False if mode == 'exact' else 2
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(report, 65536)

    def test_file_pin_exit_runtime_and_live_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in SOURCES:
                (root / name).write_bytes(name.encode())
            raw = json.dumps(fixture()).encode()
            digest = hashlib.sha256(raw).hexdigest()
            (root / 'ladder-cache.json').write_bytes(raw)
            (root / 'ladder-cache.exit-status').write_text('0')
            (root / 'simulator-runtime.txt').write_text(RUNTIME)
            qualify(root, root, digest, 65536)
            with self.assertRaises(ValueError):
                qualify(root, root, '0' * 64, 65536)
            for name, value in (('ladder-cache.exit-status', '124'),
                    ('simulator-runtime.txt', 'wrong'), ('ordered_cache.py', 'changed')):
                path = root / name
                original = path.read_bytes()
                path.write_text(value)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    qualify(root, root, digest, 65536)
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
