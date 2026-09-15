import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dspark_splitk_compile_cache as cache
from dspark_hardware_gate import digest
from dspark_runtime_cache import cache_key, store_entry


class CompileCacheTests(unittest.TestCase):
    def test_report_change_reuses_only_identical_compiled_inputs(self):
        inputs = dict(image='pinned image', backend='hardware', factory='factory hash',
            registration={'source_after': 'registration hash'}, simulator_report='old report',
            builders={name: 'codegen hash' for name in cache.CODEGEN_BUILDERS})
        inputs['builders']['dspark_splitk_sim_gate.py'] = 'old admission'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'binary'
            binary.write_bytes(b'fixture only, not hardware evidence')
            store_entry(root, inputs, binary)
            updated = copy.deepcopy(inputs)
            updated['simulator_report'] = 'new report'
            updated['builders']['dspark_splitk_sim_gate.py'] = 'new admission'
            with patch.object(cache, 'SEED_KEY', cache_key(inputs)), \
                    patch.object(cache, 'SEED_BINARY', digest(binary)):
                identity, manifest = cache.find_entry(root, updated)
                self.assertEqual(manifest['binary_sha256'], digest(binary))
                self.assertEqual(identity, cache.compile_inputs(inputs))
                updated['factory'] = 'different factory'
                self.assertIsNone(cache.find_entry(root, updated)[1])
                updated['factory'] = inputs['factory']
                updated['builders'][cache.CODEGEN_BUILDERS[0]] = 'different codegen'
                self.assertIsNone(cache.find_entry(root, updated)[1])
                (root / cache.SEED_KEY / '_ttnncpp.so').write_bytes(b'corruption')
                with self.assertRaisesRegex(ValueError, 'provenance or binary'):
                    cache.find_entry(root, updated)


if __name__ == '__main__':
    unittest.main()
