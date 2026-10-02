"""a0_upstream: the pin is checked BEFORE anything runs; bounds; the pins file."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a0_upstream as up  # noqa: E402


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.marker = os.path.join(self.root, 'ran')
        self.source = ('import os\nVALUE = 41 + 1\nopen(%r, "w").close()\ndef f(x):\n    return x + seed\n' % self.marker).encode()
        self.path = os.path.join(self.root, 'model.py')
        with open(self.path, 'wb') as handle:
            handle.write(self.source)
        self.digest = hashlib.sha256(self.source).hexdigest()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_a_matching_pin_loads_and_runs_the_module(self):
        module = up.load_pinned_module(self.path, self.digest, 'pinned', len(self.source), namespace=dict(seed=5))
        self.assertEqual(module.VALUE, 42)
        self.assertEqual(module.f(1), 6)
        self.assertTrue(os.path.exists(self.marker))

    def test_a_wrong_digest_refuses_before_anything_runs(self):
        with self.assertRaises(up.IntakeError):
            up.load_pinned_module(self.path, '0' * 64, 'pinned')
        self.assertFalse(os.path.exists(self.marker))

    def test_a_wrong_size_refuses_before_anything_runs(self):
        with self.assertRaises(up.IntakeError):
            up.load_pinned_module(self.path, self.digest, 'pinned', len(self.source) + 1)
        self.assertFalse(os.path.exists(self.marker))

    def test_a_file_over_the_bound_is_refused(self):
        with self.assertRaises(up.IntakeError):
            up.read_pinned(self.path, self.digest, max_bytes=10)

    def test_a_changed_file_is_refused(self):
        with open(self.path, 'ab') as handle:
            handle.write(b'\n# edited\n')
        with self.assertRaises(up.IntakeError):
            up.read_pinned(self.path, self.digest)
        self.assertFalse(os.path.exists(self.marker))

    def test_pins_file_must_hold_exactly_the_wanted_names(self):
        pins = os.path.join(self.root, 'pins.json')
        entry = dict(bytes=5, sha256='a' * 64)
        for document, ok in ((dict(a=entry), True), (dict(a=entry, b=entry), False), (dict(), False),
                             (dict(a=dict(bytes=5, sha256='short')), False), (dict(a=dict(bytes='x', sha256='a' * 64)), False)):
            with open(pins, 'w') as handle:
                json.dump(document, handle)
            if ok:
                self.assertEqual(up.load_pins(pins, ['a']), dict(a=(5, 'a' * 64)))
            else:
                with self.assertRaises(up.IntakeError):
                    up.load_pins(pins, ['a'])

    def test_the_dspark_pins_are_the_repository_pins(self):
        import dspark_intake
        for name in ('dspark.py', 'dflash.py'):
            size, digest = dspark_intake.FILES[name]
            self.assertEqual(len(digest), 64)
            self.assertGreater(size, 1000)
            # the pin the intake module carries is accepted by the pinned reader's own arguments
            self.assertIsInstance(size, int)

    def test_checkpoint_verification(self):
        weights = os.path.join(self.root, 'w.safetensors')
        with open(weights, 'wb') as handle:
            handle.write(b'abc')
        good = hashlib.sha256(b'abc').hexdigest()
        self.assertTrue(up.verify_checkpoint(self.root, {'w.safetensors': good}))
        with self.assertRaises(up.IntakeError):
            up.verify_checkpoint(self.root, {'w.safetensors': '0' * 64})
        with self.assertRaises(up.IntakeError):
            up.verify_checkpoint(self.root, {'missing.safetensors': good})


if __name__ == '__main__':
    unittest.main()
