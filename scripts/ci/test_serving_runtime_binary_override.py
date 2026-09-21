import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import runtime_binary_override as override


def sha256_of_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


class RequestedTests(unittest.TestCase):
    def test_unset_or_empty_is_none(self):
        self.assertIsNone(override.requested({}))
        self.assertIsNone(override.requested({override.ENV: ''}))

    def test_bad_hex_raises(self):
        with self.assertRaises(ValueError):
            override.requested({override.ENV: 'not-hex'})
        with self.assertRaises(ValueError):
            override.requested({override.ENV: 'abcd'})

    def test_valid_hex_is_lowercased(self):
        value = 'A' * 64
        self.assertEqual(override.requested({override.ENV: value}), 'a' * 64)


class InstallTests(unittest.TestCase):
    def make_sim(self, directory, binary_sha, factory_sha, binary_bytes=b'binary', factory_bytes=b'factory'):
        binaries = ('a/_x.so', 'b/_x.so')
        for name in binaries:
            path = Path(directory) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(binary_bytes)
        factory_path = Path(directory) / 'f.cpp'
        factory_path.write_bytes(factory_bytes)
        return SimpleNamespace(
            BINARY_SHA256=binary_sha,
            BINARIES=binaries,
            FACTORY='f.cpp',
            COMBINED_FACTORY=factory_sha,
            digest=sha256_of_file,
        )

    def test_unset_env_returns_none_and_rebinds_nothing(self):
        with TemporaryDirectory() as directory:
            pinned = 'p' * 64
            sim = self.make_sim(directory, pinned, 'f' * 64)
            scope = SimpleNamespace(BINARY_SHA256=pinned)
            log_calls = []
            result = override.install(directory, log=lambda *a: log_calls.append(a),
                                       environ={}, sim=sim, scopes=[scope])
            self.assertIsNone(result)
            self.assertEqual(sim.BINARY_SHA256, pinned)
            self.assertEqual(scope.BINARY_SHA256, pinned)
            self.assertEqual(log_calls, [])

    def test_bad_hex_raises_and_rebinds_nothing(self):
        with TemporaryDirectory() as directory:
            pinned = 'p' * 64
            sim = self.make_sim(directory, pinned, 'f' * 64)
            scope = SimpleNamespace(BINARY_SHA256=pinned)
            with self.assertRaises(ValueError):
                override.install(directory, log=lambda *a: None,
                                  environ={override.ENV: 'zz'}, sim=sim, scopes=[scope])
            self.assertEqual(sim.BINARY_SHA256, pinned)
            self.assertEqual(scope.BINARY_SHA256, pinned)

    def test_value_equal_to_pin_is_noop(self):
        with TemporaryDirectory() as directory:
            sim = self.make_sim(directory, None, 'f' * 64)
            actual_binary_hash = sha256_of_file(Path(directory) / sim.BINARIES[0])
            sim.BINARY_SHA256 = actual_binary_hash
            scope = SimpleNamespace(BINARY_SHA256=actual_binary_hash)
            log_calls = []
            result = override.install(directory, log=lambda *a: log_calls.append(a),
                                       environ={override.ENV: actual_binary_hash},
                                       sim=sim, scopes=[scope])
            self.assertIsNone(result)
            self.assertEqual(sim.BINARY_SHA256, actual_binary_hash)
            self.assertEqual(scope.BINARY_SHA256, actual_binary_hash)
            self.assertEqual(len(log_calls), 1)
            self.assertIn('names the pinned binary', log_calls[0][0])

    def test_one_path_hashing_differently_raises_and_rebinds_nothing(self):
        with TemporaryDirectory() as directory:
            pinned = 'p' * 64
            sim = self.make_sim(directory, pinned, 'f' * 64)
            scope = SimpleNamespace(BINARY_SHA256=pinned)
            # Overwrite only one of the two pinned paths so it hashes differently
            # from the requested value: a partial or stale mount.
            override_value = sha256_of_file(Path(directory) / sim.BINARIES[0])
            (Path(directory) / sim.BINARIES[1]).write_bytes(b'different-contents')
            with self.assertRaises(ValueError) as ctx:
                override.install(directory, log=lambda *a: None,
                                  environ={override.ENV: override_value}, sim=sim, scopes=[scope])
            self.assertIn(override.ENV, str(ctx.exception))
            self.assertEqual(sim.BINARY_SHA256, pinned)
            self.assertEqual(scope.BINARY_SHA256, pinned)

    def test_wrong_factory_raises_and_rebinds_nothing(self):
        with TemporaryDirectory() as directory:
            pinned = 'p' * 64
            sim = self.make_sim(directory, pinned, 'combined' + 'f' * 56)
            scope = SimpleNamespace(BINARY_SHA256=pinned)
            for name in sim.BINARIES:
                (Path(directory) / name).write_bytes(b'grafted-binary')
            override_value = sha256_of_file(Path(directory) / sim.BINARIES[0])
            # Factory on disk is still the plain 'factory' bytes from make_sim, whose
            # hash does not match COMBINED_FACTORY.
            with self.assertRaises(ValueError):
                override.install(directory, log=lambda *a: None,
                                  environ={override.ENV: override_value}, sim=sim, scopes=[scope])
            self.assertEqual(sim.BINARY_SHA256, pinned)
            self.assertEqual(scope.BINARY_SHA256, pinned)

    def test_success_rebinds_and_returns_record(self):
        with TemporaryDirectory() as directory:
            pinned = 'p' * 64
            sim = self.make_sim(directory, pinned, 'f' * 64)
            scope = SimpleNamespace(BINARY_SHA256=pinned)
            for name in sim.BINARIES:
                (Path(directory) / name).write_bytes(b'grafted-binary')
            (Path(directory) / sim.FACTORY).write_bytes(b'combined-factory')
            override_value = sha256_of_file(Path(directory) / sim.BINARIES[0])
            combined_factory_hash = sha256_of_file(Path(directory) / sim.FACTORY)
            sim.COMBINED_FACTORY = combined_factory_hash
            log_calls = []
            result = override.install(directory, log=lambda *a: log_calls.append(a),
                                       environ={override.ENV: override_value}, sim=sim, scopes=[scope])
            self.assertEqual(result['pinned'], pinned)
            self.assertEqual(result['override'], override_value)
            self.assertEqual(result['factory'], combined_factory_hash)
            self.assertEqual(result['binaries'], dict.fromkeys(sim.BINARIES, override_value))
            self.assertEqual(sim.BINARY_SHA256, override_value)
            self.assertEqual(scope.BINARY_SHA256, override_value)
            self.assertEqual(len(log_calls), 1)
            self.assertIn('overridden', log_calls[0][0])


class WiringTests(unittest.TestCase):
    def test_serving_runtime_calls_the_override_hook(self):
        source = Path(__file__).parent.joinpath('serving_runtime.py').read_text(encoding='utf-8')
        self.assertIn('override_runtime_binary(runtime_root, log=pindiag)', source)


if __name__ == '__main__':
    unittest.main()
