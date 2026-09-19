import subprocess
import tempfile
import unittest
from pathlib import Path

from frozen_recipe_context import REVISION
from mlp_k_block_gate import CONTROL, CANDIDATE, checked_source, stage_candidate


class KBlockGateTests(unittest.TestCase):
    def populate(self, directory):
        for name in CONTROL:
            payload = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'])
            (directory / name).write_bytes(payload.replace(b'\r\n', b'\n'))

    def test_exact_candidate_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.populate(directory)
            candidate = stage_candidate(directory)
            for name in CONTROL:
                checked_source(directory, name, CONTROL[name])
                checked_source(candidate, name, CANDIDATE[name])
            with self.assertRaises(FileExistsError):
                stage_candidate(directory)
            for name in CANDIDATE:
                original = (candidate / name).read_bytes()
                (candidate / name).write_bytes(original + b'\n')
                with self.assertRaises(ValueError):
                    checked_source(candidate, name, CANDIDATE[name])
                (candidate / name).write_bytes(original)

    def test_changed_control_rejected_before_candidate_creation(self):
        for name in CONTROL:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.populate(directory)
                with (directory / name).open('ab') as handle:
                    handle.write(b'\n')
                with self.assertRaises(ValueError):
                    stage_candidate(directory)
                self.assertFalse((directory / 'mlp-k-block-candidate').exists())
