from pathlib import Path
import unittest

from sdpa_graft_build import COMPONENTS, implementation_sources


class GraftBuildTests(unittest.TestCase):
    def test_only_host_implementations_are_registered_once(self):
        sources = implementation_sources()
        self.assertEqual(len(sources), 15)
        self.assertEqual(len(set(sources)), 15)
        self.assertTrue(all('nanobind' not in name and '/kernels/' not in name for name in sources))
        patch = (Path(__file__).resolve().parents[2] / 'optimisation/sim/sdpa-graft-registration.patch').read_text()
        added = [line.strip()[1:].strip() for line in patch.splitlines() if line.startswith('+    ')]
        self.assertEqual(added, list(sources))
        self.assertEqual(len(COMPONENTS), 5)
