import math
from pathlib import Path
import unittest
from unittest.mock import patch

from draft_selector_fixture import TENSORS, fetch
from draft_remaining_layers_fixture import specifications
from draft_projection_full_fixture import TENSORS as PROJECTION


class SelectorFixtureTests(unittest.TestCase):
    def test_selector_selection_is_bounded(self):
        self.assertEqual(len(TENSORS), 4)
        self.assertEqual(sum(2 * math.prod(shape) for shape, filename in TENSORS.values()), 256911360)
        with patch('draft_selector_fixture.fetch_subset', return_value={}) as download:
            fetch(Path('selector'))
            self.assertEqual(download.call_args.kwargs['specifications'], TENSORS)

    def test_all_fixture_families_cover_checkpoint_payload_bytes(self):
        layer_bytes = sum(2 * math.prod(shape) for shape, filename in specifications(1).values())
        projection_bytes = sum(2 * math.prod(shape) for shape, filename in PROJECTION.values())
        selector_bytes = sum(2 * math.prod(shape) for shape, filename in TENSORS.values())
        self.assertEqual(5 * layer_bytes + projection_bytes + selector_bytes + 8936, 3848817896)
