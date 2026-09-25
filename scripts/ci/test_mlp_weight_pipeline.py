import subprocess
import unittest

from frozen_recipe_context import REVISION
from mlp_weight_pipeline import END, FINISH, PIPELINE, READ, RESERVE, SETUP, START, schedule, transform


class WeightPipelineTests(unittest.TestCase):
    def test_only_weight_issue_and_publication_are_changed(self):
        original = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/fused_1d_weights.cpp'], text=True)
        candidate = transform(original)
        restored = candidate.replace(SETUP, START + RESERVE).replace(READ,
            '                    noc_async_read_tile(page, weights, tile_address);\n').replace(PIPELINE, FINISH)
        self.assertEqual(restored, original)
        self.assertEqual(candidate[candidate.index(END):], original[original.index(END):])
        self.assertIn('static_assert(pairs_per_worker == 3)', candidate)
        self.assertIn('cb_reserve_back(1, 2 * block_tiles)', candidate)
        self.assertIn('noc_async_read_barrier();\n    noc_async_read_set_trid(0);', candidate)
        with self.assertRaises(ValueError):
            transform(candidate)

    def test_no_publish_before_completion_or_slot_reuse_before_consume(self):
        issued, completed, published, consumed = set(), set(), set(), set()
        slots = {}
        for operation, block, slot, transaction in schedule():
            self.assertEqual(transaction, slot + 1)
            if operation == 'issue':
                self.assertNotIn(block, issued)
                self.assertNotIn(slot, slots)
                slots[slot] = block
                issued.add(block)
            elif operation == 'barrier':
                self.assertIn(block, issued)
                completed.add(block)
            elif operation == 'publish':
                self.assertIn(block, completed)
                self.assertEqual(block, len(published))
                published.add(block)
            else:
                self.assertIn(block, published)
                self.assertEqual(slots.pop(slot), block)
                consumed.add(block)
            self.assertLessEqual(len(slots), 2)
        self.assertEqual(issued, set(range(20)))
        self.assertEqual(consumed, issued)
        self.assertFalse(slots)

    def test_read_page_and_padding_geometry_unchanged(self):
        from mlp_weight_read_order import accesses
        for worker in range(91):
            for block in range(20):
                values = accesses(worker, block, False)
                self.assertEqual([item['destination_tile'] for item in values], list(range(48)))
                self.assertEqual(sum(item['page'] is None for item in values), 16 if worker == 90 else 0)
