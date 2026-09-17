import unittest
from unittest.mock import patch

from frozen_draft_tail import append_queries
from dspark_native_cached_layer import append_queries as reference


class Tensor:
    dtype = 'bf16'
    layout = 'tile'

    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (1, 4, len(self.rows), 128)

    def memory_config(self):
        return 'dram'


class Operations:
    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def __init__(self):
        self.pad_rows, self.concat_shapes = [], []

    def slice(self, value, start, end):
        return Tensor(value.rows[start[2]:end[2]])

    def pad(self, value, padding, fill):
        self.pad_rows.append(len(value.rows))
        return Tensor(value.rows + [fill] * padding[2][1])

    def concat(self, values, *, dim, memory_config):
        assert dim == 2 and memory_config == 'dram'
        self.concat_shapes.append([value.shape[2] for value in values])
        return Tensor([row for value in values for row in value.rows])


class DraftTailTests(unittest.TestCase):
    def test_complete_output_matches_original_and_only_small_tail_is_padded(self):
        with patch('dspark_full_attention.MAX_CONTEXT', 262400):
            for position in (32, 64, 4096, 4352, 33024, 65792, 131328, 262400):
                for proposals in (7, 15):
                    original, candidate = Operations(), Operations()
                    history = Tensor(range(position))
                    queries = Tensor(list(range(100, 100 + proposals)) + [999] * (32 - proposals))
                    expected = reference(original, history, queries, lambda value: value,
                        position=position, proposals=proposals)
                    actual = append_queries(candidate, history, queries, lambda value: value,
                        position=position, proposals=proposals)
                    self.assertEqual(actual.rows, expected.rows)
                    self.assertEqual(candidate.pad_rows, [proposals])
                    self.assertTrue(all(rows % 32 == 0 for rows in candidate.concat_shapes[0]))
                    self.assertEqual(history.rows, list(range(position)))

    def test_unaligned_or_unsupported_inputs_rejected(self):
        for position, proposals in ((31, 15), (64, 31), (0, 15), (True, 15)):
            with self.assertRaises(ValueError):
                append_queries(Operations(), Tensor(range(position)), Tensor(range(32)),
                    lambda value: value, position=position, proposals=proposals)

    def test_wrong_layout_rejected(self):
        history = Tensor(range(64))
        history.layout = 'row_major'
        with self.assertRaises(ValueError):
            append_queries(Operations(), history, Tensor(range(32)), lambda value: value,
                position=64, proposals=15)
