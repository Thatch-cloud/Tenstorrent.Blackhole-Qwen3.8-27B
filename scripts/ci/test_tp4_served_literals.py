"""The unpinned served modules that build the drafter's K/V banks read the served width, not the pair's literals.

Found by the second review of the S2 TP4 port: publication_warm (the extent block's attach-time warm) and
dflash_proposal_trace (every request's proposal capture) carried the pair's four draft KV heads and 2,048-wide query, so
a four-card attach died at the warm and every request's drafter build died in the proposal capture's warm pass. The
attach simulation fakes those modules, so these tests hold them directly: the shapes they build, and the class the warm's
scratch cache is made from. Nothing here touches a card."""

import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import publication_warm

FOUR = {'QWEN_FAST_TP': '4'}
PAIR_KV_LITERAL = re.compile(r'\(\s*1,\s*4,\s*[A-Za-z_.0-9]+,\s*128\s*\)')


class ServedShapeTests(unittest.TestCase):
    def test_the_warms_shapes_are_the_pairs_unset_and_four_cards_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('QWEN_FAST_TP', None)
            self.assertEqual(publication_warm.kv_shape(), (1, 4, 2048, 128))
            self.assertEqual(publication_warm.query_shape(), (1, 1, 32, 2048))
            self.assertEqual((publication_warm.kv_shape(), publication_warm.query_shape()),
                             (publication_warm.KV_SHAPE, publication_warm.QUERY_SHAPE))
        with patch.dict(os.environ, FOUR):
            self.assertEqual(publication_warm.kv_shape(), (1, 2, 2048, 128))
            self.assertEqual(publication_warm.query_shape(), (1, 1, 32, 1024))

    def test_the_warms_scratch_cache_is_the_served_widths_class(self):
        import draft_kv_history
        import draft_kv_history_tp

        def build():
            return publication_warm.scratch_cache(object(), object(), [object()], object(), object(), object())

        with patch.dict(os.environ, FOUR):
            self.assertIs(type(build()), draft_kv_history_tp.DraftKVHistory)
        with patch.dict(os.environ, {}):
            os.environ.pop('QWEN_FAST_TP', None)
            self.assertIs(type(build()), draft_kv_history.DraftKVHistory)

    def test_the_proposal_capture_carries_no_pair_kv_head_literal(self):
        text = (HERE / 'dflash_proposal_trace.py').read_text(encoding='utf-8')
        code = '\n'.join(line for line in text.split('\n') if not line.strip().startswith('#'))
        self.assertEqual(PAIR_KV_LITERAL.findall(code), [])
        self.assertGreaterEqual(code.count('tp_shapes.active().draft_kv_heads'), 4)


if __name__ == '__main__':
    unittest.main()
