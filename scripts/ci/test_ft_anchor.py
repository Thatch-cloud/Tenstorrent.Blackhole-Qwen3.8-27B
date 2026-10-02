"""ft_anchor: anchor sampling, block rows, labels and the dense mask on hand-built cases; and, when SPECFORGE_SRC names the recipe's
source file, bit-exact parity with its own functions (skipped otherwise: the recipe's source is not in this repository)."""
import ast
import os
import sys
import unittest
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import ft_anchor as fa  # noqa: E402


class AnchorTests(unittest.TestCase):
    def test_only_valid_positions_are_drawn_sorted_and_counted(self):
        mask = torch.tensor([[0., 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]])
        valid = [i for i in range(11) if mask[0, i] > 0.5 and mask[0, i + 1] > 0.5]
        anchors, keep = fa.sample_anchors(mask, 4, generator=torch.Generator().manual_seed(1))
        got = anchors[0][keep[0]].tolist()
        self.assertEqual(len(got), 4)
        self.assertEqual(got, sorted(got))
        self.assertTrue(set(got) <= set(valid))
        anchors, keep = fa.sample_anchors(mask, 100, generator=torch.Generator().manual_seed(1))
        self.assertEqual(anchors[0][keep[0]].tolist(), valid)            # fewer valid than asked: all of them

    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        mask = torch.ones(2, 50)
        a = fa.sample_anchors(mask, 8, generator=torch.Generator().manual_seed(3))
        b = fa.sample_anchors(mask, 8, generator=torch.Generator().manual_seed(3))
        c = fa.sample_anchors(mask, 8, generator=torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertFalse(torch.equal(a[0], c[0]))

    def test_an_explicit_draw_picks_the_smallest_values(self):
        mask = torch.ones(1, 6)                                   # valid positions 0..4
        draw = torch.tensor([[0.9, 0.1, 0.5, 0.05, 0.7]])
        anchors, keep = fa.sample_anchors(mask, 2, random_values=draw)
        self.assertEqual(anchors.tolist(), [[1, 3]])
        self.assertTrue(keep.all())

    def test_short_sequences_pad_with_dropped_slots(self):
        mask = torch.tensor([[1., 1, 1, 1, 1, 1], [1., 1, 0, 0, 0, 0]])
        anchors, keep = fa.sample_anchors(mask, 4, generator=torch.Generator().manual_seed(0))
        self.assertEqual(keep.sum(dim=1).tolist(), [4, 1])
        self.assertEqual(anchors[1][keep[1]].tolist(), [0])
        self.assertEqual(anchors[1][~keep[1]].tolist(), [0, 0, 0])

    def test_no_supervised_pair_is_refused(self):
        with self.assertRaises(ValueError):
            fa.sample_anchors(torch.tensor([[1., 0, 1, 0]]), 4)

    def test_the_draw_is_roughly_uniform(self):
        mask = torch.ones(1, 21)
        counts = torch.zeros(20)
        generator = torch.Generator().manual_seed(5)
        for _ in range(400):
            anchors, keep = fa.sample_anchors(mask, 2, generator=generator)
            for value in anchors[0][keep[0]].tolist():
                counts[value] += 1
        self.assertTrue(float(counts.min()) > 20 and float(counts.max()) < 70)       # 40 expected each


class BlockTests(unittest.TestCase):
    def test_positions_and_noise_rows(self):
        anchors = torch.tensor([[2, 5]])
        keep = torch.tensor([[True, False]])
        self.assertEqual(fa.position_ids(anchors, 3).tolist(), [[2, 3, 4, 5, 6, 7]])
        ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        noise = fa.noise_ids(ids, anchors, keep, 99, 3)
        self.assertEqual(noise.tolist(), [[12, 99, 99, 99, 99, 99]])          # a dropped slot is all mask tokens

    def test_labels_predecessors_and_weights(self):
        ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        loss_mask = torch.tensor([[1., 1, 1, 0, 1, 1]])
        anchors, keep = torch.tensor([[1, 4]]), torch.tensor([[True, True]])
        target, predecessor, weight = fa.block_labels(ids, loss_mask, anchors, keep, 4)
        self.assertEqual(target[0, 0].tolist(), [11, 12, 13, 14])
        self.assertEqual(predecessor[0, 0].tolist(), [11, 11, 12, 13])
        # block 0: position 0 excluded, label 13 is unsupervised (loss_mask 0), 12 and 14 are
        self.assertEqual(weight[0, 0].tolist(), [0., 1., 0., 1.])
        # block 1 runs off the end after two rows
        self.assertEqual(weight[0, 1].tolist(), [0., 1., 0., 0.])

    def test_a_dropped_anchor_has_no_weight(self):
        ids = torch.arange(8)[None]
        _, _, weight = fa.block_labels(ids, torch.ones(1, 8), torch.tensor([[1, 0]]), torch.tensor([[True, False]]), 3)
        self.assertEqual(weight[0, 1].sum().item(), 0.0)


class MaskTests(unittest.TestCase):
    def mask(self, window=None, rule='specforge', anchors=((2, 5),), keep=((True, True),), length=7, block=3):
        return fa.dense_mask(torch.tensor(anchors), torch.tensor(keep), length, block, window, rule)

    def test_context_is_strictly_before_the_anchor_and_the_block_is_its_own(self):
        mask = self.mask()
        self.assertEqual(tuple(mask.shape), (1, 1, 6, 7 + 6))
        row0 = mask[0, 0, 0].tolist()                       # block 0, offset 0, anchor 2
        self.assertEqual([i for i, v in enumerate(row0) if v], [0, 1, 7, 8, 9])
        row3 = mask[0, 0, 3].tolist()                       # block 1, anchor 5
        self.assertEqual([i for i, v in enumerate(row3) if v], [0, 1, 2, 3, 4, 10, 11, 12])

    def test_the_anchor_row_itself_is_never_context(self):
        mask = self.mask()
        self.assertFalse(bool(mask[0, 0, 0, 2]))             # the anchor's own context row: its block copy replaces it

    def test_sliding_window_lower_bound_follows_the_row_offset(self):
        mask = self.mask(window=3, anchors=((6,),), keep=((True,),), length=8)
        for offset in range(3):
            row = mask[0, 0, offset, :8].tolist()
            lower = 6 + offset - 2
            self.assertEqual([i for i, v in enumerate(row) if v], [i for i in range(lower, 6)])

    def test_the_block_rule_changes_only_the_block_triangle_and_only_with_a_window(self):
        causal = self.mask(window=4, rule='specforge')
        full = self.mask(window=4, rule='full')
        self.assertFalse(torch.equal(causal, full))
        self.assertTrue(torch.equal(causal[..., :7], full[..., :7]))              # the context part is the same
        upper = [(q, 7 + k) for q in range(3) for k in range(3) if k > q]
        for q, kv in upper:
            self.assertFalse(bool(causal[0, 0, q, kv]))
            self.assertTrue(bool(full[0, 0, q, kv]))
        # no window: the recipe's block is bidirectional, equal to the 'full' rule
        self.assertTrue(torch.equal(self.mask(window=None, rule='specforge'), self.mask(window=None, rule='full')))

    def test_a_dropped_block_sees_nothing(self):
        mask = self.mask(keep=((True, False),))
        self.assertTrue(mask[0, 0, 3:].sum().item() == 0)
        self.assertTrue(mask[0, 0, :3].sum().item() > 0)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            self.mask(rule='sideways')
        with self.assertRaises(ValueError):
            self.mask(window=0)


# -- parity with the recipe's own source (local only) ------------------------------------------------------------------------------

def recipe_functions():
    path = os.environ.get('SPECFORGE_SRC')
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    namespace = dict(torch=torch, Optional=__import__('typing').Optional, Tuple=__import__('typing').Tuple)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'create_dflash_sdpa_mask':
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
        if isinstance(node, ast.ClassDef) and node.name == 'OnlineDFlashModel':
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '_sample_anchor_positions':
                    exec(compile(ast.Module(body=[item], type_ignores=[]), path, 'exec'), namespace)
    return namespace


@unittest.skipUnless(recipe_functions(), 'SPECFORGE_SRC does not name the recipe source')
class RecipeParityTests(unittest.TestCase):
    def test_mask_is_bit_exact(self):
        namespace = recipe_functions()
        generator = torch.Generator().manual_seed(2)
        for window in (None, 5, 16):
            for batch, length, count, block in ((2, 40, 6, 4), (1, 64, 9, 8)):
                anchors = torch.sort(torch.randint(0, length - 1, (batch, count), generator=generator), dim=1).values
                keep = torch.rand(batch, count, generator=generator) > 0.2
                want = namespace['create_dflash_sdpa_mask'](anchors, keep, length, block, torch.device('cpu'), window)
                got = fa.dense_mask(anchors, keep, length, block, window, 'specforge')
                self.assertTrue(torch.equal(want, got), (window, batch, length))

    def test_anchor_sampling_is_exact_for_the_same_draw(self):
        namespace = recipe_functions()
        method = namespace['_sample_anchor_positions']
        model = SimpleNamespace(num_anchors=7)
        mask = (torch.rand(3, 50, generator=torch.Generator().manual_seed(8)) > 0.2).float()
        torch.manual_seed(11)
        want_anchors, want_keep = method(model, 50, mask, torch.device('cpu'))
        torch.manual_seed(11)
        draw = torch.rand(3, 49)
        got_anchors, got_keep = fa.sample_anchors(mask, 7, random_values=draw)
        self.assertTrue(torch.equal(want_anchors, got_anchors))
        self.assertTrue(torch.equal(want_keep, got_keep))


if __name__ == '__main__':
    unittest.main()
