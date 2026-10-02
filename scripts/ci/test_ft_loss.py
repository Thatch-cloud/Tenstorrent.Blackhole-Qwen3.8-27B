"""ft_loss against hand arithmetic: the decay, the hard cross-entropy, the selector term and its coverage rule, chunking."""
import math
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import dflash2_torch as d2  # noqa: E402
import ft_loss as fl  # noqa: E402


def softmax_ce(logits, target):
    top = max(logits)
    log_sum = top + math.log(sum(math.exp(value - top) for value in logits))
    return log_sum - logits[target]


class Fixture(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        self.vocab, self.hidden_size, self.block, self.count = 12, 8, 4, 3
        self.hidden = torch.randn(2, self.count, self.block, self.hidden_size, generator=generator)
        self.head = torch.randn(self.vocab, self.hidden_size, generator=generator)
        self.target = torch.randint(0, self.vocab, (2, self.count, self.block), generator=generator)
        self.predecessor = torch.cat([self.target[:, :, :1], self.target[:, :, :-1]], dim=-1)
        self.weight = torch.ones(2, self.count, self.block)
        self.weight[:, :, 0] = 0.0
        self.weight[1, 2] = 0.0                                   # one dropped block
        cfg = d2.tiny_config(hidden=self.hidden_size, vocab=self.vocab, top_k=3, selector_rank=4)
        self.selector = d2.CandidateSelector(cfg)
        with torch.no_grad():
            self.selector.predecessor_codebook.copy_(torch.randn(self.vocab, 4, generator=generator))
            self.selector.successor_codebook.copy_(torch.randn(self.vocab, 4, generator=generator))
            self.selector.hidden_projection.weight.copy_(torch.randn(4, self.hidden_size, generator=generator))


class DecayTests(unittest.TestCase):
    def test_decay_is_one_for_rows_zero_and_one_then_exponential(self):
        weights = fl.decay_weights(5, 7.0).tolist()
        self.assertEqual(weights[:2], [1.0, 1.0])
        for k in range(2, 5):
            self.assertAlmostEqual(weights[k], math.exp(-(k - 1) / 7.0), places=6)


class ObjectiveTests(Fixture):
    def test_unary_loss_equals_the_hand_computed_weighted_cross_entropy(self):
        loss, terms = fl.dflash_loss(self.hidden, self.head, None, self.target, self.predecessor, self.weight, gamma=7.0)
        logits = (self.hidden @ self.head.T).tolist()
        decay = fl.decay_weights(self.block, 7.0).tolist()
        num = den = 0.0
        for b in range(2):
            for n in range(self.count):
                for k in range(self.block):
                    weight = self.weight[b, n, k].item() * decay[k]
                    num += weight * softmax_ce(logits[b][n][k], self.target[b, n, k].item())
                    den += weight
        self.assertAlmostEqual(loss.item(), num / den, places=4)
        self.assertAlmostEqual(terms['den'].item(), den, places=4)

    def test_selector_term_is_the_cross_entropy_over_the_covered_candidates(self):
        loss, terms = fl.dflash_loss(self.hidden, self.head, self.selector, self.target, self.predecessor, self.weight, alpha=1.0)
        unary_loss, unary_terms = fl.dflash_loss(self.hidden, self.head, None, self.target, self.predecessor, self.weight)
        logits = self.hidden @ self.head.T
        decay = fl.decay_weights(self.block).tolist()
        sel_num = covered_den = 0.0
        for b in range(2):
            for n in range(self.count):
                for k in range(self.block):
                    values, ids = logits[b, n, k].topk(3)
                    target = self.target[b, n, k].item()
                    weight = self.weight[b, n, k].item() * decay[k]
                    if target not in ids.tolist() or weight == 0:
                        continue
                    scores = [values[j].item() + (self.selector.predecessor_codebook[self.predecessor[b, n, k]] *
                                                  self.selector.hidden_projection(self.hidden[b, n, k])
                                                  ).dot(self.selector.successor_codebook[ids[j]]).item() for j in range(3)]
                    sel_num += weight * softmax_ce(scores, ids.tolist().index(target))
                    covered_den += weight
        self.assertAlmostEqual(terms['selector_num'].item(), sel_num, places=3)
        self.assertAlmostEqual(terms['covered_den'].item(), covered_den, places=4)
        self.assertAlmostEqual(loss.item(), (unary_terms['ce_num'].item() + sel_num) / unary_terms['den'].item(), places=4)
        self.assertGreater(loss.item(), unary_loss.item())

    def test_alpha_zero_is_the_unary_loss_and_leaves_the_selector_without_gradient(self):
        hidden = self.hidden.clone().requires_grad_(True)
        loss, _ = fl.dflash_loss(hidden, self.head, self.selector, self.target, self.predecessor, self.weight, alpha=0.0)
        loss.backward()
        self.assertEqual(float(self.selector.successor_codebook.grad.abs().sum()), 0.0)
        unary, _ = fl.dflash_loss(self.hidden, self.head, None, self.target, self.predecessor, self.weight)
        self.assertAlmostEqual(loss.item(), unary.item(), places=6)

    def test_selector_gradients_flow_with_alpha_one(self):
        loss, _ = fl.dflash_loss(self.hidden, self.head, self.selector, self.target, self.predecessor, self.weight, alpha=1.0)
        loss.backward()
        self.assertGreater(float(self.selector.successor_codebook.grad.abs().sum()), 0.0)
        self.assertGreater(float(self.selector.hidden_projection.weight.grad.abs().sum()), 0.0)

    def test_a_target_outside_the_top_k_has_no_selector_weight(self):
        # make every target the least likely token: no candidate covers it
        logits = self.hidden @ self.head.T
        worst = logits.argmin(dim=-1)
        _, terms = fl.dflash_loss(self.hidden, self.head, self.selector, worst, worst, self.weight)
        self.assertEqual(terms['covered_den'].item(), 0.0)
        self.assertEqual(terms['selector_num'].item(), 0.0)

    def test_chunking_does_not_change_the_loss_or_the_gradient(self):
        a = self.hidden.clone().requires_grad_(True)
        b = self.hidden.clone().requires_grad_(True)
        loss_a, _ = fl.dflash_loss(a, self.head, self.selector, self.target, self.predecessor, self.weight)
        loss_b, _ = fl.dflash_loss(b, self.head, self.selector, self.target, self.predecessor, self.weight, chunk_blocks=1)
        loss_a.backward()
        loss_b.backward()
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=5)
        self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-6))

    def test_an_all_zero_weight_mask_gives_a_zero_loss_not_nan(self):
        loss, _ = fl.dflash_loss(self.hidden, self.head, self.selector, self.target, self.predecessor, torch.zeros_like(self.weight))
        self.assertEqual(loss.item(), 0.0)

    def test_the_anchor_row_and_dropped_blocks_carry_no_gradient(self):
        hidden = self.hidden.clone().requires_grad_(True)
        loss, _ = fl.dflash_loss(hidden, self.head, self.selector, self.target, self.predecessor, self.weight)
        loss.backward()
        self.assertEqual(float(hidden.grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(hidden.grad[1, 2].abs().sum()), 0.0)
        self.assertGreater(float(hidden.grad[0, 0, 1:].abs().sum()), 0.0)

    def test_accuracy_counts_weighted_rows_only(self):
        _, terms = fl.dflash_loss(self.hidden, self.head, None, self.target, self.predecessor, self.weight)
        self.assertEqual(terms['accuracy_den'].item(), float((self.weight > 0.5).sum()))
        self.assertTrue(0 <= terms['correct'].item() <= terms['accuracy_den'].item())


if __name__ == '__main__':
    unittest.main()
