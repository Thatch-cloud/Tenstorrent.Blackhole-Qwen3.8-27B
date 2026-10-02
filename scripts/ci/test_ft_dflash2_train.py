"""The packed-anchor training forward at tiny scale: one anchor equals the inference draft forward, blocks are isolated, dropped blocks
are finite, frozen parameters get no gradient, the block rule is the one visible difference between training and serving."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import dflash2_torch as d2  # noqa: E402
import ft_anchor as fa  # noqa: E402
import ft_dflash2_train as tr  # noqa: E402

VOCAB = 64


def setup(window=32, block=4, seed=0, **kwargs):
    cfg = d2.tiny_config(window=window, block=block, vocab=VOCAB, mask_token=VOCAB - 1, top_k=4, **kwargs)
    model = d2.init_random(d2.Dflash2(cfg), seed).float()
    generator = torch.Generator().manual_seed(seed + 1)
    embed = torch.randn(VOCAB, cfg.hidden, generator=generator)
    head = torch.randn(VOCAB, cfg.hidden, generator=generator)
    return cfg, model, embed, head


def sample(cfg, length=40, seed=3, supervised_from=8):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB - 1, (1, length), generator=generator)
    mask = torch.zeros(1, length)
    mask[:, supervised_from:] = 1.0
    raw = torch.randn(1, length, len(cfg.tap_ids) * cfg.hidden, generator=generator)
    return ids, mask, raw


class OneAnchorTests(unittest.TestCase):
    def one_anchor(self, rule, window=64):
        cfg, model, embed, head = setup(window=window, block=4)
        ids, mask, raw = sample(cfg)
        batch = tr.make_batch(ids, mask, cfg, 1, random_values=torch.full((1, 39), 0.5) + torch.arange(39) * 1e-3)
        anchor = int(batch.anchors[0, 0])
        with torch.no_grad():
            trained = tr.draft_hidden(model, embed, raw, batch, block_rule=rule)[0, 0]
            lo = max(0, anchor - window)
            noise = embed[batch.noise_ids[:1]]
            inference = model(noise, batch.positions[:1], raw[:, lo:anchor], (lo + torch.arange(anchor - lo))[None])[0]
        return trained, inference, anchor

    def test_a_training_anchor_equals_the_inference_forward_under_the_full_block_rule(self):
        trained, inference, anchor = self.one_anchor('full')
        self.assertGreater(anchor, 7)
        self.assertTrue(torch.allclose(trained, inference, atol=1e-5))

    def test_the_recipes_causal_block_differs_from_the_inference_model(self):
        """The finding: for a sliding-window draft the recipe's training mask is causal inside a block, the serving model is not."""
        trained, inference, _ = self.one_anchor('specforge')
        self.assertFalse(torch.allclose(trained, inference, atol=1e-4))
        # only the last block rows can differ the other way round: row 0 reads nothing of the block but itself under the causal rule
        self.assertFalse(torch.allclose(trained[0], inference[0], atol=1e-4))

    def test_without_a_window_the_two_rules_agree(self):
        cfg, model, embed, head = setup(window=None)
        ids, mask, raw = sample(cfg)
        batch = tr.make_batch(ids, mask, cfg, 3, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            a = tr.draft_hidden(model, embed, raw, batch, 'specforge', window=None)
            b = tr.draft_hidden(model, embed, raw, batch, 'full', window=None)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


class IsolationTests(unittest.TestCase):
    def test_blocks_do_not_see_each_other_or_leak_through_the_convolution(self):
        cfg, model, embed, head = setup()
        ids, mask, raw = sample(cfg)
        batch = tr.make_batch(ids, mask, cfg, 3, generator=torch.Generator().manual_seed(4))
        with torch.no_grad():
            base = tr.draft_hidden(model, embed, raw, batch, 'full')
            changed = batch._replace(noise_ids=batch.noise_ids.clone())
            changed.noise_ids[0, 0] = (changed.noise_ids[0, 0] + 1) % (VOCAB - 1)       # the first block's anchor token
            moved = tr.draft_hidden(model, embed, raw, changed, 'full')
        self.assertFalse(torch.allclose(base[0, 0], moved[0, 0], atol=1e-5))
        self.assertTrue(torch.allclose(base[0, 1:], moved[0, 1:], atol=1e-6))

    def test_dropped_blocks_are_finite_and_carry_no_gradient(self):
        cfg, model, embed, head = setup()
        ids, mask, raw = sample(cfg, length=14, supervised_from=11)          # only 2 valid anchors, 6 requested
        generator = torch.Generator().manual_seed(1)
        loss, terms, batch = tr.train_forward(model, embed, head, raw, ids, mask, 6, generator)
        self.assertEqual(int(batch.keep.sum()), 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_only_the_labels_at_or_after_the_supervised_start_count(self):
        cfg, model, embed, head = setup()
        ids, mask, raw = sample(cfg, supervised_from=30)
        batch = tr.make_batch(ids, mask, cfg, 5, generator=torch.Generator().manual_seed(5))
        for n in range(batch.anchors.shape[1]):
            if batch.keep[0, n]:
                anchor = int(batch.anchors[0, n])
                for k in range(1, cfg.block):
                    expected = 1.0 if anchor + k < 40 and anchor + k >= 30 else 0.0
                    self.assertEqual(float(batch.weight_mask[0, n, k]), expected)


class FrozenTests(unittest.TestCase):
    def test_frozen_parameters_get_no_gradient_and_the_rest_do(self):
        cfg, model, embed, head = setup()
        frozen = tr.freeze(model)
        self.assertEqual(sorted(frozen), ['fc.weight', 'hidden_norm.weight'])
        ids, mask, raw = sample(cfg)
        loss, _, _ = tr.train_forward(model, embed, head, raw, ids, mask, 4, torch.Generator().manual_seed(0))
        loss.backward()
        self.assertIsNone(model.fc.weight.grad)
        self.assertIsNone(model.hidden_norm.weight.grad)
        for name, parameter in model.named_parameters():
            if name not in frozen:
                self.assertIsNotNone(parameter.grad, name)
                if 'selector' not in name:       # random labels are rarely inside a random top-4: the selector term can be empty
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_unfrozen_fc_trains(self):
        cfg, model, embed, head = setup()
        ids, mask, raw = sample(cfg)
        loss, _, _ = tr.train_forward(model, embed, head, raw, ids, mask, 4, torch.Generator().manual_seed(0))
        loss.backward()
        self.assertGreater(float(model.fc.weight.grad.abs().sum()), 0.0)


class LossTests(unittest.TestCase):
    def test_a_fixed_batch_gives_the_same_loss_twice_and_the_batch_argument_is_honoured(self):
        cfg, model, embed, head = setup()
        ids, mask, raw = sample(cfg)
        a, _, batch = tr.train_forward(model, embed, head, raw, ids, mask, 4, torch.Generator().manual_seed(0))
        b, _, _ = tr.train_forward(model, embed, head, raw, ids, mask, 4, batch=batch)
        self.assertEqual(a.item(), b.item())

    def test_chunked_loss_equals_unchunked(self):
        cfg, model, embed, head = setup()
        ids, mask, raw = sample(cfg)
        a, _, batch = tr.train_forward(model, embed, head, raw, ids, mask, 6, torch.Generator().manual_seed(0))
        b, _, _ = tr.train_forward(model, embed, head, raw, ids, mask, 6, batch=batch, chunk_blocks=2)
        self.assertAlmostEqual(a.item(), b.item(), places=5)


if __name__ == '__main__':
    unittest.main()
