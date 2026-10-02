"""dflash2_torch at tiny scale: names, cached context equals recomputed, the sliding window, block isolation of the convolution,
position equivariance, the selector's greedy path."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import dflash2_torch as d2  # noqa: E402


def make(seed=0, **kwargs):
    cfg = d2.tiny_config(**kwargs)
    model = d2.init_random(d2.Dflash2(cfg), seed).float().eval()
    return cfg, model


def rows(cfg, count, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, count, len(cfg.tap_ids) * cfg.hidden, generator=generator)


def noise(cfg, count, seed):
    generator = torch.Generator().manual_seed(seed + 100)
    return torch.randn(1, count, cfg.hidden, generator=generator)


def positions(start, count):
    return (start + torch.arange(count))[None]


class ShapeAndNameTests(unittest.TestCase):
    def test_checkpoint_names(self):
        names = d2.parameter_names(2)
        for expected in ('fc.weight', 'hidden_norm.weight', 'norm.weight', 'candidate_selector.predecessor_codebook',
                         'candidate_selector.successor_codebook', 'candidate_selector.hidden_projection.weight',
                         'layers.1.self_attn.q_proj.weight', 'layers.1.self_attn.q_norm.weight',
                         'layers.0.attention_conv.base_kernel', 'layers.0.attention_conv.kernel_projection.weight',
                         'layers.0.mlp_conv.base_kernel', 'layers.1.mlp.down_proj.weight',
                         'layers.0.input_layernorm.weight', 'layers.0.post_attention_layernorm.weight'):
            self.assertIn(expected, names)
        self.assertEqual(len(names), 3 + 3 + 2 * (4 + 2 + 3 + 2 + 4))

    def test_real_config_shapes(self):
        cfg = d2.real_config()
        self.assertEqual((cfg.hidden, cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.window, cfg.vocab),
                         (5120, 5, 32, 8, 128, 2048, 248320))
        self.assertEqual(cfg.tap_ids, (5, 19, 33, 47, 61))

    def test_forward_shape(self):
        cfg, model = make()
        out = model(noise(cfg, cfg.block, 1), positions(40, cfg.block), rows(cfg, 40, 2), positions(0, 40))
        self.assertEqual(tuple(out.shape), (1, cfg.block, cfg.hidden))
        self.assertTrue(torch.isfinite(out).all())


class CacheAndWindowTests(unittest.TestCase):
    def test_cached_context_equals_recomputed(self):
        cfg, model = make()
        raw, ctx_pos = rows(cfg, 50, 3), positions(0, 50)
        nz, pos = noise(cfg, cfg.block, 4), positions(50, cfg.block)
        with torch.no_grad():
            direct = model(nz, pos, raw, ctx_pos)
            kv = model.context_kv(raw, ctx_pos)
            cached = model(nz, pos, ctx_positions=ctx_pos, ctx_kv=kv)
        self.assertTrue(torch.equal(direct, cached))

    def test_chunked_context_ingest_equals_one_shot(self):
        cfg, model = make()
        raw = rows(cfg, 60, 5)
        with torch.no_grad():
            whole = model.context_kv(raw, positions(0, 60))
            parts = [model.context_kv(raw[:, a:b], positions(a, b - a)) for a, b in ((0, 17), (17, 40), (40, 60))]
        for layer in range(cfg.layers):
            for which in (0, 1):
                joined = torch.cat([part[layer][which] for part in parts], dim=2)
                self.assertTrue(torch.allclose(joined, whole[layer][which], atol=1e-6))

    def test_rows_older_than_the_window_do_not_matter_and_newer_ones_do(self):
        cfg, model = make(window=16)
        raw, ctx_pos = rows(cfg, 60, 6), positions(0, 60)
        nz, pos = noise(cfg, cfg.block, 7), positions(60, cfg.block)
        with torch.no_grad():
            base = model(nz, pos, raw, ctx_pos)
            old = raw.clone()
            old[:, :30] += 5.0                       # positions 0..29: further than the window from every block row
            self.assertTrue(torch.allclose(base, model(nz, pos, old, ctx_pos), atol=1e-6))
            new = raw.clone()
            new[:, 55] += 5.0
            self.assertFalse(torch.allclose(base, model(nz, pos, new, ctx_pos), atol=1e-4))

    def test_window_slice_equals_the_explicit_mask(self):
        cfg, model = make(window=16)
        raw, ctx_pos = rows(cfg, 60, 8), positions(0, 60)
        nz, pos = noise(cfg, cfg.block, 9), positions(60, cfg.block)
        with torch.no_grad():
            full = model(nz, pos, raw, ctx_pos)
            keep = 60 - 16 - cfg.block
            sliced = model(nz, pos, raw[:, keep:], ctx_pos[:, keep:])
        self.assertTrue(torch.allclose(full, sliced, atol=1e-6))

    def test_position_equivariance(self):
        cfg, model = make()
        raw = rows(cfg, 30, 10)
        nz = noise(cfg, cfg.block, 11)
        with torch.no_grad():
            a = model(nz, positions(30, cfg.block), raw, positions(0, 30))
            b = model(nz, positions(1030, cfg.block), raw, positions(1000, 30))
        self.assertTrue(torch.allclose(a, b, atol=2e-4))

    def test_block_rows_see_each_other_both_ways(self):
        cfg, model = make()
        raw, ctx_pos = rows(cfg, 20, 12), positions(0, 20)
        nz, pos = noise(cfg, cfg.block, 13), positions(20, cfg.block)
        with torch.no_grad():
            base = model(nz, pos, raw, ctx_pos)
            changed = nz.clone()
            changed[:, -1] += 3.0
            moved = model(changed, pos, raw, ctx_pos)
        self.assertFalse(torch.allclose(base[:, 0], moved[:, 0], atol=1e-5))


class ConvolutionTests(unittest.TestCase):
    def test_identity_at_initialisation(self):
        conv = d2.GroupedDynamicCausalConv(32, 2, 16)
        torch.nn.init.zeros_(conv.kernel_projection.weight)
        value = torch.randn(2, 8, 32)
        pre, post = conv.prepare(value)
        self.assertTrue(torch.allclose(pre, value))
        self.assertTrue(torch.allclose(conv.finish(pre, post), value))

    def test_causal_and_block_isolated(self):
        generator = torch.Generator().manual_seed(1)
        conv = d2.GroupedDynamicCausalConv(32, 2, 16)
        with torch.no_grad():
            conv.base_kernel.copy_(torch.randn(conv.base_kernel.shape, generator=generator))
            conv.kernel_projection.weight.copy_(0.3 * torch.randn(conv.kernel_projection.weight.shape, generator=generator))
        value = torch.randn(1, 16, 32, generator=generator)
        changed = value.clone()
        changed[:, 7] += 2.0                                  # last row of block 0 when blocks are 8 rows
        with torch.no_grad():
            plain = conv.prepare(value)[0]
            plain_changed = conv.prepare(changed)[0]
            blocked = conv.prepare(value, block=8)[0]
            blocked_changed = conv.prepare(changed, block=8)[0]
        self.assertFalse(torch.allclose(plain[:, 8], plain_changed[:, 8]))          # leaks across the boundary without blocks
        self.assertTrue(torch.allclose(blocked[:, 8:], blocked_changed[:, 8:]))      # does not with them
        self.assertTrue(torch.allclose(plain[:, :7], plain_changed[:, :7]))          # causal: earlier rows never move

    def test_a_ragged_block_is_refused(self):
        conv = d2.GroupedDynamicCausalConv(32, 2, 16)
        with self.assertRaises(ValueError):
            conv.prepare(torch.randn(1, 10, 32), block=8)


class SelectorTests(unittest.TestCase):
    def test_greedy_path_matches_a_manual_walk(self):
        cfg, model = make(top_k=4)
        generator = torch.Generator().manual_seed(2)
        hidden = torch.randn(1, 5, cfg.hidden, generator=generator)
        head = torch.randn(cfg.vocab, cfg.hidden, generator=generator)
        anchor = torch.tensor([7])
        with torch.no_grad():
            path = model.propose(hidden, anchor, head)
            logits = hidden @ head.T
            unary, cand = torch.topk(logits, 4, dim=-1, sorted=False)
            sel = model.candidate_selector
            prev, expected = 7, []
            for at in range(5):
                scores = unary[0, at] + torch.stack([
                    (sel.predecessor_codebook[prev] * sel.hidden_projection(hidden[0, at])) @ sel.successor_codebook[c]
                    for c in cand[0, at]])
                prev = int(cand[0, at][scores.argmax()])
                expected.append(prev)
        self.assertEqual(path[0].tolist(), expected)
        self.assertTrue(all(token in cand[0, at].tolist() for at, token in enumerate(expected)))


class DSparkShapeTests(unittest.TestCase):
    def test_yarn_tables_equal_the_repository_reference(self):
        import dspark_rope_tables
        config = dict(max_position_embeddings=262144, head_dim=128, rope_parameters=dict(
            beta_fast=32.0, beta_slow=1.0, factor=32.0, original_max_position_embeddings=8192, rope_theta=10000000, rope_type='yarn'))
        original = dspark_rope_tables.validate_config
        dspark_rope_tables.validate_config = lambda value: None
        try:
            reference = dspark_rope_tables.DSparkRotary(config)
        finally:
            dspark_rope_tables.validate_config = original
        positions = torch.tensor([0, 1, 6, 7, 169, 4095, 8192, 65536, 131072, 262143])
        cos, sin = reference.positions(positions, dtype=torch.float32)
        mine_cos, mine_sin = d2.rotary(positions[None], 128, 1e7, torch.float32, (32.0, 32.0, 1.0, 8192))
        self.assertTrue(torch.allclose(mine_cos[0], cos[0, 0], atol=1e-6))
        self.assertTrue(torch.allclose(mine_sin[0], sin[0, 0], atol=1e-6))

    def test_real_dspark_config_has_no_convolution_or_selector(self):
        cfg = d2.real_dspark_config()
        self.assertEqual((cfg.conv_taps, cfg.selector_rank, cfg.window), (0, 0, None))
        self.assertEqual(cfg.yarn, (32.0, 32.0, 1.0, 8192))

    def test_a_dspark_shaped_model_runs_and_has_the_dspark_tensor_names(self):
        cfg = d2.tiny_config(window=None, taps=3)._replace(conv_taps=0, selector_rank=0, selector_top_k=0, yarn=(4.0, 32.0, 1.0, 64),
                                                           head_dim=16)
        model = d2.init_random(d2.Dflash2(cfg), 3).float().eval()
        names = sorted(model.state_dict())
        self.assertFalse(any('conv' in name or 'selector' in name for name in names))
        for expected in ('fc.weight', 'hidden_norm.weight', 'norm.weight', 'layers.0.self_attn.k_norm.weight', 'layers.1.mlp.up_proj.weight'):
            self.assertIn(expected, names)
        out = model(noise(cfg, 7, 1), positions(40, 7), rows(cfg, 40, 2), positions(0, 40))
        self.assertEqual(tuple(out.shape), (1, 7, cfg.hidden))
        self.assertTrue(torch.isfinite(out).all())
        # full attention: the oldest context row still matters
        changed = rows(cfg, 40, 2)
        changed[:, 0] += 5.0
        self.assertFalse(torch.allclose(out, model(noise(cfg, 7, 1), positions(40, 7), changed, positions(0, 40)), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
