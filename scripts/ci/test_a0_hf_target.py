"""HFTarget against the REAL transformers Qwen3_5 classes, on a tiny random model (CPU, the pinned transformers): the taps equal
`output_hidden_states`, chunked equals one-shot, snapshot / restore equals a fresh prefill, a prefix-shared group equals independent
traces. The cache layout (dict states in linear-attention layers, attention layers cropped by a negative count) is the one thing the
hand-built fakes cannot vouch for, so these run wherever transformers imports and are skipped where it does not (a laptop without it);
the CI install step pins it."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import a0_target as tgt  # noqa: E402

try:
    import transformers
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    HAVE = True
except Exception:   # pragma: no cover - environments without transformers
    HAVE = False

VOCAB = 64
LAYER_TYPES = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * 2
TAPS = (1, 3, 5)


def tiny_model(seed=0):
    config = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=len(LAYER_TYPES),
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_conv_kernel_dim=4,
                               linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
                               layer_types=LAYER_TYPES, max_position_embeddings=512)
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(config).eval()


def ids(count, seed):
    return torch.randint(0, VOCAB, (count,), generator=torch.Generator().manual_seed(seed))


@unittest.skipUnless(HAVE, 'transformers is not installed')
class HFTargetTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.target = tgt.HFTarget(cls.model, TAPS, 'cpu', head_chunk=7)

    def trace(self, tokens, chunk=1 << 20):
        state = self.target.new_state()
        rows, hidden = [], []
        for a in range(0, len(tokens), chunk):
            hidden.append(self.target.forward(state, tokens[a:a + chunk]))
            rows.append(self.target.taps.collect())
        return torch.cat(rows), torch.cat(hidden)

    def test_taps_equal_output_hidden_states(self):
        tokens = ids(23, 1)
        with torch.no_grad():
            reference = self.model.model(input_ids=tokens[None], output_hidden_states=True, use_cache=False).hidden_states
        rows, _ = self.trace(tokens)
        expected = torch.cat([reference[i + 1][0] for i in TAPS], dim=-1)
        self.assertTrue(torch.allclose(rows, expected, atol=1e-5, rtol=1e-5))

    def test_chunked_equals_one_shot(self):
        tokens = ids(37, 2)
        whole, hidden_whole = self.trace(tokens)
        for chunk in (5, 16):
            rows, hidden = self.trace(tokens, chunk)
            self.assertTrue(torch.allclose(rows, whole, atol=1e-4, rtol=1e-4))
            self.assertTrue(torch.allclose(hidden, hidden_whole, atol=1e-4, rtol=1e-4))

    def test_the_linear_layers_really_hold_dict_states(self):
        state = self.target.new_state()
        self.target.forward(state, ids(6, 3))
        self.target.taps.collect()
        linear = [layer for layer in state.layers if tgt.is_linear_layer(layer)]
        self.assertEqual(len(linear), 6)
        self.assertEqual(sum(1 for layer in state.layers if tgt.is_attention_layer(layer) and not tgt.is_linear_layer(layer)), 2)

    def test_snapshot_restore_equals_a_fresh_prefill(self):
        self.assertTrue(tgt.branch_selfcheck(self.target, ids(19, 4).tolist(), ids(7, 5).tolist(), ids(9, 6).tolist(), 3 * 32, chunk=6))

    def test_restore_returns_the_state_exactly_and_keeps_the_buffers(self):
        state = self.target.new_state()
        self.target.forward(state, ids(11, 7))
        self.target.taps.collect()
        buffers = [layer.recurrent_states[0] for layer in state.layers if tgt.is_linear_layer(layer)]
        snapshot = self.target.snapshot(state)
        before = [b.clone() for b in buffers]
        self.target.forward(state, ids(5, 8))
        self.target.taps.collect()
        self.target.restore(state, snapshot)
        for buffer, kept in zip(buffers, before):
            self.assertTrue(torch.equal(buffer, kept))
        self.assertEqual(self.target._attention_length(state), 11)
        # and the next forward from the restored state equals a fresh prefill of the same rows
        tail = ids(4, 9)
        hidden = self.target.forward(state, tail)
        rows = self.target.taps.collect()
        _, fresh_hidden = self.trace(torch.cat([ids(11, 7), tail]))
        self.assertTrue(torch.allclose(hidden, fresh_hidden[11:], atol=1e-4, rtol=1e-4))

    def test_a_prefix_group_equals_independent_traces(self):
        runner = tgt.GroupRunner(self.target, chunk=8, feature_dtype=torch.float32)
        prompt1, prompt2 = ids(14, 10).tolist(), None
        answer1 = ids(5, 11).tolist()
        prompt2 = prompt1 + answer1 + ids(6, 12).tolist()
        answer2 = ids(4, 13).tolist()
        records = [dict(k=0, prompt_ids=prompt1, output_ids=answer1), dict(k=1, prompt_ids=prompt2, output_ids=answer2)]
        seen = {}
        runner.run_group(records, lambda record, view, v1: seen.__setitem__(record['k'], view.rows(0, len(view)).clone()), 3 * 32)
        for record in records:
            full = torch.as_tensor(record['prompt_ids'] + record['output_ids'][:-1], dtype=torch.int64)
            expected, _ = self.trace(full)
            self.assertTrue(torch.allclose(seen[record['k']], expected, atol=1e-4, rtol=1e-4))

    def test_the_torch_fallback_is_not_reported_as_the_fast_kernel(self):
        module = sys.modules[type(self.model.model.layers[0].linear_attn).__module__]
        self.assertFalse(tgt.uses_fla_kernel(module.torch_chunk_gated_delta_rule))


if __name__ == '__main__':
    unittest.main()
