"""a0_target on a tiny hybrid model (recurrent and attention layers): hooks equal the model's own hidden states, chunked equals
one-shot, snapshot / restore equals a fresh prefill, prefix-shared groups equal independent traces, V1 counts."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import a0_fakes as fakes  # noqa: E402
import a0_target as tgt  # noqa: E402

VOCAB, HIDDEN = 50, 16
TAPS = (1, 3, 5)


def make(taps=TAPS, seed=0):
    model = fakes.TinyHybrid(VOCAB, HIDDEN, seed=seed)
    return model, fakes.TinyTarget(model, taps)


def greedy_continue(target, ids, count):
    """The target's own greedy tokens after `ids` (so a logged answer that V1 must fully agree with)."""
    state = target.new_state()
    hidden = target.forward(state, torch.as_tensor(ids, dtype=torch.int64))
    target.taps.collect()
    out = []
    for _ in range(count):
        token = int(target.argmax(hidden[-1:])[0])
        out.append(token)
        hidden = target.forward(state, torch.tensor([token]))
        target.taps.collect()
    return out


def fresh_trace(target, ids):
    state = target.new_state()
    target.forward(state, torch.as_tensor(ids, dtype=torch.int64))
    return target.taps.collect()


class TapTests(unittest.TestCase):
    def test_hooks_equal_the_hidden_states_in_tap_order(self):
        model, target = make(taps=(3, 1, 5))
        ids = torch.randint(0, VOCAB, (9,))
        with torch.no_grad():
            _, every = model(ids, model.new_cache(), return_all=True)
        target.forward(target.new_state(), ids)
        rows = target.taps.collect()
        expected = torch.cat([every[3][0], every[1][0], every[5][0]], dim=-1)
        self.assertTrue(torch.equal(rows, expected))
        self.assertEqual(rows.shape, (9, 3 * HIDDEN))

    def test_a_missing_tap_layer_is_an_error_and_collect_clears(self):
        model, target = make()
        with self.assertRaises(RuntimeError):
            target.taps.collect()
        target.forward(target.new_state(), torch.tensor([1, 2]))
        target.taps.collect()
        with self.assertRaises(RuntimeError):
            target.taps.collect()

    def test_detach_removes_the_hooks(self):
        model, target = make()
        target.taps.detach()
        target.forward(target.new_state(), torch.tensor([1, 2]))
        with self.assertRaises(RuntimeError):
            target.taps.collect()

    def test_chunked_prefill_equals_one_shot(self):
        model, target = make()
        ids = torch.randint(0, VOCAB, (37,))
        whole = fresh_trace(target, ids)
        for chunk in (1, 3, 7, 16, 64):
            state, parts = target.new_state(), []
            for a in range(0, 37, chunk):
                target.forward(state, ids[a:a + chunk])
                parts.append(target.taps.collect())
            self.assertTrue(torch.allclose(torch.cat(parts), whole, atol=2e-3, rtol=1e-4), chunk)


class BranchTests(unittest.TestCase):
    def test_snapshot_restore_equals_a_fresh_prefill(self):
        model, target = make()
        self.assertTrue(tgt.branch_selfcheck(target, list(range(1, 30)), [3, 4, 5, 6, 7], [9, 8, 7, 6], 3 * HIDDEN, chunk=5))

    def test_a_restore_that_forgets_the_attention_length_is_caught(self):
        class Broken(fakes.TinyTarget):
            def restore(self, state, snapshot):
                for layer, saved in zip(state['cache'], snapshot):
                    if 'h' in layer:
                        layer['h'], layer['tail'] = saved['h'].clone(), saved['tail'].clone()      # keys and values keep the branch
        model = fakes.TinyHybrid(VOCAB, HIDDEN)
        self.assertFalse(tgt.branch_selfcheck(Broken(model, TAPS), list(range(1, 30)), [3, 4, 5], [9, 8, 7], 3 * HIDDEN))

    def test_a_restore_that_forgets_the_recurrent_state_is_caught(self):
        class Broken(fakes.TinyTarget):
            def restore(self, state, snapshot):
                for layer, saved in zip(state['cache'], snapshot):
                    if 'k' in layer and layer['k'] is not None:
                        layer['k'], layer['v'] = layer['k'][:, :saved['length']], layer['v'][:, :saved['length']]
        model = fakes.TinyHybrid(VOCAB, HIDDEN)
        self.assertFalse(tgt.branch_selfcheck(Broken(model, TAPS), list(range(1, 30)), [3, 4, 5], [9, 8, 7], 3 * HIDDEN))


class GroupTests(unittest.TestCase):
    def records(self, target, prompt, extra, answers):
        out, current = [], list(prompt)
        for at, answer in enumerate(answers):
            out.append(dict(prompt_ids=list(current), output_ids=answer, order=at))
            current = current + (extra[at] if at < len(extra) else [])
        return out

    def test_group_features_equal_independent_traces(self):
        model, target = make()
        prompt = [int(x) for x in torch.randint(0, VOCAB, (20,))]
        records = self.records(target, prompt, [[5, 6, 7, 8, 9], [1, 2, 3]], [
            [11, 12, 13, 14], [21, 22, 23, 24, 25, 26], [31, 32]])
        seen = {}

        def consume(record, view, v1):
            seen[record['order']] = (view.rows(0, len(view)).float().clone(), v1.as_dict())

        tgt.GroupRunner(target, chunk=7, feature_dtype=torch.float32).run_group(records, consume, 3 * HIDDEN)
        self.assertEqual(sorted(seen), [0, 1, 2])
        for record in records:
            full = list(record['prompt_ids']) + record['output_ids'][:-1]
            want = fresh_trace(target, full)
            got = seen[record['order']][0]
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.allclose(got, want, atol=2e-3, rtol=1e-4), record['order'])

    def test_v1_counts_agreement_with_the_targets_own_argmax(self):
        model, target = make()
        prompt = [int(x) for x in torch.randint(0, VOCAB, (15,))]
        answer = greedy_continue(target, prompt, 12)
        wrong = list(answer)
        wrong[5] = (wrong[5] + 1) % VOCAB
        for given, first in ((answer, None), (wrong, 5)):
            result = {}
            tgt.GroupRunner(target, chunk=4, feature_dtype=torch.float32).run_group(
                [dict(prompt_ids=prompt, output_ids=given)], lambda r, v, v1: result.update(v1.as_dict()), 3 * HIDDEN)
            self.assertEqual(result['rows'], 12)
            if first is None:
                self.assertEqual(result['agree'], 12)
            else:        # later rows are conditioned on the wrong token, so only the prefix is certain
                self.assertTrue(first <= result['agree'] < 12)
            self.assertEqual(result['first_divergence'], first)

    def test_prompts_that_do_not_extend_are_refused(self):
        model, target = make()
        records = [dict(prompt_ids=[1, 2, 3], output_ids=[4, 5]), dict(prompt_ids=[1, 9, 3, 4], output_ids=[5, 6])]
        with self.assertRaises(ValueError):
            tgt.GroupRunner(target).run_group(records, lambda *a: None, 3 * HIDDEN)

    def test_a_one_token_answer_has_no_rows(self):
        model, target = make()
        seen = []
        tgt.GroupRunner(target, feature_dtype=torch.float32).run_group(
            [dict(prompt_ids=[1, 2, 3], output_ids=[4])], lambda r, v, v1: seen.append((len(v), v1.rows)), 3 * HIDDEN)
        self.assertEqual(seen, [(3, 1)])

    def test_bf16_features(self):
        model, target = make()
        seen = []
        tgt.GroupRunner(target).run_group([dict(prompt_ids=[1, 2, 3, 4], output_ids=[5, 6, 7])],
                                          lambda r, v, v1: seen.append(v.rows(0, 6).dtype), 3 * HIDDEN)
        self.assertEqual(seen, [torch.bfloat16])


class ViewTests(unittest.TestCase):
    def test_view_slices_across_the_prompt_answer_boundary(self):
        buffer = tgt.FeatureBuffer(6, 2, torch.float32)
        buffer.write(0, torch.arange(12.).reshape(6, 2))
        answer = 100 + torch.arange(6.).reshape(3, 2)
        view = tgt.FeatureView(buffer, 6, answer)
        self.assertEqual(len(view), 9)
        self.assertEqual(view.rows(4, 8).tolist(), [[8., 9.], [10., 11.], [100., 101.], [102., 103.]])
        self.assertEqual(view.rows(6, 9).tolist(), answer.tolist())
        with self.assertRaises(ValueError):
            view.rows(0, 10)

    def test_buffer_is_written_in_order_and_read_only_when_written(self):
        buffer = tgt.FeatureBuffer(6, 2, torch.float32)
        with self.assertRaises(ValueError):
            buffer.write(2, torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            buffer.rows(0, 1)


def linear_layer(value):
    """A cache layer in the transformers 5.x linear-attention layout: dicts of state index -> tensor."""
    return SimpleNamespace(conv_states={0: torch.full((2,), float(value))}, recurrent_states={0: torch.full((3,), float(value))},
                           has_previous_state={0: True})


class AttentionStub(object):
    def __init__(self, length):
        self.keys = torch.zeros(1, 1, length, 2)
        self.values = torch.zeros(1, 1, length, 2)
        self.cropped = []

    def get_seq_length(self):
        return self.keys.shape[-2]

    def crop(self, count):
        self.cropped.append(count)
        self.keys, self.values = self.keys[..., :count, :], self.values[..., :count, :]


class HFGuardTests(unittest.TestCase):
    def test_an_unexpected_cache_layout_is_refused_not_guessed(self):
        target = tgt.HFTarget.__new__(tgt.HFTarget)
        state = SimpleNamespace(layers=[AttentionStub(3)])
        with self.assertRaises(RuntimeError):
            tgt.HFTarget.snapshot(target, state)

    def test_the_old_tensor_attribute_layout_is_not_accepted(self):
        target = tgt.HFTarget.__new__(tgt.HFTarget)
        layer = SimpleNamespace(conv_states=torch.ones(2), recurrent_states=torch.zeros(3))
        with self.assertRaises(RuntimeError):
            tgt.HFTarget.snapshot(target, SimpleNamespace(layers=[layer]))

    def test_dict_states_are_cloned_restored_in_place_and_attention_is_cropped_by_a_negative_count(self):
        target = tgt.HFTarget.__new__(tgt.HFTarget)
        linear, attention = linear_layer(1), AttentionStub(7)
        state = SimpleNamespace(layers=[linear, attention])
        buffer = linear.conv_states[0]
        snapshot = tgt.HFTarget.snapshot(target, state)
        self.assertEqual(snapshot['length'], 7)
        linear.conv_states[0].fill_(9)
        linear.recurrent_states[0].fill_(9)
        linear.has_previous_state[0] = False
        attention.keys = torch.zeros(1, 1, 12, 2)
        attention.values = torch.zeros(1, 1, 12, 2)
        tgt.HFTarget.restore(target, state, snapshot)
        self.assertEqual(attention.cropped, [-5])
        self.assertEqual(attention.get_seq_length(), 7)
        self.assertIs(linear.conv_states[0], buffer)             # the buffer itself, not a replacement
        self.assertEqual(linear.conv_states[0].tolist(), [1.0, 1.0])
        self.assertEqual(linear.recurrent_states[0].tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(linear.has_previous_state[0])

    def test_a_shorter_attention_cache_than_the_snapshot_is_refused(self):
        target = tgt.HFTarget.__new__(tgt.HFTarget)
        state = SimpleNamespace(layers=[linear_layer(1), AttentionStub(7)])
        snapshot = tgt.HFTarget.snapshot(target, state)
        state.layers[1].keys = torch.zeros(1, 1, 3, 2)
        with self.assertRaises(RuntimeError):
            tgt.HFTarget.restore(target, state, snapshot)

    def test_a_widened_conv_state_is_put_back_by_value(self):
        target = tgt.HFTarget.__new__(tgt.HFTarget)
        linear = linear_layer(1)
        state = SimpleNamespace(layers=[linear, AttentionStub(2)])
        snapshot = tgt.HFTarget.snapshot(target, state)
        linear.conv_states[0] = torch.zeros(5)                    # past recording keeps the full state
        tgt.HFTarget.restore(target, state, snapshot)
        self.assertEqual(linear.conv_states[0].tolist(), [1.0, 1.0])


class KernelBindingTests(unittest.TestCase):
    def test_a_closure_over_an_fla_function_counts_as_fla(self):
        def fast():
            return 1
        fast.__module__ = 'fla.ops.gated_delta_rule.chunk'

        def wrapped():
            return fast()
        self.assertTrue(tgt.uses_fla_kernel(wrapped))

    def test_the_torch_fallback_alone_is_not_fla(self):
        def fallback():
            return 1
        fallback.__module__ = 'transformers.models.qwen3_5.modeling_qwen3_5'

        def wrapped():
            return fallback()
        self.assertFalse(tgt.uses_fla_kernel(wrapped))

    def test_a_module_name_that_only_starts_with_fla_is_not_fla(self):
        def other():
            return 1
        other.__module__ = 'flat_things.x'
        self.assertFalse(tgt.uses_fla_kernel(other))


if __name__ == '__main__':
    unittest.main()
