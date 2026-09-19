"""The patcher must edit the right loop, or fail loudly.

model.py has four calls to _reset_gdn_state_for_new_sequence and three
"for c in range(num_full)" loops, so a text-scoped patch could silently edit the
wrong one. These tests hold the scoping honest without needing the real 174 KB file.
"""

import ast
import unittest

from lever_n_model_patch import (function_span, patch_chunked_entry, patch_model,
                                 patch_tp_replay, patch_vllm_entry, replace_once)

# A miniature stand-in with the same ambiguity as the real source: the decoy methods
# carry identical reset calls and chunk loops.
MODEL = '''import torch
import ttnn


class Model:
    def decoy_one(self):
        self._reset_gdn_state_for_new_sequence()
        for c in range(num_full):
            pass
        if tail_real > 0:
            pass

    def prefill_paged_slots(self, token_ids_list, page_table, empty_slots, valid_lens=None):
        return []

    def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None):
        chunk_size = self._chunked_chunk_size or 2048
        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)
        if self.num_devices > 1:
            if self._chunked_trace_id is not None:
                return self._prefill_traced_chunked_tp(
                    token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=vision_tokens
                )
        return None

    def _prefill_traced_chunked_tp(
        self, token_ids, page_table, actual_len, num_full, chunk_size, tail_real, vision_tokens=None
    ):
        """Docstring."""
        # Re-zero GDN once; carries across replays + tail (chunk_start>0 skips reset).
        self._reset_gdn_state_for_new_sequence()
        for c in range(num_full):
            pass
        if tail_real > 0:
            return self.prefill_masked_bucket()
        return None

    def decoy_two(self):
        self._reset_gdn_state_for_new_sequence()
        for c in range(num_full):
            pass
'''

VLLM = '''class Runner:
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        if True:
            return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))
        return None

    def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots):
        N = tokens.shape[0]
        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)
        return host_logits
'''


class ScopingTests(unittest.TestCase):
    def test_span_isolates_the_named_method(self):
        start, end = function_span(MODEL, '_prefill_traced_chunked_tp')
        region = ''.join(MODEL.splitlines(keepends=True)[start:end])
        self.assertIn('Docstring.', region)
        self.assertNotIn('def decoy_two', region)
        self.assertEqual(region.count('for c in range(num_full):'), 1)

    def test_decoys_are_untouched(self):
        out = patch_tp_replay(MODEL)
        self.assertEqual(out.count('for c in range(chunk_from, chunk_to):'), 1)
        self.assertEqual(out.count('for c in range(num_full):'), 2)  # both decoys intact
        self.assertEqual(out.count('if do_tail and tail_real > 0:'), 1)

    def test_replace_once_refuses_an_ambiguous_region(self):
        lines = MODEL.splitlines(keepends=True)
        with self.assertRaises(ValueError):
            replace_once(lines, (0, len(lines)), 'for c in range(num_full):', 'x', 'ambiguous')

    def test_missing_method_is_an_error(self):
        with self.assertRaises(ValueError):
            function_span(MODEL, 'not_a_method')


class OutputTests(unittest.TestCase):
    def test_model_patch_is_valid_python_and_complete(self):
        out = patch_model(MODEL)
        ast.parse(out)
        for probe in ('chunk_from=0, chunk_to=None, do_reset=True, do_tail=True',
                      'for c in range(chunk_from, chunk_to):',
                      'if do_tail and tail_real > 0:',
                      'def prefill_paged_slots_range(',
                      'start=0',
                      'assert start % chunk_size == 0'):
            self.assertIn(probe, out, probe)

    def test_reset_becomes_conditional_on_the_first_step(self):
        out = patch_tp_replay(MODEL)
        self.assertIn('if do_reset:\n            self._reset_gdn_state_for_new_sequence()', out)

    def test_rope_is_staged_only_on_the_first_step(self):
        out = patch_chunked_entry(MODEL)
        self.assertIn('if start == 0:\n            self._build_request_rope', out)

    def test_vllm_entry_threads_start_pos(self):
        out = patch_vllm_entry(VLLM)
        ast.parse(out)
        for probe in ('start_pos=kwargs.get("start_pos")', 'start_pos=None):',
                      'model.prefill_paged_slots_range('):
            self.assertIn(probe, out, probe)

    def test_dispatch_keys_off_a_nonzero_start_not_a_present_start_pos(self):
        """model_runner.submit_prefill always sends start_pos, so "is None" never fires.

        Keying off presence would send the unchunked path through the range method too,
        which is what made run 35415521079's baseline arm indistinguishable from its
        resumable arm.
        """
        out = patch_vllm_entry(VLLM)
        self.assertIn('if not any(s > 0 for s in starts):', out)
        self.assertNotIn('if start_pos is None:', out)

    def test_is_last_is_not_threaded_because_the_runner_does_not_send_it(self):
        out = patch_vllm_entry(VLLM)
        self.assertNotIn('is_last', out)
        model = patch_model(MODEL)
        self.assertNotIn('is_last=', model)

    def test_unpatched_path_is_preserved_when_start_pos_is_absent(self):
        """Without chunked prefill the old call must still run unchanged."""
        out = patch_vllm_entry(VLLM)
        self.assertIn('model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)', out)

    def test_patch_is_not_idempotent_and_says_so(self):
        out = patch_model(MODEL)
        with self.assertRaises(ValueError):
            patch_model(out)


if __name__ == '__main__':
    unittest.main()
