"""The patcher must edit the right loop, or fail loudly.

model.py has four calls to _reset_gdn_state_for_new_sequence and three
"for c in range(num_full)" loops, so a text-scoped patch could silently edit the
wrong one. These tests hold the scoping honest without needing the real 174 KB file.
"""

import ast
import unittest

from lever_n_model_patch import (function_span, patch_chunked_entry, patch_model,
                                 patch_platform, patch_prefill_chunk, patch_tp_replay,
                                 patch_vllm_entry, replace_once)

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



# The real policy, reduced to the shape patch_platform depends on.
PLATFORM = """import os

_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}


def _apply_chunked_prefill_policy(vllm_config: "VllmConfig") -> None:
    \"\"\"Restrict token-chunked prefill to the model types that support it.\"\"\"
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config
    model_type = getattr(model_config.hf_config, "model_type", None)

    if model_type in _CHUNKED_PREFILL_MODEL_TYPES:
        scheduler_config.disable_chunked_mm_input = True
        return

    if scheduler_config.enable_chunked_prefill:
        scheduler_config.enable_chunked_prefill = False
    scheduler_config.long_prefill_token_threshold = 0


def _unrelated(vllm_config):
    if model_type in _CHUNKED_PREFILL_MODEL_TYPES:
        return True
"""


class PlatformTests(unittest.TestCase):
    def test_the_allowlist_gains_an_explicit_opt_in(self):
        out = patch_platform(PLATFORM)
        ast.parse(out)
        self.assertIn('def _m1_chunked_prefill_opt_in():', out)
        self.assertIn('if model_type in _CHUNKED_PREFILL_MODEL_TYPES or '
                      '_m1_chunked_prefill_opt_in():', out)

    def test_a_lookalike_condition_elsewhere_is_untouched(self):
        """_unrelated carries the same line; only the policy function may change."""
        out = patch_platform(PLATFORM)
        self.assertEqual(out.count('if model_type in _CHUNKED_PREFILL_MODEL_TYPES:'), 1)
        self.assertIn('def _unrelated(vllm_config):', out)

    def test_the_opt_in_reads_the_documented_variable(self):
        out = patch_platform(PLATFORM)
        self.assertIn('os.environ.get("TT_M1_FORCE_CHUNKED_PREFILL") == "1"', out)

    def test_patching_twice_raises(self):
        out = patch_platform(PLATFORM)
        with self.assertRaises(ValueError):
            patch_platform(out)

    def test_a_source_without_the_policy_raises(self):
        with self.assertRaises(ValueError):
            patch_platform('import os' + chr(10))



class PrefillChunkTests(unittest.TestCase):
    SOURCE = ('_PREFILL_WARMUP_CHUNK = 2048' + chr(10)
              + '_PREFILL_WARMUP_BUCKET = 4096' + chr(10))

    def test_retunes_only_the_chunk(self):
        out = patch_prefill_chunk(self.SOURCE, 4096)
        self.assertIn('_PREFILL_WARMUP_CHUNK = 4096', out)
        self.assertIn('_PREFILL_WARMUP_BUCKET = 4096', out)

    def test_rejects_sizes_model_py_would_assert_on(self):
        """model.py asserts chunk_size % 128 == 0, so catch it here rather than on device."""
        for bad in (2000, 100, 0, -2048, 32768, '4096', 4096.0):
            with self.assertRaises(ValueError):
                patch_prefill_chunk(self.SOURCE, bad)

    def test_a_source_without_the_constant_raises(self):
        with self.assertRaises(ValueError):
            patch_prefill_chunk('nothing here', 4096)

    def test_refuses_an_ambiguous_source(self):
        with self.assertRaises(ValueError):
            patch_prefill_chunk(self.SOURCE + self.SOURCE, 4096)


if __name__ == '__main__':
    unittest.main()
