"""The patcher must edit the right loop, or fail loudly.

model.py has four calls to _reset_gdn_state_for_new_sequence and three
"for c in range(num_full)" loops, so a text-scoped patch could silently edit the
wrong one. These tests hold the scoping honest without needing the real 174 KB file.
"""

import ast
import os
import unittest
from unittest.mock import patch
from pathlib import Path
import types

import lever_n_model_patch

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



# scripts/ci/fixtures/platform_chunked_prefill_policy.py is the REAL function, copied
# verbatim out of run 35681324335's graft artifact (_CHUNKED_PREFILL_MODEL_TYPES plus
# _apply_chunked_prefill_policy, nothing else). The reduction below stays because it is
# a readable statement of the shape the patch depends on, but it is no longer the only
# thing under test: the reduction had silently dropped the max_num_batched_tokens bump,
# and a reduction can never tell me what the real branch also does. Run 35681324335
# died because the real gemma4 branch sets disable_chunked_mm_input and this file had
# that line in front of it without a single test asking what the opt-in path does with
# it.
REAL_POLICY = Path(__file__).parent / 'fixtures' / 'platform_chunked_prefill_policy.py'


class Box(object):
    """A stand-in for the vllm config objects, which cannot be imported here."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


UNSET = 'UNSET'


def apply_policy(source, model_type, env):
    """Execute the patched policy and hand back the scheduler config it produced.

    Executing it is the point. Every assertion that v37 passed was about the TEXT of
    the patched source - that the condition had gained an `or`. None of them ran it, so
    none could see that the branch it now entered sets disable_chunked_mm_input.
    """
    module = types.ModuleType('patched_platform')
    module.__dict__['logger'] = Box(info=lambda *a, **k: None,
                                    warning=lambda *a, **k: None)
    exec(compile(source, 'patched_platform', 'exec'), module.__dict__)

    scheduler = Box(enable_chunked_prefill=True, max_num_batched_tokens=2048,
                    long_prefill_token_threshold=2048, disable_chunked_mm_input=UNSET)
    config = Box(scheduler_config=scheduler,
                 model_config=Box(hf_config=Box(model_type=model_type),
                                  max_model_len=33024))
    environ = {} if env is None else {'TT_M1_FORCE_CHUNKED_PREFILL': env}
    with patch.dict(os.environ, environ, clear=True):
        module._apply_chunked_prefill_policy(config)
    return scheduler


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
    def test_the_opt_in_is_a_separate_return_not_a_widened_allowlist(self):
        """Widening the allowlist condition is what broke run 35681324335: qwen3_5 then
        entered the branch written for gemma4, which sets disable_chunked_mm_input."""
        out = patch_platform(PLATFORM)
        ast.parse(out)
        self.assertIn('def _m1_chunked_prefill_opt_in():', out)
        self.assertIn('    if _m1_chunked_prefill_opt_in():', out)
        self.assertNotIn('_CHUNKED_PREFILL_MODEL_TYPES or _m1_chunked_prefill_opt_in()',
                         out)

    def test_the_opt_in_path_leaves_disable_chunked_mm_input_alone(self):
        """THE test v37 needed and did not have.

        vLLM 0.25.1 raises 'Chunked MM input disabled but max_tokens_per_mm_item
        (16384) is larger than max_num_batched_tokens (2048)' when that flag is set,
        and probe run 35681729538 confirmed the raise is guarded by it and that its
        default is False. Qwen36ForCausalLM is text-only, so the 16384-token item can
        never exist and gemma4's reason for the flag cannot apply here.
        """
        for source in (PLATFORM, REAL_POLICY.read_text(encoding='utf-8')):
            with self.subTest(source='reduced' if source is PLATFORM else 'real'):
                scheduler = apply_policy(patch_platform(source), 'qwen3_5', '1')
                self.assertEqual(scheduler.disable_chunked_mm_input, UNSET)
                self.assertTrue(scheduler.enable_chunked_prefill)
                self.assertEqual(scheduler.max_num_batched_tokens, 2048)
                self.assertEqual(scheduler.long_prefill_token_threshold, 2048)

    def test_without_the_env_the_policy_is_todays_behaviour(self):
        """Every arm that does not opt in must be unchanged: chunked prefill off, the
        batched budget bumped back to max_model_len, the threshold zeroed."""
        real = REAL_POLICY.read_text(encoding='utf-8')
        for env in (None, '0'):
            with self.subTest(env=env):
                scheduler = apply_policy(patch_platform(real), 'qwen3_5', env)
                self.assertFalse(scheduler.enable_chunked_prefill)
                self.assertEqual(scheduler.max_num_batched_tokens, 33024)
                self.assertEqual(scheduler.long_prefill_token_threshold, 0)
                self.assertEqual(scheduler.disable_chunked_mm_input, UNSET)

    def test_gemma4_is_untouched_in_both_env_states(self):
        """The allowlisted model types keep the branch and the flag they shipped with,
        whether or not the Qwen graft's variable happens to be set."""
        real = REAL_POLICY.read_text(encoding='utf-8')
        for model_type in ('gemma4', 'gemma4_unified'):
            for env in ('1', None):
                with self.subTest(model_type=model_type, env=env):
                    scheduler = apply_policy(patch_platform(real), model_type, env)
                    self.assertTrue(scheduler.disable_chunked_mm_input)
                    self.assertTrue(scheduler.enable_chunked_prefill)
                    self.assertEqual(scheduler.long_prefill_token_threshold, 2048)

    def test_the_real_captured_policy_is_the_one_the_graft_will_see(self):
        """A drift guard on the fixture itself: if the shipped function stops matching
        the shape the patch anchors to, this fails here rather than on the rig."""
        real = REAL_POLICY.read_text(encoding='utf-8')
        self.assertIn('def _apply_chunked_prefill_policy(', real)
        self.assertIn('_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}', real)
        self.assertIn('        scheduler_config.disable_chunked_mm_input = True', real)
        self.assertIn('    if scheduler_config.enable_chunked_prefill:', real)
        ast.parse(patch_platform(real))

    def test_a_lookalike_function_elsewhere_is_byte_identical(self):
        """_unrelated carries the same condition line; only the policy may change.

        This used to assert that the policy's condition had been rewritten and so
        appeared once instead of twice. The patch no longer rewrites that condition -
        it inserts a separate return - so counting the line says nothing about
        scoping. Comparing _unrelated's own source before and after does.
        """
        out = patch_platform(PLATFORM)

        def body_of(source, name):
            tree = ast.parse(source)
            lines = source.splitlines(keepends=True)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return ''.join(lines[node.lineno - 1:node.end_lineno])
            raise AssertionError('no function named %s' % name)

        self.assertEqual(body_of(out, '_unrelated'), body_of(PLATFORM, '_unrelated'))
        self.assertEqual(out.count('    if _m1_chunked_prefill_opt_in():'), 1)
        self.assertIn('if _m1_chunked_prefill_opt_in():',
                      body_of(out, '_apply_chunked_prefill_policy'))

    def test_the_opt_in_reads_the_documented_variable(self):
        out = patch_platform(PLATFORM)
        self.assertIn('os.environ.get("TT_M1_FORCE_CHUNKED_PREFILL") == "1"', out)

    def test_patching_twice_raises(self):
        """Idempotence has to be checked, not inherited from anchor uniqueness.

        The old patch rewrote the allowlist condition, so a second pass could not find
        the original text and failed by accident. A plain insertion leaves the anchor
        intact, so without an explicit guard a double patch would quietly stack two
        opt-in blocks and two helper definitions.
        """
        for source in (PLATFORM, REAL_POLICY.read_text(encoding='utf-8')):
            with self.subTest(source='reduced' if source is PLATFORM else 'real'):
                out = patch_platform(source)
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


class ScratchOccupancyGuardTests(unittest.TestCase):
    """The GDN prefill scratch is single-occupancy and the emitted method used to say so
    in prose - "The scheduler enforces that; this method assumes it." Nothing downstream
    checks it: docs/lever-n-build-plan-2026-09-22.md section 3 found two of the three
    guards the design cited are not guards, and the corruption this method introduces is
    invisible to all of them, so a wrong resumption yields wrong tokens rather than an
    error. These tests execute the guards that replaced the prose.

    The method body is extracted and run against stubs; the device work (mesh, ttnn,
    traced replay) is stubbed out, because what is under test is the sequencing
    arithmetic, not the prefill.
    """

    def method(self):
        """The emitted prefill_paged_slots_range, bound into a throwaway class."""
        import ast
        import textwrap
        source = 'class Host:\n' + lever_n_model_patch.SLOTS_RANGE
        ast.parse(source)          # the graft must be valid where it lands
        namespace = {'torch': _StubTorch(), 'ttnn': _StubTtnn()}
        exec(compile(textwrap.dedent(source), '<slots-range>', 'exec'), namespace)
        return namespace['Host']

    def host(self, **overrides):
        Host = self.method()
        host = Host()
        host.num_devices = 2
        host.mesh_device = object()
        host.layers = []
        host.args = type('A', (), {'vocab_size': 8})()
        host._bind_gdn_prefill_scratch = lambda: 'prev'
        host._unbind_gdn_prefill_scratch = lambda prev: None
        host._write_gdn_slot = lambda slot, rec, conv: None
        host.prefill_traced_chunked = lambda toks, pt, actual_len, start: 'logits'
        for name, value in overrides.items():
            setattr(host, name, value)
        return host

    def call(self, host, starts, lengths=None, slots=None):
        n = len(starts)
        lengths = lengths or [2048] * n
        toks = [_StubTensor((1, length)) for length in lengths]
        return host.prefill_paged_slots_range(
            toks, _StubTensor((n, 4)), slots or list(range(n)), starts,
            [s + l for s, l in zip(starts, lengths)], valid_lens=lengths)

    def test_a_resumed_prefill_cannot_share_a_call_with_another_request(self):
        """Fatal within one call: every request runs through the same scratch, so a
        sibling either re-zeroes it or advances it with its own tokens."""
        host = self.host(_qwen_lever_n_next_start=2048)
        with self.assertRaisesRegex(ValueError, 'cannot share a call with another request'):
            self.call(host, [2048, 0])
        with self.assertRaisesRegex(ValueError, 'cannot share a call with another request'):
            self.call(host, [0, 2048])

    def test_several_fresh_prompts_in_one_call_are_still_allowed(self):
        """Unchanged behaviour: N fresh prompts each reset and complete within the call,
        which is what prefill_paged_slots always did. Only RESUMPTION is exclusive."""
        host = self.host()
        self.assertEqual(len(self.call(host, [0, 0, 0])), 3)

    def test_a_continuation_must_arrive_at_the_offset_the_last_step_left(self):
        host = self.host()
        self.call(host, [0], lengths=[2048])              # fresh chunk, cursor -> 2048
        self.assertEqual(host._qwen_lever_n_next_start, 2048)
        self.call(host, [2048], lengths=[2048])           # in sequence
        self.assertEqual(host._qwen_lever_n_next_start, 4096)
        with self.assertRaisesRegex(ValueError, 'the scratch was left at'):
            self.call(host, [2048], lengths=[2048])       # replayed chunk
        with self.assertRaisesRegex(ValueError, 'the scratch was left at'):
            self.call(host, [8192], lengths=[2048])       # skipped chunk

    def test_resuming_with_no_prefill_in_flight_is_refused(self):
        """start > 0 with an empty cursor means the scratch holds nothing, or holds
        another prompt whose steps this process never saw."""
        host = self.host()
        with self.assertRaisesRegex(ValueError, 'the scratch was left at None'):
            self.call(host, [2048])

    def test_a_fresh_prompt_resets_the_cursor_which_this_guard_does_NOT_police(self):
        """Stated so the limit is on the record rather than assumed away: a fresh prompt
        admitted between two continuations of another legitimately resets the cursor, so
        this guard cannot see that interleaving. Only the scheduler can prevent it -
        build plan step 8, capping prefill capacity at partials rather than partials + 1."""
        host = self.host()
        self.call(host, [0], lengths=[2048])
        self.call(host, [2048], lengths=[2048])
        self.call(host, [0], lengths=[512])               # another prompt barges in
        self.assertEqual(host._qwen_lever_n_next_start, 512)
        # The first prompt's next continuation is now refused, but only because the
        # offsets disagree - not because the guard understood what happened.
        with self.assertRaisesRegex(ValueError, 'the scratch was left at 512'):
            self.call(host, [4096], lengths=[2048])


class _StubTensor:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, item):
        return self


class _StubTorch:
    Tensor = _StubTensor


class _StubTtnn:
    @staticmethod
    def to_torch(value, mesh_composer=None):
        return _StubHostLogits()

    @staticmethod
    def deallocate(value):
        return None

    @staticmethod
    def ConcatMeshToTensor(mesh, dim=0):
        return object()


class _StubHostLogits:
    def reshape(self, *shape):
        return self

    def __getitem__(self, item):
        return self

    def float(self):
        return self

    def view(self, *shape):
        return self


if __name__ == '__main__':
    unittest.main()
