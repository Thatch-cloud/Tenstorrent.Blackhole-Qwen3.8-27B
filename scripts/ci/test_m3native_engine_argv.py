"""The engine argv this gate serves, pinned. Two rig slots paid for this file.

Run 35679222511 (v36) served the chunked arm with:

    --no-enable-chunked-prefill --max-num-batched-tokens 33024

because M3NATIVE_PREFILL_CHUNK_TOKENS was read on the runner host and never passed
into the container. Every prompt was prefilled whole and the resumable path never ran.

Run 35681324335 (v37) served the right flags and the engine refused to start:

    ValueError: Chunked MM input disabled but max_tokens_per_mm_item (16384) is
    larger than max_num_batched_tokens (2048)

because nothing zeroed the phantom multimodal item that Qwen3_5ForConditionalGeneration
declares and Qwen36ForCausalLM, the text-only class it resolves to, can never consume.

Both are facts about a list of strings. Both were knowable on CPU in milliseconds. The
reason neither was caught is that start_server built the argv and immediately Popened
it, so asserting the argv meant launching a server. engine_argv is now the pure part.
"""

import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from lever_n_m3native_gate import engine_argv

FLAG = 'M3NATIVE_PREFILL_CHUNK_TOKENS'
CONTEXT = 33024
USERS = 4

# The zeros lever_n_m1_gate has passed since its v8 run (35416319586). Written out
# rather than imported so that a change to either gate's value fails the parity test
# below instead of both drifting together.
MM_ZEROS = json.dumps(dict(image=0, video=0))


def argv(chunk=None):
    environ = {} if chunk is None else {FLAG: chunk}
    with patch.dict(os.environ, environ, clear=True):
        return engine_argv(8000, USERS, CONTEXT)


def value_of(args, flag):
    """The argument following flag, or None when flag is a bare switch or absent."""
    if flag not in args:
        return None
    index = args.index(flag)
    if index + 1 >= len(args) or args[index + 1].startswith('--'):
        return None
    return args[index + 1]


class UnchunkedArmTests(unittest.TestCase):
    """The arm every tag before v36 serves, and the v39 control. Must not move."""

    def test_chunked_prefill_is_explicitly_off(self):
        args = argv()
        self.assertIn('--no-enable-chunked-prefill', args)
        self.assertNotIn('--enable-chunked-prefill', args)

    def test_the_batched_budget_is_the_whole_context(self):
        self.assertEqual(value_of(argv(), '--max-num-batched-tokens'), str(CONTEXT))

    def test_no_chunk_threshold_is_passed(self):
        self.assertNotIn('--long-prefill-token-threshold', argv())


class ChunkedArmTests(unittest.TestCase):
    """The v40 arm. The invariant is stated in the gate: max_num_batched_tokens must
    EQUAL the model chunk size, because a larger budget hands the model a window it
    cannot replay as whole traced chunks and a smaller one starts a continuation
    mid-chunk, breaking start % chunk_size == 0."""

    def test_the_v36_regression_cannot_recur(self):
        """With the flag set, the argv must never claim chunked prefill is disabled."""
        for chunk in ('1024', '2048', '4096'):
            with self.subTest(chunk=chunk):
                args = argv(chunk)
                self.assertIn('--enable-chunked-prefill', args)
                self.assertNotIn('--no-enable-chunked-prefill', args)

    def test_batched_budget_and_threshold_both_equal_the_chunk(self):
        for chunk in ('1024', '2048', '4096'):
            with self.subTest(chunk=chunk):
                args = argv(chunk)
                self.assertEqual(value_of(args, '--max-num-batched-tokens'), chunk)
                self.assertEqual(value_of(args, '--long-prefill-token-threshold'), chunk)

    def test_an_unqualified_chunk_is_refused(self):
        for bad in ('0', '512', '3000', '8192', 'yes', ''):
            with self.subTest(chunk=bad), self.assertRaises(ValueError):
                argv(bad)


class MultimodalBudgetTests(unittest.TestCase):
    """The v37 refusal. Zeroing the declared modalities makes
    compute_mm_encoder_budget return (0, 0) before it can raise, and stops vLLM sizing
    an encoder cache for a modality this checkpoint has no TT weights for."""

    def test_both_arms_zero_the_declared_modalities(self):
        for chunk in (None, '2048'):
            with self.subTest(chunk=chunk):
                self.assertEqual(value_of(argv(chunk), '--limit-mm-per-prompt'), MM_ZEROS)

    def test_the_zeros_match_the_m1_gate_exactly(self):
        """Parity with the lane that has served these keys successfully since v8. If
        either gate changes its modality set, this fails rather than letting the two
        drift apart silently."""
        m1 = (Path(__file__).parent / 'lever_n_m1_gate.py').read_text(encoding='utf-8')
        self.assertIn("'--limit-mm-per-prompt', json.dumps(dict(image=0, video=0))", m1)
        m3 = (Path(__file__).parent / 'lever_n_m3native_gate.py').read_text(encoding='utf-8')
        self.assertIn("'--limit-mm-per-prompt', json.dumps(dict(image=0, video=0))", m3)


class ArmInvariantTests(unittest.TestCase):
    def test_the_two_chunked_switches_are_never_both_present(self):
        for chunk in (None, '1024', '2048', '4096'):
            with self.subTest(chunk=chunk):
                args = argv(chunk)
                both = '--enable-chunked-prefill' in args and '--no-enable-chunked-prefill' in args
                self.assertFalse(both)

    def test_the_arm_geometry_is_unchanged_by_chunking(self):
        """Chunking must move the prefill budget and nothing else: same users, same
        context, same served name, same speculative config. A v40 result is only
        comparable to the v39 control if these agree."""
        base, chunked = argv(), argv('2048')
        for flag in ('--max-num-seqs', '--max-model-len', '--served-model-name',
                     '--block-size', '--num-gpu-blocks-override', '--speculative-config',
                     '--additional-config', '--dtype', '--shutdown-timeout'):
            with self.subTest(flag=flag):
                self.assertEqual(value_of(base, flag), value_of(chunked, flag))

    def test_prefix_caching_and_async_scheduling_stay_off(self):
        """Both are load-bearing for the fast path: FastRunnerBridge requires
        non_dp_async_scheduling false, and a cached prefix would break the
        uncached-prompt contract the lifecycle asserts."""
        for chunk in (None, '2048'):
            with self.subTest(chunk=chunk):
                args = argv(chunk)
                self.assertIn('--no-enable-prefix-caching', args)
                self.assertIn('--no-async-scheduling', args)


if __name__ == '__main__':
    unittest.main()
