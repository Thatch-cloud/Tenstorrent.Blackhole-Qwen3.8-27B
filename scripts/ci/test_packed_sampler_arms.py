"""The packed block's two sampler isolation arms (the audits-off hang's factorial, tp4/sampler): default off, each reusing the T1 audit's
sampler call (PackedVerifierEngine.sample_shards -> force_argmax.sample_rows) without its per-round readback and compare.

QWEN_FAST_PACKED_SAMPLER_PREWARM=1 runs the sampler once in the eager warm forward only; QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1 also records it
in the verify capture, so every replay runs it. Neither changes a token and neither reads output[3] back."""

import unittest
from unittest.mock import Mock, patch

import packed_verifier
import test_verify_trace_t1 as base
import verify_trace_t1 as t1

ON = base.ON
PREWARM = dict(ON, QWEN_FAST_PACKED_SAMPLER_PREWARM='1')
IN_TRACE = dict(ON, QWEN_FAST_PACKED_SAMPLER_IN_TRACE='1')


class SamplerArmTests(base.PackedBlockTests):
    """Reuses the T1 packed-block fixtures (fake model batch, fake shards); the inherited tests are switched off here."""

    def setUp(self):
        super().setUp()
        self.events = []
        sample = packed_verifier.sample_rows
        sample.side_effect = lambda *args, **options: self.events.append('sample') or self.ids
        capture = packed_verifier.capture_operation
        original = capture.side_effect

        def first_capture_only(operations, mesh, operation):
            # the verify capture is the first capture_operation call; the 64 commit-trace captures after it are not the sampler's business
            if 'capture' not in self.events:
                self.events.append('capture')
            return original(operations, mesh, operation)

        capture.side_effect = first_capture_only

    def arm_lines(self):
        return [line for line in self.lines if packed_verifier.SAMPLER_ARM_MARKER in line]

    def verified(self, block, environ):
        """One verify round, and whether the round read the sampler's output (output[3]) back."""
        reads = []
        operations = block.operations
        original = operations.get_device_tensors

        def watched(tensor):
            reads.append(tensor)
            return original(tensor)

        with patch.object(operations, 'get_device_tensors', side_effect=watched):
            predictions, metrics = self.verify(block, environ)
        return predictions, any(read is self.ids for read in reads)

    def test_both_flags_default_off_and_t1_alone_runs_no_sampler(self):
        block = self.build_with(ON)
        self.assertFalse(block.sampler_prewarm or block.sampler_in_trace)
        self.assertEqual(self.events, ['capture'])
        self.assertEqual(len(block.output), 3)
        self.assertEqual(self.arm_lines(), [])

    def test_a_flag_that_is_not_exactly_one_is_off(self):
        for value in ('0', '', 'true', 'yes', '2'):
            for flag in (packed_verifier.SAMPLER_PREWARM_FLAG, packed_verifier.SAMPLER_IN_TRACE_FLAG):
                with self.subTest(flag=flag, value=value):
                    self.assertFalse(packed_verifier.sampler_arm_enabled(flag, {flag: value}))
        self.assertTrue(packed_verifier.sampler_arm_enabled(packed_verifier.SAMPLER_PREWARM_FLAG,
                                                            {packed_verifier.SAMPLER_PREWARM_FLAG: '1'}))
        self.assertFalse(packed_verifier.sampler_arm_requested({}))

    def test_prewarm_runs_the_sampler_once_in_the_eager_warm_forward_before_the_capture_and_never_in_the_trace(self):
        block = self.build_with(PREWARM)
        self.assertTrue(block.sampler_prewarm)
        self.assertFalse(block.sampler_in_trace)
        self.assertEqual(self.events, ['sample', 'capture'], 'the sampler before the capture, nothing after it')
        self.assertEqual(packed_verifier.sample_rows.call_count, 1)
        # T1's own call shape: the pinned sampler over the 64-row pre-gather logits, as in the audit.
        args, options = packed_verifier.sample_rows.call_args
        self.assertEqual((args[2], options), (64, dict(native_rows=False)))
        self.assertEqual(self.shard_calls, [((1, 1, 64, 124160), 64)] * 2, 'the shard argmax is unchanged')
        self.assertEqual(len(block.output), 3, 'the capture holds no sampler output')
        self.assertEqual(len(self.arm_lines()), 1)
        self.assertIn('prewarm=1 in_trace=0 requested=1 shard_argmax=1 audit=0', self.arm_lines()[0])

    def test_in_trace_records_the_sampler_in_the_warm_forward_and_the_capture_like_the_audit(self):
        block = self.build_with(IN_TRACE)
        self.assertTrue(block.sampler_in_trace)
        self.assertEqual(self.events, ['sample', 'capture', 'sample'])
        self.assertEqual(packed_verifier.sample_rows.call_count, 2)
        self.assertEqual(len(block.output), 4, 'output[3] is held by the capture, as the audit holds it')
        self.assertIn('prewarm=0 in_trace=1 requested=1 shard_argmax=1 audit=0', self.arm_lines()[0])

    def test_the_audit_gives_the_same_events_the_in_trace_arm_copies(self):
        block = self.build_with(dict(ON, QWEN_FAST_VERIFY_T1_AUDIT='1'))
        self.assertEqual(self.events, ['sample', 'capture', 'sample'])
        self.assertEqual(len(block.output), 4)

    def no_token_change(self, environ):
        block = self.build_with(environ)
        logged = []
        t1._AUDIT.update(rounds=0, rows=0)
        with patch.object(t1, 'log_line', side_effect=logged.append),                 patch.object(t1, 'audit_round', side_effect=AssertionError('no per-round compare')):
            predictions, read_back = self.verified(block, environ)
        self.assertEqual(predictions, [self.COMBINED[16 * user:16 * user + 16] for user in range(4)])
        self.assertFalse(read_back, 'output[3] is never read to the host')
        self.assertEqual(logged, [])

    def test_prewarm_changes_no_token_and_reads_nothing_back(self):
        self.no_token_change(PREWARM)

    def test_in_trace_changes_no_token_and_reads_nothing_back_or_compares(self):
        self.no_token_change(IN_TRACE)

    def test_with_the_audit_on_the_arms_add_nothing_and_the_audit_is_unchanged(self):
        audit = dict(ON, QWEN_FAST_VERIFY_T1_AUDIT='1', QWEN_FAST_PACKED_SAMPLER_PREWARM='1', QWEN_FAST_PACKED_SAMPLER_IN_TRACE='1')
        block = self.build_with(audit)
        self.assertFalse(block.sampler_prewarm or block.sampler_in_trace)
        self.assertTrue(block.shard_audit)
        self.assertEqual(self.events, ['sample', 'capture', 'sample'])
        self.assertEqual(len(block.output), 4)
        self.assertIn('requested=1 shard_argmax=1 audit=1', self.arm_lines()[0])
        self.assertIn('prewarm=0 in_trace=0', self.arm_lines()[0])

    def test_both_flags_together_are_the_in_trace_arm(self):
        both = dict(ON, QWEN_FAST_PACKED_SAMPLER_PREWARM='1', QWEN_FAST_PACKED_SAMPLER_IN_TRACE='1')
        block = self.build_with(both)
        self.assertEqual(self.events, ['sample', 'capture', 'sample'])
        self.assertEqual(len(block.output), 4)

    def test_with_the_shard_argmax_skipped_the_arms_do_nothing_and_say_so(self):
        block = self.build_with(dict(PREWARM, QWEN_FAST_VERIFY_T1_SKIP='shard_argmax'))
        self.assertFalse(block.sampler_prewarm or block.sampler_in_trace)
        self.assertEqual(self.events, ['sample', 'capture', 'sample'], 'the pinned sampler runs as before, twice')
        self.assertEqual(len(block.output), 2)
        self.assertIn('prewarm=0 in_trace=0 requested=1 shard_argmax=0', self.arm_lines()[-1])

    def test_without_t1_the_arms_do_nothing(self):
        block = self.build_with({'QWEN_FAST_PACKED_SAMPLER_IN_TRACE': '1'})
        self.assertFalse(block.sampler_prewarm or block.sampler_in_trace)
        self.assertEqual(self.events, ['sample', 'capture', 'sample'], 'the pinned sampler runs as before, twice')
        self.assertEqual(len(block.output), 2)

    def test_a_sampler_that_is_not_plain_greedy_keeps_the_pinned_path_in_the_arms_too(self):
        block = self.build_with(IN_TRACE, sampler=base.greedy_sampler(_penalties_active=True))
        self.assertFalse(block.sampler_in_trace)
        self.assertEqual(len(block.output), 2)
        self.assertEqual(self.shard_calls, [])


# The inherited packed-block tests run in test_verify_trace_t1: not again here.
for _name in dir(base.PackedBlockTests):
    if _name.startswith('test_') and _name not in SamplerArmTests.__dict__:
        setattr(SamplerArmTests, _name, None)


if __name__ == '__main__':
    unittest.main()
