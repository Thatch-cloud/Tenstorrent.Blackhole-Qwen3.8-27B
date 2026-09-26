"""S2 W6 in the request factory (s2-design.md section 4, W6a/W6b/W6d), under QWEN_FAST_EXTENT_REPLAY=1 only:
one 2048 proposal bucket whatever the prompt (single_bucket_contexts, a scoped patch of
dflash_proposal_trace.proposal_contexts), the post-prefill DRAM backstop and the scheduler hold's registration
(dram_backstop, register_dram_admission), and the engine build's ledger before point. Without the flag the
engine captures today's ladder and nothing else runs.

Its own module rather than test_serving_request_factory: that one is imported by test_serving_lifecycle, which
runs inside the C2 image, so changing it would put it on the image's in-image test path
(test_c2_image_overlay.test_the_in_image_tests_import_current_test_modules).

    py -3.11 -B -m unittest test_serving_extent_memory      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

import serving_request_factory
from serving_request_factory import from_prefill
import test_serving_request_factory   # a module import, so its TestCase is not collected here again


EXTENT = {'QWEN_FAST_EXTENT_REPLAY': '1'}
MB = 10 ** 6


class Pool(object):
    """A pool whose allocator statistics read (free, largest_free) bytes per chip."""

    def __init__(self, *chips):
        self.chips = chips

    def dram_statistics(self):
        return [dict(chip=index, free=free, largest_free=largest) for index, (free, largest) in enumerate(self.chips)]


class ExtentMemoryTests(unittest.TestCase):
    """S2 W6 in the request factory, under QWEN_FAST_EXTENT_REPLAY=1 only: one 2048 proposal bucket (W6a), the
    post-prefill DRAM backstop and the hold's registration (W6b), the engine build's ledger point (W6d)."""

    def setUp(self):
        import serving_prefill_admission

        saved = sys.modules.pop(serving_prefill_admission.DRAM_KEY, None)

        def restore():
            sys.modules.pop(serving_prefill_admission.DRAM_KEY, None)
            if saved is not None:
                sys.modules[serving_prefill_admission.DRAM_KEY] = saved

        self.addCleanup(restore)

    def environ(self, on):
        """The flag set, or absent (the flag-off byte path)."""
        environ = {name: value for name, value in os.environ.items() if name != 'QWEN_FAST_EXTENT_REPLAY'}
        if on:
            environ.update(EXTENT)
        return patch.dict(os.environ, environ, clear=True)

    def short_request(self, pool=None):
        """RequestFactoryTests' fixture for a 60-token prompt, with a proposal that reads the ladder the way
        PreparedDFlashProposal does: dflash_proposal_trace.proposal_contexts at call time."""
        import dflash_proposal_trace

        components, device, engines, arguments = test_serving_request_factory.RequestFactoryTests().fixture()
        arguments['state'].prompt_token_ids = [1] * 60
        arguments['state'].block_ids = ([0, 1],)
        device.position = 60
        ladders = []
        components.proposal.side_effect = lambda drafter, max_new_tokens: ladders.append(
            dflash_proposal_trace.proposal_contexts(drafter.position, max_new_tokens))
        pages = torch.tensor([[0, 1] + [0] * 66], dtype=torch.int32)
        helpers = [Mock(spec=['adopt_slot'], **{'adopt_slot.return_value': 2}) for _ in range(48)]

        def build():
            with patch('serving_request_factory.device_components', return_value=components):
                return from_prefill(object(), SimpleNamespace(args=SimpleNamespace(vocab_size=100), mesh_device=object()),
                                    object(), pages, helpers, buffer_pool=pool, **arguments)

        return components, helpers, ladders, build

    def test_the_flag_is_strict(self):
        for value, expected in ((None, False), ('0', False), ('1', True)):
            with self.subTest(value=value):
                environ = {} if value is None else {'QWEN_FAST_EXTENT_REPLAY': value}
                self.assertIs(serving_request_factory.extent_replay_enabled(environ), expected)
        for value in ('true', '2', '', ' 1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                serving_request_factory.extent_replay_enabled({'QWEN_FAST_EXTENT_REPLAY': value})

    def test_inside_the_scope_every_position_takes_the_one_2048_bucket_and_outside_the_ladder_is_unchanged(self):
        import dflash_proposal_inputs
        import dflash_proposal_trace

        ladder = dflash_proposal_trace.proposal_contexts
        self.assertIs(ladder, dflash_proposal_inputs.proposal_contexts)
        with serving_request_factory.single_proposal_bucket():
            for position in range(1, 4097):
                for budget in (1, 256, 16383):
                    self.assertEqual(dflash_proposal_trace.proposal_contexts(position, budget), (2048,),
                                     (position, budget))
        self.assertIs(dflash_proposal_trace.proposal_contexts, ladder, 'restored after the scope')
        self.assertEqual(dflash_proposal_trace.proposal_contexts(60, 4096), (256, 512, 1024, 2048))
        with self.assertRaises(RuntimeError), serving_request_factory.single_proposal_bucket():
            raise RuntimeError('capture failed')
        self.assertIs(dflash_proposal_trace.proposal_contexts, ladder, 'restored after a failed capture')

    def test_the_single_bucket_keeps_the_ladders_refusals(self):
        for position, budget in ((True, 513), (0, 513), (262112, 1), (4096, 0), (4096, True), (262111, 2)):
            with self.subTest(position=position, budget=budget), self.assertRaises(ValueError):
                serving_request_factory.single_bucket_contexts(position, budget)

    def test_a_2048_bucket_under_2048_history_rows_is_a_mask_the_t16_gate_accepts(self):
        """proposal_inputs masks the rows past history_rows with -inf; the T16 validate_mask accepts the holes."""
        from dflash_proposal_inputs import proposal_inputs
        from dflash_t16_native_attention import validate_mask

        for position in (1, 60, 255, 2047):
            with self.subTest(position=position):
                mask = proposal_inputs(248044, position, min(position, 2048), 16, 2048)['mask']
                self.assertEqual(tuple(mask.shape), (1, 1, 32, 2080))
                validate_mask(mask)
                self.assertTrue(torch.isneginf(mask[..., position:2048]).all(), 'the holes are masked')

    def test_under_the_flag_the_engine_captures_one_2048_bucket_and_off_the_ladder(self):
        from dflash_proposal_inputs import proposal_contexts

        for on, expected in ((True, [(2048,)]), (False, [proposal_contexts(60, 256)])):
            with self.subTest(flag=on), self.environ(on):
                _, _, ladders, build = self.short_request()
                build().close('request')
                self.assertEqual(ladders, expected)
        self.assertEqual(proposal_contexts(60, 256), (256, 512), 'off, the short prompt takes two buckets')

    def test_the_ladder_line_names_the_single_bucket_under_the_flag(self):
        with self.environ(True):
            self.assertEqual(serving_request_factory._proposal_ladder(60, 4096), (2048,))
        with self.environ(False):
            self.assertEqual(serving_request_factory._proposal_ladder(60, 4096), (256, 512, 1024, 2048))

    def test_the_backstop_refuses_below_the_build_peak_and_the_reserve_on_the_smallest_largest_block(self):
        import serving_prefill_admission

        reserve = 256 * 2 ** 20
        need = serving_prefill_admission.backstop_need(reserve)
        log = Mock()
        self.assertEqual(serving_request_factory.dram_backstop(Pool((9_000 * MB, need), (9_000 * MB, need + 1)),
                                                               request_id='r', reserve=reserve, log=log), need)
        for pool in (Pool((9_000 * MB, need - 1), (9_000 * MB, need)),
                     Pool((40_000 * MB, 100 * MB), (40_000 * MB, 40_000 * MB))):
            with self.subTest(chips=pool.chips), self.assertRaisesRegex(serving_request_factory.RequestRefused,
                                                                        'DRAM backstop'):
                serving_request_factory.dram_backstop(pool, request_id='r', reserve=reserve, log=log)
        self.assertTrue(log.call_args_list[-1].args[0].startswith(serving_request_factory.DRAM_BACKSTOP_REFUSED))
        self.assertIsNone(serving_request_factory.dram_backstop(object(), request_id='r', reserve=reserve, log=log),
                          'an unreadable pool is a diagnostic, not a refusal')
        self.assertEqual(log.call_args_list[-1].args[1:], ('r', 'pool without device statistics'))

    def test_under_the_flag_a_short_pool_refuses_the_request_before_any_device_state(self):
        pool = Pool((9_000 * MB, 100 * MB), (9_000 * MB, 100 * MB))
        with self.environ(True):
            components, helpers, _, build = self.short_request(pool)
            with patch('serving_request_factory._log'), \
                    self.assertRaisesRegex(serving_request_factory.RequestRefused, 'DRAM backstop'):
                build()
            components.device.assert_not_called()
            components.engine.assert_not_called()
            self.assertEqual([helper.adopt_slot.call_count for helper in helpers], [0] * 48)
        with self.environ(False):
            components, _, _, build = self.short_request(pool)
            build().close('request')
            components.device.assert_called_once()

    def test_the_engine_build_is_a_ledger_before_point_under_the_flag_only(self):
        import memory_ledger
        import serving_prefill_admission

        for on in (True, False):
            with self.subTest(flag=on), self.environ(on):
                components, _, _, build = self.short_request()
                order = []
                components.device.side_effect = lambda *args, **kwargs: order.append('device') or components.device.return_value
                with patch.object(memory_ledger, 'before', side_effect=lambda *args, **kwargs: order.append(('before', args, kwargs))):
                    build().close('request')
                if on:
                    self.assertEqual(order, [('before', ('engine',), dict(estimate=serving_prefill_admission.engine_build_peak(), point='req=request',
                                                                          request='request')), 'device'])
                else:
                    self.assertEqual(order, ['device'])

    def test_registration_parks_the_hold_and_logs_what_it_reads(self):
        import serving_prefill_admission

        log = Mock()
        pool = Pool((9_000 * MB, 1_500 * MB), (9_000 * MB, 2_000 * MB))
        with self.environ(True):
            unregister = serving_request_factory.register_dram_admission(pool, log=log)
        holder = sys.modules[serving_prefill_admission.DRAM_KEY]
        reserve = 256 * 2 ** 20
        self.assertEqual(holder.admits(60), (True, dict(largest_free=1_500 * MB, need=1_000 * MB + reserve)))
        self.assertFalse(holder.admits(4096)[0], 'a long prompt needs 1.3 GB and the reserve')
        self.assertTrue(log.call_args.args[0].startswith(serving_request_factory.DRAM_REGISTERED))
        self.assertEqual(log.call_args.args[1:], (800 * MB, 200 * MB, 300 * MB, 2048, reserve, 1_500 * MB))
        unregister()
        self.assertNotIn(serving_prefill_admission.DRAM_KEY, sys.modules)


if __name__ == '__main__':
    unittest.main()
