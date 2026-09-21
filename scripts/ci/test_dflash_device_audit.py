"""The first-divergent-stage audit behind QWEN_FAST_PROPOSAL_AUDIT.

Off by default and invisible to the proposal: no observer is passed down and no
readback happens. On, every replicated input and intermediate of the EAGER
proposal is read back from both chips at the stage that made it and reported on
one short '[AUDIT]' line; the first stage that differs is marked with which chip's
copy carries the larger norm, and the summary names it. At proposal start the
state a device reads but does not own - lent weights, pool histories, the
collectives and its counters, the target model's tensors - is listed with id()
and device addresses, so two devices' listings can be compared for a collision.
"""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice, ProposalAudit, compare_shards, proposal_audit_enabled

# The log capture truncates around 250 characters and loguru's prefix takes some;
# every audit message stays within the budget the shard check is held to.
LINE_BUDGET = 180


class FakeShard:
    def __init__(self, data, address):
        self.data, self.address = data, address

    def buffer_address(self):
        return self.address


class FakeTensor:
    def __init__(self, shards):
        self.shards = shards
        self.shape = tuple(shards[0].data.shape)


def replicated(values, address=0x1000, *, chip1=None):
    """One mesh tensor: chip 0 holds `values`, chip 1 the same unless `chip1` is given."""
    return FakeTensor([FakeShard(values, address), FakeShard(values if chip1 is None else chip1, address + 0x100)])


class FakeOperations(SimpleNamespace):
    """DFlashDevice.operations stand-in: chip shards through get_device_tensors and
    to_torch, counted, plus whatever Mocks a test attaches."""

    def __init__(self, **attributes):
        super().__init__(**attributes)
        self.reads = 0

    def get_device_tensors(self, value):
        return value.shards

    def to_torch(self, shard):
        self.reads += 1
        return shard.data


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, template, *values):
        self.lines.append(template.format(*values))

    def stages(self):
        return [line.split(' stage=')[1].split(' ')[0] for line in self.lines if ' stage=' in line]


def audit_device(**overrides):
    device = SimpleNamespace(operations=FakeOperations(), proposal_calls=2, position=32780, history_rows=2048,
        block_rows=16, native_proposal_attention=True, fused_convolution=True, shared_weights=None,
        pool_slot=None, collectives=None, model=None, kv_history=None, history=None, spare_history=None,
        validated_native_proposal_masks=set(), mesh='mesh', audit_digest=None)
    device.__dict__.update(overrides)
    return device


class SwitchTests(unittest.TestCase):
    def test_off_by_default_and_read_from_the_environment_each_time(self):
        self.assertFalse(proposal_audit_enabled({}))
        self.assertFalse(proposal_audit_enabled({'QWEN_FAST_PROPOSAL_AUDIT': '0'}))
        self.assertTrue(proposal_audit_enabled({'QWEN_FAST_PROPOSAL_AUDIT': '1'}))
        with patch.dict(os.environ, {'QWEN_FAST_PROPOSAL_AUDIT': '1'}):
            self.assertTrue(proposal_audit_enabled())
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(proposal_audit_enabled())


class CompareShardsTests(unittest.TestCase):
    def test_equal_copies_differ_nowhere(self):
        values = torch.arange(8, dtype=torch.float32).bfloat16()
        result = compare_shards(values, values.clone())
        self.assertEqual((result['differing'], result['total'], result['max_abs'], result['mean_abs']), (0, 8, 0.0, 0.0))
        self.assertIsNone(result['larger_norm'])
        self.assertEqual(result['finite'], (True, True))

    def test_a_difference_is_counted_bit_for_bit_and_sized_and_the_larger_norm_named(self):
        left = torch.arange(8, dtype=torch.float32).bfloat16()
        right = left.clone()
        right[3] += 1
        result = compare_shards(left, right)
        self.assertEqual((result['differing'], result['total']), (1, 8))
        self.assertEqual(result['max_abs'], 1.0)
        self.assertAlmostEqual(result['mean_abs'], 1 / 8)
        self.assertEqual(result['larger_norm'], 'chip1')
        self.assertEqual(compare_shards(right, left)['larger_norm'], 'chip0')

    def test_signed_zero_counts_as_bits_that_differ_and_float32_and_integers_compare_too(self):
        zeros = torch.zeros(4, dtype=torch.bfloat16)
        negative = zeros.clone()
        negative[0] = -0.0
        result = compare_shards(zeros, negative)
        self.assertEqual((result['differing'], result['max_abs']), (1, 0.0))
        wide = torch.arange(4, dtype=torch.float32)
        self.assertEqual(compare_shards(wide, wide + 0.5)['differing'], 4)
        identifiers = torch.tensor([[17, 248070, 248070]], dtype=torch.int32)
        changed = identifiers.clone()
        changed[0, 2] = 5
        self.assertEqual(compare_shards(identifiers, changed)['differing'], 1)
        self.assertEqual(compare_shards(identifiers, identifiers.clone())['differing'], 0)

    def test_non_finite_values_are_reported_and_geometry_mismatches_refused(self):
        left = torch.zeros(4, dtype=torch.bfloat16)
        right = left.clone()
        right[1] = float('inf')
        self.assertEqual(compare_shards(left, right)['finite'], (True, False))
        with self.assertRaises(AssertionError):
            compare_shards(torch.zeros(4, dtype=torch.bfloat16), torch.zeros(5, dtype=torch.bfloat16))
        with self.assertRaises(AssertionError):
            compare_shards(torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.float32))


class StageLineTests(unittest.TestCase):
    def test_one_short_line_per_stage_and_only_the_first_divergence_is_marked(self):
        log, device = Lines(), audit_device()
        audit = ProposalAudit(device, log=log)
        # Values below one, so adding a half is exact in BF16 and the count is the rows touched.
        equal = (torch.arange(2048, dtype=torch.float32) % 32 / 32).reshape(1, 1, 32, 64).bfloat16()
        audit.observe('input.history-window', replicated(equal))
        larger = equal.clone()
        larger[..., :16, :] += 0.5
        audit.observe('layer2.attn.gathered', replicated(equal, chip1=larger))
        audit.observe('layer2.attn.reduced', replicated(larger, chip1=equal))
        audit.summary()
        audit.summary()
        self.assertEqual(log.stages(), ['input.history-window', 'layer2.attn.gathered', 'layer2.attn.reduced'])
        self.assertEqual(device.operations.reads, 6)
        first, second, third, summary = log.lines
        prefix = '[AUDIT] dev=%x call=2 ' % id(device)
        for line in log.lines:
            self.assertTrue(line.startswith(prefix), line)
            self.assertLessEqual(len(line), LINE_BUDGET, line)
        self.assertTrue(first.endswith('shape=1x1x32x64 differing=0 of 2048 max_abs=0 mean_abs=0'), first)
        self.assertTrue(second.endswith('differing=1024 of 2048 max_abs=0.5 mean_abs=0.25 FIRST larger_norm=chip1'), second)
        self.assertTrue(third.endswith('differing=1024 of 2048 max_abs=0.5 mean_abs=0.25'),
                        'only the first divergent stage carries the marker: ' + third)
        self.assertEqual(audit.first_divergent, 'layer2.attn.gathered')
        self.assertIn('summary stages=3 diverged=2 first_divergent=layer2.attn.gathered larger_norm=chip1 norms=', summary)
        self.assertEqual(len(log.lines), 4, 'a second summary is not written')

    def test_non_finite_copies_are_flagged_on_the_stage_line(self):
        log, device = Lines(), audit_device()
        audit = ProposalAudit(device, log=log)
        broken = torch.zeros(4, dtype=torch.bfloat16)
        broken[2] = float('nan')
        audit.observe('layer0.mlp.reduced', replicated(torch.zeros(4, dtype=torch.bfloat16), chip1=broken))
        self.assertIn(' finite=1/0 FIRST larger_norm=', log.lines[0])

    def test_a_summary_without_divergence_says_none(self):
        log, device = Lines(), audit_device()
        audit = ProposalAudit(device, log=log)
        audit.observe('selector.features', replicated(torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16)))
        audit.summary()
        self.assertIn('summary stages=1 diverged=0 first_divergent=none', log.lines[-1])

    def test_a_tensor_without_both_chips_is_refused(self):
        audit = ProposalAudit(audit_device(), log=Lines())
        with self.assertRaises(AssertionError):
            audit.observe('input.mask', FakeTensor([FakeShard(torch.zeros(2, dtype=torch.bfloat16), 0x10)]))

    def test_a_long_message_continues_on_further_lines_rather_than_truncating(self):
        log = Lines()
        audit = ProposalAudit(audit_device(), log=log, line_budget=80)
        audit.line('{}', 'x' * 200)
        self.assertGreater(len(log.lines), 1)
        for line in log.lines:
            self.assertLessEqual(len(line), 80)
            self.assertTrue(line.startswith('[AUDIT] dev='))
        joined = log.lines[0] + ''.join(line.split('...', 1)[1] for line in log.lines[1:])
        self.assertEqual(joined.count('x'), 200)


class NotOwnedListingTests(unittest.TestCase):
    def fixture(self):
        tensors = [replicated(torch.zeros(2, dtype=torch.bfloat16), address) for address in (0x1000, 0x2000, 0x3000)]
        names = ['layer0.attention.norm', 'layer0.attention.convolution', 'selector_projection']
        weights = SimpleNamespace(tensors=tensors, borrowers=[object(), object()], names=lambda: list(names))
        model = SimpleNamespace(lm_head_weight=replicated(torch.zeros(2, dtype=torch.bfloat16), 0x9000),
                                embd=SimpleNamespace(weights=replicated(torch.zeros(2, dtype=torch.bfloat16), 0xa000)))
        collectives = SimpleNamespace(ag_index=[3, 0], barrier_index=7, handles=object(), armed=True)
        device = audit_device(shared_weights=weights, pool_slot=SimpleNamespace(index=1, owner='B'),
            collectives=collectives, model=model, kv_history=SimpleNamespace(active=[{}] * 5),
            history=replicated(torch.zeros(2, dtype=torch.bfloat16), 0x6000),
            spare_history=replicated(torch.zeros(2, dtype=torch.bfloat16), 0x7000))
        return device, weights, names, collectives, model

    def begin(self, device):
        log = Lines()
        with patch('dflash_device.projection_links', return_value=1):
            ProposalAudit(device, log=log).begin()
        for line in log.lines:
            self.assertLessEqual(len(line), LINE_BUDGET, line)
        return log.lines

    def test_every_lent_tensor_and_every_shared_object_is_listed_with_id_and_addresses(self):
        device, weights, names, collectives, model = self.fixture()
        lines = self.begin(device)
        text = '\n'.join(lines)
        self.assertIn('begin position=32780 history_rows=2048 block_rows=16 path=eager,uncached-history '
                      'native_attention=True fused_convolution=True', text)
        self.assertIn('sharded per chip, not compared: embedding.local, attn q/k/v heads+output, o-proj partial, '
                      'mlp gate/up/act, down-proj partial, vocab head top16', text)
        self.assertIn('not-owned weights=SimpleNamespace@%x borrowers=2 model=SimpleNamespace@%x collectives=SimpleNamespace@%x'
                      % (id(weights), id(model), id(collectives)), text)
        self.assertIn('pool slot=SimpleNamespace@%x index=1 owner=B' % id(device.pool_slot), text)
        state = next(line for line in lines if 'collectives state=' in line)
        self.assertIn("'ag_index': [3, 0]", state)
        self.assertIn("'barrier_index': 7", state)
        self.assertNotIn('handles', state)
        self.assertNotIn('armed', state, 'booleans are not counters')
        self.assertIn('histories lent by the pool: history=6000/6100 spare_history=7000/7100; kv banks 5 layers '
                      '(not read by the eager proposal)', text)
        self.assertIn('model tensors: lm_head_weight=9000/9100 embd=SimpleNamespace@%x embd.weights=a000/a100'
                      % id(model.embd), text)
        self.assertIn('module state: projection_links=1 (lru_cache projection_link_policy._resolve) validated_masks=0 '
                      'program_cache_entries=n/a', text)
        self.assertIn('t16 admission=', text)
        self.assertIn('shared weights: 3 tensors, digest ', text)
        for name, tensor, address in zip(names, weights.tensors, ('1000/1100', '2000/2100', '3000/3100')):
            entry = 'shared-tensor %s id=%x addr=%s' % (name, id(tensor), address)
            self.assertEqual(text.count(entry), 1, entry)

    def test_an_unchanged_set_is_one_line_and_a_changed_address_is_listed_again(self):
        device, weights, names, collectives, model = self.fixture()
        self.begin(device)
        again = '\n'.join(self.begin(device))
        self.assertIn('shared weights: 3 tensors, ids and addresses unchanged since listed (digest ', again)
        self.assertNotIn('layer0.attention.norm id=', again)
        weights.tensors[1] = replicated(torch.zeros(2, dtype=torch.bfloat16), 0x2200)
        relisted = self.begin(device)
        text = '\n'.join(relisted)
        self.assertIn('layer0.attention.convolution id=%x addr=2200/2300' % id(weights.tensors[1]), text)
        self.assertEqual(sum('shared-tensor ' in line for line in relisted), 3, 'one line per tensor')

    def test_a_device_with_its_own_weights_and_histories_says_so(self):
        device = audit_device(history=replicated(torch.zeros(2, dtype=torch.bfloat16), 0x6000))
        text = '\n'.join(self.begin(device))
        self.assertIn('not-owned weights=none borrowers=0 model=none collectives=none', text)
        self.assertIn('pool slot=none index=None owner=None', text)
        self.assertIn('collectives state=none', text)
        self.assertIn('histories owned: history=6000/6100 spare_history=n/a; kv banks none', text)
        self.assertIn("shared weights: none; the weights are this device's own uploads", text)


class ProposeTests(unittest.TestCase):
    """DFlashDevice.propose over a stand-in `self`: the eager path with the trace absent."""

    def device(self, *, tensors):
        operations = FakeOperations(bfloat16='bf16', uint32='u32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='row',
            DRAM_MEMORY_CONFIG='dram', ReplicateTensorToMesh=Mock(return_value='map'), synchronize_device=Mock())
        small = torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)
        if tensors:
            operations.from_torch = Mock(side_effect=lambda value, **kwargs: replicated(value))
            operations.slice = Mock(side_effect=lambda *args, **kwargs: replicated(small, 0x6100))
            operations.pad = Mock(side_effect=lambda *args, **kwargs: replicated(small, 0x6200))
            history = replicated(small, 0x6000)
        else:
            operations.from_torch = Mock(side_effect=lambda *args, **kwargs: object())
            operations.slice = Mock(side_effect=lambda *args, **kwargs: object())
            operations.pad = Mock(side_effect=lambda *args, **kwargs: object())
            history = object()
        owned = []
        device = SimpleNamespace(operations=operations, mesh='mesh', closed=False, pending=None, max_drafts=15,
            proposal_capture=None, history=history, spare_history=None, owned=[], progress=None,
            history_rows=2048, block_rows=16, native_proposal_attention=False, validated_native_proposal_masks=set(),
            position=4096, proposal_calls=2, shared_weights=None, pool_slot=None, collectives=None, model=None,
            kv_history=None, fused_convolution=True, audit_digest=None,
            temporaries=lambda protected: (owned, lambda value: value),
            execute_proposal=Mock(return_value='outputs'), select_proposal=Mock(return_value=(1, 2, 3)))
        return device, operations

    def test_off_the_proposal_gets_no_observer_and_nothing_is_read_back(self):
        device, operations = self.device(tensors=False)
        log = Lines()
        with patch.dict(os.environ, {'QWEN_FAST_PROPOSAL_AUDIT': '0'}), patch('dflash_device.release_owned'), \
                patch('dflash_device.pindiag', log):
            self.assertEqual(DFlashDevice.propose(device, 17, 15), (1, 2, 3))
        device.execute_proposal.assert_called_once()
        self.assertNotIn('observe', device.execute_proposal.call_args.kwargs)
        self.assertEqual(operations.reads, 0)
        self.assertEqual(log.lines, [])
        self.assertEqual(device.proposal_calls, 3)

    def test_on_the_inputs_are_audited_in_order_and_the_observer_reaches_execute_proposal(self):
        device, operations = self.device(tensors=True)
        log = Lines()

        def execute(identifiers, history, mask, rope, **kwargs):
            kwargs['observe']('layer0.attn.output', replicated(torch.zeros(1, 1, 32, 8, dtype=torch.bfloat16)))
            return 'outputs'

        device.execute_proposal = Mock(side_effect=execute)
        with patch.dict(os.environ, {'QWEN_FAST_PROPOSAL_AUDIT': '1'}), patch('dflash_device.release_owned'), \
                patch('dflash_device.pindiag', log), patch('dflash_device.projection_links', return_value=1):
            self.assertEqual(DFlashDevice.propose(device, 17, 15), (1, 2, 3))
        self.assertEqual(log.stages(), ['input.identifiers', 'input.history-buffer', 'input.history-window', 'input.mask',
            'input.rope.q.cos', 'input.rope.q.sin', 'input.rope.k.cos', 'input.rope.k.sin', 'layer0.attn.output'])
        first_stage = next(index for index, line in enumerate(log.lines) if ' stage=' in line)
        self.assertIn('begin position=4096 history_rows=2048 block_rows=16', log.lines[0])
        self.assertTrue(all('[AUDIT]' in line for line in log.lines[:first_stage]), 'the listing precedes the stages')
        self.assertIn('shape=1x1x32x2080 differing=0 of 66560', next(line for line in log.lines if 'stage=input.mask' in line))
        self.assertIn('shape=1x1x2080x128', next(line for line in log.lines if 'stage=input.rope.k.cos' in line))
        summaries = [line for line in log.lines if 'summary stages=9 diverged=0 first_divergent=none' in line]
        self.assertEqual(len(summaries), 1, 'one summary, before the selector runs, not again from finally')
        self.assertLess(log.lines.index(summaries[0]), len(log.lines))
        for line in log.lines:
            self.assertLessEqual(len(line), LINE_BUDGET, line)
        self.assertEqual(device.proposal_calls, 3)

    def test_on_a_failure_inside_the_proposal_still_gets_its_summary(self):
        device, operations = self.device(tensors=True)
        log = Lines()

        def execute(identifiers, history, mask, rope, **kwargs):
            kwargs['observe']('layer1.mlp.gathered', replicated(torch.zeros(4, dtype=torch.bfloat16),
                chip1=torch.ones(4, dtype=torch.bfloat16)))
            raise RuntimeError('device fault')

        device.execute_proposal = Mock(side_effect=execute)
        with patch.dict(os.environ, {'QWEN_FAST_PROPOSAL_AUDIT': '1'}), patch('dflash_device.release_owned'), \
                patch('dflash_device.pindiag', log), patch('dflash_device.projection_links', return_value=1):
            with self.assertRaises(RuntimeError):
                DFlashDevice.propose(device, 17, 15)
        self.assertEqual(log.stages()[-1], 'layer1.mlp.gathered')
        self.assertIn('summary stages=9 diverged=1 first_divergent=layer1.mlp.gathered', log.lines[-1])

    def test_a_trace_replay_says_the_audit_needs_the_eager_proposal(self):
        device, operations = self.device(tensors=False)
        device.proposal_capture = SimpleNamespace(propose=Mock(return_value=(4, 5)))
        log = Lines()
        with patch.dict(os.environ, {'QWEN_FAST_PROPOSAL_AUDIT': '1'}), patch('dflash_device.pindiag', log):
            self.assertEqual(DFlashDevice.propose(device, 17, 15), (4, 5))
        self.assertEqual(len(log.lines), 1)
        self.assertIn('proposal is a trace replay', log.lines[0])
        self.assertIn('QWEN_FAST_EAGER_PROPOSAL=1', log.lines[0])
        device.execute_proposal.assert_not_called()

    def test_a_prepared_pending_proposal_is_finished_instead_of_redone(self):
        """QWEN_FAST_PIPELINED_PROPOSALS: prepare_device(seed) already ran this
        proposal's device work and left it pending - propose() must finish it
        (the deferred assertion/bookkeeping/release and the readback) rather than
        replay the trace and redo the copies a second time."""
        device, operations = self.device(tensors=False)
        device.proposal_capture = SimpleNamespace(has_pending=Mock(return_value=True),
            finish=Mock(return_value=(7, 8)), propose=Mock())
        log = Lines()
        with patch('dflash_device.pindiag', log):
            self.assertEqual(DFlashDevice.propose(device, 17, 15), (7, 8))
        device.proposal_capture.has_pending.assert_called_once_with(17)
        device.proposal_capture.finish.assert_called_once_with(15)
        device.proposal_capture.propose.assert_not_called()
        self.assertEqual(device.proposal_calls, 3)
        self.assertEqual(log.lines, [], 'a finished prewarm is expected, not a surprise the audit line reports')

    def test_no_pending_proposal_falls_back_to_the_normal_trace_replay(self):
        device, operations = self.device(tensors=False)
        device.proposal_capture = SimpleNamespace(has_pending=Mock(return_value=False),
            finish=Mock(), propose=Mock(return_value=(4, 5)))
        self.assertEqual(DFlashDevice.propose(device, 17, 15), (4, 5))
        device.proposal_capture.has_pending.assert_called_once_with(17)
        device.proposal_capture.propose.assert_called_once_with(17, 15)
        device.proposal_capture.finish.assert_not_called()

    def test_prepare_device_delegates_to_the_captured_trace(self):
        device, operations = self.device(tensors=False)
        device.proposal_capture = SimpleNamespace(prepare_device=Mock(return_value=True))
        self.assertTrue(DFlashDevice.prepare_device(device, 17))
        device.proposal_capture.prepare_device.assert_called_once_with(17)

    def test_prepare_device_is_a_no_op_without_a_captured_trace_mid_publication_or_closed(self):
        device, operations = self.device(tensors=False)
        # No captured trace at all - the eager QWEN_FAST_EAGER_PROPOSAL path.
        self.assertFalse(DFlashDevice.prepare_device(device, 17))
        device.proposal_capture = SimpleNamespace(prepare_device=Mock(return_value=True))
        device.pending = object()
        self.assertFalse(DFlashDevice.prepare_device(device, 17))
        device.proposal_capture.prepare_device.assert_not_called()
        device.pending = None
        device.closed = True
        self.assertFalse(DFlashDevice.prepare_device(device, 17))
        device.proposal_capture.prepare_device.assert_not_called()

    def test_prepare_device_rejects_an_unbounded_seed(self):
        device, operations = self.device(tensors=False)
        device.proposal_capture = SimpleNamespace(prepare_device=Mock(return_value=True))
        with self.assertRaises(ValueError):
            DFlashDevice.prepare_device(device, -1)
        device.proposal_capture.prepare_device.assert_not_called()


class ExecuteProposalTests(unittest.TestCase):
    """execute_proposal over a stand-in `self`: the observer is scoped per layer and
    the selector stages are named; without it the branches see no observer."""

    def device(self):
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            Topology=SimpleNamespace(Linear='linear'),
            experimental=SimpleNamespace(all_gather_async=Mock(side_effect=lambda *a, **k: object())))
        for name in ('reshape', 'pad', 'rms_norm', 'matmul', 'typecast', 'slice', 'MatmulMultiCoreReuseMultiCast1DProgramConfig'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))
        collectives = SimpleNamespace(get_and_cycle_ag_semaphore_handles=Mock(return_value='ag'),
                                      get_and_cycle_barrier_semaphore_handle=Mock(return_value='barrier'))
        model = SimpleNamespace(embd=Mock(side_effect=lambda *a, **k: object()))
        layers = [(dict(), dict(), object(), object()) for _ in range(5)]
        return SimpleNamespace(operations=operations, mesh='mesh', collectives=collectives, model=model, layers=layers,
            block_rows=16, live_query_qk=False, native_proposal_attention=False, kv_history=None,
            fused_convolution=False, progress=None, proposal_calls=0, kernel='kernel', final_norm='final',
            selector_projection='selector', validated_live_masks=set(), validated_native_proposal_masks=set(),
            convolution_checks=[], position=4096)

    def run_proposal(self, device, **extra):
        rope = {name: (object(), object()) for name in ('q', 'k')}
        with patch('dflash_device.projection_links', return_value=1), \
                patch('dflash_device.shared_head_candidates', return_value=['chunks']), \
                patch('dflash_device.execute_attention_branch') as attention, \
                patch('dflash_device.execute_mlp_branch') as mlp:
            def attend(*args, **kwargs):
                if 'observe' in kwargs:
                    kwargs['observe']('output', object())
                return object()

            def feed(*args, **kwargs):
                if 'observe' in kwargs:
                    kwargs['observe']('gathered', object())
                return dict(output=object())

            attention.side_effect, mlp.side_effect = attend, feed
            outputs = DFlashDevice.execute_proposal(device, object(), object(), object(), rope, context=2048,
                owned=[], retain=lambda value: value, stage=lambda name, **values: None, **extra)
        return outputs, attention, mlp

    def test_the_observer_is_scoped_per_layer_and_the_selector_stages_are_named_in_order(self):
        seen = []
        device = self.device()
        outputs, attention, mlp = self.run_proposal(device, observe=lambda name, value: seen.append(name))
        expected = ['embedding.gathered', 'embedding.padded']
        for layer in range(5):
            expected.extend(['layer%d.attn.output' % layer, 'layer%d.mlp.gathered' % layer])
        expected.extend(['selector.normalized', 'selector.projected-fp32', 'selector.features', 'head.input'])
        self.assertEqual(seen, expected)
        self.assertEqual(outputs.chunks, ['chunks'])
        self.assertEqual(attention.call_count, 5)
        self.assertEqual(mlp.call_count, 5)

    def test_without_the_observer_the_branches_are_called_exactly_as_before(self):
        device = self.device()
        outputs, attention, mlp = self.run_proposal(device)
        for call in [*attention.call_args_list, *mlp.call_args_list]:
            self.assertNotIn('observe', call.kwargs)
        self.assertEqual(outputs.chunks, ['chunks'])


class BranchStageTests(unittest.TestCase):
    """The stage names each branch reports, in the order the eager uncached path
    makes them; the collective reports the gathered partials from inside."""

    def operations(self):
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='row',
            DRAM_MEMORY_CONFIG='dram', MathFidelity=SimpleNamespace(HiFi4='hifi4'),
            Topology=SimpleNamespace(Linear='linear'),
            experimental=SimpleNamespace(rotary_embedding_hf=Mock(return_value=object()),
                                         all_gather_async=Mock(side_effect=lambda *a, **k: object())))
        for name in ('from_torch', 'ShardTensorToMesh', 'ReplicateTensorToMesh', 'WormholeComputeKernelConfig',
                     'MatmulMultiCoreReuseMultiCast1DProgramConfig', 'matmul', 'rms_norm', 'typecast', 'add',
                     'zeros_like', 'reshape', 'transpose', 'pad', 'slice', 'concat', 'silu', 'multiply'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))
        # The real collective checks the partial's geometry, so every projection carries one.
        operations.matmul = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(shape=(1, 1, 32, 5120), dtype='fp32'))
        return operations

    def test_attention_branch_on_the_uncached_native_path(self):
        from draft_attention_branch import execute_attention_branch, prepare_attention_branch

        operations, mesh, collectives = self.operations(), SimpleNamespace(shape=[1, 2]), Mock()
        weights = {'layers.0.self_attn.%s_proj.weight' % name: torch.ones(2, 2) for name in ('q', 'k', 'v', 'o')}
        weights.update({'layers.0.self_attn.%s_norm.weight' % name: torch.ones(128) for name in ('q', 'k')})
        convolution = {'layers.0.input_layernorm.weight': torch.ones(5120),
                       'layers.0.attention_conv.kernel_projection.weight': torch.ones(2, 2),
                       'layers.0.attention_conv.base_kernel': torch.ones(2, 2, 5120)}
        key_rows = 2080
        hidden = SimpleNamespace(shape=(1, 1, 32, 5120), dtype='bf16')
        history = SimpleNamespace(shape=(1, 1, key_rows, 5120), dtype='bf16')
        mask = SimpleNamespace(shape=(1, 1, 32, key_rows), dtype='bf16')
        rope = {'q': tuple(SimpleNamespace(shape=(1, 1, 32, 128), dtype='bf16') for _ in range(2)),
                'k': tuple(SimpleNamespace(shape=(1, 1, key_rows, 128), dtype='bf16') for _ in range(2))}
        for observing in (True, False):
            seen = []
            with patch('draft_attention_branch.grouped_causal_convolution', side_effect=lambda *a, **k: object()), \
                    patch('feature_collective.projection_links', return_value=1), \
                    patch('draft_attention_branch.concatenate_query_heads', return_value=object()), \
                    patch('draft_attention_branch.split_projected_heads',
                          side_effect=lambda *a, **k: {name: object() for name in ('q', 'k', 'v')}), \
                    patch('dflash_t16_native_attention.attention', return_value=object()), \
                    patch('dflash_t16_native_scope.require_active', return_value=None):
                parameters = prepare_attention_branch(operations, mesh, weights, convolution, lambda value: value,
                    native_head_layout=True, block_rows=16)
                parameters.update(native_proposal_attention=True)
                execute_attention_branch(operations, mesh, collectives, hidden, history, mask, rope, lambda value: value,
                    parameters=parameters, context=2048, native_proposal_mask_validated=True,
                    **(dict(observe=lambda name, value: seen.append(name)) if observing else {}))
            self.assertEqual(seen, ['normalized', 'conv-kernels', 'conv-in', 'keys', 'gathered', 'reduced',
                                    'conv-out', 'output'] if observing else [])

    def test_mlp_branch(self):
        from draft_mlp_branch import execute_mlp_branch

        operations, mesh, collectives = self.operations(), SimpleNamespace(shape=[1, 2]), Mock()
        weights, convolution = object(), object()
        parameters = dict(operations=operations, mesh=mesh, source_weights=weights, source_convolution=convolution,
            kernel='kernel', device_norm=object(), device_conv=object(), bases=[object() for _ in range(4)],
            device_projections=[object() for _ in range(3)], shards=None, norm_weight=None, conv_weight=None,
            base_weight=None)
        hidden = SimpleNamespace(shape=(1, 1, 32, 5120), dtype='bf16')
        for observing in (True, False):
            seen = []
            with patch('draft_mlp_branch.grouped_causal_convolution', side_effect=lambda *a, **k: object()), \
                    patch('draft_mlp_branch.swiglu_device', return_value=object()), \
                    patch('feature_collective.projection_links', return_value=1):
                result = execute_mlp_branch(operations, mesh, collectives, hidden, weights, convolution, lambda value: value,
                    parameters=parameters, trace_safe=True,
                    **(dict(observe=lambda name, value: seen.append(name)) if observing else {}))
            self.assertEqual(seen, ['normalized', 'conv-kernels', 'conv-in', 'gathered', 'reduced', 'conv-out', 'output']
                             if observing else [])
            self.assertIn('output', result)

    def test_gather_add_reports_the_gathered_partials_before_the_add_only_when_asked(self):
        from feature_collective import gather_add_projection

        operations = self.operations()
        gathered, output = object(), object()
        operations.experimental.all_gather_async = Mock(return_value=gathered)
        operations.add = Mock(return_value=output)
        operations.synchronize_device = Mock()
        value = SimpleNamespace(shape=(1, 1, 32, 5120), dtype='fp32')
        seen = []
        with patch('feature_collective.projection_links', return_value=1):
            result = gather_add_projection(operations, SimpleNamespace(shape=[1, 2]), Mock(), value,
                retain_temporaries=lambda tensor: tensor,
                observe=lambda name, tensor: seen.append((name, tensor, operations.add.call_count)))
            self.assertIs(result, output)
            self.assertEqual(seen, [('gathered', gathered, 0)])
            gather_add_projection(operations, SimpleNamespace(shape=[1, 2]), Mock(), value,
                retain_temporaries=lambda tensor: tensor)
            self.assertEqual(len(seen), 1)


if __name__ == '__main__':
    unittest.main()
