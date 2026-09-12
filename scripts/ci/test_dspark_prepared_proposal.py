from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import dspark_prepared_proposal as prepared

EXECUTE = prepared.execute


class FakeTensor:
    def __init__(self, value, dtype=None, layout='tile'):
        self.value = value.clone()
        self.dtype = value.dtype if dtype is None else dtype
        self.layout = layout


class PreparedProposalTests(unittest.TestCase):
    def test_deferred_capture_does_not_allocate_inputs_after_trace(self):
        proposal = prepared.PreparedDSparkProposal(self.device, 10, defer_capture=True)
        self.capture.assert_not_called()
        allocated = self.operations.from_torch.call_count
        with self.assertRaisesRegex(ValueError, 'captured'):
            proposal.propose(20, 7)
        proposal.capture()
        self.assertEqual(self.operations.from_torch.call_count, allocated)
        self.capture.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'One capture'):
            proposal.capture()
        proposal.close()

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.operations = SimpleNamespace(bfloat16=torch.bfloat16, float32=torch.float32, uint32=torch.int64,
            ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=Mock(side_effect=lambda value, **options: FakeTensor(value, options.get('dtype'), options.get('layout'))),
            clone=lambda value, **options: FakeTensor(value.value), ReplicateTensorToMesh=lambda mesh: mesh,
            copy_host_to_device_tensor=Mock(side_effect=lambda source, destination: destination.value.copy_(source.value)),
            copy=Mock(side_effect=lambda source, destination: destination.value.copy_(source.value)),
            get_device_tensors=lambda value: (value, value), to_torch=lambda value: value.value,
            synchronize_device=Mock(), release_trace=Mock())
        self.bank = tuple(tuple(FakeTensor(torch.zeros(1, 4, 64, 128, dtype=torch.bfloat16))
            for operand in range(2)) for layer in range(5))
        self.device = SimpleNamespace(closed=False, position=32, max_drafts=15, operations=self.operations,
            mesh=object(), parameters={}, layer_weights=[], predecessor=object(), successor=object(),
            target=SimpleNamespace(lm_head_weight=object()), history=SimpleNamespace(capacity=64,
                layers=self.bank, spare_layers=self.bank, pending=None),
            rotary=SimpleNamespace(tables=lambda position, rows: (torch.ones(1, 1, rows, 128, dtype=torch.bfloat16),
                torch.zeros(1, 1, rows, 128, dtype=torch.bfloat16))))
        self.scopes = []
        self.stack.enter_context(patch.object(prepared, 'TensorScope', side_effect=self.scope))
        self.stack.enter_context(patch.object(prepared, 'addresses', side_effect=lambda operations, value: (id(value), id(value))))
        self.run = self.stack.enter_context(patch.object(prepared, 'execute', side_effect=self.execute))
        self.capture = self.stack.enter_context(patch.object(prepared, 'capture_operation', side_effect=self.capture_operation))
        self.operations.execute_trace = Mock(side_effect=self.replay)
        self.stale = False

    def scope(self, operations, borrowed):
        scope = SimpleNamespace(retain=lambda value: value, release=Mock())
        self.scopes.append(scope)
        return scope

    def execute(self, device, inputs, history, retain):
        anchor = int(inputs['anchor'].value.item())
        tokens = torch.arange(anchor + 1, anchor + 16).reshape(1, 1, 15, 1)
        return dict(normalized=FakeTensor(torch.full((1, 1, 32, 5120), anchor, dtype=torch.bfloat16)),
            logits=FakeTensor(torch.full((1, 1, 15, 32), anchor, dtype=torch.float32)), tokens=FakeTensor(tokens))

    def capture_operation(self, operations, mesh, operation):
        self.operation = operation
        self.outputs = operation()
        return 71, self.outputs

    def replay(self, *args, **options):
        if not self.stale:
            current = self.operation()
            for name in current:
                self.outputs[name].value.copy_(current[name].value)

    def test_changing_anchor_and_banks_update_stable_inputs_and_pack_all_rows(self):
        proposal = prepared.PreparedDSparkProposal(self.device, 10)
        inputs = tuple(id(value) for value in proposal.input_values())
        self.device.history.layers = tuple(tuple(FakeTensor(torch.full((1, 4, 64, 128), 5, dtype=torch.bfloat16))
            for operand in range(2)) for layer in range(5))
        self.device.position = 47
        self.assertEqual(proposal.propose(20, 7), tuple(range(21, 28)))
        self.assertEqual(inputs, tuple(id(value) for value in proposal.input_values()))
        self.assertEqual(self.operations.copy.call_count, 10)
        self.assertEqual(self.operations.copy_host_to_device_tensor.call_count, 6)
        self.assertEqual(len(self.scopes), 3)
        self.assertEqual(proposal.checks, [])
        self.capture.assert_called_once()
        proposal.close()
        proposal.close()
        self.operations.release_trace.assert_called_once_with(self.device.mesh, 71)
        self.assertTrue(all(scope.release.call_count == 1 for scope in self.scopes))

    def test_audit_compares_complete_fixed_layout_outputs_on_both_replicas(self):
        proposal = prepared.PreparedDSparkProposal(self.device, 10, audit=True)
        self.assertEqual(proposal.propose(20, 15), tuple(range(21, 36)))
        self.assertEqual(proposal.checks, [dict(position=32, tensors=6, exact=True)])
        self.stale = True
        with self.assertRaisesRegex(AssertionError, 'Changing-input'):
            proposal.propose(30, 15)
        proposal.close()

    def test_input_rebinding_pending_history_and_unqualified_count_fail(self):
        proposal = prepared.PreparedDSparkProposal(self.device, 10)
        for count in (True, 0, 16):
            with self.subTest(count=count), self.assertRaises(ValueError):
                proposal.propose(20, count)
        self.device.history.pending = object()
        with self.assertRaises(ValueError):
            proposal.propose(20, 15)
        self.device.history.pending = None
        proposal.inputs['anchor'] = FakeTensor(torch.zeros(1, 1, 1, 1, dtype=torch.int64))
        with self.assertRaisesRegex(AssertionError, 'addresses'):
            proposal.propose(20, 15)
        proposal.close()

    def test_constructor_failure_releases_owned_inputs_and_never_frees_borrowed_device(self):
        self.run.side_effect = RuntimeError('injected warm failure')
        with self.assertRaisesRegex(RuntimeError, 'warm failure'):
            prepared.PreparedDSparkProposal(self.device, 10)
        self.assertTrue(all(scope.release.called for scope in self.scopes))
        self.operations.release_trace.assert_not_called()
        self.assertFalse(self.device.closed)

    def test_all_five_layers_use_physical_capacity_but_pass_actual_absolute_rotary_tables(self):
        inputs = {name: object() for name in ('identifiers', 'anchor', 'cosine', 'sine', 'mask', 'live')}
        self.device.collectives = object()
        self.device.layer_weights = [object() for index in range(5)]
        self.device.parameters = {'norm.weight': object()}
        retained = []
        with patch.object(prepared, 'noise_embeddings', return_value='noise') as noise, \
                patch.object(prepared, 'layer', side_effect=[dict(finish=dict(output=index)) for index in range(5)]) as layer, \
                patch.object(prepared, 'norm', return_value='normalized') as normalize, \
                patch.object(prepared, 'shared_head_logits', return_value='logits'), \
                patch.object(prepared, 'markov', return_value='records') as markov, \
                patch.object(prepared, 'pack_tokens', return_value='tokens'):
            result = EXECUTE(self.device, inputs, self.bank, retained.append)
        self.assertEqual(result, dict(normalized='normalized', logits='logits', tokens='tokens'))
        self.assertEqual(layer.call_count, 5)
        for index, call in enumerate(layer.call_args_list):
            self.assertEqual(call.kwargs, dict(position=64, proposals=15, mask_validated=True))
            self.assertEqual(call.args[3], 'noise' if index == 0 else index - 1)
            self.assertIs(call.args[4], self.bank[index])
            self.assertEqual(call.args[6], (inputs['cosine'], inputs['sine']))
            self.assertIs(call.args[7], inputs['mask'])
        self.assertEqual(noise.call_args.kwargs['proposals'], 15)
        self.assertEqual(normalize.call_args.args[1], 4)
        self.assertEqual(markov.call_args.args[1:3], (inputs['anchor'], 'logits'))


if __name__ == '__main__':
    unittest.main()
