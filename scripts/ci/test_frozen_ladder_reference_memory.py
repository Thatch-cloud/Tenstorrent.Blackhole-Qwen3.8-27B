from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from frozen_ladder_reference_memory import audit_scope, bounded_reference


class ReferenceMemoryTests(unittest.TestCase):
    def test_only_audit_eager_execution_is_wrapped(self):
        execute = Mock(return_value='original')
        module = SimpleNamespace(execute=execute)

        class Prepared:
            def __init__(self, audit):
                self.audit = audit

            def propose(self):
                return module.execute('device', 'inputs', 'history', 'retain') if self.audit else 'trace-only'

        module.PreparedDSparkProposal = Prepared
        original = Prepared.propose
        with patch.dict(sys.modules, dspark_prepared_proposal=module), \
                patch('frozen_ladder_reference_memory.bounded_reference', return_value='bounded') as bounded:
            with audit_scope() as evidence:
                self.assertEqual(Prepared(False).propose(), 'trace-only')
                bounded.assert_not_called()
                self.assertEqual(Prepared(True).propose(), 'bounded')
                self.assertIs(module.execute, execute)
            self.assertIs(Prepared.propose, original)
            self.assertTrue(evidence['restored'])
            self.assertFalse(evidence['captured_execution_changed'])
            self.assertEqual(evidence['calls'], 1)
            bounded.assert_called_once_with(execute, 'device', 'inputs', 'history', 'retain', evidence)

    def test_layer_temporaries_released_but_outputs_and_borrowed_inputs_preserved(self):
        released, scratch, outputs, events = [], [], [], []
        operations = SimpleNamespace(
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2,
            synchronize_device=lambda mesh: events.append('sync'),
            deallocate=lambda value: (events.append('free'), released.append(value)))
        borrowed = [object() for unused in range(7)]

        def backend(operations, mesh, collectives, noise, cached, weights, tables, mask, live, retain, **options):
            self.assertTrue(all(value in released for value in scratch))
            for value in [noise, *cached, *weights.values(), *tables, mask, live]:
                retain(value)
            scratch.append(retain(object()))
            output = retain(SimpleNamespace(value=noise.value + 1))
            outputs.append(output)
            return dict(finish=dict(output=output), temporary=scratch[-1])

        device = SimpleNamespace(proposal_layer=backend)

        def execute(device, inputs, history, retain):
            hidden = SimpleNamespace(value=0)
            for unused in range(5):
                hidden = device.proposal_layer(operations, 'mesh', None, hidden,
                    borrowed[:2], {'weight': borrowed[2]}, borrowed[3:5], borrowed[5], borrowed[6], retain,
                    position=131328, proposals=15)['finish']['output']
            return hidden

        retained = []
        evidence = dict(layers=0)
        result = bounded_reference(execute, device, {}, (), lambda value: retained.append(value) or value, evidence)
        self.assertEqual(result.value, 5)
        self.assertEqual(evidence['layers'], 5)
        self.assertEqual(released, scratch)
        self.assertEqual(retained, outputs)
        self.assertEqual(events, ['sync', 'free'] * 5)
        self.assertIs(device.proposal_layer, backend)

    def test_layer_error_releases_owned_storage_and_restores_backend(self):
        temporary = object()
        operations = SimpleNamespace(
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2,
            synchronize_device=Mock(), deallocate=Mock())

        def backend(*args, **kwargs):
            args[9](temporary)
            raise RuntimeError('layer failed')

        device = SimpleNamespace(proposal_layer=backend)

        def execute(device, inputs, history, retain):
            return device.proposal_layer(operations, None, None, object(), (), {}, (), object(), object(), retain)

        with self.assertRaisesRegex(RuntimeError, 'layer failed'):
            bounded_reference(execute, device, {}, (), lambda value: value, dict(layers=0))
        operations.deallocate.assert_called_once_with(temporary)
        self.assertIs(device.proposal_layer, backend)


if __name__ == '__main__':
    unittest.main()
