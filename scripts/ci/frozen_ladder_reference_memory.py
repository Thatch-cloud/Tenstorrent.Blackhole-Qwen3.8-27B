"""Bound eager audit temporaries per draft layer without altering captured execution."""

from contextlib import contextmanager
from unittest.mock import patch

from dspark_history import TensorScope


def bounded_reference(execute, device, inputs, history, retain, evidence):
    backend = device.proposal_layer

    def layer(operations, mesh, collectives, noise, cached, weights, tables, mask, live,
            outer_retain, **options):
        scope = TensorScope(operations, [noise, *cached, *weights.values(), *tables, mask, live])
        output = None
        try:
            result = backend(operations, mesh, collectives, noise, cached, weights, tables,
                mask, live, scope.retain, **options)
            operations.synchronize_device(mesh)
            output = outer_retain(result['finish']['output'])
            evidence['layers'] += 1
            return dict(finish=dict(output=output))
        finally:
            scope.release(keep=() if output is None else (output,))

    with patch.object(device, 'proposal_layer', layer):
        return execute(device, inputs, history, retain)


@contextmanager
def audit_scope():
    import dspark_prepared_proposal as prepared
    execute = prepared.execute
    propose = prepared.PreparedDSparkProposal.propose
    evidence = dict(calls=0, layers=0, restored=False, captured_execution_changed=False)

    def audited(instance, *args, **kwargs):
        if not instance.audit:
            return propose(instance, *args, **kwargs)

        def bounded(device, inputs, history, retain):
            evidence['calls'] += 1
            return bounded_reference(execute, device, inputs, history, retain, evidence)

        with patch.object(prepared, 'execute', bounded):
            return propose(instance, *args, **kwargs)

    try:
        with patch.object(prepared.PreparedDSparkProposal, 'propose', audited):
            yield evidence
    finally:
        evidence['restored'] = prepared.execute is execute and prepared.PreparedDSparkProposal.propose is propose
