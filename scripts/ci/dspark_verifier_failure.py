"""Failure-only first-block comparison; never resumes or qualifies the failed request."""

from gdn_multitoken_conv import addresses, release_owned
from target_features import LayerOutputCapture
from verifier_inputs import stage_inputs
from dspark_projection import tensor_digest


def compare_first_block(engine, decode, snapshot, initial_state, traced_feature, *, tap, chip):
    import torch

    ticket = engine.pending
    if engine.session.committed_blocks != 0 or ticket.position != engine.position:
        raise ValueError('Failure comparison requires the initial request frontier')
    operations, model = engine.operations, engine.model
    layers = tuple(range(tap + 1))

    def restore():
        engine.restore_initial()
        operations.synchronize_device(engine.mesh)
        if snapshot() != initial_state:
            raise AssertionError('Diagnostic restore does not match independently saved prefilled state')

    def capture(operation):
        observer = LayerOutputCapture(model, layers,
            snapshot=lambda value: operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG),
            release=operations.deallocate,
            storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))
        try:
            with observer.capture():
                operation()
            return [[operations.to_torch(shard).clone() for shard in operations.get_device_tensors(value)]
                for value in observer.outputs()]
        finally:
            observer.close()

    fixture = None
    try:
        restore()
        native = capture(lambda: decode(ticket.tokens[0], ticket.position, False))
        restore()
        bucket = engine.buckets[engine.pending_key]
        fixture = engine.fixture(len(ticket.tokens), bucket['checkpoints'], retain=False, position=ticket.position)
        stage_inputs(fixture, ticket.tokens, ticket.position)

        def run_batch():
            output = fixture.run(sharded_logits=engine.sampler is not None)
            operations.synchronize_device(engine.mesh)
            release_owned(operations, [output])

        batched = capture(run_batch)
        records = []
        for layer, (expected, actual) in enumerate(zip(native, batched, strict=True)):
            if len(expected) != 2 or len(actual) != 2:
                raise AssertionError('Both physical chips required in failure comparison')
            for shard_index in range(2):
                reference = expected[shard_index][..., :1, :]
                value = actual[shard_index][..., :1, :]
                records.append(dict(layer=layer, chip=shard_index, exact=torch.equal(reference, value),
                    native_sha256=tensor_digest(reference), eager_batch_sha256=tensor_digest(value),
                    max_abs=str(float((reference.float() - value.float()).abs().max()))))
        return dict(scope='Failure-only native T1 versus identical-anchor eager batch; no throughput qualification',
            layers=records, traced_equals_eager_batch=torch.equal(traced_feature, batched[tap][chip][..., :1, :]),
            traced_equals_native=torch.equal(traced_feature, native[tap][chip][..., :1, :]))
    finally:
        if fixture is not None:
            fixture.close()
        restore()
