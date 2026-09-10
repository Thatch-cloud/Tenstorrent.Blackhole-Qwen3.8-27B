"""Read-only full-cache lifetime checks for the instrumented DSpark request."""

from dspark_projection import tensor_digest


class HistoryMismatch(AssertionError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"DSpark history changed during {evidence['phase']}: {evidence}")


class AuditedHistoryDrafter:
    def __init__(self, operations, drafter, records):
        self.operations, self.drafter, self.records = operations, drafter, records
        self.expected = self.snapshot(drafter.history.layers, drafter.position, 'initial_history')
        self.prepared = None

    def __getattr__(self, name):
        return getattr(self.drafter, name)

    def snapshot(self, layers, position, phase):
        import torch

        values = []
        if len(layers) != 5 or any(len(pair) != 2 for pair in layers):
            raise ValueError('Every learned layer and both history operands required')
        for layer, pair in enumerate(layers):
            for operand, value in zip(('key', 'value'), pair, strict=True):
                shards = self.operations.get_device_tensors(value)
                if len(shards) != 2:
                    raise ValueError('Both physical history shards required')
                for chip, shard in enumerate(shards):
                    actual = self.operations.to_torch(shard).clone()
                    capacity = getattr(self.drafter.history, 'capacity', position)
                    if tuple(actual.shape) != (1, 4, capacity, 128) or capacity < position or not torch.isfinite(actual).all():
                        raise HistoryMismatch(dict(phase=phase, layer=layer, operand=operand, chip=chip,
                            position=position, shape=list(actual.shape), finite=bool(torch.isfinite(actual).all())))
                    values.append(actual[..., :position, :])
        return tuple(values)

    def compare(self, actual, expected, phase, position):
        import torch

        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            if not torch.equal(value, reference):
                raise HistoryMismatch(dict(phase=phase, position=position, layer=index // 4,
                    operand=('key', 'value')[(index // 2) % 2], chip=index % 2,
                    actual_sha256=tensor_digest(value), expected_sha256=tensor_digest(reference)))
        self.records.append(dict(phase=phase, position=position, tensor_checks=len(actual), exact=True))

    def check_current(self, phase):
        self.compare(self.snapshot(self.drafter.history.layers, self.position, phase),
            self.expected, phase, self.position)

    def propose(self, anchor, count):
        self.check_current('before_proposal_after_capture_or_publication')
        result = self.drafter.propose(anchor, count)
        self.check_current('after_proposal')
        return result

    def prepare_publication(self, features, prefix, *, position):
        self.check_current('after_target_verifier')
        publication = self.drafter.prepare_publication(features, prefix, position=position)
        try:
            self.check_current('after_history_projection')
            prepared = self.snapshot(publication.layers, position + prefix, 'prepared_history')
            self.compare(tuple(value[..., :position, :] for value in prepared), self.expected,
                'prepared_history_preserves_committed_prefix', position)
            self.prepared = prepared
            return publication
        except BaseException:
            self.drafter.discard_publication(publication)
            raise

    def commit_publication(self, publication):
        position = publication.position + publication.prefix
        self.compare(self.snapshot(publication.layers, position, 'after_target_publication'),
            self.prepared, 'after_target_publication', position)
        self.drafter.commit_publication(publication)
        self.expected, self.prepared = self.prepared, None

    def discard_publication(self, publication):
        self.drafter.discard_publication(publication)
        self.prepared = None
