"""Diagnostic-only shadow B1 SDPA on the exact query and KV used by a replay."""

import hashlib
from pathlib import Path

from gdn_multitoken_conv import release_owned


class AttentionMismatch(AssertionError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"Folded attention differs from native B1 at position {evidence['position']}, "
                         f"attention layer {evidence['attention_index']}, chip {evidence['chip']}")


def tensor_summary(value):
    import torch

    raw = value.contiguous().view(torch.uint8).numpy().tobytes()
    finite = torch.isfinite(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest(),
                finite=bool(finite.all()), maximum=float(value[finite].abs().max()) if finite.any() else None)


class AttentionReplayAudit:
    def __init__(self, operations, oracle, *, pages=None, output_directory=None):
        self.operations, self.oracle = operations, oracle
        self.pages, self.output_directory = pages, output_directory
        self.records, self.owned = [], []
        self.closed = False

    def capture(self, query, keys, values, candidate, **kwargs):
        if self.closed or len(self.records) >= 16:
            raise ValueError('One captured forward of at most sixteen attention layers required')
        reference = self.oracle(query, keys, values, **kwargs)
        self.owned.append(reference)
        snapshots = []
        for tensor in (query, candidate):
            snapshots.append(self.operations.clone(tensor, memory_config=self.operations.DRAM_MEMORY_CONFIG))
            self.owned.append(snapshots[-1])
        self.records.append(dict(query=snapshots[0], candidate=snapshots[1], reference=reference,
                                 keys=keys, values=values, scale=kwargs['scale'],
                                 program_config=str(kwargs.get('program_config'))))

    def check(self, position, tokens):
        import torch

        if self.closed or len(self.records) != 16:
            raise ValueError('All sixteen captured attention layers required before checking')
        for index, record in enumerate(self.records):
            actual_parts = self.operations.get_device_tensors(record['candidate'])
            reference_parts = self.operations.get_device_tensors(record['reference'])
            if len(actual_parts) != 2 or len(reference_parts) != 2:
                raise AssertionError('Two independent attention replicas required')
            for chip, (actual, expected) in enumerate(zip(actual_parts, reference_parts, strict=True)):
                actual, expected = self.operations.to_torch(actual), self.operations.to_torch(expected)
                if not torch.equal(actual, expected):
                    mismatches = (actual != expected).nonzero()
                    coordinates = [tuple(entry) for entry in mismatches[:8].tolist()]
                    query = self.operations.to_torch(self.operations.get_device_tensors(record['query'])[chip])
                    evidence = dict(kind='attention-output-mismatch', position=position,
                        input_tokens=list(tokens), attention_index=index, chip=chip,
                        differing_elements=len(mismatches), total_elements=actual.numel(),
                        max_absolute_error=float((actual.float() - expected.float()).abs().max()),
                        coordinates=[list(entry) for entry in coordinates],
                        actual_values=[float(actual[entry]) for entry in coordinates],
                        expected_values=[float(expected[entry]) for entry in coordinates],
                        query=tensor_summary(query), actual=tensor_summary(actual), expected=tensor_summary(expected),
                        scale=record['scale'], program_config=record['program_config'])
                    if self.output_directory is not None:
                        evidence['fixture'] = self.save_fixture(record, chip, query, actual, expected, position)
                    raise AttentionMismatch(evidence)

    def save_fixture(self, record, chip, query, actual, expected, position):
        import torch

        if self.pages is None or self.pages.ndim != 2 or self.pages.shape[0] != 1:
            raise ValueError('One complete bounded physical page table required for diagnostic export')
        page_count = int(self.pages.max()) + 1
        if not 0 < page_count <= 12 or int(self.pages.min()) < 0:
            raise ValueError('Diagnostic export is bounded to the twelve short-context pages')
        tensors = dict(query=query, actual=actual, expected=expected, pages=self.pages.cpu().clone(),
                       position=position, scale=record['scale'], chip=chip)
        for name in ('keys', 'values'):
            source = record[name]
            end = (page_count, *tuple(source.shape)[1:])
            selected = self.operations.slice(source, (0, 0, 0, 0), end,
                                             memory_config=self.operations.DRAM_MEMORY_CONFIG)
            try:
                tensors[name] = self.operations.to_torch(self.operations.get_device_tensors(selected)[chip]).clone()
            finally:
                self.operations.deallocate(selected)
        directory = Path(self.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'real-query.pt'
        torch.save(tensors, path)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    keys=tensor_summary(tensors['keys']), values=tensor_summary(tensors['values']))

    def close(self):
        if not self.closed:
            release_owned(self.operations, self.owned)
            self.owned.clear()
            self.records.clear()
            self.closed = True
