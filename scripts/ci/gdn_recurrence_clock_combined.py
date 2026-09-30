"""Preallocated all-layer recurrence diagnostics inside one complete T16 request."""

from contextlib import contextmanager
import hashlib
from unittest.mock import patch

from gdn_recurrence_clock import instrument
from gdn_recurrence_clock_gate import KERNEL, REPORT_SHA256


class CombinedRecurrenceCapture:
    def __init__(self, operations, model, captures, candidate, admission):
        self.indices = tuple(index for index, layer in enumerate(model.layers) if not layer.is_full_attention)
        if (len(model.layers) != 64 or len(self.indices) != 48 or len(captures) != 48
                or admission.get('report_sha256') != REPORT_SHA256 or admission.get('kernel') != KERNEL):
            raise ValueError('All 48 GDN layers and source-admitted samples required')
        self.operations, self.model, self.candidate = operations, model, candidate
        self.captures = tuple(captures)
        self.builds, self.engines, self.records = [], [], []
        self.active = self.failed = False
        identities = set()
        for capture in self.captures:
            if (capture.mesh is not model.mesh_device or capture.pending or capture.records
                    or capture.token != 8 or len(capture.addresses()) != 2):
                raise ValueError('Fresh middle-token captures on both target chips required')
            for chip, address in enumerate(capture.addresses()):
                if (chip, address) in identities:
                    raise ValueError('Diagnostic layers must not alias')
                identities.add((chip, address))

    def build(self, original, pipeline, operations, mesh, tensors, *, root):
        if (not self.active or self.failed or operations is not self.operations
                or mesh is not self.model.mesh_device or len(tensors) != 11 or self.records):
            raise ValueError('Only owned pre-capture three-stage programs may be instrumented')
        ordinal = len(self.builds) % 48
        capture = self.captures[ordinal]
        constructed = []

        def recurrence(ttnn, target_mesh, shards, kernels):
            if constructed or ttnn is not operations or target_mesh is not mesh:
                raise ValueError('One recurrence per three-stage build required')
            before = kernels['recurrence']['compute']
            after = instrument(before)
            if (hashlib.sha256(before.encode()).hexdigest() != KERNEL['control_sha256']
                    or hashlib.sha256(after.encode()).hexdigest() != KERNEL['candidate_sha256']):
                raise ValueError('Runtime recurrence differs from simulator')
            modified = {stage: dict(parts) for stage, parts in kernels.items()}
            modified['recurrence']['compute'] = after
            program = self.candidate.build_recurrence(ttnn, mesh,
                shards + [operations.get_device_tensors(capture.buffer)], modified, capture.token)
            constructed.append(program)
            return program

        with patch.object(pipeline, 'build_recurrence', recurrence):
            programs = original(operations, mesh, tensors, root=root)
        if len(programs) != 3 or len(constructed) != 1 or programs[1][1] is not constructed[0]:
            raise ValueError('Unchanged three-stage program order required')
        self.builds.append(dict(layer=self.indices[ordinal], ordinal=ordinal, kernel=dict(KERNEL)))
        return (programs[0], (list(programs[1][0]) + [capture.buffer], programs[1][1]), programs[2])

    def verify(self, engine, ticket, execute):
        if not self.active or self.failed or self.engines != [engine]:
            raise ValueError('Owned active verifier required')
        if len(ticket.tokens) != 16 or len(self.records) == 2:
            return execute()
        if not self.builds or len(self.builds) % 48:
            raise ValueError('Complete all-layer recurrence binding required')
        try:
            for capture in self.captures:
                capture.prepare()
            result = execute()
            layers = [dict(layer=layer, capture=capture.collect(f'verify-{len(self.records)}-layer-{layer}'))
                for layer, capture in zip(self.indices, self.captures, strict=True)]
            self.records.append(dict(position=ticket.position, rows=16, layers=layers))
            return result
        except BaseException:
            self.failed = True
            for capture in self.captures:
                capture.pending = False
            raise

    def assert_releasable(self):
        if self.active or any(getattr(engine, 'phase', None) != 'closed' for engine in self.engines):
            raise ValueError('Release verifier traces before diagnostic sample storage')

    def summary(self):
        self.assert_releasable()
        if self.failed or len(self.engines) != 1 or len(self.records) != 2 or not self.builds or len(self.builds) % 48:
            raise ValueError('Complete two-replay all-layer diagnostic required')
        return dict(diagnostic_only=True, committed_tg=None, report_sha256=REPORT_SHA256,
            layers=list(self.indices), builds=list(self.builds), records=list(self.records),
            sampled_verifier_replays=2, samples_releasable=True)

    @contextmanager
    def install(self, pipeline, engine_class):
        if self.active or self.failed or self.engines or self.builds or self.records:
            raise ValueError('Fresh diagnostic scope required')
        original_build = pipeline.build
        original_init, original_verify = engine_class.__init__, engine_class.verify

        def build(*args, **kwargs):
            return self.build(original_build, pipeline, *args, **kwargs)

        def initialize(engine, model, *args, **kwargs):
            if model is not self.model or self.engines:
                raise ValueError('One owned verifier required')
            self.engines.append(engine)
            original_init(engine, model, *args, **kwargs)

        def verify(engine, ticket):
            return self.verify(engine, ticket, lambda: original_verify(engine, ticket))

        self.active = True
        try:
            with patch.object(pipeline, 'build', build), patch.object(engine_class, '__init__', initialize), \
                    patch.object(engine_class, 'verify', verify):
                yield self
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False
