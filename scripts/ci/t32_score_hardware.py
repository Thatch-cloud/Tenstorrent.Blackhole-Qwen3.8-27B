"""Opt-in scoped execution of the simulator-covered T32 score body on allocated hardware."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from dspark_markov_score_layout import execute as fused
from dspark_projection import require_tensor
from dspark_t32_prepared import TracedDSparkDevice
from projection_link_policy import validate
from t32_hardware_kernel import KERNEL_DIRECTORY, PATCHED, RUNTIME
from t32_score_composition import audit as composition_audit


_ACTIVE = ContextVar('qwen_t32_hardware_score', default=None)


def mathematical_body(source):
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'execute']
    if len(functions) != 1:
        raise ValueError('One explicit feedback execution function required')
    return ast.dump(ast.Module(body=functions[0].body[1:], type_ignores=[]), include_attributes=False)


def sources(directory):
    directory = Path(directory)
    original = (directory / 'dspark_t32_score_layout.py').read_text()
    candidate = Path(__file__).read_text()
    if mathematical_body(original) != mathematical_body(candidate):
        raise ValueError('Hardware feedback math must exactly match the simulator-covered body')
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ('t32_score_hardware.py', Path(__file__)), ('dspark_t32_score_layout.py', directory / 'dspark_t32_score_layout.py'))}


def environment():
    if (os.environ.get('QWEN_T32_FUSED_SCORE_HARDWARE') != '1'
            or any(os.environ.get(name) != '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))
            or any(os.environ.get(name) == '1' for name in ('QWEN_SIM_ONLY', 'QWEN_CONTEXT_LADDER_SIM'))
            or any(os.environ.get(name) for name in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH'))):
        raise ValueError('Explicit allocated fused-score hardware experiment required')
    links = validate(os.environ)
    if links['backend'] != 'hardware':
        raise ValueError('Physical hardware link admission required')
    return links


def native_sources(root):
    root = Path(root)
    expected = dict(RUNTIME, **{str(KERNEL_DIRECTORY / name): checksum for name, checksum in PATCHED.items()})
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError('Installed simulator-qualified T32 arithmetic and hardware runtime required')
    return actual


def require_active(mesh=None):
    state = _ACTIVE.get()
    if state is None or environment() != state.links or (mesh is not None and mesh is not state.mesh):
        raise ValueError('Active owned two-chip fused-score hardware scope required')
    return state


@contextmanager
def hardware_scope(root, directory, proposal, score, mesh, installation):
    if _ACTIVE.get() is not None or list(mesh.shape) != [1, 2]:
        raise ValueError('One exclusive two-chip fused-score hardware scope required')
    links = environment()
    composition = composition_audit(directory, proposal, score)
    if (installation.get('proposal', {}).get('score_composition') != composition
            or installation.get('runtime') != RUNTIME or installation.get('patched') != PATCHED
            or installation.get('links') != links or installation.get('full_request_qualified') is not False):
        raise ValueError('Matching installed arithmetic and source-bound composition required')
    before, native = sources(directory), native_sources(root)
    record = dict(calls=0, native_proposal_checks=[], restored=False, sources=before, hardware_qualified=False,
        performance_qualified=False, full_request_qualified=False, serving_qualified=False)
    state = SimpleNamespace(mesh=mesh, links=links, record=record)
    token = _ACTIVE.set(state)
    try:
        yield record
        if record['calls'] <= 0:
            raise ValueError('Fused-score candidate was not executed')
    finally:
        _ACTIVE.reset(token)
        record['restored'] = _ACTIVE.get() is None
        record['sources_after'] = sources(directory)
        if (record['sources_after'] != before or native_sources(root) != native
                or composition_audit(directory, proposal, score) != composition or environment() != links):
            raise ValueError('Hardware score experiment bindings or sources changed')


def execute(operations, anchor, base_logits, predecessor, successor, owned, *, mesh, on_step_enqueued=None):
    require_active(mesh).record['calls'] += 1
    shape = tuple(base_logits.shape)
    if (len(shape) != 4 or shape[:3] != (1, 1, 31) or shape[3] not in (64, 248320)
            or not isinstance(owned, list)
            or (on_step_enqueued is not None and not callable(on_step_enqueued))):
        raise ValueError('Explicit experimental 31-query full-vocabulary feedback required')
    require_tensor(operations, base_logits, shape, operations.float32)
    records = []
    previous = anchor
    for start, width in ((0, 7), (7, 7), (14, 7), (21, 7), (28, 3)):
        local = operations.slice(base_logits, (0, 0, start, 0), (1, 1, start + width, shape[3]))
        owned.append(local)
        observer = None if on_step_enqueued is None else lambda step, offset=start: on_step_enqueued(offset + step)
        segment = fused(operations, mesh, previous, local, predecessor, successor, owned,
            on_step_enqueued=observer)
        if len(segment) != width:
            raise AssertionError('Every query must produce exactly one feedback token')
        records.extend(segment)
        previous = operations.reshape(segment[-1]['token'], (1, 1, 1, 1))
        owned.append(previous)
    return records


class HardwareTracedDSparkDevice(TracedDSparkDevice):
    def __init__(self, *args, fused_score_layout=False, **options):
        from functools import partial

        if fused_score_layout is not True:
            raise ValueError('Explicit fused-score candidate selection required')
        require_active()
        super().__init__(*args, fused_score_layout=False, **options)
        try:
            require_active(self.mesh)
            self.proposal_markov = partial(execute, mesh=self.mesh)
        except BaseException:
            self.close()
            raise

    def propose(self, anchor, count):
        import torch
        from dspark_t32_markov import execute as native_markov
        from dspark_t32_prepared import execute as complete_proposal

        state = require_active(self.mesh)
        if self.prepared is None or self.prepared.audit is not True:
            raise ValueError('Complete hardware proposal comparison requires an audited prepared trace')
        self.prepared.update(anchor)
        scope = self.prepared.output_owner()
        candidate = self.proposal_markov
        try:
            self.proposal_markov = native_markov
            reference = self.prepared.snapshot(complete_proposal(self, self.prepared.inputs,
                self.prepared.history, scope.retain))
        finally:
            self.proposal_markov = candidate
            scope.release()
        tokens = super().propose(anchor, count)
        actual = self.prepared.snapshot(self.prepared.outputs)
        if len(reference) != 6 or len(actual) != 6 or any(
                not torch.equal(expected, current) for expected, current in zip(reference, actual, strict=True)):
            raise AssertionError('Complete fused-score proposal differs from native-score reference')
        state.record['native_proposal_checks'].append(dict(position=self.position, anchor=anchor,
            rows=31, tensors=6, exact=True))
        return tokens
