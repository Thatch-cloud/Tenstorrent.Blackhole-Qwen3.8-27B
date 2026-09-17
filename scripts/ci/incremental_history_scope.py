"""Opt-in captured-publication replacement; requires separate hardware admission."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from history_append_dma import prepare as prepare_dma
from history_append_plan import BankAppendPlanner


@contextmanager
def incremental_history(history_class, publication_module, records, *, writer_factory=prepare_dma):
    sessions = {}
    original_commit = history_class.commit_publication
    original_discard = history_class.discard_publication

    def session(history):
        key = id(history)
        if key not in sessions:
            record = dict(initial_position=history.position, capacity=history.capacity,
                prepared=0, committed=0, discarded=0, max_touched_rows=0, failed=False, restored=False)
            records.append(record)
            sessions[key] = SimpleNamespace(history=history,
                planner=BankAppendPlanner(history.position, history.capacity), record=record)
        result = sessions[key]
        if result.record['failed'] or result.planner.position != history.position:
            raise ValueError('Live matching history frontier required')
        return result

    def assemble(history, added, prefix, *, position):
        history.check_prefix(prefix, position)
        if len(added) != 5 or any(len(pair) != 2 for pair in added):
            raise ValueError('All five fixed-tile projected K/V pairs required')
        current = session(history)
        plan = current.planner.prepare(prefix)
        try:
            operations = [writer_factory(history.mesh, active, delta, spare, plan)
                for active_pair, delta_pair, spare_pair in zip(history.layers, added, history.spare_layers, strict=True)
                for active, delta, spare in zip(active_pair, delta_pair, spare_pair, strict=True)]
            if len(operations) != 10:
                raise ValueError('Ten persistent history banks required')
            for operation in operations:
                operation()
            history.operations.synchronize_device(history.mesh)
        except BaseException:
            current.record['failed'] = True
            raise
        history.pending = SimpleNamespace(owner=history, position=position, prefix=prefix,
            layers=history.spare_layers, status='prepared')
        current.record['prepared'] += 1
        current.record['max_touched_rows'] = max(current.record['max_touched_rows'], plan.end_row - plan.first_row)
        return history.pending

    def publication(history, projection, features, tables, prefix, *, position):
        if not isinstance(history, history_class):
            raise ValueError('Scoped persistent history required')
        history.check_prefix(prefix, position)
        if projection.operations is not history.operations or projection.mesh is not history.mesh:
            raise ValueError('Projection and history must share runtime and mesh')
        outputs = projection.project(features, tables)
        return history.prepare_projected(outputs, prefix, position=position)

    def commit(history, publication):
        history.validate_publication(publication)
        current = session(history)
        plan = current.planner.pending
        if plan is None or plan.prefix != publication.prefix:
            raise ValueError('Matching incremental transaction required')
        original_commit(history, publication)
        current.planner.resolve(plan, commit=True)
        current.record['committed'] += 1

    def discard(history, publication):
        if getattr(publication, 'owner', None) is history and publication.status == 'committed':
            return original_discard(history, publication)
        history.validate_publication(publication)
        current = session(history)
        plan = current.planner.pending
        if plan is None or plan.prefix != publication.prefix:
            raise ValueError('Matching incremental transaction required')
        original_discard(history, publication)
        current.planner.resolve(plan, commit=False)
        current.record['discarded'] += 1

    try:
        with patch.object(history_class, 'prepare_projected', assemble), \
                patch.object(history_class, 'commit_publication', commit), \
                patch.object(history_class, 'discard_publication', discard), \
                patch.object(publication_module, 'prepare', publication):
            yield
    finally:
        for current in sessions.values():
            current.record['restored'] = True
