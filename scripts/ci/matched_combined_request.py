"""Fresh maxima/incremental-history bounded request audit; no timing admission."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter
from unittest.mock import patch

from dspark_splitk_combined_build import digest
from history_append_hardware_gate import qualify as qualify_history
from incremental_history_scope import incremental_history
from matched_combined_build import admission, identity_scope
from qwen_lazy_weight_gate import qualify as qualify_loader


@contextmanager
def publication_scope(history_class, publication_module, updates, warmups):
    arm_class = publication_module.CapturedPublicationArm
    original = arm_class.__init__

    def initialize(arm, history, *, audit=False):
        if not audit:
            raise ValueError('Fresh feature/state audited publication required')
        original(arm, history, audit=audit)
        started = perf_counter()
        try:
            publication = history.prepare_projected(arm.projection.outputs, 32, position=history.position)
            history.discard_publication(publication)
        except BaseException:
            arm.projection.close()
            raise
        warmups.append(dict(position=history.position, committed=False,
            elapsed_ms=(perf_counter() - started) * 1000))

    with incremental_history(history_class, publication_module, updates), \
            patch.object(arm_class, '__init__', initialize):
        yield


def validate_result(report, updates, warmups):
    checks = report.get('request_checks', [])
    if (report.get('passed') is not True or report.get('correctness_screen_passed') is not True
            or report.get('closed_cleanly') is not True or len(checks) != 1
            or checks[0].get('instrumented_timing') is not True):
        raise ValueError('One complete clean feature-audited request required')
    blocks = checks[0].get('blocks', [])
    if (not blocks or len(updates) != 1 or len(warmups) != 1
            or warmups[0]['committed'] or updates[0]['failed'] or not updates[0]['restored']
            or updates[0]['committed'] != len(blocks)
            or updates[0]['prepared'] != len(blocks) + 1 or updates[0]['discarded'] != 1
            or updates[0]['max_touched_rows'] > 96):
        raise ValueError('Every audited commit must use the bounded incremental writer')


def main():
    required = ('QWEN_MATCHED_COMBINED', 'QWEN_LAZY_WEIGHT_LOAD', 'QWEN_64K_SCORE_AUDIT',
        'QWEN_DSPARK_SFPU_REQUEST_SCREEN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Fresh combined audit required; previous timing admission cannot be reused')
    directory = Path(__file__).parent
    components = admission(directory)
    history = qualify_history(directory, directory / 'history-append-hardware.json')
    loader = qualify_loader(directory, directory / 'qwen-lazy-weight-audit.json')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh combined audit report required')
    names = (Path(__file__).name, 'matched_combined_build.py', 'incremental_history_scope.py',
        'history_append_dma.py', 'history_append_dma.cpp', 'history_append_plan.py',
        'history_append_hardware_gate.py', 'qwen_lazy_weight_gate.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    updates, warmups, failure = [], [], None
    try:
        import dspark_stable_history
        import dspark_publication_scope

        with identity_scope(), publication_scope(dspark_stable_history.StableHistoryKV,
                dspark_publication_scope, updates, warmups):
            runpy.run_path(str(directory / 'dspark-64k-lazy-load-request.py'), run_name='__main__')
        validate_result(json.loads(output.read_text()), updates, warmups)
        if before != sources():
            raise ValueError('Combined runtime source changed during audit')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['matched_combined'] = dict(components=components, history_admission=history,
            loader_admission=loader, updates=updates, warmups=warmups, sources=before,
            sources_after=sources(), failure=failure, full_request_qualified=False,
            performance_qualified=False, serving_qualified=False)
        if failure is not None:
            report.update(passed=False, correctness_screen_passed=False)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
