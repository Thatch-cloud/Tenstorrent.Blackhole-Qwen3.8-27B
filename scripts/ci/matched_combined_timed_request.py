"""Two complete repeated native-score requests on the audited maxima runtime."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter
from unittest.mock import patch

from dspark_splitk_combined_build import digest
from incremental_history_scope import incremental_history
from matched_combined_gate import SCREEN_RUN, qualify
from matched_combined_timed import identity_scope
from qwen_lazy_weight_load import lazy_weight_load


def validate_updates(requests, updates, warmups, loaders):
    if len(requests) != 2 or len(updates) != 2 or len(warmups) != 2:
        raise ValueError('Two complete incremental-history requests required')
    for request, update, warmup in zip(requests, updates, warmups, strict=True):
        blocks = request.get('blocks', [])
        if (not blocks or update['failed'] or not update['restored'] or warmup['committed']
                or update['committed'] != len(blocks) or update['prepared'] != len(blocks) + 1
                or update['discarded'] != 1 or update['max_touched_rows'] > 96):
            raise ValueError('Every timed publication must use the audited incremental writer')
    if (len(loaders) != 1 or not loaders[0]['restored'] or loaders[0]['calls'] != 320
            or loaders[0]['packed_calls'] != 64 or loaders[0]['materializations'] != 0
            or loaders[0]['packed_materializations'] != 0):
        raise ValueError('Same complete cache-hit lazy loading required')


@contextmanager
def publication_scope(history_class, module, updates, warmups):
    arm_class = module.CapturedPublicationArm
    original = arm_class.__init__

    def initialize(arm, history, *, audit=False):
        if audit:
            raise ValueError('Timing cannot include per-block feature auditing')
        original(arm, history, audit=False)
        started = perf_counter()
        try:
            publication = history.prepare_projected(arm.projection.outputs, 32, position=history.position)
            history.discard_publication(publication)
        except BaseException:
            arm.projection.close()
            raise
        warmups.append(dict(position=history.position, committed=False,
            elapsed_ms=(perf_counter() - started) * 1000))

    with incremental_history(history_class, module, updates), patch.object(arm_class, '__init__', initialize):
        yield


def main():
    directory = Path(__file__).parent
    evidence = qualify(directory, directory / 'dspark-sfpu-request-screen.json')
    entry = directory / 'dspark-64k-shared-timed-request.py'
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(entry), run_name='__main__')
        return
    if (any(os.environ.get(name) != '1' for name in
            ('QWEN_MATCHED_COMBINED', 'QWEN_LAZY_WEIGHT_LOAD', 'QWEN_DSPARK_SFPU_TIMED'))
            or os.environ.get('QWEN_DSPARK_SFPU_REQUEST_SCREEN') != '0'):
        raise ValueError('Explicit clean combined timing required')
    import torch
    import ttnn
    import dspark_stable_history
    import dspark_publication_scope
    from models.demos.blackhole.qwen36.tt import mlp, tp_common
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh full-response timing report required')
    names = (Path(__file__).name, 'matched_combined_timed.py', 'matched_combined_gate.py',
        'incremental_history_scope.py', 'qwen_lazy_weight_load.py')

    def sources():
        return {name: digest(directory / name) for name in names}

    before = sources()
    updates, warmups, loaders, failure = [], [], [], None
    try:
        with identity_scope(), lazy_weight_load(tp_common, ttnn, torch, loaders, mlp, prepare_for_fused_swiglu), \
                publication_scope(dspark_stable_history.StableHistoryKV, dspark_publication_scope, updates, warmups):
            runpy.run_path(str(entry), run_name='__main__')
        report = json.loads(output.read_text())
        validate_updates(report['request_checks'], updates, warmups, loaders)
        if (report.get('passed') is not True or report.get('full_request_passed') is not True
                or report.get('closed_cleanly') is not True or before != sources()):
            raise ValueError('Complete clean source-stable requests required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        report = json.loads(output.read_text()) if output.exists() else {}
        report['matched_timed'] = dict(audit_run=SCREEN_RUN,
            audit_sha256=evidence['report_sha256'], updates=updates, warmups=warmups, loaders=loaders,
            sources=before, sources_after=sources(), failure=failure,
            score_path='native', repetitions=2, held_out_coding_quality=False, serving_qualified=False)
        if failure is not None:
            report.update(passed=False, full_request_passed=False, pp=None, committed_tg=None)
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
