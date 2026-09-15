"""One model load, unchanged control versus hardware-admitted incremental history."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter
from unittest.mock import patch

from dspark_64k_score_timed import identity_scope
from history_append_hardware_gate import qualify as qualify_history
from history_append_pair import paired_history
from incremental_history_scope import incremental_history
from qwen_lazy_weight_gate import qualify as qualify_loader
from qwen_lazy_weight_load import lazy_weight_load


def main():
    directory = Path(__file__).parent
    qualify_history(directory, directory / 'history-append-hardware.json')
    qualify_loader(directory, directory / 'qwen-lazy-weight-audit.json')
    entry = directory / 'dspark-64k-shared-timed-request.py'
    if sys.argv[1:] == ['--source-preflight']:
        with identity_scope():
            runpy.run_path(str(entry), run_name='__main__')
        return
    if any(os.environ.get(name) != '1' for name in
            ('QWEN_HISTORY_APPEND_PAIR', 'QWEN_LAZY_WEIGHT_LOAD', 'QWEN_DSPARK_SFPU_TIMED')):
        raise ValueError('Explicit offline incremental history comparison required')
    import torch
    import ttnn
    import full_dspark_request
    import dspark_stable_history
    import dspark_publication_scope
    from models.demos.blackhole.qwen36.tt import mlp, tp_common
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh paired report required')
    names = (Path(__file__).name, 'history_append_pair.py', 'incremental_history_scope.py',
        'history_append_hardware_gate.py', 'history_append_dma.py', 'history_append_dma.cpp', 'history_append_plan.py')
    def sources():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
    before = sources()
    records, updates, loaders, warmups, failure = [], [], [], [], None

    @contextmanager
    def candidate():
        arm_class = dspark_publication_scope.CapturedPublicationArm
        original_init = arm_class.__init__

        def initialize(arm, history, *, audit=False):
            original_init(arm, history, audit=audit)
            started = perf_counter()
            try:
                publication = history.prepare_projected(arm.projection.outputs, 32, position=history.position)
                history.discard_publication(publication)
            except BaseException:
                arm.projection.close()
                raise
            warmups.append(dict(position=history.position, elapsed_ms=(perf_counter() - started) * 1000,
                committed=False, scope='Writer compilation before verifier capture and decode timing'))

        with incremental_history(dspark_stable_history.StableHistoryKV, dspark_publication_scope, updates), \
                patch.object(arm_class, '__init__', initialize):
            yield

    try:
        with lazy_weight_load(tp_common, ttnn, torch, loaders, mlp, prepare_for_fused_swiglu), \
                identity_scope(), paired_history(full_dspark_request, candidate, records,
                    lambda record: print(json.dumps(record), flush=True)):
            runpy.run_path(str(entry), run_name='__main__')
        if (len(updates) != 1 or len(warmups) != 1 or updates[0]['failed'] or not updates[0]['restored']
                or updates[0]['committed'] != len(records[1]['request']['blocks'])
                or updates[0]['prepared'] != updates[0]['committed'] + 1 or updates[0]['discarded'] != 1
                or updates[0]['max_touched_rows'] > 96 or before != sources()):
            raise ValueError('Every candidate publication must use the source-stable incremental writer')
        if (len(loaders) != 1 or not loaders[0]['restored'] or loaders[0]['calls'] != 320
                or loaders[0]['packed_calls'] != 64 or loaders[0]['materializations'] != 0
                or loaders[0]['packed_materializations'] != 0):
            raise ValueError('Unchanged audited cache-hit loading required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['history_pair'] = dict(records=records, updates=updates, loaders=loaders, warmups=warmups,
                sources=before, sources_after=sources(), failure=failure,
                order=['control', 'incremental_history'], repetitions_per_arm=1,
                hardware_admission_run=35024279412, serving_qualified=False, held_out_coding_quality=False)
            report.update(pp=None, committed_tg=None, performance_qualified=False)
            for arm in report.get('request_comparison', {}).get('arms', {}).values():
                arm.update(pp=None, committed_tg=None,
                    qualification_scope='Different paired arms; use history_pair, never pooled throughput')
            if failure is not None:
                report.update(passed=False, full_request_passed=False)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
