"""Same-process control/score comparison using the hardware-audited lazy loader."""

import json
import os
from pathlib import Path
import runpy
import sys

from qwen_lazy_weight_gate import qualify
from qwen_lazy_weight_load import lazy_weight_load


def main():
    directory = Path(__file__).parent
    qualify(directory, directory / 'qwen-lazy-weight-audit.json')
    entry = directory / 'dspark-64k-score-paired-request.py'
    if sys.argv[1:] == ['--source-preflight']:
        runpy.run_path(str(entry), run_name='__main__')
        return
    if any(os.environ.get(name) != '1' for name in (
            'QWEN_LAZY_WEIGHT_LOAD', 'QWEN_SCORE_PAIRED', 'QWEN_DSPARK_SFPU_TIMED')):
        raise ValueError('Explicit audited lazy-loader paired timing required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt import mlp, tp_common
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    records, failure = [], None
    try:
        with lazy_weight_load(tp_common, ttnn, torch, records, mlp, prepare_for_fused_swiglu):
            runpy.run_path(str(entry), run_name='__main__')
        if (len(records) != 1 or not records[0]['restored'] or records[0]['calls'] != 320
                or records[0]['packed_calls'] != 64 or records[0]['materializations'] != 0
                or records[0]['packed_materializations'] != 0):
            raise ValueError('Matched cached loader behavior required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report['lazy_timing'] = dict(records=records, failure=failure, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, full_request_passed=False)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
