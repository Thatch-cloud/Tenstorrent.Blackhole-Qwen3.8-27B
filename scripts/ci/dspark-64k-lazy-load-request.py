"""Combined correctness audit with cache-miss-only TP weight preprocessing."""

import json
import os
from pathlib import Path
import runpy
import sys

from dspark_splitk_combined_build import digest
from qwen_lazy_weight_load import lazy_weight_load


def main():
    required = ('QWEN_LAZY_WEIGHT_LOAD', 'QWEN_64K_SCORE_AUDIT', 'QWEN_DSPARK_SFPU_REQUEST_SCREEN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Explicit combined lazy-weight correctness audit required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt import mlp, tp_common
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh lazy-loading report required')
    dependencies = (Path(__file__).name, 'qwen_lazy_weight_load.py')
    sources = {name: digest(directory / name) for name in dependencies}
    records, failure = [], None
    try:
        with lazy_weight_load(tp_common, ttnn, torch, records, mlp, prepare_for_fused_swiglu):
            runpy.run_path(str(directory / 'dspark-64k-score-request.py'), run_name='__main__')
        if (len(records) != 1 or not records[0]['restored'] or records[0]['packed_calls'] != 64
                or sources != {name: digest(directory / name) for name in dependencies}):
            raise ValueError('One restored source-stable model with all 64 packed MLP loads required')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        print(json.dumps(dict(stage='lazy_weight_load', records=records, failure=failure)), flush=True)
        if output.exists():
            report = json.loads(output.read_text())
            report['lazy_weight_load'] = dict(records=records, sources=sources, failure=failure,
                performance_qualified=False, serving_qualified=False)
            if failure is not None:
                report.update(passed=False, correctness_screen_passed=False)
            output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
