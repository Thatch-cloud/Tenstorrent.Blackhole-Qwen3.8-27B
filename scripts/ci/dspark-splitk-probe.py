"""Simulator-only native split-K candidate with unchanged mixed attention gates."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_score_bitwise
import dspark_splitk_attention
from dspark_splitk_attention import splitk_scope
from dspark_splitk_layout import scheduling


def main():
    if os.environ.get('QWEN_SPLITK_ATTENTION') != '1':
        raise ValueError('Explicit split-K experiment required')
    directory = Path(__file__).resolve().parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_entrypoint = dspark_score_bitwise.candidate_entrypoint

    def entrypoint(unused_script):
        return original_entrypoint(Path(__file__).resolve())

    with splitk_scope(), patch.object(dspark_score_bitwise, 'candidate_entrypoint', entrypoint), \
            patch.object(dspark_splitk_attention, 'execute_folded',
                wraps=dspark_splitk_attention.execute_folded) as execution:
        runpy.run_path(str(directory / 'dspark-center-tile-fill-probe.py'), run_name='__main__')
        if execution.call_count < 4:
            raise ValueError('Split-K adapter must execute every eager fixture, not the old candidate')
    report = json.loads(output.read_text())
    report.update(candidate='native-decode-split-k', performance_qualified=False,
        draft_attention_backend='scaled_dot_product_attention_decode',
        draft_math='native decode reduction; prefill scalar selectors do not apply',
        scheduling_model=scheduling(), splitk_execution_calls=execution.call_count)
    report['candidate_sources'].update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('dspark_splitk_attention.py', 'dspark_splitk_layout.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
