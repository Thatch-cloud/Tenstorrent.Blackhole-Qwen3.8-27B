"""Small SFPU centering simulation; no hardware or performance qualification."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from dspark_score_bitwise import bitwise_infinity_checks, candidate_entrypoint
from dspark_score_smoke_geometry import small_score_fixture
from dspark_score_sfpu import factory_scope, kernel_scope


def main():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_LADDER_CONTEXT') != '128'
            or os.environ.get('QWEN_LADDER_SCORE_SMOKE') != '1'
            or '--hardware' in sys.argv or Path('/dev/tenstorrent').exists()):
        raise ValueError('Small CPU-only simulator required')
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    with small_score_fixture(), bitwise_infinity_checks(), factory_scope(), kernel_scope(), \
            candidate_entrypoint(Path(__file__).resolve()):
        runpy.run_path(str(directory / 'dspark-ladder-attention-probe.py'), run_name='__main__')
    report = json.loads(output.read_text())
    report.update(candidate='sfpu-score-centering', performance_qualified=False,
        candidate_sources={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('dspark_score_bitwise.py', 'dspark_score_smoke_geometry.py',
                'dspark_score_sfpu.py', 'dspark_score_sfpu_build.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
