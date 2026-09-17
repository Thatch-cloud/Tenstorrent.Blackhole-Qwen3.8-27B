"""Opt-in SFPU scope for bounded hardware diagnostics, not serving admission."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

import dspark_64k_build
from dspark_score_bitwise import bitwise_infinity_checks
from dspark_score_sfpu import factory_scope, kernel_scope
from dspark_score_sfpu_gate import qualify


@contextmanager
def hardware_scope(directory):
    if (os.environ.get('QWEN_DSPARK_SCORE_SFPU') != '1'
            or os.environ.get('QWEN_DSPARK_PHASE_PROBE') != '1'
            or os.environ.get('QWEN_DSPARK_64K_TRIAL') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Explicit bounded 64K SFPU hardware diagnostic required')
    qualify(directory, Path(directory) / 'dspark-score-sfpu.json')
    builders = (*dspark_64k_build.BUILDERS, 'dspark_score_sfpu.py',
        'dspark_score_sfpu_gate.py', 'dspark_score_sfpu_hardware.py')
    with patch.object(dspark_64k_build, 'BUILDERS', builders), factory_scope(), kernel_scope(), bitwise_infinity_checks():
        yield


if __name__ == '__main__':
    from dspark_runtime_cache import main

    with hardware_scope(Path(__file__).parent):
        main()
