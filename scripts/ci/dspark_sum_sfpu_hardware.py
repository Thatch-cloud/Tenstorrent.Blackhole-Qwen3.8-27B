"""Opt-in sum-update hardware factory and kernel scope; no serving changes."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

import dspark_64k_build
from dspark_score_sfpu_hardware import hardware_scope as base_scope
from dspark_sum_sfpu import sum_scope
from dspark_sum_sfpu_gate import qualify


@contextmanager
def hardware_scope(directory):
    if os.environ.get('QWEN_DSPARK_SUM_SFPU') != '1':
        raise ValueError('Explicit sum-update hardware experiment required')
    qualify(directory, Path(directory) / 'dspark-sum-sfpu.json')
    builders = (*dspark_64k_build.BUILDERS, 'dspark_sum_sfpu.py',
        'dspark_sum_sfpu_gate.py', 'dspark_sum_sfpu_hardware.py')
    with patch.object(dspark_64k_build, 'BUILDERS', builders), sum_scope(), base_scope(directory):
        yield


if __name__ == '__main__':
    from dspark_runtime_cache import main
    with hardware_scope(Path(__file__).parent):
        main()
