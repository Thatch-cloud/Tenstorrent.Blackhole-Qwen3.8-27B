"""Isolated simulator factory cache for SFPU centering."""

from pathlib import Path
from unittest.mock import patch

import dspark_sim_build_cache
from dspark_hardware_gate import digest
from dspark_score_sfpu import factory_scope


def main():
    original = dspark_sim_build_cache.build_inputs

    def inputs(root, scripts):
        result = original(root, scripts)
        result['sfpu_center_builders'] = {'dspark_score_sfpu_build.py':
            digest(Path(scripts) / 'dspark_score_sfpu_build.py')}
        return result

    with factory_scope(), patch.object(dspark_sim_build_cache, 'build_inputs', inputs):
        dspark_sim_build_cache.main()


if __name__ == '__main__':
    main()
