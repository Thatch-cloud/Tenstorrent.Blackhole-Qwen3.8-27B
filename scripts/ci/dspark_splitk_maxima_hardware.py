"""Separate hardware build identity for the simulator-qualified maxima candidate."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_splitk_hardware_build as baseline
from dspark_splitk_maxima_factory import transform
from dspark_splitk_maxima_gate import qualify


@contextmanager
def hardware_identity():
    original = baseline.transform
    builders = (*baseline.BUILDERS, 'dspark_splitk_maxima_factory.py',
        'dspark_splitk_maxima_gate.py', 'dspark_splitk_maxima_hardware.py')
    with patch.object(baseline, 'qualify', qualify), patch.object(baseline, 'BUILDERS', builders), \
            patch.object(baseline, 'transform', lambda source: transform(original(source))):
        yield


if __name__ == '__main__':
    with hardware_identity():
        baseline.main()
