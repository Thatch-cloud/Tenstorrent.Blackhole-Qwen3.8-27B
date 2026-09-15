"""Separate simulator build identity for the unqualified local-maxima ablation."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_splitk_fp32_build as baseline
from dspark_splitk_maxima_factory import transform


BUILDERS = (*baseline.BUILDERS, 'dspark_splitk_maxima_factory.py', 'dspark_splitk_maxima_build.py')


@contextmanager
def builder_identity():
    with patch.object(baseline, 'BUILDERS', BUILDERS):
        yield


def main():
    original = baseline.transform
    with builder_identity(), patch.object(baseline, 'transform', lambda source: transform(original(source))):
        baseline.main()


if __name__ == '__main__':
    main()
