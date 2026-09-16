"""Request tests must reuse an exact combined build, never silently compile."""

from contextlib import contextmanager
from pathlib import Path
import runpy
from unittest.mock import patch

import dspark_runtime_cache


@contextmanager
def cached_only(module=dspark_runtime_cache):
    original = module.inspect_entry

    def inspect(cache, inputs):
        result = original(cache, inputs)
        if result is None:
            raise ValueError('Exact combined runtime cache miss: prepare the build separately before request testing')
        return result

    with patch.object(module, 'inspect_entry', inspect):
        yield


if __name__ == '__main__':
    with cached_only():
        runpy.run_path(str(Path(__file__).with_name('matched_combined_build.py')), run_name='__main__')
