"""Full 64K numerical fixtures on the simulator-qualified SFPU hardware path."""

from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import dspark_ladder_build
import dspark_ladder_fixtures
from dspark_combined_probe_build import fingerprints, validate_combined
from dspark_score_bitwise import candidate_entrypoint
from dspark_score_sfpu_hardware import hardware_scope
from native_draft_sdpa import audit_active_kernel


def main():
    if (os.environ.get('QWEN_LADDER_CONTEXT') != '65536'
            or os.environ.get('QWEN_LADDER_BACKEND') != 'hardware'
            or os.environ.get('QWEN_LADDER_SCORE_SMOKE', '0') != '0'
            or '--hardware' not in sys.argv):
        raise ValueError('Full 64K hardware numerical fixture required')
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_fixture = dspark_ladder_fixtures.fixture_probe

    @contextmanager
    def combined_fixture(context):
        with original_fixture(context) as probe:
            probe.SOURCES += ('dspark_combined_probe_build.py', Path(__file__).name,
                'dspark_score_sfpu.py', 'dspark_score_sfpu_hardware.py', 'dspark_score_bitwise.py')
            checked = partial(fingerprints, packer=probe.NATIVE.PACKER,
                expected_packer=probe.NATIVE.ORIGINAL_PACKER, audit_kernel=audit_active_kernel)
            with patch.object(probe, 'runner_fingerprints', checked):
                yield probe

    with hardware_scope(directory), candidate_entrypoint(Path(__file__).resolve()), \
            patch.object(dspark_ladder_fixtures, 'fixture_probe', combined_fixture), \
            patch.object(dspark_ladder_build, 'validate_manifest', validate_combined):
        runpy.run_path(str(directory / 'dspark-ladder-attention-probe.py'), run_name='__main__')
    report = json.loads(output.read_text())
    report.update(candidate='sfpu-score-centering', performance_qualified=False,
        committed_tg=None, full_request_passed=False,
        candidate_sources={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ('dspark_score_sfpu.py', 'dspark_score_bitwise.py',
                'dspark_combined_probe_build.py', 'dspark_score_sfpu_hardware.py', Path(__file__).name)})
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
