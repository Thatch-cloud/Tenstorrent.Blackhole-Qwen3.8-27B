"""Check T32 drafter report coverage; callers must verify immutable source identity."""

import math
import re

from sim_memory_budget import require_clean
from t32_ci_runtime import PINS


NAMES = ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask')


def coverage(report, field, columns, expected, flag):
    records = report.get(field, [])
    keys = [tuple(entry.get(name) for name in columns) for entry in records]
    if (len(keys) != len(expected) or set(keys) != expected
            or any(entry.get(flag) is not True for entry in records)):
        raise ValueError(f'Complete distinct passing {field} required')
    return records


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('capacity') != 4384 or report.get('proposal_rows') != 31
            or list(report.get('positions', [])) != [4096, 4109]
            or report.get('key_chunk_size') != 64
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)
            or report.get('target_integrated') is not False or report.get('committed_tg') is not None):
        raise ValueError('Complete isolated T32 drafter attention result required')
    for field in ('sources', 'native_sources'):
        hashes = report.get(field, {})
        if (not hashes or hashes != report.get(field + '_after')
                or any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in hashes.values())):
            raise ValueError('Stable complete source fingerprints required')
    if any(report['native_sources'].get(name) != value for name, value in PINS.items()):
        raise ValueError('Pinned simulator binaries and packer required')
    before, after = report.get('resources_before', {}), report.get('resources_after', {})
    for resource in (before, after):
        period = resource.get('cpu_period', 0)
        if (resource.get('bounded') is not True or not resource.get('boot_id')
                or type(period) is not int or period <= 0 or resource.get('cpu_quota') != 16 * period
                or resource.get('limits', {}).get('memory.max') != 64 * 1024**3):
            raise ValueError('Enforced 16-CPU and 64-GiB budget required')
    require_clean(before, after)
    coordinates = {(mode, ordinal, case, chip)
                   for mode, cases in (('eager', (0, 1)), ('replay', (1, 0)))
                   for ordinal, case in enumerate(cases) for chip in range(2)}
    eager = {}
    for mode in ('eager', 'replay'):
        expected = {key[1:] for key in coordinates if key[0] == mode}
        records = coverage(report, mode + '_checks', ('ordinal', 'case', 'chip'), expected, 'passed')
        for entry in records:
            maximum = entry.get('max_abs')
            if (entry.get('numerical_close') is not True or entry.get('failed_elements') != 0
                    or type(maximum) not in (float, int) or not math.isfinite(maximum) or maximum < 0
                    or any(re.fullmatch('[0-9a-f]{64}', entry.get(name, '')) is None
                           for name in ('sha256', 'expected_sha256'))):
                raise ValueError('Finite numerical evidence required')
            key = entry['case'], entry['chip']
            hashes = entry['sha256'], entry['expected_sha256']
            if mode == 'eager':
                eager[key] = hashes
            elif entry.get('replay_exact') is not True or hashes != eager[key]:
                raise ValueError('Replay must match its own eager case and oracle')
    coverage(report, 'input_checks', ('mode', 'ordinal', 'case', 'chip', 'name'),
             {(*key, name) for key in coordinates for name in NAMES}, 'exact')
    layouts = coverage(report, 'layout_checks', ('mode', 'ordinal', 'case', 'chip', 'name'),
                       {(*key, name) for key in coordinates for name in ('key', 'value')}, 'passed')
    if any(entry.get('sha256') != entry.get('expected_sha256')
           or re.fullmatch('[0-9a-f]{64}', entry.get('sha256', '')) is None for entry in layouts):
        raise ValueError('Exact retained key/value layout required')
    coverage(report, 'fixture_controls', ('name', 'case', 'chip'),
             {(name, int(name == 'frontier_update'), chip)
              for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)},
             'detected')
    coverage(report, 'stale_controls', ('chip',), {(0,), (1,)}, 'detected')
    if any(eager[0, chip][0] == eager[1, chip][0] for chip in range(2)):
        raise ValueError('Changed frontier must change each chip output')
    return dict(scope='T32 drafter attention component only', eager_checks=4, replay_checks=4,
                input_checks=48, layout_checks=16, full_request_qualified=False,
                immutable_source_identity_verified=False)
