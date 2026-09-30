"""Device-free plan and host-predicted cache for the ordered-cache wide page-table probe.

ordered-cache-hw-probe.py writes BF8 K/V through ordered_cache.update at page-table width
2,052 (and a width-1,024 control) on both Blackhole chips, then compares the COMPLETE
cache on each chip against the cache predicted here. The prediction reads only the host
page table this module builds. It never reads a page table back from the device and never
calls a native op. Every earlier 2,052 check compared against code that shares the
native page-table read (the simulator against native paged_update_cache, the hardware run
against a native-decode oracle). A misread of page-table bytes 8192-8207 in that shared
code would pass both of those checks and fails this one.

Only payload(), bf8_exact(), ExpectedCache and compare_cache need torch, and they import
it lazily, so `--check REPORT` runs on any Python 3.8+ without it.
"""

import argparse
import hashlib
import json
import sys

BLOCK_SIZE = 64
HEADS = 2
PADDED_HEADS = 32
HEAD_DIM = 256
BF8_GROUP = 16
ROWS = 16
MIN_ROWS = 8
BLOCKS = 2064
MIN_BLOCKS = 2060
WIDE_WIDTH = 2052
CONTROL_WIDTH = 1024
CHIPS = (0, 1)
KERNEL_ROLES = ('reader', 'writer', 'compute')

# Entries cycled through by every row (row r at step s takes entries[(r + s) % n]), so each
# row - on its own DRAM page of the page-table tensor - reads every listed entry once.
WIDE_ENTRIES = (0, 1, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051)
CONTROL_ENTRIES = (0, 1, 511, 512, 1022, 1023)
# Positions written by rows 0..3 in the final anchor step of each case.
WIDE_ANCHORS = (131072, 131136, 131200, 131327)
CONTROL_ANCHORS = (0, 64, 65472, 65535)
REQUIRED_WIDE_ENTRIES = (0, 1023, 1024, 2048, 2049, 2050, 2051)
REQUIRED_CONTROL_ENTRIES = (0, 1023)

CASES = (
    dict(name='control-1024-eager', mode='eager', width=CONTROL_WIDTH, table_seeds=(1024,),
         entries=CONTROL_ENTRIES, anchors=CONTROL_ANCHORS, required_entries=REQUIRED_CONTROL_ENTRIES),
    dict(name='wide-2052-eager', mode='eager', width=WIDE_WIDTH, table_seeds=(2052,),
         entries=WIDE_ENTRIES, anchors=WIDE_ANCHORS, required_entries=REQUIRED_WIDE_ENTRIES),
    # Two tables alternate by step: the probe rewrites the page-table buffer in place
    # between replays, so a replay that reused a stale table would mispredict.
    dict(name='wide-2052-trace', mode='trace', width=WIDE_WIDTH, table_seeds=(20521, 20522),
         entries=WIDE_ENTRIES, anchors=WIDE_ANCHORS, required_entries=REQUIRED_WIDE_ENTRIES),
)

# Environment the probe reads inside the container. Each `required` name must be passed
# with `docker -e` by the workflow (test_ordered_cache_hw_plan checks it does).
PROBE_ENV = dict(required=('QWEN_HARDWARE_TESTS', 'TT_METAL_HOME', 'ORDERED_CACHE_SHA256'),
                 forbidden=('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE'))

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1


def _splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, value ^ (value >> 31)


def page_table(seed, rows, width, blocks):
    """Per-row partial permutations of range(blocks): distinct ids within a row, a distinct
    row per user. Pure integer arithmetic, so the table is identical on every Python."""
    if type(width) is not int or not 1 <= width <= blocks or type(rows) is not int or rows < 1:
        raise ValueError('Page-table width must fit the cache')
    state = seed
    table = []
    for unused in range(rows):
        ids = list(range(blocks))
        for index in range(blocks - 1, 0, -1):
            state, value = _splitmix(state)
            other = value % (index + 1)
            ids[index], ids[other] = ids[other], ids[index]
        table.append(ids[:width])
    return table


def schedule(entries, anchors, rows):
    """Positions per step. Offsets (row * 4 + step * 5) % 64 are distinct across rows within
    a step, so two rows never target the same (block, offset) in one step."""
    count = len(entries)
    steps = [[entries[(row + step) % count] * BLOCK_SIZE + (row * 4 + step * 5) % BLOCK_SIZE
              for row in range(rows)] for step in range(count)]
    steps.append([anchors[row] if row < len(anchors)
                  else entries[(row + 3) % count] * BLOCK_SIZE + (row * 4 + 1) % BLOCK_SIZE
                  for row in range(rows)])
    return steps


def payload_seed(case_index, step, row):
    return (case_index + 1) * 1000000 + step * 1000 + row + 1


def table_digest(table):
    return hashlib.sha256(json.dumps(table, separators=(',', ':')).encode()).hexdigest()


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_plan(blocks=BLOCKS, rows=ROWS):
    cases = []
    for case_index, spec in enumerate(CASES):
        tables = [page_table(seed, rows, spec['width'], blocks) for seed in spec['table_seeds']]
        steps = []
        for number, positions in enumerate(schedule(spec['entries'], spec['anchors'], rows)):
            table_index = number % len(tables)
            table = tables[table_index]
            steps.append(dict(step=number, table=table_index, positions=positions,
                entries=[position // BLOCK_SIZE for position in positions],
                offsets=[position % BLOCK_SIZE for position in positions],
                blocks=[table[row][position // BLOCK_SIZE] for row, position in enumerate(positions)],
                payload_seeds=[payload_seed(case_index, number, row) for row in range(rows)]))
        cases.append(dict(name=spec['name'], mode=spec['mode'], width=spec['width'],
            table_seeds=list(spec['table_seeds']), required_entries=list(spec['required_entries']),
            required_positions=list(spec['anchors']), tables=tables,
            table_sha256=[table_digest(table) for table in tables], steps=steps))
    plan = dict(version=1, blocks=blocks, rows=rows, heads=HEADS, padded_heads=PADDED_HEADS,
        block_size=BLOCK_SIZE, head_dim=HEAD_DIM, cache_dtype='bfloat8_b',
        cache_shape=[blocks, HEADS, BLOCK_SIZE, HEAD_DIM], cases=cases)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Refuse a plan that could not tell a page-table misread from a correct write."""
    blocks, rows = plan['blocks'], plan['rows']
    if type(blocks) is not int or blocks < MIN_BLOCKS:
        raise ValueError('At least %d cache blocks required' % MIN_BLOCKS)
    if rows not in (8, 16) or rows < MIN_ROWS:
        raise ValueError('8 or 16 rows required (distinct offsets need rows <= 16)')
    if plan['cache_shape'] != [blocks, HEADS, BLOCK_SIZE, HEAD_DIM]:
        raise ValueError('Cache geometry must be (blocks, 2, 64, 256)')
    names = [case['name'] for case in plan['cases']]
    if len(set(names)) != len(names):
        raise ValueError('Case names must be unique')
    seen = set()
    for case in plan['cases']:
        width = case['width']
        if type(width) is not int or not 1 <= width <= blocks or case['mode'] not in ('eager', 'trace'):
            raise ValueError('Case width and mode required: ' + case['name'])
        seen.add((width, case['mode']))
        if not case['tables'] or len(case['tables']) != len(case['table_sha256']):
            raise ValueError('Page tables required: ' + case['name'])
        for table, digest in zip(case['tables'], case['table_sha256']):
            if len(table) != rows or any(len(row) != width for row in table):
                raise ValueError('Page-table shape must be (rows, width): ' + case['name'])
            if any(len(set(row)) != width or min(row) < 0 or max(row) >= blocks for row in table):
                raise ValueError('Each page row must be distinct block ids within the cache: ' + case['name'])
            if len(set(tuple(row) for row in table)) != rows:
                raise ValueError('Every row needs its own page row: ' + case['name'])
            if table_digest(table) != digest:
                raise ValueError('Page-table digest mismatch: ' + case['name'])
        hits = set()
        positions_seen = set()
        for number, step in enumerate(case['steps']):
            positions = step['positions']
            if step['step'] != number or not 0 <= step['table'] < len(case['tables']) or len(positions) != rows:
                raise ValueError('Step shape required: %s step %d' % (case['name'], number))
            table = case['tables'][step['table']]
            if any(type(position) is not int or not 0 <= position < width * BLOCK_SIZE for position in positions):
                raise ValueError('Positions must lie inside the page-table window: %s step %d' % (case['name'], number))
            if (step['entries'] != [position // BLOCK_SIZE for position in positions]
                    or step['offsets'] != [position % BLOCK_SIZE for position in positions]
                    or step['blocks'] != [table[row][position // BLOCK_SIZE] for row, position in enumerate(positions)]):
                raise ValueError('Step bookkeeping disagrees with its table: %s step %d' % (case['name'], number))
            targets = list(zip(step['blocks'], step['offsets']))
            if len(set(targets)) != rows:
                raise ValueError('Two rows write one (block, offset) in a step: %s step %d' % (case['name'], number))
            if len(step['payload_seeds']) != rows or len(set(step['payload_seeds'])) != rows:
                raise ValueError('Distinct payload per row required: %s step %d' % (case['name'], number))
            hits.update((row, position // BLOCK_SIZE) for row, position in enumerate(positions))
            positions_seen.update(positions)
        for entry in case['required_entries']:
            if any((row, entry) not in hits for row in range(rows)):
                raise ValueError('Every row must hit page-table entry %d: %s' % (entry, case['name']))
        if not set(case['required_positions']) <= positions_seen:
            raise ValueError('Required anchor positions missing: ' + case['name'])
    wide_required = set(REQUIRED_WIDE_ENTRIES)
    for mode in ('eager', 'trace'):
        cases = [case for case in plan['cases'] if case['width'] == WIDE_WIDTH and case['mode'] == mode]
        if not cases or not any(wide_required <= set(case['required_entries'])
                                and set(WIDE_ANCHORS) <= set(case['required_positions']) for case in cases):
            raise ValueError('A %d-wide %s case covering every tail entry is required' % (WIDE_WIDTH, mode))
    if (CONTROL_WIDTH, 'eager') not in seen:
        raise ValueError('A width-%d control case is required' % CONTROL_WIDTH)
    payload_seeds = [seed for case in plan['cases'] for step in case['steps'] for seed in step['payload_seeds']]
    if len(set(payload_seeds)) != len(payload_seeds):
        raise ValueError('Payload seeds must be unique across the plan')
    return plan


def _mix32(values):
    """A bijective 32-bit integer hash on int64 tensors; every product stays below 2**63."""
    values = values & MASK32
    values = (((values >> 16) ^ values) * 0x45D9F3B) & MASK32
    values = (((values >> 16) ^ values) * 0x45D9F3B) & MASK32
    return (values >> 16) ^ values


def payload(seed, padded_heads=PADDED_HEADS, head_dim=HEAD_DIM):
    """(padded_heads, head_dim) BF16 values exactly representable in BF8: every 16-wide
    group shares one power-of-two scale and magnitudes are 64..120 in steps of 8 (four
    significant bits), so the BF8 round trip is lossless. No value is zero, so a written row
    is never mistaken for an unwritten one. Padding heads 2..31 carry values too: a writer
    that copied a padding head into the cache would mispredict."""
    import torch

    if head_dim % BF8_GROUP:
        raise ValueError('Head dimension must be whole BF8 groups')
    salt = _splitmix(seed)[1] & MASK32
    groups = head_dim // BF8_GROUP
    index = torch.arange(padded_heads * head_dim, dtype=torch.int64).reshape(padded_heads, head_dim)
    bits = _mix32(index * 40503 + salt)
    group_index = torch.arange(padded_heads * groups, dtype=torch.int64).reshape(padded_heads, groups)
    group_bits = _mix32(group_index * 40503 + (salt ^ 0x5BD1E995))
    magnitude = 64 + 8 * (bits & 7)
    sign = 1 - 2 * ((bits >> 3) & 1)
    exponent = (2 + group_bits % 10).to(torch.float64)
    scale = torch.pow(torch.full_like(exponent, 2.0), -exponent).repeat_interleave(BF8_GROUP, dim=1)
    return ((sign * magnitude).to(torch.float64) * scale).to(torch.bfloat16)


def step_payloads(step):
    import torch

    return torch.stack([payload(seed) for seed in step['payload_seeds']])


def bf8_exact(values, mantissa_bits=7):
    """True when every 16-wide group is representable with one shared exponent and
    `mantissa_bits` significant bits - the BF8_b tile format."""
    import torch

    flat = values.to(torch.float64).reshape(-1, BF8_GROUP)
    if not bool(torch.isfinite(flat).all()):
        return False
    unused, exponent = torch.frexp(flat)
    nonzero = flat != 0
    if not bool(nonzero.any()):
        return True
    floor = torch.full_like(exponent, -100000)
    shared = torch.where(nonzero, exponent, floor).amax(dim=1, keepdim=True)
    scaled = flat * torch.pow(torch.full_like(flat, 2.0), (mantissa_bits - shared).to(torch.float64))
    representable = (scaled == scaled.round()) & (scaled.abs() < 2 ** mantissa_bits)
    return bool(representable[nonzero].all())


class ExpectedCache:
    """The cache the kernel must leave behind, built from the host page table alone."""

    def __init__(self, blocks, heads=HEADS, block_size=BLOCK_SIZE, head_dim=HEAD_DIM):
        import torch

        self.heads = heads
        self.block_size = block_size
        self.values = torch.zeros(blocks, heads, block_size, head_dim, dtype=torch.bfloat16)
        self.predicted = set()

    def apply(self, table, positions, payloads):
        if len(positions) != len(table) or payloads.shape[0] != len(positions):
            raise ValueError('One payload and one page row per position required')
        for row, position in enumerate(positions):
            block = table[row][position // self.block_size]
            self.values[block, :, position % self.block_size, :] = payloads[row, :self.heads, :]
            self.predicted.add(block)


def compare_cache(actual, expected, chunk=128, limit=8):
    """Full-cache comparison: unpredicted blocks must be zero, predicted blocks exact."""
    import torch

    reference = expected.values
    result = dict(exact=False, shape=list(actual.shape), predicted_blocks=len(expected.predicted),
                  mismatched_blocks=0, unpredicted_nonzero_blocks=0, predicted_mismatch_blocks=0,
                  samples=[])
    if tuple(actual.shape) != tuple(reference.shape):
        result['samples'].append(dict(reason='shape', expected=list(reference.shape)))
        return result
    blocks = reference.shape[0]
    predicted = torch.zeros(blocks, dtype=torch.bool)
    if expected.predicted:
        predicted[torch.tensor(sorted(expected.predicted), dtype=torch.int64)] = True
    for start in range(0, blocks, chunk):
        end = min(start + chunk, blocks)
        left = actual[start:end].to(torch.float32)
        right = reference[start:end].to(torch.float32)
        differs = left != right
        block_differs = differs.reshape(end - start, -1).any(dim=1)
        if not bool(block_differs.any()):
            continue
        local_predicted = predicted[start:end]
        result['mismatched_blocks'] += int(block_differs.sum())
        result['unpredicted_nonzero_blocks'] += int((block_differs & ~local_predicted).sum())
        result['predicted_mismatch_blocks'] += int((block_differs & local_predicted).sum())
        for local in block_differs.nonzero().flatten().tolist():
            if len(result['samples']) >= limit:
                break
            rows = differs[local].any(dim=-1).nonzero().tolist()
            result['samples'].append(dict(block=start + local, predicted=bool(local_predicted[local]),
                head_rows=rows[:8], differing_rows=len(rows)))
    # Both halves are required: an extra write to an unpredicted block fails the case even
    # when every predicted block is exact.
    result['exact'] = (result['mismatched_blocks'] == 0 and result['unpredicted_nonzero_blocks'] == 0
                       and result['predicted_mismatch_blocks'] == 0)
    return result


def required_checks(plan):
    """Every (case, step, chip, name) the probe must record as exact for a pass."""
    required = []
    for case in plan['cases']:
        for chip in CHIPS:
            for name in ('zero_baseline', 'pages_uploaded', 'input_unchanged', 'pages_unchanged'):
                required.append((case['name'], None, chip, name))
            for step in case['steps']:
                required.append((case['name'], step['step'], chip, 'complete_cache'))
    return required


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def predicted_counts(case):
    """Blocks the host predicts after each step: the running union of every step's targets."""
    predicted, counts = set(), {}
    for step in case['steps']:
        predicted.update(step['blocks'])
        counts[step['step']] = len(predicted)
    return counts


CACHE_CHECKS = ('zero_baseline', 'complete_cache')
MISMATCH_COUNTERS = ('mismatched_blocks', 'unpredicted_nonzero_blocks', 'predicted_mismatch_blocks')


def check_passes(check, plan, counts):
    """A check passes only when it reports `exact` AND, for a cache comparison, carries the
    evidence of a real full-cache comparison: the plan's cache shape, the predicted-block
    count the plan implies at that step, and zero mismatches of every kind. A report whose
    checks were written without comparing the read-back cache cannot fake these fields
    without recomputing the plan."""
    if check.get('exact') is not True:
        return False
    if check.get('name') not in CACHE_CHECKS:
        return True
    if check.get('shape') != plan['cache_shape']:
        return False
    if any(check.get(counter) != 0 for counter in MISMATCH_COUNTERS):
        return False
    wanted = 0 if check.get('name') == 'zero_baseline' else counts.get(check.get('step'))
    return wanted is not None and check.get('predicted_blocks') == wanted


def summarise_cases(report, plan):
    checks = report.get('checks') or []
    summary = []
    for case in plan['cases']:
        own = [check for check in checks if check.get('case') == case['name']]
        needed = [key for key in required_checks(plan) if key[0] == case['name']]
        counts = predicted_counts(case)
        present = {}
        for check in own:
            key = (check.get('case'), check.get('step'), check.get('chip'), check.get('name'))
            # A duplicated key passes only if every copy passes.
            present[key] = present.get(key, True) and check_passes(check, plan, counts)
        failed = ['%s step=%s chip=%s' % (key[3], key[1], key[2]) for key in needed if not present.get(key)]
        summary.append(dict(name=case['name'], mode=case['mode'], width=case['width'],
            steps=len(case['steps']), checks=len(own), passed=not failed, failed=failed[:16]))
    return summary


def check_report(report, plan=None):
    """Return the reasons a probe report does not qualify width 2,052 (empty when it does)."""
    plan = build_plan() if plan is None else plan
    failures = []
    if report.get('backend') != 'hardware':
        failures.append('backend is not hardware')
    if report.get('error'):
        failures.append('probe error: %s' % report['error'])
    if report.get('closed_cleanly') is not True:
        failures.append('mesh not closed cleanly')
    if report.get('weight_free') is not True or report.get('prefill') is not False:
        failures.append('probe must be weight-free with no prefill')
    if report.get('plan') != plan:
        failures.append('report plan differs from the reviewed plan')
    elif report.get('plan_sha256') != plan_digest(plan):
        failures.append('plan digest mismatch')
    for field in ('native_hashes', 'generated_hashes'):
        hashes = report.get(field) or {}
        if sorted(hashes) != sorted(KERNEL_ROLES) or not all(_hex64(hashes[role]) for role in KERNEL_ROLES):
            failures.append('%s must name reader, writer and compute sha256' % field)
    if not _hex64(report.get('ordered_cache_sha256')):
        failures.append('ordered_cache.py sha256 missing')
    elif report.get('ordered_cache_expected_sha256') != report.get('ordered_cache_sha256'):
        failures.append('baked ordered_cache.py differs from the checkout')
    if WIDE_WIDTH not in (report.get('wide_page_widths') or []):
        failures.append('writer under test does not admit width %d' % WIDE_WIDTH)
    if not isinstance(report.get('tt_metal'), dict):
        failures.append('tt-metal provenance missing')
    for case in summarise_cases(report, plan):
        if not case['passed']:
            failures.append('%s failed: %s' % (case['name'], ', '.join(case['failed'])))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description='Re-check an ordered-cache hardware probe report.')
    parser.add_argument('--check', required=True, help='probe JSON report')
    options = parser.parse_args(argv)
    try:
        with open(options.check, encoding='utf-8') as handle:
            report = json.load(handle)
    except (OSError, ValueError) as error:
        print(json.dumps(dict(passed=False, failures=['unreadable report: %s' % error])))
        return 1
    plan = build_plan()
    failures = check_report(report, plan)
    print(json.dumps(dict(passed=not failures, probe_reported_passed=report.get('passed'),
        cases=[dict((key, case[key]) for key in ('name', 'mode', 'width', 'steps', 'checks', 'passed'))
               for case in summarise_cases(report, plan)],
        native_hashes=report.get('native_hashes'), tt_metal=report.get('tt_metal'),
        failures=failures), indent=2))
    return 0 if not failures and report.get('passed') is True else 1


if __name__ == '__main__':
    sys.exit(main())
