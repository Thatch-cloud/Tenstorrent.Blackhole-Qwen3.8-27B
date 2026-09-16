"""Validate sampled waits by exact execution identity; never infer throughput."""

from collections import Counter, defaultdict
import csv
from pathlib import Path

from frozen_mlp_wait_zones import ZONES


EXPECTED = Counter({name: (2 if name == 'QWEN_MLP_INPUT_FREE' else 1)
    for entries in ZONES.values() for _, name, _ in entries})

RAW_FIELDS = {'chip': 'PCIe slot', 'host_id': 'run host ID', 'trace_id': 'trace id',
    'replay': 'trace id counter', 'core_x': 'core_x', 'core_y': 'core_y',
    'cycle': 'time[cycles since reset]', 'risc': 'RISC processor type',
    'zone': 'zone name', 'phase': 'type'}


def read_raw_trace_events(path):
    with Path(path).open(newline='') as stream:
        if not stream.readline().startswith('ARCH: blackhole,'):
            raise ValueError('Pinned Blackhole device log required')
        rows = csv.DictReader(stream, skipinitialspace=True)
        if not set(RAW_FIELDS.values()).issubset(rows.fieldnames or []):
            raise ValueError('Raw device marker columns required, not aggregate durations')
        for row in rows:
            if not row['zone name'].startswith('QWEN_MLP_') or not row['trace id']:
                continue
            event = {field: row[column] for field, column in RAW_FIELDS.items()}
            for field in ('chip', 'host_id', 'trace_id', 'replay', 'core_x', 'core_y', 'cycle'):
                event[field] = int(event[field])
            yield event


def summarize(events, expected_executions, console):
    if 'markers were dropped' in console.lower():
        raise ValueError('Dropped markers invalidate wait attribution')
    expected = set(expected_executions)
    if not expected or {key[0] for key in expected} != {0, 1}:
        raise ValueError('Explicit executions on both chips required')
    if any(len(key) != 4 or any(type(value) is not int or value < 0 or value >= 2**63
            for value in key) for key in expected):
        raise ValueError('Valid chip, host program, trace and replay identities required')
    grouped = defaultdict(list)
    for event in events:
        name = event['zone']
        if not name.startswith('QWEN_MLP_'):
            continue
        if name not in EXPECTED:
            raise ValueError('Unexpected diagnostic scope')
        execution = tuple(event[field] for field in ('chip', 'host_id', 'trace_id', 'replay'))
        if execution not in expected:
            continue
        if event['phase'] not in ('ZONE_START', 'ZONE_END'):
            raise ValueError('Explicit zone endpoints required')
        if type(event['cycle']) is not int or event['cycle'] < 0:
            raise ValueError('Integer device timestamps required')
        if event['risc'] not in ('BRISC', 'NCRISC'):
            raise ValueError('MLP dataflow scopes must identify a dataflow processor')
        key = (execution, event['core_x'], event['core_y'], event['risc'], name)
        grouped[key].append(event)
    counts = defaultdict(Counter)
    records = []
    for (execution, core_x, core_y, risc, name), endpoints in grouped.items():
        if len(endpoints) != 2 or Counter(item['phase'] for item in endpoints) != Counter(
                ZONE_START=1, ZONE_END=1):
            raise ValueError('Exactly one matched scope pair per sampled worker required')
        start = next(item['cycle'] for item in endpoints if item['phase'] == 'ZONE_START')
        end = next(item['cycle'] for item in endpoints if item['phase'] == 'ZONE_END')
        if end < start:
            raise ValueError('Scope end precedes its start')
        counts[execution][name] += 1
        records.append(dict(execution=execution, core=(core_x, core_y), risc=risc,
            zone=name, cycles=end - start))
    if set(counts) != expected or any(counts[key] != EXPECTED for key in expected):
        raise ValueError('Complete sampled scopes for every requested execution required')
    return dict(passed=True, diagnostic_only=True, committed_tg=None, samples=records,
        scope='Sampled remaining waits, not transfer durations, utilization or additive critical-path time')
