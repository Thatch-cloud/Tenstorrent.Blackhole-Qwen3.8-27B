"""Validate synthetic bank lifetime evidence, not learned-model qualification."""

import hashlib
import json
from pathlib import Path


SOURCES = ('dspark-banked-trace-probe.py', 'dspark_banked_proposal.py', 'dspark_prepared_proposal.py',
    'dspark_history.py', 'dspark_fixed_inputs.py', 'dspark_wide_target.py', 'attention_batch.py',
    'gdn_multitoken_conv.py', 'feature_projection.py')


def qualify(path, directory):
    path, directory = Path(path), Path(directory)
    raw = path.read_bytes()
    report = json.loads(raw)
    if (path.with_suffix('.exit-status').read_text().strip() != '0'
            or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('learned_model_executed') is not False
            or report.get('replay_counts') != [2, 2]):
        raise ValueError('Complete clean two-bank simulator evidence required')
    expected_sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCES}
    if report.get('sources') != expected_sources or report.get('sources_after') != expected_sources:
        raise ValueError('Bank simulator sources changed or incomplete')
    expected_outputs = {(ordinal, bank, operand, chip)
        for ordinal, bank in enumerate((0, 1, 1, 0)) for operand in range(10) for chip in range(2)}
    expected_banks = {(ordinal, bank, operand, chip)
        for ordinal in (0, 1, 2, 3, 'after_close') for bank in range(2)
        for operand in range(10) for chip in range(2)}
    for field, expected in (('checks', expected_outputs), ('bank_checks', expected_banks)):
        records = report.get(field, [])
        observed = {(record.get('ordinal'), record.get('bank'), record.get('operand'), record.get('chip'))
            for record in records}
        if len(records) != len(expected) or observed != expected or any(record.get('exact') is not True for record in records):
            raise ValueError('Every bank, chip, operand and replay must be exact: ' + field)
    if report.get('eager_replay_checks') != [dict(position=position, tensors=6, exact=True) for position in range(31, 35)]:
        raise ValueError('All eager/replay audits required')
    return dict(simulator_report_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Synthetic bank lifetime only; learned request audits still required')
