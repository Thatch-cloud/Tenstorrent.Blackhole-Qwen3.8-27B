"""Prepare historical target replay for selected geometry and runtime BF16 KV."""

import hashlib
from pathlib import Path

from frozen_context_geometry import geometry
from frozen_recipe_context import replace_once
from target_t16_attention_8k_gate import SOURCES


def validate_target_report(report, directory, context):
    capacity = geometry(context)['capacity']
    if (report.get('passed') is not True or report.get('closed') is not True
            or report.get('backend') != 'simulator' or report.get('context') != context
            or report.get('kv_dtype') != 'bfloat16' or report.get('error')):
        raise ValueError('Complete selected-context BF16 target replay required')
    hashes = report.get('sources', {})
    if set(hashes) != SOURCES | {'frozen_context_geometry.py'} or hashes != report.get('sources_after'):
        raise ValueError('Complete stable target source identities required')
    for name, checksum in hashes.items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Target source changed: ' + name)
    expected = {(capacity, start, ticket, chip)
        for ticket, start in enumerate((context, context + 17, capacity - 16, context))
        for chip in range(2)}
    for field, copies in (('checks', 1), ('mask_checks', 2)):
        records = report.get(field, [])
        coordinates = [(record.get('capacity'), record.get('start'), record.get('ticket'), record.get('chip'))
            for record in records]
        if (len(coordinates) != len(expected) * copies or set(coordinates) != expected
                or any(coordinates.count(coordinate) != copies for coordinate in expected)
                or any(record.get('exact') is not True for record in records)):
            raise ValueError('Complete exact target replay and mask coverage required')
    records = report.get('source_checks', [])
    if (len(records) != 4 or sorted(record.get('chip', -1) for record in records) != [0, 0, 1, 1]
            or any(record.get('capacity') != capacity or record.get('exact') is not True for record in records)):
        raise ValueError('Unchanged target KV evidence required')
    records = report.get('unpoisoned_replay', [])
    if (len(records) != 2 or {record.get('chip') for record in records} != {0, 1}
            or any(record.get('exact') is not True or record.get('nonfinite') != 0
                or record.get('mismatches') != 0 for record in records)
            or report.get('stale_controls') != 2 or report.get('mask_poison_controls') != 8):
        raise ValueError('Target replay negative controls incomplete')
    return dict(context=context, capacity=capacity, kv_dtype='bfloat16',
        component_qualified=True, full_request_qualified=False, performance_qualified=False)


def adapt_target_probe(source):
    source = replace_once(source, 'from pathlib import Path',
        'from pathlib import Path\nfrom frozen_context_geometry import selected_geometry')
    source = replace_once(source,
        "scope='T16 long-context attention component, at CTX8192; not full-model correctness or TG',",
        "scope='Selected-context BF16 KV target replay; not full-model correctness or TG',\n"
        "        context=selected_geometry()['context'], kv_dtype='bfloat16',")
    source = replace_once(source, "'target-t16-attention-8k-probe.py')}",
        "'target-t16-attention-8k-probe.py', 'frozen_context_geometry.py')}")
    source = replace_once(source, 'for capacity in (8448,):',
        "for capacity in (selected_geometry()['capacity'],):")
    for name in ('keys', 'values'):
        source = replace_once(source,
            f'{name} = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)',
            f'{name} = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat16)')
    compile(source, 'target-t16-attention-8k-probe.py', 'exec')
    return source
