"""Prepare historical target replay for selected geometry and runtime BF16 KV."""

import hashlib
from pathlib import Path

from frozen_context_geometry import geometry
from target_t16_attention_8k_gate import SOURCES


def validate_target_report(report, directory, context, *, compact_scratch=False):
    if type(compact_scratch) is not bool:
        raise ValueError('Explicit target scratch qualification selection required')
    if report.get('compact_tree_scratch', False) is not compact_scratch:
        raise ValueError('Target scratch variant differs from requested qualification')
    if compact_scratch:
        from sdpa_tree_scratch import HASHES, PATCHED_FACTORY_SHA256
        expected_native = dict(HASHES)
        expected_native['sdpa_decode_program_factory.cpp'] = PATCHED_FACTORY_SHA256
        if (report.get('native_sources') != expected_native
                or report.get('native_sources_after') != expected_native):
            raise ValueError('Pinned compact scratch native sources required')
        build = report.get('factory_build', {})
        binaries = build.get('binaries_after', {})
        if (build.get('passed') is not True or build.get('import_passed') is not True
                or set(binaries) != {'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'}
                or len(set(binaries.values())) != 1
                or any(not isinstance(value, str) or len(value) != 64
                    or any(character not in '0123456789abcdef' for character in value)
                    for value in binaries.values())):
            raise ValueError('Audited matching compact scratch binary identities required')
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
        compact_tree_scratch=compact_scratch,
        component_qualified=True, full_request_qualified=False, performance_qualified=False)


def adapt_target_probe(source):
    from frozen_recipe_context import replace_once
    source = replace_once(source, '    mesh = None\n',
        "    scratch = os.environ.get('QWEN_FROZEN_TARGET_SCRATCH', '0')\n"
        "    if scratch not in ('0', '1'):\n"
        "        raise ValueError('Explicit scratch variant required')\n"
        "    report['compact_tree_scratch'] = scratch == '1'\n"
        "    if scratch == '1':\n"
        '        from sdpa_tree_scratch import audit\n'
        '        from dspark_fp32_build import validate_manifest\n'
        "        if os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':\n"
        "            raise ValueError('Compiled scratch candidate must be explicitly enabled')\n"
        "        report['native_sources'] = audit('/opt/tt-metal', patched=True)\n"
        "        report['factory_build'] = validate_manifest('/opt/tt-metal', '/experiment/results/dspark-fp32-build.json')\n"
        '    mesh = None\n')
    source = replace_once(source, "        report['closed'] = mesh is not None",
        "        if scratch == '1':\n"
        "            report['native_sources_after'] = audit('/opt/tt-metal', patched=True)\n"
        "        report['closed'] = mesh is not None")
    reader = '                reader = ReplayAttentionReader(ttnn, mesh, rows, capacity, pages_host, upload, short_context=False)\n'
    source = replace_once(source, reader, '')
    source = replace_once(source, '                for start, ticket_query in zip(starts, queries, strict=True):',
        reader +
        '                allocation_check = reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)\n'
        '                allocation_host = host(allocation_check)\n'
        '                ttnn.deallocate(allocation_check)\n'
        "                print(json.dumps(dict(stage='target-allocation-ready', capacity=capacity)), flush=True)\n"
        '                for start, ticket_query in zip(starts, queries, strict=True):')
    source = replace_once(source, '                    outputs = []\n',
        '                    if gold and start == starts[0] and torch.equal(ticket_query, queries[0]):\n'
        '                        gold.append([value.clone() for value in gold[0]])\n'
        "                        print(json.dumps(dict(stage='native-reference-reused', capacity=capacity, start=start)), flush=True)\n"
        '                        continue\n'
        '                    outputs = []\n')
    source = replace_once(source,
        '                warm = reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)\n'
        '                try:\n'
        '                    if any(not torch.equal(actual, expected) for actual, expected in zip(host(warm), gold[0], strict=True)):\n'
        "                        raise AssertionError('T16 long-context warm output differs from native B1')\n"
        '                finally:\n'
        '                    ttnn.deallocate(warm)\n',
        '                failures = []\n'
        '                for index, (actual, expected) in enumerate(\n'
        '                        zip(allocation_host, gold[0], strict=True)):\n'
        '                    if actual.shape != expected.shape or actual.dtype != expected.dtype:\n'
        "                        raise AssertionError('T16 and native B1 disagree on shape or dtype')\n"
        '                    left = actual.to(torch.float32)\n'
        '                    right = expected.to(torch.float32)\n'
        '                    difference = (left - right).abs()\n'
        '                    differing = int((actual != expected).sum())\n'
        '                    nonfinite = int((~torch.isfinite(left)).sum())\n'
        '                    # One ulp at the REFERENCE TENSOR SCALE, not per element. A\n'
        '                    # per-element denominator collapses on near-zero entries: run\n'
        '                    # 35665484092 reported max_ulp 91393 for a worst difference of\n'
        '                    # 3.05e-05 on a tensor whose largest value is 0.0028, because the\n'
        '                    # zeros divided by almost nothing. Against the tensor scale that\n'
        '                    # same difference is about 2 ulp, which is the interpretable number.\n'
        '                    magnitude = float(right.abs().max())\n'
        '                    ulp = magnitude * 2.0 ** -7 if magnitude > 0 else 2.0 ** -133\n'
        '                    error = difference / ulp\n'
        '                    worst = float(error.max())\n'
        '                    rows = torch.nonzero(error.reshape(-1, error.shape[-1]).amax(dim=-1) > MAX_ULP,\n'
        '                        as_tuple=False).flatten()\n'
        "                    print(json.dumps(dict(stage='t16-b1-ulp', tensor=index, start=start,\n"
        '                        shape=list(actual.shape), elements=int(actual.numel()),\n'
        '                        differing=differing,\n'
        '                        max_ulp=worst, mean_ulp=float(error.mean()), budget_ulp=MAX_ULP,\n'
        '                        scale=magnitude, ulp_of_scale=ulp,\n'
        '                        max_abs=float(difference.max()), mean_abs=float(difference.mean()),\n'
        '                        nonfinite=nonfinite, rows_over_budget=int(rows.numel()),\n'
        '                        first_rows=[int(value) for value in rows[:8]])), flush=True)\n'
        '                    if nonfinite or differing:\n'
        '                        failures.append(index)\n'
        '                if failures:\n'
        "                    raise AssertionError('T16 long-context warm output differs from native B1')\n")
    source = replace_once(source, 'from pathlib import Path',
        'from pathlib import Path\n'
        '# The bar is bit-exactness, restored 2026-09-22. Run 35663000515 showed T16 and\n'
        '# native B1 agree exactly at k_chunk_size 256 - the value B1 itself picks - at all\n'
        '# four start positions. The earlier disagreement came entirely from the 65536-only\n'
        '# override to 128, which folds the online softmax in a different order; run\n'
        '# 35662960713 measured that at one ulp across all 192 rows. Exactness is free here,\n'
        '# so a tolerance would be accepting a numerics regression for nothing.\n'
        '#\n'
        '# MAX_ULP is NOT the pass condition. It classifies a failure the moment one happens:\n'
        '# max_ulp near 1 with every row affected is a merge-order difference, while rows over\n'
        '# this threshold are structural - a wrong row moves by order the value itself,\n'
        '# hundreds of ulp. The stats print on every comparison, pass or fail, so an exact run\n'
        '# says so with max_ulp 0.0 and drift is visible the moment it appears rather than\n'
        '# arriving as a bare inequality with no magnitude.\n'
        'MAX_ULP = 4.0\n')
    source = replace_once(source, '    import torch\n',
        '    from attention_mask_replay import validate_ticket\n'
        "    capacity = selected_geometry()['capacity']\n"
        '    for start in (capacity - 256, capacity - 239, capacity - 16):\n'
        '        validate_ticket(start, 16, capacity, short_context=False)\n'
        '    import torch\n')
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


def adapt_target_mask(source):
    from frozen_recipe_context import replace_once
    source = replace_once(source, 'import hashlib',
        'import hashlib\nfrom frozen_context_geometry import selected_geometry')
    return replace_once(source,
        '    minimum, maximum = (256, 768) if short_context else (4096, 16640)',
        '    minimum, maximum = (256, 768) if short_context else (4096, 16640)\n'
        "    if not short_context and capacity == selected_geometry()['capacity']:\n"
        '        maximum = max(maximum, capacity)')
