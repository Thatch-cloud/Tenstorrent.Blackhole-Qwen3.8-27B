"""Weight-free hardware probe: ordered BF8 cache writes at page-table width 2,052.

Opens the two-chip mesh once, allocates a zero BF8 paged cache per case, and drives
ordered_cache.update (the writer baked into the serving image) through the plan in
ordered_cache_hw_plan: a width-1,024 eager control, a width-2,052 eager case, and a
width-2,052 trace capture + replay case whose page table is rewritten in place between
replays. After every step the COMPLETE cache on BOTH chips is compared with the cache
predicted on the host from the host page table. No native page-table read feeds the
expectation. No model weights are loaded and no prefill runs.

All device-free logic (plan, payloads, prediction, comparison, pass/fail) is in
ordered_cache_hw_plan and unit-tested on CPU. run_probe() takes its modules as arguments,
and test_ordered_cache_hw_plan drives it with a fake ttnn, so the tensor plumbing (which
tensor is read back, from which chip, when the prediction is applied) is tested on CPU too.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance(ttnn, torch, home):
    evidence = dict(home=home, ttnn_path=ttnn.__file__, torch_version=torch.__version__)
    try:
        evidence['revision'] = subprocess.check_output(['git', '-C', home, 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as error:
        evidence['revision_error'] = '%s: %s' % (type(error).__name__, error)
    binaries = {}
    extension = getattr(getattr(ttnn, '_ttnn', None), '__file__', None)
    candidates = [extension] if extension else []
    candidates += [os.path.join(home, 'build_Release', 'lib', '_ttnncpp.so'),
                   os.path.join(home, 'ttnn', 'ttnn', '_ttnncpp.so')]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            binaries[candidate] = sha256_file(candidate)
    evidence['binaries_sha256'] = binaries
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence-tree', type=Path, default=Path('/experiment-scripts/ci'))
    parser.add_argument('--fabric', choices=('1d', 'none'), default='1d')
    parser.add_argument('--trace-region-bytes', type=int, default=1048576)
    options = parser.parse_args(argv)

    import ordered_cache_hw_plan as hw

    if Path(hw.__file__).resolve().parent != HERE:
        raise ValueError('ordered_cache_hw_plan must load from beside the probe')
    missing = [name for name in hw.PROBE_ENV['required'] if not os.environ.get(name)]
    forbidden = [name for name in hw.PROBE_ENV['forbidden'] if os.environ.get(name)]
    if missing or forbidden:
        raise ValueError('Environment missing %s, forbidden %s' % (missing, forbidden))
    if (os.environ['QWEN_HARDWARE_TESTS'] != '1' or not Path('/dev/tenstorrent').exists()
            or options.output.exists()):
        raise ValueError('Fresh hardware run required (QWEN_HARDWARE_TESTS=1, /dev/tenstorrent, new output)')
    tree = options.evidence_tree.resolve()
    if str(tree) not in sys.path:
        sys.path.append(str(tree))
    import ordered_cache

    if Path(ordered_cache.__file__).resolve().parent != tree:
        raise ValueError('ordered_cache must be the baked copy in ' + str(tree))
    ordered_sha = sha256_file(ordered_cache.__file__)
    if ordered_sha != os.environ['ORDERED_CACHE_SHA256']:
        raise ValueError('Baked ordered_cache.py %s is not the checkout %s'
                         % (ordered_sha, os.environ['ORDERED_CACHE_SHA256']))
    if not ordered_cache.page_width_admitted(hw.WIDE_WIDTH):
        raise ValueError('Baked ordered_cache.py does not admit width %d' % hw.WIDE_WIDTH)
    home = os.environ['TT_METAL_HOME']
    kernels = ordered_cache.load_kernels(home)

    import torch
    import ttnn

    provenance = dict(ordered_cache_path=str(ordered_cache.__file__), ordered_cache_sha256=ordered_sha,
        ordered_cache_expected_sha256=os.environ['ORDERED_CACHE_SHA256'],
        tt_metal=runtime_provenance(ttnn, torch, home))
    report, error = run_probe(ttnn, torch, ordered_cache, hw, kernels, options.output, provenance,
                              fabric=options.fabric, trace_region_bytes=options.trace_region_bytes)
    if error is not None:
        raise error
    if not report['passed']:
        sys.exit(1)


def run_probe(ttnn, torch, ordered_cache, hw, kernels, output, provenance, fabric='1d',
              trace_region_bytes=1048576):
    """Everything that touches the mesh. Returns (report, error); the report is also
    written to `output` after every step, so a hang still leaves evidence behind."""
    plan = hw.build_plan()
    report = dict(probe='ordered-cache-hw-probe', passed=False, closed_cleanly=False, backend='hardware',
        scope=__doc__, weight_free=True, prefill=False, fabric=fabric,
        trace_region_bytes=trace_region_bytes,
        sources={path.name: sha256_file(path) for path in (Path(__file__), Path(hw.__file__))},
        wide_page_widths=sorted(ordered_cache.WIDE_PAGE_WIDTHS), native_hashes=dict(ordered_cache.HASHES),
        generated_hashes={role: hashlib.sha256(source.encode()).hexdigest() for role, source in kernels.items()},
        plan_sha256=hw.plan_digest(plan), widths=sorted({case['width'] for case in plan['cases']}),
        plan=plan, checks=[], cases=[], timings=[])
    report.update(provenance)

    def write(stage, **details):
        report['stage'] = stage
        output.write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps(dict(stage=stage, **details)), flush=True)

    mesh, trace, owned = None, None, []
    error = None

    def upload(value, dtype):
        result = ttnn.from_torch(value, device=mesh, dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        owned.append(result)
        return result

    def replace(value, destination, dtype):
        host = ttnn.from_torch(value, dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        ttnn.copy_host_to_device_tensor(host, destination)

    def chips(value):
        parts = ttnn.get_device_tensors(value)
        if len(parts) != len(hw.CHIPS):
            raise AssertionError('Both chips required')
        return parts

    def check_cache(case, step, name, cache, expected, **extra):
        for chip, part in enumerate(chips(cache)):
            result = hw.compare_cache(ttnn.to_torch(part), expected)
            report['checks'].append(dict(case=case, step=step, chip=chip, name=name, **extra, **result))

    def check_equal(case, name, value, host):
        for chip, part in enumerate(chips(value)):
            actual = ttnn.to_torch(part)
            exact = tuple(actual.shape) == tuple(host.shape) and torch.equal(actual.to(host.dtype), host)
            report['checks'].append(dict(case=case, step=None, chip=chip, name=name, exact=bool(exact)))

    try:
        if fabric == '1d':
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
                                     trace_region_size=trace_region_bytes)
        mesh.enable_program_cache()
        for case in plan['cases']:
            name = case['name']
            write('allocate', case=name, width=case['width'], blocks=plan['blocks'])
            expected = hw.ExpectedCache(plan['blocks'])
            cache = upload(expected.values.clone(), ttnn.bfloat8_b)
            check_cache(name, None, 'zero_baseline', cache, expected)
            tables = [torch.tensor(table, dtype=torch.int32) for table in case['tables']]
            pages = upload(tables[0], ttnn.int32)
            check_equal(name, 'pages_uploaded', pages, tables[0])
            first = case['steps'][0]
            positions = upload(torch.tensor(first['positions'], dtype=torch.int32), ttnn.int32)
            packed = upload(hw.step_payloads(first).unsqueeze(0).contiguous(), ttnn.bfloat16)
            current = 0

            def operation():
                ordered_cache.update(mesh, cache, packed, positions, pages, kernels)

            for step in case['steps']:
                started = time.monotonic()
                payloads = hw.step_payloads(step)
                if step['table'] != current:
                    replace(tables[step['table']], pages, ttnn.int32)
                    current = step['table']
                replace(torch.tensor(step['positions'], dtype=torch.int32), positions, ttnn.int32)
                replace(payloads.unsqueeze(0).contiguous(), packed, ttnn.bfloat16)
                if case['mode'] == 'eager':
                    operation()
                else:
                    if trace is None:
                        # Warm-up compiles the program on these exact tensors; the step-0
                        # write is idempotent, and steps >= 1 are written by replay alone.
                        operation()
                        ttnn.synchronize_device(mesh)
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        try:
                            operation()
                        finally:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh)
                expected.apply(case['tables'][step['table']], step['positions'], payloads)
                check_cache(name, step['step'], 'complete_cache', cache, expected, mode=case['mode'],
                            table=step['table'], positions=step['positions'])
                report['timings'].append(dict(case=name, step=step['step'],
                                              seconds=round(time.monotonic() - started, 3)))
                latest = report['checks'][-len(hw.CHIPS):]
                write('step', case=name, step=step['step'], exact=[check['exact'] for check in latest],
                      mismatched_blocks=[check['mismatched_blocks'] for check in latest])
            check_equal(name, 'input_unchanged', packed, payloads.unsqueeze(0).contiguous())
            check_equal(name, 'pages_unchanged', pages, tables[current])
            if trace is not None:
                ttnn.release_trace(mesh, trace)
                trace = None
            ttnn.synchronize_device(mesh)
            for value in reversed(owned):
                ttnn.deallocate(value)
            owned.clear()
    except BaseException as caught:
        error = caught
        report['error'] = '%s: %s' % (type(caught).__name__, caught)
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                for value in reversed(owned):
                    ttnn.deallocate(value)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            report['cases'] = hw.summarise_cases(report, plan)
            report['failures'] = hw.check_report(report, plan)
            report['passed'] = not report['failures']
            write('complete', passed=report['passed'], cases=[(case['name'], case['passed']) for case in report['cases']])
    return report, error


if __name__ == '__main__':
    main()
