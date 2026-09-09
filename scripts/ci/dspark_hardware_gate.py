"""Scope the hardware integration experiment to unchanged simulator-tested arithmetic."""

import json
from pathlib import Path

from dspark_projection import digest


REPORTS = {
    'dspark-projection-composed-simulator.json':'082e2ba464320b7ad91312e196b8c5f75406a6d2494ed7f1ebafd184b55303dc',
    'dspark-layer-mesh-simulator.json':'1374f6cbd1704a04069d29b8fa0b32db523972298e19699d0b45df2848a8607a',
}


def simulator_preflight(root):
    root = Path(root)
    evidence = {}
    for name,sha in REPORTS.items():
        path = root/name
        if digest(path)!=sha:
            raise ValueError('Pinned complete simulator evidence required: '+name)
        report = json.loads(path.read_text())
        if (report['passed'] is not True or report['closed_cleanly'] is not True
                or report['sources']!=report['sources_after'] or report['native_sources']!=report['native_sources_after']):
            raise ValueError('Closed unchanged simulator component required')
        evidence[name] = report
    qualified = {}
    for report in evidence.values():
        qualified.update(report['sources'])
    required = set(qualified)-{'../../optimisation/sim/run-dispatch-probe.sh'}
    for name in required:
        if name not in qualified or digest(root/name)!=qualified[name]:
            raise ValueError('Changed arithmetic requires targeted simulator evidence: '+name)
    return dict(reports=REPORTS,arithmetic={name:qualified[name] for name in sorted(required)},
        complete_pipeline_simulated=False,retained_cpu_numerical_gate_passed=False)


def native_fingerprints(root, simulator_report):
    root = Path(root)
    names = {name for name in simulator_report['native_sources'] if not name.startswith('simulator/')}
    names.update(('ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async.cpp',
        'ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.cpp',
        'tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto'))
    return {name:digest(root/name) for name in sorted(names)}


def require_compatible_native(native, simulator):
    rebuilt = {'build_Release/lib/_ttnncpp.so','build_Release/ttnn/_ttnncpp.so'}
    for name,sha in simulator.items():
        if name not in rebuilt and not name.startswith('simulator/') and native.get(name)!=sha:
            raise ValueError('Hardware numerical source differs from simulated arithmetic: '+name)
    if any(name not in native for name in rebuilt) or len({native[name] for name in rebuilt})!=1:
        raise ValueError('Both runtime library paths must resolve to the same audited build')
