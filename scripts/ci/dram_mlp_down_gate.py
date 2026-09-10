"""Require the exact simulator and two independent qualifying hardware MLPS before target integration."""

import hashlib
import json

from dram_mlp_gate import NATIVE_SOURCES, qualify, qualify_hardware, variant_sources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify_repeated(root, native_root):
    sources = {name: digest(root / name) for name in variant_sources(False, True)}
    native = {name: digest(native_root / name) for name in NATIVE_SOURCES}
    simulator_path = root / 'dram-mlp-down-simulator.json'
    simulator = json.loads(simulator_path.read_text())
    qualify(simulator, sources, native, (root / 'dram-mlp-down-simulator.exit-status').read_text().strip(),
        hardware=True, down=True)
    results, fingerprints = [], []
    for name in ('dram-mlp-down-hardware.json', 'dram-mlp-down-repeat.json'):
        path = root / name
        report = json.loads(path.read_text())
        if (report.get('down_only') is not True or report.get('sharded_product') is not False
                or report.get('sources') != sources or report.get('native_sources') != native
                or report.get('hardware_script_sha256') != digest(root / 'dram-mlp-hardware.py')
                or report.get('simulator_report_sha256') != digest(simulator_path)):
            raise ValueError('Exact unchanged down-only simulator, native implementation and hardware harness required')
        result = qualify_hardware(report)
        if result['eligible_for_full_model_gate'] is not True:
            raise ValueError('Both independent hardware results must qualify')
        results.append(result)
        fingerprints.append(digest(path))
    if len(set(fingerprints)) != 2:
        raise ValueError('Distinct repeat artifacts required')
    return dict(passed=True, hardware_sha256=fingerprints, results=results,
        scope='Repeated single-layer improvement; full-model exactness and PP/CTX/TG still required')
