"""Qualify complete learned DSpark layer arithmetic and replay, never host transport or hardware throughput."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_layer import ADDITIONAL, EXACT, INPUT_COUNTS, PHASES, POLICY, REPLAY_CASES, SPECIFICATIONS
from dspark_layer_reference import CASES, PROJECTION_OPERANDS_SHA256, PROJECTION_REPORT_SHA256
from dspark_markov_gate import coordinates
from dspark_projection import TOLERANCE, reference_metadata


def qualify(report, *, sources, native, reference, exit_status):
    if (exit_status.strip() != '0' or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('checkpoint_closed') is not True or report.get('stage') != 'complete'
            or report.get('error') or report.get('cleanup_error') or report.get('backend') != 'simulator'
            or report.get('mode') != 'matrix' or report.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or any(type(report.get(key)) is not int or report[key] != value for key,value in
                (('layer',0),('context_rows',32),('proposal_rows',7)))
            or report.get('cases') != [list(value) for value in CASES] or report.get('policy') != POLICY
            or report.get('tolerance') != TOLERANCE or report.get('reference') != reference
            or report.get('precise_native') is not True or report.get('packer_compat') is not True
            or report.get('projection_report_sha256') != PROJECTION_REPORT_SHA256
            or report.get('projection_operands_sha256') != PROJECTION_OPERANDS_SHA256
            or not sources or not native or report.get('sources') != sources or report.get('sources_after') != sources
            or report.get('native_sources') != native or report.get('native_sources_after') != native
            or any(report.get(key) is not False for key in ('fabric_tested','full_pipeline_captured','target_integrated','eligible_for_hardware'))):
        raise ValueError('Complete source-bound layer matrix, preserved numerical policy and non-hardware scope required')
    parameters = {name:reference['tensor_sha256']['layers.0.'+name] for name in SPECIFICATIONS}
    if report.get('parameter_sha256') != parameters:
        raise ValueError('All eleven pinned layer parameters required')
    coordinates(report.get('cpu_checks'),('case','stage'),
        {(case,stage) for case in range(3) for stage in ('attention_residual','output')},('exact',))
    coordinates(report.get('eager_checks'),('phase','case','chip','stage'),
        {(phase,case,chip,stage) for phase,stages in PHASES.items() for case in range(3) for chip in range(2)
            for stage in (*stages,*ADDITIONAL[phase])},('passed',))
    for entry in report['eager_checks']:
        exact = (entry['phase'],entry['stage']) in EXACT
        if (entry.get('exact_required') is not exact or type(entry.get('bitwise_exact')) is not bool
                or (exact and entry['bitwise_exact'] is not True) or type(entry.get('failed_elements')) is not int
                or entry['failed_elements'] != 0 or type(entry.get('max_abs')) not in (int,float)
                or not math.isfinite(entry['max_abs']) or entry['max_abs'] < 0):
            raise ValueError('Every numerical and exact layer comparison must pass unchanged')
    coordinates(report.get('replay_checks'),('phase','ordinal','case','chip','stage'),
        {(phase,ordinal,case,chip,stage) for phase,stages in PHASES.items() for ordinal,case in enumerate(REPLAY_CASES)
            for chip in range(2) for stage in stages},('exact','bindings_stable'))
    coordinates(report.get('input_checks'),('phase','mode','ordinal','case','chip','tensor'),
        {(phase,mode,ordinal,case,chip,index) for phase,count in INPUT_COUNTS.items()
            for mode,cases in (('eager',range(3)),('replay',REPLAY_CASES)) for ordinal,case in enumerate(cases)
            for chip in range(2) for index in range(count)},('exact',))
    coordinates(report.get('parameter_checks'),('phase','chip','tensor'),
        {(phase,chip,name) for phase in ('before','after') for chip in range(2) for name in SPECIFICATIONS},('exact','bindings_stable'))
    coordinates(report.get('stale_controls'),('phase','chip'),
        {(phase,chip) for phase in PHASES for chip in range(2)},('detected',))
    coordinates(report.get('padding_checks'),('phase','case','chip'),
        {(phase,case,chip) for phase in ('attention','finish') for case in range(3) for chip in range(2)},('zero',))
    counts = {name:len(report[name]) for name in ('cpu_checks','eager_checks','replay_checks','input_checks',
        'parameter_checks','stale_controls','padding_checks')}
    return dict(passed=True,checks=sum(counts.values()),counts=counts,layer=0,fabric_tested=False,
        full_pipeline_captured=False,target_integrated=False,eligible_for_hardware=False,
        scope='Complete layer-zero arithmetic and per-phase replay only; host-staged TP handoffs are not fabric')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report','exit-status','metal-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('dspark_layer_probe',Path(__file__).with_name('dspark-layer-probe.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(qualify(json.loads(options.report.read_text()),sources=module.source_hashes(),
        native=module.fingerprints(options.metal_root,active=False),reference=reference_metadata(),
        exit_status=options.exit_status.read_text())))


if __name__ == '__main__':
    main()
