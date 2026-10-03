"""Source-bound learned DSpark FC and normalization evidence; never qualify simulator host transport."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_intake import TAPS
from dspark_markov_gate import coordinates
from dspark_projection import COMPOSED_POLICY, EAGER_STAGES, EXACT_STAGES, HANDOFF, POLICY, PROJECTION_STAGES, TAIL_STAGES, TOLERANCE


def qualify(report, *, sources, native, reference, exit_status, packer_compat=False, composed_norm=False):
    if (exit_status.strip() != '0' or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('error') or report.get('cleanup_error')
            or report.get('backend') != 'simulator' or report.get('mode') != 'matrix'
            or report.get('checkpoint_sha256') != CHECKPOINT_SHA256
            or report.get('reference') != reference or report.get('policy') != (COMPOSED_POLICY if composed_norm else POLICY)
            or type(composed_norm) is not bool or report.get('composed_norm') is not composed_norm
            or report.get('tolerance') != TOLERANCE
            or report.get('handoff') != HANDOFF or report.get('taps') != list(TAPS)
            or type(packer_compat) is not bool or report.get('packer_compat') is not packer_compat
            or any(report.get(key) is not False for key in ('fabric_tested','full_pipeline_captured',
                'target_integrated','eligible_for_hardware'))
            or any(type(report.get(key)) is not int or report[key] != expected for key, expected in
                (('rows',32),('input_width',25600),('output_width',5120),('fixtures',2)))
            or not sources or not native or report.get('sources') != sources or report.get('sources_after') != sources
            or report.get('native_sources') != native or report.get('native_sources_after') != native
            or report.get('checkpoint_closed') is not True):
        raise ValueError('Complete source-bound learned FC/norm simulation and explicit non-fabric scope required')
    expected_parameters = {name: reference['tensor_sha256'][name] for name in ('fc.weight','hidden_norm.weight')}
    if report.get('parameter_sha256') != expected_parameters:
        raise ValueError('Both complete learned parameter tensors must match the frozen CPU backbone')
    coordinates(report.get('eager_checks'), ('pattern','chip','stage'),
        {(pattern,chip,stage) for pattern in range(2) for chip in range(2) for stage in EAGER_STAGES}, ('passed',))
    for row in report['eager_checks']:
        if (row.get('exact_required') is not (row['stage'] in EXACT_STAGES)
                or type(row.get('bitwise_exact')) is not bool
                or (row['stage'] in EXACT_STAGES and row['bitwise_exact'] is not True)
                or type(row.get('failed_elements')) is not int or row['failed_elements'] != 0
                or type(row.get('max_abs')) not in (int,float) or not math.isfinite(row['max_abs']) or row['max_abs'] < 0):
            raise ValueError('Every declared exact stage and unchanged numerical threshold must pass')
    coordinates(report.get('replay_checks'), ('phase','ordinal','pattern','chip','stage'),
        {(phase,ordinal,pattern,chip,stage) for phase,stages in (('projection',PROJECTION_STAGES),('tail',TAIL_STAGES))
            for ordinal,pattern in enumerate((0,1,0)) for chip in range(2) for stage in stages}, ('exact','bindings_stable'))
    coordinates(report.get('input_checks'), ('phase','mode','ordinal','pattern','chip','tensor'),
        {(phase,mode,ordinal,pattern,chip,index) for phase,count in (('projection',5),('tail',3))
            for mode,patterns in (('eager',(0,1)),('replay',(0,1,0))) for ordinal,pattern in enumerate(patterns)
            for chip in range(2) for index in range(count)}, ('exact',))
    coordinates(report.get('parameter_checks'), ('phase','chip','tensor'),
        {(phase,chip,name) for phase in ('before','after') for chip in range(2) for name in ('weight','gamma')}, ('exact',))
    coordinates(report.get('stale_controls'), ('phase','chip'),
        {(phase,chip) for phase in ('projection','tail') for chip in range(2)}, ('detected',))
    coordinates(report.get('rounding_controls'), ('pattern',), {(pattern,) for pattern in range(2)}, ('distinguished',))
    counts = {name:len(report[name]) for name in ('eager_checks','replay_checks','input_checks',
        'parameter_checks','stale_controls','rounding_controls')}
    return dict(passed=True, checks=sum(counts.values()), counts=counts, fabric_tested=False,
        full_pipeline_captured=False, target_integrated=False, eligible_for_hardware=False,
        scope='Learned projection and normalization arithmetic only; host handoff is not fabric or model integration')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--exit-status', type=Path, required=True)
    parser.add_argument('--metal-root', type=Path, required=True)
    parser.add_argument('--packer-compat', action='store_true')
    parser.add_argument('--composed-norm', action='store_true')
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location('dspark_projection_probe', Path(__file__).with_name('dspark-projection-probe.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    native = probe.fingerprints(options.metal_root,packer_compat=False,composed_norm=options.composed_norm)
    if options.packer_compat:
        native[probe.PACKER] = probe.COMPAT_PACKER
    print(json.dumps(qualify(json.loads(options.report.read_text()), sources=probe.source_hashes(), native=native,
        reference=probe.reference_metadata(), exit_status=options.exit_status.read_text(),
        packer_compat=options.packer_compat,composed_norm=options.composed_norm)))


if __name__ == '__main__':
    main()
