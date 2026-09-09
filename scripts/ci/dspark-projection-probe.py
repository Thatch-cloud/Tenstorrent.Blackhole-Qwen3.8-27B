"""Learned DSpark full FC and normalization simulation; separate traces and explicit host-staged peer partials."""

import argparse
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_checkpoint import CHECKPOINT_SHA256
from dspark_intake import TAPS
from dspark_projection import EXACT_STAGES, HANDOFF, POLICY, PROJECTION_STAGES, TAIL_STAGES, TOLERANCE
from dspark_projection import difference, digest, load_reference, norm_references, normalize_partials, project, reference_metadata, tensor_digest
from dspark_weights import VerifiedWeights
from feature_projection import projection_shards, require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dspark-projection-probe.py','dspark_projection.py','dspark_projection_gate.py',
    'dspark-backbone-cpu.py','dspark_backbone_reference.py','dspark_weights.py','dspark_checkpoint.py',
    'dspark_intake.py','dspark_markov_fixture.py','dspark_rope_tables.py','feature_projection.py',
    'attention_batch.py','gdn_multitoken_conv.py','dspark_markov_gate.py',
    'dspark-backbone-cpu-reference.json','dspark-backbone-upstream-reference.json',
    '../../optimisation/sim/run-dispatch-probe.sh')
PACKER = 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
ORIGINAL_PACKER = '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181'
COMPAT_PACKER = '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7'
BINARY_SHA256 = 'd2652fc01a6836b4d567a788a9c11d8f6cb238bb480bbf68d0e32ee4037c3e24'


def source_hashes():
    return {name:digest(Path(__file__).parent/name) for name in SOURCES}


def fingerprints(root, *, packer_compat=False):
    binaries = ('build_Release/lib/_ttnncpp.so','build_Release/ttnn/_ttnncpp.so')
    paths = {Path(PACKER), *map(Path,binaries), *map(Path,('tt_metal/hw/inc/api/dataflow/dataflow_api.h',
        'tt_metal/hw/inc/api/dataflow/noc.h','tt_metal/hw/inc/api/tensor/tensor_accessor.h'))}
    for name in ('matmul','normalization/rmsnorm','normalization/layernorm','normalization/kernel_util',
            'eltwise/unary','eltwise/binary','eltwise/binary_ng','data_movement/concat'):
        directory = root / 'ttnn/cpp/ttnn/operations' / name
        if not directory.is_dir():
            raise ValueError('Complete native operation source trees required')
        paths.update(path.relative_to(root) for path in directory.rglob('*')
            if path.is_file() and path.suffix in ('.cpp','.hpp','.h'))
    result = {str(path):digest(root/path) for path in sorted(paths)}
    if (type(packer_compat) is not bool or result[PACKER] != (COMPAT_PACKER if packer_compat else ORIGINAL_PACKER)
            or any(result[name] != BINARY_SHA256 for name in binaries)):
        raise ValueError('Pinned binaries and explicit original/compatibility packer scope required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cpu-outputs', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if os.environ.get('QWEN_HARDWARE_TESTS') == '1' or os.environ.get('QWEN_CARDS_ALLOCATED') == '1':
        raise RuntimeError('Simulator-only component; no hardware allocation accepted')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME'])
    packer_compat = os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') == '1'
    report = dict(passed=False, closed_cleanly=False, checkpoint_closed=False, backend='simulator', scope=__doc__,
        rows=32, input_width=25600, output_width=5120, fixtures=2, taps=list(TAPS), policy=POLICY,
        tolerance=TOLERANCE, handoff=HANDOFF, fabric_tested=False, full_pipeline_captured=False,
        target_integrated=False, eligible_for_hardware=False, checkpoint_sha256=CHECKPOINT_SHA256,
        packer_compat=packer_compat, sources=source_hashes(), native_sources=fingerprints(root, packer_compat=packer_compat),
        eager_checks=[], replay_checks=[], input_checks=[], parameter_checks=[], stale_controls=[], rounding_controls=[])
    mesh = trace = weights = None
    owned, transient = [], []

    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)

    def retain(value):
        transient.append(value)
        return value

    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()

    def check(actual, expected, pattern, chip, stage):
        report['eager_checks'].append(dict(pattern=pattern, chip=chip, stage=stage,
            **difference(actual, expected, exact=stage in EXACT_STAGES)))

    try:
        progress('verifying_frozen_cpu_reference_and_checkpoint')
        features, contexts, report['reference'] = load_reference(options.cpu_outputs)
        weights = VerifiedWeights(options.checkpoint)
        weight, gamma = (weights.tensor(name) for name in ('fc.weight','hidden_norm.weight'))
        report['parameter_sha256'] = {name:weights.fingerprints()[name] for name in ('fc.weight','hidden_norm.weight')}
        if any(report['parameter_sha256'][name] != report['reference']['tensor_sha256'][name]
                for name in report['parameter_sha256']):
            raise ValueError('Learned parameters differ from the frozen CPU backbone')
        packed = projection_shards(weight)
        del weight
        local_inputs = {pattern:[torch.cat([features[pattern][layer][...,chip*2560:(chip+1)*2560]
            for layer in TAPS], dim=-1).unsqueeze(1) for chip in range(2)] for pattern in range(2)}
        partial_references = {pattern:[value.float() @ shard.float() for value,shard in zip(local_inputs[pattern],packed,strict=True)]
            for pattern in range(2)}
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()

        def upload(value, dtype, mapper, device=True):
            return ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper,
                **(dict(device=mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))

        def keep(value):
            owned.append(value)
            return value

        def disjoint():
            bindings = [addresses(ttnn,value) for value in owned]
            if any(len({binding[chip] for binding in bindings}) != len(bindings) for chip in range(2)):
                raise AssertionError('All persistent feature, weight and handoff buffers must be disjoint')

        device_weight = keep(upload(torch.stack(packed).unsqueeze(1), ttnn.bfloat16, ttnn.ShardTensorToMesh(mesh,dim=0)))
        device_gamma = keep(upload(gamma.reshape(1,1,1,5120), ttnn.bfloat16, ttnn.ReplicateTensorToMesh(mesh)))
        feature_values = {layer:keep(upload(features[0][layer].unsqueeze(1),ttnn.bfloat16,
            ttnn.ShardTensorToMesh(mesh,dim=-1))) for layer in TAPS}
        payloads = {pattern:[upload(features[pattern][layer].unsqueeze(1),ttnn.bfloat16,
            ttnn.ShardTensorToMesh(mesh,dim=-1),False) for layer in TAPS] for pattern in range(2)}

        def audit_parameters(phase):
            for name,value in (('weight',device_weight),('gamma',device_gamma)):
                for chip in range(2):
                    expected = packed[chip].reshape(1,1,12800,5120) if name == 'weight' else gamma.reshape(1,1,1,5120)
                    if tensor_digest(host(value,chip)) != tensor_digest(expected):
                        raise AssertionError('Borrowed learned parameter bits changed')
                    report['parameter_checks'].append(dict(phase=phase,chip=chip,tensor=name,exact=True))

        disjoint()
        audit_parameters('before')
        measured_partials, eager = {}, {}
        tail_inputs = tail_payloads = None

        for phase, stages in (('projection',PROJECTION_STAGES),('tail',TAIL_STAGES)):
            if phase == 'projection':
                borrowed = list(feature_values.values())
                expected_inputs = {pattern:[[features[pattern][layer][...,chip*2560:(chip+1)*2560].unsqueeze(1)
                    for chip in range(2)] for layer in TAPS] for pattern in range(2)}

                def update(pattern):
                    for source,destination in zip(payloads[pattern],borrowed,strict=True):
                        ttnn.copy_host_to_device_tensor(source,destination)
                    ttnn.synchronize_device(mesh)

                def run():
                    return project(ttnn,feature_values,device_weight,retain)
            else:
                tail_inputs = [keep(upload(value,ttnn.float32,ttnn.ReplicateTensorToMesh(mesh)))
                    for value in measured_partials[0]]
                tail_payloads = {pattern:[upload(value,ttnn.float32,ttnn.ReplicateTensorToMesh(mesh),False)
                    for value in measured_partials[pattern]] for pattern in range(2)}
                borrowed = [*tail_inputs,device_gamma]
                expected_inputs = {pattern:[[value,value] for value in measured_partials[pattern]]
                    + [[gamma.reshape(1,1,1,5120)]*2] for pattern in range(2)}
                disjoint()

                def update(pattern):
                    for source,destination in zip(tail_payloads[pattern],tail_inputs,strict=True):
                        ttnn.copy_host_to_device_tensor(source,destination)
                    ttnn.synchronize_device(mesh)

                def run():
                    return normalize_partials(ttnn,*tail_inputs,device_gamma,retain)

            input_bindings = [addresses(ttnn,value) for value in borrowed]

            def audit_inputs(mode,ordinal,pattern):
                if [addresses(ttnn,value) for value in borrowed] != input_bindings:
                    raise AssertionError('Input tensor bindings changed')
                for index,value in enumerate(borrowed):
                    for chip in range(2):
                        if tensor_digest(host(value,chip)) != tensor_digest(expected_inputs[pattern][index][chip]):
                            raise AssertionError('Borrowed input bits changed')
                        report['input_checks'].append(dict(phase=phase,mode=mode,ordinal=ordinal,pattern=pattern,
                            chip=chip,tensor=index,exact=True))

            for pattern in range(2):
                progress(f'{phase}_eager_{pattern}')
                update(pattern)
                result = run()
                ttnn.synchronize_device(mesh)
                observed = {stage:[host(result[stage],chip) for chip in range(2)] for stage in stages}
                eager[phase,pattern] = observed
                if phase == 'projection':
                    measured_partials[pattern] = observed['partial']
                    for chip in range(2):
                        check(observed['joined'][chip],local_inputs[pattern][chip],pattern,chip,'joined')
                        check(observed['partial'][chip],partial_references[pattern][chip],pattern,chip,'partial')
                else:
                    expected = norm_references(*measured_partials[pattern],gamma)
                    report['rounding_controls'].append(dict(pattern=pattern,distinguished=True))
                    for chip in range(2):
                        for stage in ('sum','narrowed','unweighted_norm'):
                            check(observed[stage][chip],expected[stage],pattern,chip,stage)
                        check(observed['context'][chip],observed['unweighted_norm'][chip]*gamma,pattern,chip,'context')
                        check(observed['context'][chip],expected['native_input_norm'],pattern,chip,'native_input_norm')
                        check(observed['context'][chip],contexts[pattern],pattern,chip,'backbone_context')
                audit_inputs('eager',pattern,pattern)
                release_owned(ttnn,transient)
                transient.clear()
            update(0)
            progress(f'{phase}_capture')
            trace,result = capture_operation(ttnn,mesh,run)
            output_bindings = {stage:addresses(ttnn,result[stage]) for stage in stages}
            for ordinal,pattern in enumerate((0,1,0)):
                progress(f'{phase}_replay_{ordinal}')
                update(pattern)
                ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
                if output_bindings != {stage:addresses(ttnn,result[stage]) for stage in stages}:
                    raise AssertionError('Captured output bindings changed')
                for stage in stages:
                    for chip in range(2):
                        if tensor_digest(host(result[stage],chip)) != tensor_digest(eager[phase,pattern][stage][chip]):
                            raise AssertionError('Replay differs from its complete eager output')
                        report['replay_checks'].append(dict(phase=phase,ordinal=ordinal,pattern=pattern,chip=chip,
                            stage=stage,exact=True,bindings_stable=True))
                audit_inputs('replay',ordinal,pattern)
            ttnn.execute_trace(mesh,trace,cq_id=0,blocking=True)
            for chip in range(2):
                actual = tensor_digest(host(result[stages[-1]],chip))
                if (actual != tensor_digest(eager[phase,0][stages[-1]][chip])
                        or actual == tensor_digest(eager[phase,1][stages[-1]][chip])):
                    raise AssertionError('Omitted input update is not distinguishable')
                report['stale_controls'].append(dict(phase=phase,chip=chip,detected=True))
            ttnn.release_trace(mesh,trace)
            trace = None
            release_owned(ttnn,transient)
            transient.clear()
            progress(f'{phase}_complete')
        audit_parameters('after')
        failed = sum(not entry['passed'] for entry in report['eager_checks'])
        if failed:
            raise AssertionError(f'{failed} learned comparisons fail the declared accuracy policy')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh,trace)
                release_owned(ttnn,transient)
                release_owned(ttnn,owned)
                ttnn.close_mesh_device(mesh)
            if weights is not None:
                weights.__exit__(None,None,None)
            report['checkpoint_closed'] = weights is not None and weights.source is None
            report['closed_cleanly'] = True
            report['sources_after'] = source_hashes()
            report['native_sources_after'] = fingerprints(root,packer_compat=packer_compat)
            if report['sources_after'] != report['sources'] or report['native_sources_after'] != report['native_sources']:
                raise ValueError('Projection experiment or native source changed during execution')
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            if not report['closed_cleanly']:
                report['passed'] = False
            progress('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
