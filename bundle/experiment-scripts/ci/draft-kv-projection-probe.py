"""Simulator learned K/V row-separability gate, not a complete cache or speed measurement."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dflash_proposal_inputs import proposal_inputs
from draft_convolution_fixture import verified_bytes, verified_tensor
from draft_head_preparation import rope_tables
from draft_kv_projection import project_key_value
from draft_remaining_layers_fixture import specifications, TENSOR_SHA256
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--layer', type=int, choices=(1, 2, 3, 4), default=1)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, layer=options.layer, checks=[], negative_controls=[],
        hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
            'draft-kv-projection-probe.py', 'draft_kv_projection.py', 'draft_attention_branch.py',
            'draft_head_layout.py', 'draft_head_preparation.py', 'dflash_proposal_inputs.py')})
    mesh = trace = None
    persistent, owned = [], []
    try:
        selected = {name: value for name, value in specifications(options.layer).items()
            if name.endswith(('self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.k_norm.weight'))}
        hashes = {name: TENSOR_SHA256[str(options.layer)][name] for name in selected}
        manifest, raw = verified_bytes(options.fixture, specifications=selected, hashes=hashes)
        weights = {name: verified_tensor(raw[name], shape, hashes[name], name) for name, (shape, filename) in selected.items()}
        report['checkpoint'] = dict(model=manifest['model'], revision=manifest['revision'], tensor_sha256=hashes)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def retain(value):
            owned.append(value)
            return value

        def upload(value, *, row_major=False, device=True):
            return ttnn.from_torch(value, device=mesh if device else None, dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)

        def replicate(value):
            return torch.cat((value, value), dim=0)

        parameters = dict(operations=ttnn, native_head_layout=True,
            kernel=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False), projections={}, head_norms={})
        for name in ('k', 'v'):
            weight = weights[f'layers.{options.layer}.self_attn.{name}_proj.weight']
            parameters['projections'][name] = upload(torch.cat([part.T.contiguous() for part in weight.chunk(2, dim=0)], dim=0))
            persistent.append(parameters['projections'][name])
        norm = weights[f'layers.{options.layer}.self_attn.k_norm.weight'].reshape(1, 1, 4, 32)
        parameters['head_norms']['k'] = upload(replicate(norm), row_major=True)
        query = upload(torch.zeros((2, 1, 32, 2048), dtype=torch.bfloat16))
        persistent.extend((parameters['head_norms']['k'], query))

        def execute(source, tables):
            inputs = retain(upload(source))
            rope = tuple(retain(upload(replicate(table))) for table in tables)
            return project_key_value(ttnn, inputs, query, rope, retain, parameters=parameters)

        def snapshot(result):
            ttnn.synchronize_device(mesh)
            return {name: [ttnn.to_torch(shard).contiguous().clone() for shard in ttnn.get_device_tensors(result[name])]
                for name in ('k', 'v')}

        def compare(actual, expected, **case):
            for name in ('k', 'v'):
                if len(actual[name]) != 2 or len(expected[name]) != 2:
                    raise AssertionError('Both learned K/V shards required')
                for chip in range(2):
                    left, right = actual[name][chip], expected[name][chip]
                    if left.shape != right.shape or not torch.equal(left.view(torch.int16), right.view(torch.int16)):
                        difference = (left.float() - right.float()).abs().max().item() if left.shape == right.shape else None
                        raise AssertionError(f'Incremental {name} differs from full projection: {case}, chip={chip}, max_abs={difference}')
                    report['checks'].append(dict(**case, head=name, chip=chip, exact=True))

        def joined(history, proposal):
            return {name: retain(ttnn.concat([history[name], proposal[name]], dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)) for name in ('k', 'v')}

        for position, history_rows, context in ((170, 170, 256), (4093, 2048, 2048)):
            print(json.dumps(dict(stage='full-versus-split-begin', position=position, context=context)), flush=True)
            generator = torch.Generator().manual_seed(position)
            history = torch.randn((2, 1, context, 5120), generator=generator).bfloat16()
            history[..., history_rows:, :] = 0
            proposal = torch.randn((2, 1, 32, 5120), generator=generator).bfloat16()
            proposal[..., 8:, :] = 0
            tables = proposal_inputs(17, position, history_rows, 8, context)['rope']['k']
            full = execute(torch.cat((history, proposal), dim=2), tables)
            cached = execute(history, tuple(table[..., :context, :] for table in tables))
            live = execute(proposal, tuple(table[..., context:, :] for table in tables))
            compare(snapshot(joined(cached, live)), snapshot(full), stage='split', position=position, context=context)
            if position == 4093:
                prefix = 7
                committed = torch.randn((2, 1, 32, 5120), generator=generator).bfloat16()
                committed[..., prefix:, :] = 0
                new_tables = tuple(table.clone() for table in rope_tables(position, 32))
                for table in new_tables:
                    table[..., prefix:, :] = 0
                appended = execute(committed, new_tables)
                next_history = torch.cat((history[..., prefix:, :], committed[..., :prefix, :]), dim=2)
                shifted = {}
                for name in ('k', 'v'):
                    retained = retain(ttnn.slice(cached[name], (0, 0, prefix, 0), (1, 4, context, 128)))
                    added = retain(ttnn.slice(appended[name], (0, 0, 0, 0), (1, 4, prefix, 128)))
                    shifted[name] = retain(ttnn.concat([retained, added], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                next_position = position + prefix
                next_proposal = torch.randn((2, 1, 32, 5120), generator=generator).bfloat16()
                next_proposal[..., 8:, :] = 0
                next_rope = proposal_inputs(29, next_position, context, 8, context)['rope']['k']
                print(json.dumps(dict(stage='eviction-begin', position=next_position, prefix=prefix)), flush=True)
                reference = snapshot(execute(torch.cat((next_history, next_proposal), dim=2), next_rope))
                live_next = execute(next_proposal, tuple(table[..., context:, :] for table in next_rope))
                compare(snapshot(joined(shifted, live_next)), reference, stage='evict-append', position=next_position, prefix=prefix)
                stale = snapshot(joined(cached, live_next))
                for name in ('k', 'v'):
                    for chip in range(2):
                        if torch.equal(stale[name][chip].view(torch.int16), reference[name][chip].view(torch.int16)):
                            raise AssertionError('Stale historical cache negative control was not detected')
                        report['negative_controls'].append(dict(kind='stale-history', head=name, chip=chip, detected=True))
            print(json.dumps(dict(stage='full-versus-split-end', position=position, exact=True)), flush=True)
            release_owned(ttnn, owned)
            owned.clear()

        generator = torch.Generator().manual_seed(761)
        payloads = [torch.randn((2, 1, 32, 5120), generator=generator).bfloat16() for unused in range(2)]
        tables = [rope_tables(position, 32) for position in (4093, 4100)]
        expected = [snapshot(execute(source, table)) for source, table in zip(payloads, tables, strict=True)]
        inputs = retain(upload(payloads[0]))
        rope = tuple(retain(upload(replicate(table))) for table in tables[0])

        def operation():
            return project_key_value(ttnn, inputs, query, rope, retain, parameters=parameters)

        operation()
        ttnn.synchronize_device(mesh)
        trace, output = capture_operation(ttnn, mesh, operation)
        for index in (1, 0):
            for source, destination in zip((payloads[index], *(replicate(table) for table in tables[index])), (inputs, *rope), strict=True):
                ttnn.copy_host_to_device_tensor(upload(source, device=False), destination)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            compare(snapshot(output), expected[index], stage='changed-input-trace', replay=index)
        report['passed'] = len(report['checks']) == 20 and len(report['negative_controls']) == 4
        if not report['passed']:
            raise AssertionError('All learned projection, eviction and replay comparisons required')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        release_owned(ttnn, owned)
        release_owned(ttnn, persistent)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
