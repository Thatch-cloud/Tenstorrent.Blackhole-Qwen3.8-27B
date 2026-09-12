"""Learned TP2 cached-attention operands and real publication/replay ownership; not full-model throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch

from attention_batch import capture_operation
from dflash_device import DFlashDevice
from dflash_proposal_inputs import proposal_inputs
from dflash_proposal_trace import PreparedDFlashProposal
from draft_attention_branch import prepare_attention_branch, execute_attention_branch
from draft_attention_fixture import TENSORS as ATTENTION
from draft_convolution_fixture import TENSORS as CONVOLUTION, verified_bytes, verified_tensor
from draft_convolution_fused import checked_convolution
from draft_kv_history import DraftKVHistory
from draft_kv_projection import project_key_value
from draft_remaining_layers_fixture import specifications, TENSOR_SHA256
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--layer', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--capture-projection', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, layer=options.layer, head_checks=[], replay_checks=[],
        unchanged_checks=[], negative_controls=[], projection_checks=[], capture_projection=options.capture_projection, hashes={name:
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
                'draft-kv-history-probe.py', 'draft_kv_history.py', 'draft_kv_projection.py', 'draft_kv_projection_trace.py', 'draft_attention_branch.py',
                'dflash_device.py', 'dflash_proposal_trace.py', 'dflash_proposal_inputs.py', 'draft_head_layout.py',
                'draft_convolution_fused.py', 'draft_convolution_fused_compute.cpp', 'draft_convolution_fused_io.cpp')})
    mesh = device = prepared = other_trace = None
    owned = []
    try:
        names = {name.replace('layers.0.', f'layers.{options.layer}.', 1) for name in ATTENTION | CONVOLUTION}
        selected = {name: value for name, value in specifications(options.layer).items() if name in names}
        hashes = {name: TENSOR_SHA256[str(options.layer)][name] for name in selected}
        manifest, raw = verified_bytes(options.fixture, specifications=selected, hashes=hashes)
        weights = {name.replace(f'layers.{options.layer}.', 'layers.0.', 1):
            verified_tensor(raw[name], shape, hashes[name], name) for name, (shape, filename) in selected.items()}
        report['checkpoint'] = dict(model=manifest['model'], revision=manifest['revision'], tensor_sha256=hashes)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()

        def retain(value):
            owned.append(value)
            return value

        def upload(value, *, sharded=False):
            return retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
                    if sharded else ttnn.ReplicateTensorToMesh(mesh)))

        def snapshot(values):
            ttnn.synchronize_device(mesh)
            return tuple(ttnn.to_torch(shard).contiguous().clone() for value in values for shard in ttnn.get_device_tensors(value))

        def compare(left, right, checks, **description):
            if len(left) != len(right):
                raise AssertionError('Complete TP2 operand snapshots required')
            for index, (actual, expected) in enumerate(zip(left, right, strict=True)):
                if actual.shape != expected.shape or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
                    raise AssertionError(f'Cached operand or replay differs: {description}, tensor={index}')
                checks.append(dict(**description, tensor=index, exact=True))

        generator = torch.Generator().manual_seed(991)
        host_history = torch.randn((2, 1, 2048, 5120), generator=generator).bfloat16()
        history = upload(host_history, sharded=True)
        parameters = prepare_attention_branch(ttnn, mesh, {name: weights[name] for name in ATTENTION},
            {name: weights[name] for name in CONVOLUTION}, retain, native_head_layout=True)
        print(json.dumps(dict(stage='initialize-learned-cache', position=4093)), flush=True)
        device = SimpleNamespace(operations=ttnn, mesh=mesh, position=4093, history_rows=2048, block_rows=8,
            history=history, spare_history=upload(torch.zeros_like(host_history), sharded=True),
            owned=[], progress=None, pending=None, closed=False, published_rows=0, proposal_capture=None, kv_history=None)
        for name in ('temporaries', 'prepare_publication', 'commit_publication', 'discard_publication', 'close'):
            setattr(device, name, MethodType(getattr(DFlashDevice, name), device))
        device.kv_history = DraftKVHistory(ttnn, mesh, [parameters], history, position=4093, history_rows=2048,
            capture_projection=options.capture_projection)

        def project_features(features, prefix):
            sliced = ttnn.slice(features[0], (0, 0, 0, 0), (1, 1, prefix, 5120))
            output = ttnn.clone(sliced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if addresses(ttnn, sliced) != addresses(ttnn, features[0]):
                ttnn.deallocate(sliced)
            return output

        device.project_features = project_features
        hidden_host = torch.randn((1, 1, 32, 5120), generator=generator).bfloat16()
        hidden_host[..., 8:, :] = 0
        hidden = upload(hidden_host)
        host_inputs = proposal_inputs(17, 4093, 2048, 8, 2048)
        mask = upload(host_inputs['mask'])
        rope = {name: tuple(upload(value) for value in host_inputs['rope'][name]) for name in ('q', 'k')}
        padded_history = retain(ttnn.pad(history, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0))

        class AttentionBoundary(Exception):
            pass

        def operands(cached):
            transient, keep = device.temporaries([*owned, *device.kv_history.owned])
            captured = []
            def intercept(operations, mesh, query, key, value, mask, **kwargs):
                captured.extend((query, key, value, mask))
                raise AttentionBoundary()
            try:
                with patch('draft_attention_branch.composed_draft_attention', side_effect=intercept):
                    try:
                        execute_attention_branch(ttnn, mesh, None, hidden, padded_history, mask, rope, keep,
                            parameters=parameters, context=2048, convolution_operation=checked_convolution,
                            cached_history=device.kv_history.active[0] if cached else None)
                    except AttentionBoundary:
                        pass
                if len(captured) != 4:
                    raise AssertionError('Actual attention branch must reach the unchanged attention boundary')
                return snapshot(captured)
            finally:
                release_owned(ttnn, transient)

        print(json.dumps(dict(stage='compare-actual-attention-operands', position=4093)), flush=True)
        compare(operands(True), operands(False), report['head_checks'], position=4093)

        def execute(identifiers, history, mask, rope, *, retain, cached_history=None, **kwargs):
            if cached_history is None or len(cached_history) != 1:
                raise AssertionError('This replay fixture must consume the actual prepared cache inputs')
            values = [cached_history[0]['k'], cached_history[0]['v'], identifiers, mask, *rope['q']]
            return tuple(retain(ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)) for value in values)

        device.execute_proposal = execute
        device.proposal_snapshot = snapshot
        device.select_proposal = lambda outputs, seed, count: snapshot(outputs)
        prepared = PreparedDFlashProposal(device, max_new_tokens=64)
        other_input = upload(torch.ones((1, 1, 32, 32), dtype=torch.bfloat16))
        warm = ttnn.add(other_input, 1.0)
        ttnn.deallocate(warm)
        other_trace, other_output = capture_operation(ttnn, mesh, lambda: ttnn.add(other_input, 1.0))
        owned.append(other_output)
        ttnn.execute_trace(mesh, other_trace, cq_id=0, blocking=True)
        other_before = snapshot((other_output,))

        def replay(seed, stage):
            actual = prepared.propose(seed, 7)
            host = proposal_inputs(seed, device.position, device.history_rows, 8, 2048)
            expected = [*snapshot(device.kv_history.active[0].values())]
            for value in (host['identifiers'].to(torch.uint32), host['mask'], *host['rope']['q']):
                expected.extend((value.contiguous(), value.contiguous()))
            compare(actual, expected, report['replay_checks'], stage=stage, position=device.position)
            compare(snapshot((other_output,)), other_before, report['unchanged_checks'], stage=stage, target='other-trace')

        replay(17, 'initial')
        def audit_projection():
            projection = device.kv_history.projection
            if projection is not None:
                with device.kv_history.temporaries([]) as retain:
                    expected = project_key_value(ttnn, projection.inputs[0], projection.query, projection.inputs[1:],
                        retain, parameters=parameters)
                    compare(snapshot(projection.outputs[0][name] for name in ('k', 'v')),
                        snapshot(expected[name] for name in ('k', 'v')), report['projection_checks'],
                        position=device.position, replay=projection.calls)
        before = snapshot([device.history, *device.kv_history.active[0].values()])
        rejected = upload(torch.randn((2, 1, 32, 5120), generator=generator).bfloat16(), sharded=True)
        publication = device.prepare_publication([rejected], 7, position=device.position)
        audit_projection()
        compare(snapshot([device.history, *device.kv_history.active[0].values()]), before,
            report['unchanged_checks'], stage='prepared', target='active-banks')
        device.discard_publication(publication)
        replay(27, 'discarded')
        accepted = upload(torch.randn((2, 1, 32, 5120), generator=generator).bfloat16(), sharded=True)
        publication = device.prepare_publication([accepted], 7, position=device.position)
        audit_projection()
        device.commit_publication(publication)
        if device.position != 4100 or device.kv_history.position != 4100:
            raise AssertionError('Feature and K/V banks must commit to the same absolute frontier')
        print(json.dumps(dict(stage='audit-committed-cache', position=device.position)), flush=True)
        device.kv_history.audit(device.history)
        report['history_checks'] = list(device.kv_history.checks)
        replay(37, 'committed')
        if options.capture_projection:
            committed = snapshot([device.history, *device.kv_history.active[0].values()])
            publication = device.prepare_publication([rejected], 1, position=device.position)
            audit_projection()
            compare(snapshot([device.history, *device.kv_history.active[0].values()]), committed,
                report['unchanged_checks'], stage='changed-position-prepared', target='active-banks')
            device.discard_publication(publication)
            report['projection_replays'] = device.kv_history.projection.calls
        after = snapshot(device.kv_history.active[0].values())
        for index, (current, stale) in enumerate(zip(after, before[2:], strict=True)):
            if torch.equal(current.view(torch.int16), stale.view(torch.int16)):
                raise AssertionError('Stale cache must be distinguishable after a committed update')
            report['negative_controls'].append(dict(tensor=index, detected=True))
        report['passed'] = (len(report['head_checks']) == 8 and len(report['replay_checks']) == 36
            and len(report['unchanged_checks']) == (18 if options.capture_projection else 12) and len(report['history_checks']) == 4
            and len(report['negative_controls']) == 4 and len(report['projection_checks']) == (12 if options.capture_projection else 0)
            and (not options.capture_projection or report['projection_replays'] == 3))
        if not report['passed']:
            raise AssertionError('Complete operand, publication and replay evidence required')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if other_trace is not None:
            ttnn.release_trace(mesh, other_trace)
        if prepared is not None:
            prepared.close()
        if device is not None:
            protected = [addresses(ttnn, value) for value in (device.history, device.spare_history)]
            owned = [value for value in owned if addresses(ttnn, value) not in protected]
            device.close()
        release_owned(ttnn, owned)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
