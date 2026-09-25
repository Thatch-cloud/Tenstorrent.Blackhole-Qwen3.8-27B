"""Synthetic cache publication plus real T32 proposal trace and native attention; no learned qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from dflash_device import DFlashDevice
from dflash_proposal_trace import PreparedDFlashProposal
from dflash_t32_cached_adapter import require_simulator
from dflash_t32_native_attention import attention
from dflash_combined_sim_runtime import binary_hashes
from dflash_t16_native_attention_gate import native_hashes
from draft_kv_history import DraftKVHistory
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dflash-t32-cache-replay-probe.py', 'dflash_t32_cached_adapter.py',
    'dflash_t32_cache_gate.py', 'attention_batch.py', 'gdn_multitoken_conv.py',
    'dflash_device.py', 'dflash_proposal_trace.py', 'draft_kv_history.py',
    'dflash_proposal_inputs.py', 'dflash_attention_mask.py',
    'dflash_t32_native_attention.py', 'dflash_t16_native_attention.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_simulator()
    import torch
    import ttnn

    root = os.environ['TT_METAL_HOME']
    report = dict(passed=False, closed_cleanly=False, scope=__doc__, block_rows=32,
        learned_qualified=False, hardware_qualified=False, performance_qualified=False,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES},
        native_sources=native_hashes(root, simulator=True), runtime_binaries=binary_hashes(root), checks=[])
    mesh = cache = prepared = None
    owned = []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        generator = torch.Generator().manual_seed(3832)

        def upload(value):
            result = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.append(result)
            return result

        def snapshot(value):
            ttnn.synchronize_device(mesh)
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]

        def synthetic_projection(operations, inputs, query, tables, retain, *, parameters):
            rows = inputs.shape[2]
            flat = retain(operations.slice(inputs, (0, 0, 0, 0), (1, 1, rows, 512)))
            shaped = retain(operations.reshape(flat, (1, rows, 4, 128)))
            heads = retain(operations.permute(shaped, (0, 2, 1, 3)))
            return dict(k=heads, v=heads)

        history = torch.randn((1, 1, 2048, 5120), generator=generator).bfloat16()
        history_device = upload(history)
        query = upload(torch.randn((1, 16, 32, 128), generator=generator).bfloat16())
        tail = upload(torch.zeros((1, 4, 32, 128), dtype=torch.bfloat16))
        with patch('draft_kv_history.project_key_value', side_effect=synthetic_projection):
            cache = DraftKVHistory(ttnn, mesh, [{}], history_device, position=2048, history_rows=2048)
            device = SimpleNamespace(operations=ttnn, mesh=mesh, position=2048, history_rows=2048,
                block_rows=32, history=history_device, spare_history=history_device, owned=owned,
                progress=None, kv_history=cache, native_proposal_attention=True,
                validated_native_proposal_masks=set())
            device.temporaries = MethodType(DFlashDevice.temporaries, device)

            def execute(identifiers, history, mask, rope, *, retain, cached_history, **kwargs):
                if addresses(ttnn, mask) not in device.validated_native_proposal_masks:
                    raise AssertionError('Validated full-query T32 mask required')
                keys = retain(ttnn.concat([cached_history[0]['k'], tail], dim=2))
                values = retain(ttnn.concat([cached_history[0]['v'], tail], dim=2))
                result = retain(attention(ttnn, query, keys, values, mask, mask_validated=True))
                return (result, *(retain(ttnn.clone(value)) for value in
                    (cached_history[0]['k'], cached_history[0]['v'], identifiers)))

            device.execute_proposal = execute
            device.select_proposal = lambda outputs, seed, count: [snapshot(value) for value in outputs]
            prepared = PreparedDFlashProposal(device, max_new_tokens=64)

            def replay(stage, seed):
                actual = prepared.propose(seed, 31)
                bucket = prepared.buckets[2048]
                temporary, retain = device.temporaries([*owned, *prepared.owned, *cache.owned])
                try:
                    expected = execute(bucket.identifiers, bucket.history, bucket.mask, bucket.rope,
                        retain=retain, cached_history=bucket.cached_history)
                    reference = [snapshot(value) for value in expected]
                finally:
                    release_owned(ttnn, temporary)
                projected = history[..., :512].reshape(1, 2048, 4, 128).transpose(1, 2).contiguous()
                for chip in range(2):
                    if not all(torch.equal(left[chip], right[chip]) for left, right in zip(actual, reference, strict=True)):
                        raise AssertionError('Cached native T32 replay differs from eager')
                    if not all(torch.equal(actual[index][chip], projected) for index in (1, 2)):
                        raise AssertionError('Trace consumed stale or incorrect committed K/V rows')
                    if not torch.isfinite(actual[0][chip]).all():
                        raise AssertionError('Non-finite T32 attention output')
                    if actual[3][chip].flatten().tolist() != [seed, *([248070] * 31)]:
                        raise AssertionError('Captured proposal identifiers did not update')
                    report['checks'].append(dict(stage=stage, chip=chip, position=device.position,
                        exact=True, finite=True, rows=32))
                print(json.dumps(dict(stage=stage, position=device.position)), flush=True)
                return actual

            previous = replay('initial', 17)
            for prefix in (1, 16, 32):
                candidate = torch.randn((1, 1, 32, 5120), generator=generator).bfloat16()
                candidate_device = upload(candidate)
                publication = cache.prepare(candidate_device, prefix, position=device.position)
                try:
                    prepared.propose(19, 31)
                except ValueError:
                    report['checks'].append(dict(stage='pending-rejected', prefix=prefix, exact=True))
                else:
                    raise AssertionError('Replay accepted an unpublished cache frontier')
                cache.discard(publication)
                discarded = replay(f'discard-{prefix}', 17)
                if any(not torch.equal(left, right) for left, right in zip(previous[0], discarded[0], strict=True)):
                    raise AssertionError('Discard changed visible attention output')
                publication = cache.prepare(candidate_device, prefix, position=device.position)
                cache.commit(publication)
                device.position, device.history_rows = cache.position, cache.history_rows
                history = torch.cat((history, candidate[..., :prefix, :]), dim=2)[..., -2048:, :].contiguous()
                current = replay(f'commit-{prefix}', 27)
                for chip in range(2):
                    if torch.equal(current[0][chip], previous[0][chip]):
                        raise AssertionError('Changed-cache negative control did not distinguish stale replay')
                previous = current
            report['passed'] = len(report['checks']) == 17
            if not report['passed']:
                raise AssertionError('Incomplete cache replay matrix')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if prepared is not None:
                prepared.close()
            if cache is not None:
                cache.close()
            if mesh is not None:
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['native_sources_after'] = native_hashes(root, simulator=True)
            report['runtime_binaries_after'] = binary_hashes(root)
            report['closed_cleanly'] = True
            if (report['native_sources_after'] != report['native_sources']
                    or report['runtime_binaries_after'] != report['runtime_binaries']):
                report['passed'] = False
                raise AssertionError('Runtime changed during cache replay')
        finally:
            if not report['closed_cleanly']:
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
