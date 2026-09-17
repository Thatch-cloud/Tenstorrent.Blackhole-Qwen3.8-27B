"""Synthetic real-device trace/bank lifetime diagnostic, not learned drafting or throughput."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dspark_banked_proposal import BankedDSparkProposal
from dspark_history import leaves
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dspark-banked-trace-probe.py', 'dspark_banked_proposal.py', 'dspark_prepared_proposal.py',
    'dspark_history.py', 'dspark_fixed_inputs.py', 'dspark_wide_target.py', 'attention_batch.py',
    'gdn_multitoken_conv.py', 'feature_projection.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh simulator report required')
    import torch
    import ttnn
    root = Path(__file__).parent

    def hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}

    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=hashes(), checks=[], bank_checks=[], learned_model_executed=False,
        hardware_qualified=False, serving_qualified=False)
    mesh, candidate, owned = None, None, []

    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)

    def fixture(bank, operand, pattern):
        values = torch.full((2, 4, 64, 128), bank * 32 + operand + pattern * 64, dtype=torch.bfloat16)
        values[1] += 128
        return values

    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        banks = []
        for bank in range(2):
            values = []
            for operand in range(10):
                tensor = ttnn.from_torch(fixture(bank, operand, 0), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
                owned.append(tensor)
                values.append(tensor)
            banks.append(tuple(tuple(values[index:index + 2]) for index in range(0, 10, 2)))
        bindings = [addresses(ttnn, value) for value in owned]
        device = SimpleNamespace(operations=ttnn, mesh=mesh, closed=False, position=31, max_drafts=15,
            history=SimpleNamespace(layers=banks[0], spare_layers=banks[1], capacity=64, pending=None),
            parameters={}, layer_weights=[], predecessor=owned[0], successor=owned[1],
            target=SimpleNamespace(lm_head_weight=owned[2]), rotary=SimpleNamespace(
                tables=lambda position, rows: (torch.ones(1, 1, rows, 128, dtype=torch.bfloat16),
                    torch.zeros(1, 1, rows, 128, dtype=torch.bfloat16))))

        def execute(device, inputs, history, retain):
            copies = [retain(ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG)) for value in leaves(history)]
            tokens = retain(ttnn.clone(inputs['identifiers'], memory_config=ttnn.DRAM_MEMORY_CONFIG))
            return dict(normalized=copies[0], logits=copies[-1], tokens=tokens, bank_outputs=copies)

        def compare(tensor, expected, field, **metadata):
            for chip, shard in enumerate(ttnn.get_device_tensors(tensor)):
                if not torch.equal(ttnn.to_torch(shard), expected[chip:chip + 1]):
                    raise AssertionError('Bank trace output or borrowed storage changed')
                report[field].append(dict(metadata, chip=chip, exact=True))

        with patch('dspark_prepared_proposal.execute', execute):
            save('capture_both_banks')
            candidate = BankedDSparkProposal(device, 10, audit=True)
            for ordinal, bank in enumerate((0, 1, 1, 0)):
                save(f'replay_{ordinal}_bank_{bank}')
                for owner, contents in enumerate(banks):
                    for operand, tensor in enumerate(leaves(contents)):
                        host = ttnn.from_torch(fixture(owner, operand, ordinal + 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                        ttnn.copy_host_to_device_tensor(host, tensor)
                device.history.layers, device.history.spare_layers = banks[bank], banks[1 - bank]
                device.position = 31 + ordinal
                tokens = candidate.propose(10 + ordinal, 15)
                if tokens[0] != 10 + ordinal:
                    raise AssertionError('Changing anchor did not reach selected trace')
                for operand, output in enumerate(candidate.proposals[bank].outputs['bank_outputs']):
                    compare(output, fixture(bank, operand, ordinal + 1), 'checks', ordinal=ordinal, bank=bank, operand=operand)
                for owner, contents in enumerate(banks):
                    for operand, tensor in enumerate(leaves(contents)):
                        compare(tensor, fixture(owner, operand, ordinal + 1), 'bank_checks',
                            ordinal=ordinal, bank=owner, operand=operand)
                if bindings != [addresses(ttnn, value) for value in owned]:
                    raise AssertionError('Borrowed bank binding changed')
            report['replay_counts'] = candidate.replay_counts
            report['eager_replay_checks'] = candidate.checks
            candidate.close()
            candidate = None
            for owner, contents in enumerate(banks):
                for operand, tensor in enumerate(leaves(contents)):
                    compare(tensor, fixture(owner, operand, 4), 'bank_checks', ordinal='after_close', bank=owner, operand=operand)
        if len(report['checks']) != 80 or len(report['bank_checks']) != 200:
            raise AssertionError('Incomplete two-bank lifetime diagnostic')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if candidate is not None:
                candidate.close()
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
