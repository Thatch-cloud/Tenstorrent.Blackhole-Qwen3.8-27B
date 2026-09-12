"""Simulator reproduction of consecutive DFlash2 key-input assembly at CTX170/178."""

import argparse
import faulthandler
import json
import os
from pathlib import Path

from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    mesh = None
    owned = []
    report = dict(passed=False, scope=__doc__, checks=[])
    def save(stage, **values):
        report.update(stage=stage, **values)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage, **values)), flush=True)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        def retain(value):
            owned.append(value)
            return value
        def upload(value):
            return retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        host_history = (torch.arange(192).reshape(1, 1, 192, 1) % 64).expand(1, 1, 192, 5120).bfloat16()
        host_proposal = torch.full((1, 1, 32, 5120), 99, dtype=torch.bfloat16)
        history, proposal = upload(host_history), upload(host_proposal)
        zeros = retain(ttnn.zeros_like(history))
        for context in (170, 178, 170):
            faulthandler.dump_traceback_later(90, exit=True)
            save('slice', context=context)
            context_input = retain(ttnn.slice(history, (0, 0, 0, 0), (1, 1, context, 5120)))
            proposal_input = retain(ttnn.slice(proposal, (0, 0, 0, 0), (1, 1, 8, 5120)))
            padding = retain(ttnn.slice(zeros, (0, 0, 0, 0), (1, 1, 192 - context - 8, 5120)))
            ttnn.synchronize_device(mesh)
            save('concat', context=context, input_shapes=[list(value.shape) for value in (context_input, proposal_input, padding)])
            output = retain(ttnn.concat([context_input, proposal_input, padding], dim=2))
            save('synchronize-concat', context=context)
            ttnn.synchronize_device(mesh)
            expected = torch.cat((host_history[..., :context, :], host_proposal[..., :8, :],
                torch.zeros((1, 1, 192 - context - 8, 5120), dtype=torch.bfloat16)), dim=2)
            for chip, shard in enumerate(ttnn.get_device_tensors(output)):
                if not torch.equal(ttnn.to_torch(shard), expected):
                    raise AssertionError('Consecutive key concatenation changed logical rows')
                report['checks'].append(dict(context=context, chip=chip, exact=True))
            save('checked', context=context)
            faulthandler.cancel_dump_traceback_later()
        report['passed'] = len(report['checks']) == 6
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save('closing')
        release_owned(ttnn, owned)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
