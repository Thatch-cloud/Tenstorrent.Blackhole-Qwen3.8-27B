"""Capability query only; never starts a DRAM prefetcher or claims correctness.

Defaults to the simulator. --hardware runs the same query against allocated cards,
which is the only way to see the firmware and harvesting gates as the runtime sees
them. It never sets the force-enable override: a true here must be native.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from feature_projection import require_projection_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true',
                        help='Query allocated cards instead of the simulator')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    import ttnn

    report = dict(scope=__doc__, backend='hardware' if options.hardware else 'simulator',
        kernel_correctness_qualified=False,
        override_set=bool(os.environ.get('TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES')),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(mesh)
        report['workers'] = [mesh.compute_with_storage_grid_size().x, mesh.compute_with_storage_grid_size().y]
        report['dram_grid'] = [mesh.dram_grid_size().x, mesh.dram_grid_size().y]
        print(json.dumps(report), flush=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report['closed_cleanly'] = True
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
