"""Scoped sampler-only workaround for the audited four-link P150A pair."""

from contextlib import contextmanager
import hashlib
from pathlib import Path


DESCRIPTOR = 'tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto'
SOURCES = {
    DESCRIPTOR: 'e5d25de80648816ed8305960cb4ea1c887a8157d595d348719bc3b5ade6f8682',
    'models/common/modules/tt_ccl.py': 'ec24c3abd732a1b8c4dff836b992892ac35917e529843cdb3772cf2ae064bedd',
    'models/common/sampling/tt_sampling.py': 'd3d32ea9370e7f51f771f13f44e6aa06429299d64e08d3ae9a3793271a55d730',
}


def audit(root, environment):
    root = Path(root)
    if (environment.get('QWEN_HARDWARE_TESTS') != '1' or environment.get('QWEN_CARDS_ALLOCATED') != '1'
            or environment.get('QWEN_FABRIC_LINK_PROBE') != '1'
            or any(environment.get(key) for key in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE', 'TT_METAL_MOCK_CLUSTER_DESC_PATH'))
            or environment.get('TT_MESH_GRAPH_DESC_PATH') != str(root / DESCRIPTOR)):
        raise ValueError('Explicit allocated four-channel hardware experiment required')
    found = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    if found != SOURCES:
        raise ValueError('Pinned sampling, link helper or descriptor source changed')
    return found


@contextmanager
def sampler_links(sampling, links):
    if type(links) is not int or links not in (1, 2, 4):
        raise ValueError('Explicit one, two or four sampler links required')
    if list(sampling.mesh_device.shape) != [1, 2]:
        raise ValueError('Scoped workaround requires the two-chip mesh')
    collective = sampling.tt_ccl
    original_links = sampling.num_argmax_gather_links
    had_override = 'get_num_links' in vars(collective)
    original_override = vars(collective).get('get_num_links')

    def pair_capacity(cluster_axis=None):
        if cluster_axis is not None and (type(cluster_axis) is not int or cluster_axis not in (0, 1)):
            raise ValueError('Unsupported collective axis')
        return 4

    sampling.num_argmax_gather_links = links
    collective.get_num_links = pair_capacity
    try:
        yield
    finally:
        sampling.num_argmax_gather_links = original_links
        if had_override:
            collective.get_num_links = original_override
        else:
            del collective.get_num_links
