"""Link policy for every mesh the stack opens: the audited p150a pair (1, 2) and the four-card ring (1, 4).

sampling_link_policy, projection_link_policy and model_link_policy are the pair's policy, and each refuses
any mesh but [1, 2]. Their bytes are sha256-pinned by recorded evidence (sampling_link_policy.py f9e8d30c in
the dspark evidence JSONs and tensix_mlp_hardware_gate.HARDWARE_SOURCES; model_link_policy.py in
target_link_request.SOURCES), so editing them to take a second shape would void that evidence - the
tp_common lesson (memory qualification-pins-model-sources). This module generalises them instead:

  MESHES[(1, 2)]  is theirs: the p150_x2 descriptor under TT_METAL_HOME at the sha256 sampling_link_policy
                  pins, links 1, 2 or 4. At (1, 2) every function here DELEGATES to the pinned module, so the
                  pair's behaviour is theirs call for call.
  MESHES[(1, 4)]  is the four-card ring's (tp4_mesh): its descriptor at the image path the TP4 profiles name,
                  held to the checked-in file's sha256, links 1 or 2 (what each ring edge trains; tt_ccl's
                  P150x4 entry is (2, 2)).

Nothing on the TP4 G1 path calls these (the stock decode path takes tt_ccl's table); they are the policy the
fast path's collectives will take when it is ported past two chips, and what the four-card fabric probe
states its link counts against. Stdlib only at import (sampling_link_policy is); projection_link_policy and
model_link_policy are imported where delegated to (model_link_policy pulls in the fast path's model_batch).
"""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from sampling_link_policy import DESCRIPTOR as PAIR_DESCRIPTOR, SOURCES as PAIR_SOURCES
import tp4_mesh
import tp_shapes

HERE = Path(__file__).resolve().parent
# The pair's descriptor and its audited bytes are sampling_link_policy's own (one source of truth).
PAIR_DESCRIPTOR_SHA256 = PAIR_SOURCES[PAIR_DESCRIPTOR]


# The four-card descriptor's audited bytes (scripts/ci/qwen_p150x4_ring_mesh_graph_descriptor.textproto, which the
# image lays ONLY at tp4_mesh.DESCRIPTOR_PATH, not beside this module). Pinned here so the startup audit compares
# the file the runtime reads against bytes a review saw, not against itself; test_mesh_link_policy holds this
# constant to the checked-in file.
RING_DESCRIPTOR_SHA256 = '3603ef30556a92305739592b13064a30f2c65372f90f8e8dbd9292d966fd0bc0'


def ring_descriptor_sha256(path=None):
    """The checked-in four-card descriptor's sha256, or `path`'s when one is named (a test's copy)."""
    if path is None:
        return RING_DESCRIPTOR_SHA256
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MESHES = {
    (1, 2): dict(name='p150a pair', mesh_device='P300', descriptor=PAIR_DESCRIPTOR, under_runtime=True,
                 sha256=PAIR_DESCRIPTOR_SHA256, links=(1, 2, 4), capacity=4),
    (1, 4): dict(name='four p150a, (1, 4) ring', mesh_device=tp4_mesh.MESH_DEVICE, descriptor=tp4_mesh.DESCRIPTOR_PATH,
                 under_runtime=False, sha256=None, links=(1, tp4_mesh.LINKS), capacity=tp4_mesh.LINKS),
}


def entry(shape):
    """The MESHES entry for a mesh shape, or ValueError."""
    key = tuple(int(dim) for dim in shape)
    if key not in MESHES:
        raise ValueError('No link policy for mesh %s (known: %s)' % (list(key), ', '.join(
            str(list(known)) for known in sorted(MESHES))))
    return MESHES[key]


def descriptor_path(shape, runtime_root):
    found = entry(shape)
    return str(Path(runtime_root) / found['descriptor']) if found['under_runtime'] else found['descriptor']


def expected_sha256(shape):
    found = entry(shape)
    return found['sha256'] if found['sha256'] is not None else ring_descriptor_sha256()


def audit_descriptor(environment, shape, read=None):
    """The descriptor TT_MESH_GRAPH_DESC_PATH names must be `shape`'s, at its audited bytes; returns its sha256.
    `read` (path -> bytes) is injectable."""
    path = descriptor_path(shape, environment.get('TT_METAL_HOME', ''))
    if environment.get('TT_MESH_GRAPH_DESC_PATH') != path:
        raise ValueError('Mesh %s requires TT_MESH_GRAPH_DESC_PATH=%s, got %r'
                         % (list(shape), path, environment.get('TT_MESH_GRAPH_DESC_PATH')))
    data = (read or (lambda name: Path(name).read_bytes()))(path)
    found = hashlib.sha256(data).hexdigest()
    if found != expected_sha256(shape):
        raise ValueError('Mesh %s descriptor %s differs from its audited bytes' % (list(shape), path))
    return found


def requested_links(shape, links):
    """`links` if it is an explicit count `shape`'s fabric carries, else ValueError."""
    allowed = entry(shape)['links']
    if type(links) is not int or links not in allowed:
        raise ValueError('Mesh %s takes explicit %s links, got %r' % (list(shape), allowed, links))
    return links


@contextmanager
def sampler_links(sampling, links):
    """sampling_link_policy.sampler_links at (1, 2), unchanged; at (1, 4) the same scoped override of the
    sampler's gather links and its collective's reported capacity, against the ring's table."""
    shape = tuple(sampling.mesh_device.shape)
    if list(shape) == [1, 2]:
        import sampling_link_policy

        with sampling_link_policy.sampler_links(sampling, links):
            yield
        return
    found = entry(shape)
    requested_links(shape, links)
    collective = sampling.tt_ccl
    original_links = sampling.num_argmax_gather_links
    had_override = 'get_num_links' in vars(collective)
    original_override = vars(collective).get('get_num_links')

    def ring_capacity(cluster_axis=None):
        if cluster_axis is not None and (type(cluster_axis) is not int or cluster_axis not in (0, 1)):
            raise ValueError('Unsupported collective axis')
        return found['capacity']

    sampling.num_argmax_gather_links = links
    collective.get_num_links = ring_capacity
    try:
        yield
    finally:
        sampling.num_argmax_gather_links = original_links
        if had_override:
            collective.get_num_links = original_override
        else:
            del collective.get_num_links


def projection_validate(environment, shape):
    """projection_link_policy.validate at (1, 2), unchanged; at (1, 4) the same explicit-links rule against the
    ring: a simulator claims one link only, hardware needs the allocation flags and the ring's audited
    descriptor."""
    if list(shape) == [1, 2]:
        import projection_link_policy

        return projection_link_policy.validate(environment)
    requested = environment.get('QWEN_PROJECTION_LINKS', '1')
    allowed = tuple(str(count) for count in entry(shape)['links'])
    if requested not in allowed:
        raise ValueError('Projection links on mesh %s must be explicitly %s' % (list(shape), ' or '.join(allowed)))
    simulated = bool(environment.get('TT_METAL_SIMULATOR'))
    if simulated:
        if requested != '1':
            raise ValueError('Simulator cannot qualify physical multi-link operation')
    elif requested != '1':
        if (environment.get('QWEN_HARDWARE_TESTS') != '1' or environment.get('QWEN_CARDS_ALLOCATED') != '1'
                or environment.get('TT_METAL_MOCK_CLUSTER_DESC_PATH') or environment.get('TT_METAL_SLOW_DISPATCH_MODE')):
            raise ValueError('Allocated four-card hardware required')
        audit_descriptor(environment, shape)
    backend = 'simulator' if simulated else 'hardware' if environment.get('QWEN_HARDWARE_TESTS') == '1' else 'unallocated'
    return dict(requested_links=int(requested), backend=backend, mesh=list(shape),
                selection='explicit; not native discovery or physical-link validation')


AXES = (('default', None), ('axis0', 0), ('axis1', 1))


@contextmanager
def target_links(model, links, layers=64):
    """model_link_policy.target_links at (1, 2), unchanged; at (1, 4) the same request-local override of the
    complete target's shared collective (every layer, attention/GDN and MLP owner must share it), restored on
    exit, with the ring's link counts."""
    shape = tuple(model.mesh_device.shape)
    if list(shape) == [1, 2]:
        import model_link_policy

        with model_link_policy.target_links(model, links) as report:
            yield report
        return
    requested_links(shape, links)
    if len(model.layers) != layers:
        raise ValueError('Explicit link experiment on the complete %d-layer target required' % layers)
    collective = model.tt_ccl
    if hasattr(model, '_qwen_target_link_scope') or hasattr(collective, '_qwen_target_link_scope'):
        raise RuntimeError('Only one target link-policy scope may own this collective')
    owners = [model] + [owner for layer in model.layers for owner in (layer, layer.attention, layer.feed_forward)]
    if any(owner.tt_ccl is not collective for owner in owners):
        raise ValueError('All target layers, attention/GDN and MLP owners must share the audited collective')
    before = {name: collective.get_num_links(axis) for name, axis in AXES}
    if any(type(value) is not int or value not in entry(shape)['links'] for value in before.values()):
        raise ValueError('Original target link requests must be explicit supported integer counts')
    report = dict(requested_links=links, mesh=list(shape), owners_validated=len(owners), original_requests=before,
                  effective_requests={name: links for name, _ in AXES}, calls={name: 0 for name, _ in AXES},
                  restored=False, scope='Target model only; sampler and drafter policies unchanged')

    def requested(cluster_axis=None):
        if cluster_axis is not None and (type(cluster_axis) is not int or cluster_axis not in (0, 1)):
            raise ValueError('Unsupported target collective axis')
        name = 'default' if cluster_axis is None else 'axis%d' % cluster_axis
        report['calls'][name] += 1
        return links

    bindings = [(model, '_qwen_target_link_scope', report), (collective, '_qwen_target_link_scope', report),
                (collective, 'get_num_links', requested)]
    previous = []
    try:
        for instance, name, value in bindings:
            previous.append((instance, name, name in instance.__dict__, instance.__dict__.get(name)))
            setattr(instance, name, value)
        yield report
    finally:
        for instance, name, existed, value in reversed(previous):
            if existed:
                setattr(instance, name, value)
            else:
                delattr(instance, name)
        if {name: collective.get_num_links(axis) for name, axis in AXES} != before:
            raise AssertionError('Target collective policy was not restored')
        report['restored'] = True


PROJECTION_KEYS = ('QWEN_FAST_TP', 'QWEN_PROJECTION_LINKS', 'TT_METAL_SIMULATOR', 'TT_METAL_HOME',
                   'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED', 'TT_METAL_MOCK_CLUSTER_DESC_PATH',
                   'TT_METAL_SLOW_DISPATCH_MODE', 'TT_MESH_GRAPH_DESC_PATH')


@lru_cache(maxsize=8)
def _ring_projection_links(values):
    report = projection_validate(dict(values), tp4_mesh.MESH_SHAPE)
    print(json.dumps(dict(stage='mesh_link_policy', **report)), flush=True)
    return report['requested_links']


def projection_links():
    """The fast path's collective link count for the width this process serves at (QWEN_FAST_TP; unset is the
    pair): projection_link_policy.projection_links() itself at (1, 2), unchanged, and at (1, 4) the same explicit
    QWEN_PROJECTION_LINKS rule against the ring (1 or 2 links, the audited descriptor), validated once per
    environment and printed as one JSON line as the pinned resolver does. Read from os.environ, like it."""
    if tp_shapes.requested_tp(os.environ) == tp_shapes.PAIR:
        import projection_link_policy

        return projection_link_policy.projection_links()
    return _ring_projection_links(tuple((key, os.environ[key]) for key in PROJECTION_KEYS if key in os.environ))


TOPOLOGY_SWITCH = 'QWEN_FAST_CCL_TOPOLOGY'
TOPOLOGIES = ('linear', 'ring')
# The four-card default until the fabric probe (tp4_fabric_probe: Ring vs Linear at 32 x 5120, exact and timed)
# says otherwise: the descriptor's fabric is FABRIC_1D, which upstream exercises with Linear at four devices.
RING_DEFAULT_TOPOLOGY = 'linear'


def fast_ccl_topology(operations, environment=None):
    """The ttnn Topology the fast path's own collectives (feature projection, drafter gathers) run with.

    The pair is always Linear, as the code it replaces hard-coded, and refuses any other request: its bytes are
    what the TP2 evidence qualified. At four cards QWEN_FAST_CCL_TOPOLOGY picks 'linear' or 'ring' (default
    RING_DEFAULT_TOPOLOGY); an unknown value is refused, not read as the default."""
    environment = os.environ if environment is None else environment
    wanted = environment.get(TOPOLOGY_SWITCH)
    if tp_shapes.requested_tp(environment) == tp_shapes.PAIR:
        if wanted not in (None, 'linear'):
            raise ValueError('%s=%r: the pair runs Linear only' % (TOPOLOGY_SWITCH, wanted))
        return operations.Topology.Linear
    choice = RING_DEFAULT_TOPOLOGY if wanted is None else wanted
    if choice not in TOPOLOGIES:
        raise ValueError('%s must be one of %s, got %r' % (TOPOLOGY_SWITCH, ', '.join(TOPOLOGIES), wanted))
    return operations.Topology.Ring if choice == 'ring' else operations.Topology.Linear


def environment_shape(environment=None):
    """The mesh shape a TT_MESH_GRAPH_DESC_PATH names, or None for one this table does not know."""
    environment = os.environ if environment is None else environment
    path = environment.get('TT_MESH_GRAPH_DESC_PATH')
    for shape in sorted(MESHES):
        if path == descriptor_path(shape, environment.get('TT_METAL_HOME', '')):
            return shape
    return None
