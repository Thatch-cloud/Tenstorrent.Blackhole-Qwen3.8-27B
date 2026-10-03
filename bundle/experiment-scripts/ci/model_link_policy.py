"""Request-local target-model link-count experiment, separate from sampler and drafter collectives."""

from contextlib import contextmanager

from model_batch import instance_overrides


AXES = (('default', None), ('axis0', 0), ('axis1', 1))


@contextmanager
def target_links(model, links):
    if type(links) is not int or links not in (2, 4) or list(model.mesh_device.shape) != [1, 2] or len(model.layers) != 64:
        raise ValueError('Explicit two-versus-four-link experiment on the complete TP2 target required')
    collective = model.tt_ccl
    if hasattr(model, '_qwen_target_link_scope') or hasattr(collective, '_qwen_target_link_scope'):
        raise RuntimeError('Only one target link-policy scope may own this collective')
    owners = [model, *(owner for layer in model.layers for owner in (layer, layer.attention, layer.feed_forward))]
    if any(owner.tt_ccl is not collective for owner in owners):
        raise ValueError('All target layers, attention/GDN and MLP owners must share the audited collective')
    before = {name: collective.get_num_links(axis) for name, axis in AXES}
    if any(type(value) is not int or value not in (1, 2, 4) for value in before.values()):
        raise ValueError('Original target link requests must be explicit supported integer counts')
    report = dict(requested_links=links, owners_validated=len(owners), original_requests=before,
        effective_requests={name: links for name, unused in AXES}, calls={name: 0 for name, unused in AXES},
        restored=False, scope='Target model only; sampler and five-layer drafter policies unchanged')
    def requested(cluster_axis=None):
        if cluster_axis is not None and (type(cluster_axis) is not int or cluster_axis not in (0, 1)):
            raise ValueError('Unsupported target collective axis')
        name = 'default' if cluster_axis is None else f'axis{cluster_axis}'
        report['calls'][name] += 1
        return links
    try:
        with instance_overrides([(model, '_qwen_target_link_scope', report),
                (collective, '_qwen_target_link_scope', report), (collective, 'get_num_links', requested)]):
            yield report
    finally:
        if {name: collective.get_num_links(axis) for name, axis in AXES} != before:
            raise AssertionError('Target collective policy was not restored')
        report['restored'] = True
