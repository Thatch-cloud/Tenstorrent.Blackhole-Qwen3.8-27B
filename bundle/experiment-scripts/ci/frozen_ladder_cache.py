"""Separate runtime capacity metadata from unchanged native factory build inputs."""

from frozen_context_geometry import CONTEXTS, geometry


def build_identity(factory):
    if (type(factory.get('capacity')) is not int
            or factory['capacity'] not in {geometry(context)['capacity'] for context in CONTEXTS}):
        raise ValueError('Explicit supported ladder capacity required')
    required = {'report_sha256', 'source', 'source_before', 'factory_sha256',
        'key_chunk_size', 'capacity', 'target_tree_scratch'}
    if set(factory) != required:
        raise ValueError('Reviewed factory build fields required')
    return {name: value for name, value in factory.items() if name != 'capacity'}
