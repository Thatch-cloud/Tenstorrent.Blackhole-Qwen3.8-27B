"""Unqualified shared-Q/K composition of the retained direct-scatter reader."""

from unittest.mock import patch

from gdn_norm_scatter import replace_reader


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline

    original = pipeline.load_kernels

    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original(requested_root).items()}
        result['norm_gate']['reader'] = replace_reader(result['norm_gate']['reader'])
        return result

    with patch.object(pipeline, 'load_kernels', kernels):
        return pipeline.build(operations, mesh, tensors, root=root)
