"""Unqualified native replacement for full-history proposal attention; simulator use only."""

from draft_attention import draft_sdpa
from dspark_full_attention import validate_inputs


def execute(operations, mesh, query, key, value, mask, owned, *, context_rows,
        proposals, mask_validated=False):
    if mesh is None or not isinstance(owned, list):
        raise ValueError('Explicit mesh and caller-owned attention output required')
    validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated)
    output = draft_sdpa(operations, query, key, value, mask, key_chunk_size=64)
    owned.append(output)
    return output
