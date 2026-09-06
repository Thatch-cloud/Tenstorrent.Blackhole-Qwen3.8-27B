"""Narrow input-projection batching over the pinned native GDN decode path."""

import ast
import inspect
import textwrap
from types import MethodType


def split_gated_source(source):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one native decode function")
    function = tree.body[0]
    anchor = 'partial = self._row_proj(gated, tw["out"])'
    expected = ast.dump(ast.parse(anchor).body[0])
    matches = [index for index, node in enumerate(function.body) if ast.dump(node) == expected]
    if len(matches) != 1 or len(function.body) - matches[0] != 5:
        raise ValueError("Native output projection tail changed")
    function.name = "forward_gated"
    function.body = function.body[:matches[0]] + [ast.Return(value=ast.Name(id="gated", ctx=ast.Load()))]
    return ast.fix_missing_locations(tree)


def gated_decode(gdn, profiler=None):
    native = type(gdn).forward_decode
    namespace = dict(native.__globals__)
    if profiler:
        from stage_profile import gdn_namespace
        namespace = gdn_namespace(namespace, profiler)
    exec(compile(split_gated_source(inspect.getsource(native)), "<audited-native-gdn-prefix>", "exec"), namespace)
    return MethodType(namespace["forward_gated"], gdn)


def validate_rows(shape):
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 5120 or shape[1] not in (1, 2, 4, 8, 16):
        raise ValueError("Expected [1, T, 5120], T=1/2/4/8/16")
    return shape[1]


def decode_projected(gdn, packed_input, token_inputs, checkpoint, operations, forward=None, profiler=None):
    rows = validate_rows(tuple(packed_input.shape))
    if len(token_inputs) != rows or any(tuple(token.shape) != (1, 1, 5120) for token in token_inputs):
        raise ValueError("Expected one B1 input per verification row")
    original = gdn._project_qkvzab_raw
    packed = original(packed_input, rows, operations.L1_MEMORY_CONFIG)
    cursor = 0

    def projected_row(unused_input, batch, memory):
        nonlocal cursor
        if batch != 1 or cursor >= rows:
            raise ValueError("Native projection callback did not consume exactly one row")
        sliced = operations.slice(packed, (0, cursor, 0), (1, cursor + 1, packed.shape[-1]),
                                  memory_config=memory)
        output = operations.clone(sliced, memory_config=memory)
        cursor += 1
        return output

    outputs = []
    try:
        gdn._project_qkvzab_raw = profiler.wrap("gdn.projected_row_copy", projected_row) if profiler else projected_row
        for index, token in enumerate(token_inputs):
            outputs.append((forward or gdn.forward_decode)(token))
            checkpoint(index + 1)
        if cursor != rows:
            raise ValueError("Native direct-projection path was not engaged")
        return outputs
    finally:
        gdn._project_qkvzab_raw = original
        operations.deallocate(packed)
