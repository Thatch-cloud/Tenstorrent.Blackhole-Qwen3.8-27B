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
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 5120 or shape[1] not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Expected [1, T, 5120], T=1/2/4/8/16/32")
    return shape[1]


def validate_reused_input(source):
    tree = ast.parse(textwrap.dedent(source))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    allowed = {"self._project_qkvzab_raw", "self._project_qkvzab", "ttnn.reshape"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != "x" or not isinstance(node.ctx, ast.Load):
            continue
        parent = parents[node]
        if isinstance(parent, ast.Attribute) and parent.attr == "shape":
            continue
        if isinstance(parent, ast.Call) and parent.args and parent.args[0] is node and ast.unparse(parent.func) in allowed:
            continue
        raise ValueError("Native GDN now consumes input outside shape/projection preparation")


def prepare_token_rows(operations, packed, reuse=False):
    rows = validate_rows(tuple(packed.shape))
    owned = [operations.slice(packed, (0, index, 0), (1, index + 1, 5120),
                              memory_config=operations.L1_MEMORY_CONFIG)
             for index in range(1 if reuse else rows)]
    return (owned * rows if reuse else owned), owned


def independent_row(operations, packed, sliced):
    source = operations.get_device_tensors(packed)
    destination = operations.get_device_tensors(sliced)
    if len(source) != 2 or len(destination) != 2:
        raise ValueError("Both chips required for projected-row ownership")
    if any(first.buffer_address() == second.buffer_address()
           for first, second in zip(source, destination, strict=True)):
        raise ValueError("Projected row aliases its live packed projection")
    return sliced


def decode_projected(gdn, packed_input, token_inputs, checkpoint, operations, forward=None, profiler=None,
                     clone_skipped=None, hoist_row_layout=False):
    rows = validate_rows(tuple(packed_input.shape))
    if len(token_inputs) != rows or any(tuple(token.shape) != (1, 1, 5120) for token in token_inputs):
        raise ValueError("Expected one B1 input per verification row")
    if hoist_row_layout and (rows == 1 or clone_skipped is None):
        raise ValueError("Hoisted layout requires multirow selective-clone control")
    original = gdn._project_qkvzab_raw
    packed = original(packed_input, rows, operations.L1_MEMORY_CONFIG)
    row_source = packed
    cursor = 0

    def projected_row(unused_input, batch, memory):
        nonlocal cursor
        if batch != 1 or cursor >= rows:
            raise ValueError("Native projection callback did not consume exactly one row")
        sliced = operations.slice(row_source, (0, cursor, 0), (1, cursor + 1, packed.shape[-1]),
                                  memory_config=memory)
        if hoist_row_layout:
            independent_row(operations, row_source, sliced)
            tiled = operations.to_layout(sliced, operations.TILE_LAYOUT, memory_config=memory)
            independent_row(operations, sliced, tiled)
            independent_row(operations, packed, tiled)
            operations.deallocate(sliced)
            sliced = tiled
        if clone_skipped is not None and cursor > 0:
            output = independent_row(operations, packed, sliced)
            clone_skipped()
        else:
            output = operations.clone(sliced, memory_config=memory)
            if hoist_row_layout:
                operations.deallocate(sliced)
        cursor += 1
        return output

    outputs = []
    try:
        if hoist_row_layout:
            converted = operations.to_layout(packed, operations.ROW_MAJOR_LAYOUT, memory_config=operations.L1_MEMORY_CONFIG)
            row_source = independent_row(operations, packed, converted)
        gdn._project_qkvzab_raw = profiler.wrap("gdn.projected_row_copy", projected_row) if profiler else projected_row
        for index, token in enumerate(token_inputs):
            outputs.append((forward or gdn.forward_decode)(token))
            checkpoint(index + 1)
        if cursor != rows:
            raise ValueError("Native direct-projection path was not engaged")
        return outputs
    finally:
        gdn._project_qkvzab_raw = original
        if row_source is not packed:
            operations.deallocate(row_source)
        operations.deallocate(packed)
