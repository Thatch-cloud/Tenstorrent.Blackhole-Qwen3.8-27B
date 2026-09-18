"""Source-exact publication-chain substitution; not an admission or automatic install."""

import ast
import hashlib
import textwrap

from frozen_recipe_context import replace_once


ORIGINAL = '''                    historical = retain(operations.slice(active[name], (0, 0, 0, 0), (1, 4, self.history_rows, 128)))
                    accepted = retain(operations.slice(result[name], (0, 0, 0, 0), (1, 4, prefix, 128)))
                    combined = retain(operations.concat([historical, accepted], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
                    tail = retain(operations.slice(combined, (0, 0, self.history_rows + prefix - rows, 0),
                        (1, 4, self.history_rows + prefix, 128)))
                    padded = retain(operations.pad(tail, [(0, 0), (0, 0), (0, 2048 - rows), (0, 0)], 0.0))
                    operations.copy(padded, spare[name])'''
REPLACEMENT = '''                    prepare_slide(self.mesh, active[name], result[name], spare[name],
                        history_rows=self.history_rows, prefix=prefix)()'''


def build_prepare(source, namespace, transport):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DraftKVHistory']
    if len(classes) != 1:
        raise ValueError('One native DraftKVHistory class required')
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == 'prepare']
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError('One undecorated native preparation method required')
    method = methods[0]
    original = '\n'.join(source.splitlines()[method.lineno - 1:method.end_lineno])
    candidate = textwrap.dedent(replace_once(original, ORIGINAL, REPLACEMENT))
    bindings = dict(namespace, prepare_slide=transport)
    exec(compile(candidate, '<draft-kv-slide-prepare>', 'exec'), bindings)
    return bindings['prepare'], dict(
        original_prepare_sha256=hashlib.sha256(original.encode()).hexdigest(),
        candidate_prepare_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        default_changed=False, simulator_qualified=False, hardware_qualified=False)
