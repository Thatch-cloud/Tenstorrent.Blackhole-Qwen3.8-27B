"""Stage explicit fast-serving hooks against the pinned TT plugin source."""

import argparse
import ast
from pathlib import Path
import shutil
import subprocess


PLUGIN_REVISION = 'bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c'


def patch_capacity(source):
    tree = ast.parse(source)
    matches = [node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'get_tt_max_batch_size']
    if len(matches) != 1:
        raise ValueError('Expected exactly one native TT batch capacity function')
    function = matches[0]
    returns = [node for node in function.body if isinstance(node, ast.Return)]
    expected = ast.parse('int(vllm_config.scheduler_config.max_num_seqs)', mode='eval').body
    if len(function.body) != 2 or len(returns) != 1 or ast.dump(returns[0].value) != ast.dump(expected):
        raise ValueError('Pinned native TT batch capacity contract changed or already patched')
    lines = source.splitlines(keepends=True)
    replacement = (
        '    from vllm_tt_plugin.qwen_fast_policy import internal_batch_capacity\n'
        '    return internal_batch_capacity(vllm_config)\n'
    )
    lines[returns[0].lineno - 1:returns[0].end_lineno] = [replacement]
    result = ''.join(lines)
    ast.parse(result)
    return result


def patch_platform(source):
    if 'qwen_fast_policy' in source:
        raise ValueError('Platform already patched')
    expected = ast.parse('not vllm_config.speculative_config', mode='eval').body
    matches = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)
        and ast.dump(node.test) == ast.dump(expected)]
    if len(matches) != 1 or matches[0].col_offset != 8:
        raise ValueError('Pinned speculative platform guard changed')
    original = matches[0]
    lines = source.splitlines(keepends=True)
    baseline = ''.join('    ' + line for line in lines[original.lineno - 1:original.end_lineno])
    replacement = (
        '        from vllm_tt_plugin.qwen_fast_policy import fast_requested, validate_fast_config\n'
        '        if fast_requested(vllm_config):\n'
        '            validate_fast_config(vllm_config)\n'
        '        else:\n' + baseline)
    lines[original.lineno - 1:original.end_lineno] = [replacement]
    result = ''.join(lines)
    ast.parse(result)
    return result


def patch_worker(source):
    if 'serving_startup' in source:
        raise ValueError('Worker already patched')
    tree = ast.parse(source)
    insertions = []
    for name, code in (
            ('compile_or_warm_up_model',
             '        from vllm_tt_plugin.qwen_fast_policy import fast_requested\n'
             '        if fast_requested(self.vllm_config):\n'
             '            from serving_startup import warmup\n'
             '            return warmup(self)\n'),
            ('__del__',
             '        if getattr(self, "_qwen_fast_resources", None) is not None:\n'
             '            from serving_startup import stop\n'
             '            stop(self)\n')):
        matches = [node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name and node.col_offset == 4]
        if len(matches) != 1:
            raise ValueError(f'Pinned worker method missing or ambiguous: {name}')
        method = matches[0]
        offset = 1 if isinstance(method.body[0], ast.Expr) and isinstance(method.body[0].value, ast.Constant) else 0
        insertions.append((method.body[offset].lineno - 1, code))
    lines = source.splitlines(keepends=True)
    for index, code in sorted(insertions, reverse=True):
        lines.insert(index, code)
    result = ''.join(lines)
    ast.parse(result)
    return result


def stage(root):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PLUGIN_REVISION:
        raise ValueError('Exact pinned plugin checkout required')
    package = Path(root) / 'src' / 'vllm_tt_plugin'
    policy = package / 'qwen_fast_policy.py'
    if policy.exists():
        raise ValueError('Refusing to overwrite an existing fast policy')
    edits = {package / name: operation((package / name).read_text(encoding='utf-8'))
        for name, operation in (('config.py', patch_capacity), ('platform.py', patch_platform), ('worker.py', patch_worker))}
    shutil.copyfile(Path(__file__).with_name('serving_fast_policy.py'), policy)
    for path, source in edits.items():
        path.write_text(source, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plugin_root', type=Path)
    arguments = parser.parse_args()
    stage(arguments.plugin_root)
