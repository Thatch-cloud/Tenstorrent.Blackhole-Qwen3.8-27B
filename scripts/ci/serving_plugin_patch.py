"""Stage the native batch-capacity hook against the pinned TT plugin source."""

import argparse
import ast
from pathlib import Path
import shutil


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


def stage(root):
    package = Path(root) / 'src' / 'vllm_tt_plugin'
    config = package / 'config.py'
    policy = package / 'qwen_fast_policy.py'
    if policy.exists():
        raise ValueError('Refusing to overwrite an existing fast policy')
    patched = patch_capacity(config.read_text(encoding='utf-8'))
    shutil.copyfile(Path(__file__).with_name('serving_fast_policy.py'), policy)
    config.write_text(patched, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plugin_root', type=Path)
    arguments = parser.parse_args()
    stage(arguments.plugin_root)
