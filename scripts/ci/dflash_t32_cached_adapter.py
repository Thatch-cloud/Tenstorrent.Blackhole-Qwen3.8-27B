"""Simulator-only cached T32 source preparation; admitted T8/T16 files stay unchanged."""

import os

from frozen_recipe_context import replace_once


def require_simulator():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('Cached T32 drafting remains simulator-only')


def adapt_sources(originals):
    if any('dflash_t32_cached_adapter' in source for source in originals.values()):
        raise ValueError('Cached T32 source already adapted')
    device = originals['dflash_device.py']
    device = replace_once(device, '        window = prefill_window(position)',
        '        if block_rows == 32 and cache_history:\n'
        '            from dflash_t32_cached_adapter import require_simulator\n'
        '            require_simulator()\n'
        '        window = prefill_window(position)')
    if device.count('block_rows not in (8, 16)') != 2:
        raise ValueError('Both original cached/native width guards required')
    device = device.replace('block_rows not in (8, 16)', 'block_rows not in (8, 16, 32)')
    branch = originals['draft_attention_branch.py']
    branch = replace_once(branch, '    if native_proposal_attention and block_rows == 16:',
        '    if native_proposal_attention and block_rows == 32:\n'
        '        from dflash_t32_cached_adapter import require_simulator\n'
        '        require_simulator()\n'
        '    if native_proposal_attention and block_rows == 16:')
    branch = replace_once(branch, "    if parameters.get('native_proposal_attention') and parameters.get('block_rows', 8) == 16:",
        "    if parameters.get('native_proposal_attention') and parameters.get('block_rows', 8) == 32:\n"
        '        from dflash_t32_cached_adapter import require_simulator\n'
        '        require_simulator()\n'
        "    if parameters.get('native_proposal_attention') and parameters.get('block_rows', 8) == 16:")
    branch = replace_once(branch, 'block_rows not in (8, 16)', 'block_rows not in (8, 16, 32)')
    branch = replace_once(branch, "parameters.get('block_rows', 8) not in (8, 16)",
        "parameters.get('block_rows', 8) not in (8, 16, 32)")
    branch = replace_once(branch, '            if block_rows == 16:',
        '            if block_rows == 32:\n'
        '                from dflash_t32_native_attention import attention as native_proposal\n'
        '            elif block_rows == 16:')
    trace = replace_once(originals['dflash_proposal_trace.py'], '            if device.block_rows == 16:',
        '            if device.block_rows == 32:\n'
        '                from dflash_t32_cached_adapter import require_simulator\n'
        '                from dflash_t32_native_attention import validate_mask\n'
        '                require_simulator()\n'
        '            elif device.block_rows == 16:')
    result = {'dflash_device.py': device, 'draft_attention_branch.py': branch, 'dflash_proposal_trace.py': trace}
    for name, source in result.items():
        compile(source, name, 'exec')
    return result
