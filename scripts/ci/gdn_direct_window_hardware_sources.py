"""Exact source adaptations for a request-scoped direct-window hardware experiment."""

import hashlib
from pathlib import Path


BATCH_SHA256 = '6d5fe07d4fbabd953fa5681b3f055578e18f3719a23a128edbb7fb5de5b1af27'
GUARD = "    if os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR'):\n"
HARDWARE_GUARD = (
    "    if (any(os.environ.get(name) != '1' for name in ('QWEN_GDN_DIRECT_WINDOW',\n"
    "            'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS', 'QWEN_FROZEN_COMBINED_RUNTIME'))\n"
    "            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('QWEN_SIM_ONLY') == '1'):\n"
)


def device(source):
    if source.count(GUARD) != 1 or source.count('Direct convolution windows are simulator-only') != 1:
        raise ValueError('Exact simulator device guard required')
    result = source.replace(GUARD, HARDWARE_GUARD).replace('Direct convolution windows are simulator-only',
        'Allocated direct-window hardware experiment required')
    compile(result, 'direct-window-hardware-device', 'exec')
    return result


def batch(source):
    if hashlib.sha256(source.encode()).hexdigest() != BATCH_SHA256:
        raise ValueError('Exact frozen batched GDN implementation required')
    start, end = '        if dma_windows:\n', '        prefixes = [None] * rows\n'
    if source.count(start) != 1 or source.count(end) != 1 or source.index(start) >= source.index(end):
        raise ValueError('Exact native window and convolution boundaries required')
    replacement = '''        if not (rows == 16 and dma_windows and packed_checkpoints and norm_batch and defer_conv_publication):
            raise ValueError('Direct windows require complete deferred T16 checkpoint publication')
        from gdn_direct_window_hardware_device import execute as direct_windows
        direct = direct_windows(operations, mesh, projected, conv_states, taps, dt_bias, neg_exp_A, own)
        packed, windows = direct[:3], direct[3:]
'''
    result = source[:source.index(start)] + replacement + source[source.index(end):]
    compile(result, 'direct-window-hardware-batch', 'exec')
    return result


def payloads(directory):
    directory = Path(directory)
    return {'gdn_direct_window_hardware_device.py': device((directory / 'gdn_direct_window_device.py').read_text()),
            'gdn_direct_window_hardware_batch.py': batch((directory / 'gdn_batched_conv.py').read_text())}
