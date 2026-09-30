"""Prepare separate T32 direct-window sources with full tile-face addressing."""


def payloads(originals):
    from frozen_recipe_context import replace_once

    reader = originals['gdn_direct_window.py']
    for before, after in (
        ('0 <= token < 16', '0 <= token < 32'),
        ('T16 causal convolution', 'T32 causal convolution'),
        ('B == 16', 'B == 32'),
        ('token < 16;', 'token < 32;'),
        ('source_row * 32 + face * 512',
            '(source_row / 16) * 1024 + (source_row % 16) * 32 + face * 512'),
        ('token * 32 + face * 512',
            '(token / 16) * 1024 + (token % 16) * 32 + face * 512')):
        reader = replace_once(reader, before, after)
    device = originals['gdn_direct_window_device.py']
    for before, after in (
        ('from gdn_direct_window import reader', 'from gdn_direct_window_t32 import reader'),
        ('(1, 16, 8240)', '(1, 32, 8240)'),
        ('T16 projection/history/gate', 'T32 projection/history/gate'),
        ('for shape in [(1, 16, 5120), (1, 16, 24), (1, 16, 24)] + [(1, 16, 5120)] * 4',
            'for shape in [(1, 32, 5120), (1, 32, 24), (1, 32, 24)] + [(1, 32, 5120)] * 4'),
        ('reader=[4, 160, 1, 16, 1, 258', 'reader=[4, 160, 1, 32, 1, 258')):
        device = replace_once(device, before, after)
    result = {'gdn_direct_window_t32.py': reader,
        'gdn_direct_window_t32_device.py': device}
    for name, source in result.items():
        compile(source, name, 'exec')
    return result
