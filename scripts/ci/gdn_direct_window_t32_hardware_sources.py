"""Preserve native checkpoint publication around the simulator-checked T32 windows."""

from pathlib import Path

from frozen_recipe_context import replace_once
from gdn_direct_window_hardware_sources import batch, device
from gdn_direct_window_t32_adapter import payloads as simulator_payloads


def payloads(directory):
    directory = Path(directory)
    simulator = simulator_payloads({name: (directory / name).read_text() for name in
        ('gdn_direct_window.py', 'gdn_direct_window_device.py')})
    pipeline = batch((directory / 'gdn_batched_conv.py').read_text())
    for before, after in (
        ('rows == 16 and dma_windows', 'rows == 32 and dma_windows'),
        ('deferred T16 checkpoint', 'deferred T32 checkpoint'),
        ('from gdn_direct_window_hardware_device import execute',
            'from gdn_direct_window_t32_hardware_device import execute')):
        pipeline = replace_once(pipeline, before, after)
    result = {'gdn_direct_window_t32.py': simulator['gdn_direct_window_t32.py'],
        'gdn_direct_window_t32_hardware_device.py': device(simulator['gdn_direct_window_t32_device.py']),
        'gdn_direct_window_t32_hardware_batch.py': pipeline}
    for name, source in result.items():
        compile(source, name, 'exec')
    return result
