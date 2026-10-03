"""Request-owned direct-window convolution; native shorter tails and rollback stay intact."""

from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
from unittest.mock import patch

from gdn_direct_window_gate import REPORT_SHA256
from gdn_direct_window_hardware_sources import payloads
from gdn_direct_window_report import validate


@contextmanager
def scoped_direct_windows(admission, directory):
    import gdn_batched_conv
    import gdn_device_loop_state

    directory = Path(directory)
    if admission.get('report_sha256') != REPORT_SHA256 or admission.get('passed') is not True:
        raise ValueError('Pinned direct-window simulator admission required')
    validate(admission['report'], directory, '/opt/tt-metal')
    generated = payloads(directory)
    for name, source in generated.items():
        if (directory / name).read_bytes() != source.encode():
            raise ValueError('Hardware source differs from qualified transformation: ' + name)
    device = importlib.import_module('gdn_direct_window_hardware_device')
    candidate = importlib.import_module('gdn_direct_window_hardware_batch')
    for module, name in ((device, 'gdn_direct_window_hardware_device.py'),
                         (candidate, 'gdn_direct_window_hardware_batch.py')):
        if Path(module.__file__).resolve() != (directory / name).resolve():
            raise ValueError('Direct-window adapter loaded from another checkout')
    original = gdn_batched_conv.run_batched_projected
    if (gdn_device_loop_state.run_batched_projected is not original
            or getattr(original, '_direct_window_override', False)):
        raise ValueError('Original unmodified GDN bindings required')
    audit = dict(hits=0, fallbacks=0, restored=False, shapes={}, hardware_sources={
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in generated.items()})

    def run(mesh, projected, *arguments, **options):
        shape = tuple(projected.shape)
        label = str(shape)
        audit['shapes'][label] = audit['shapes'].get(label, 0) + 1
        if len(shape) == 3 and shape[1] == 16 and shape != (1, 16, 8240):
            raise ValueError('Unqualified T16 direct-window projection')
        if shape != (1, 16, 8240):
            audit['fallbacks'] += 1
            return original(mesh, projected, *arguments, **options)
        if any(options.get(name) is not True for name in
               ('dma_windows', 'packed_checkpoints', 'norm_batch', 'defer_conv_publication')):
            raise ValueError('Deferred packed T16 checkpoints and shared-QK-compatible normalization required')
        import ttnn
        if projected.memory_config() != ttnn.L1_MEMORY_CONFIG:
            raise ValueError('Simulator-qualified L1 projection required')
        result = candidate.run_batched_projected(mesh, projected, *arguments, **options)
        audit['hits'] += 1
        return result

    run._direct_window_override = True
    try:
        with patch.object(gdn_batched_conv, 'run_batched_projected', run), \
                patch.object(gdn_device_loop_state, 'run_batched_projected', run):
            try:
                yield audit
            finally:
                if any(module.run_batched_projected is not run for module in (gdn_batched_conv, gdn_device_loop_state)):
                    raise ValueError('Direct-window binding changed outside owning request')
    finally:
        audit['restored'] = all(module.run_batched_projected is original
                                for module in (gdn_batched_conv, gdn_device_loop_state))
        validate(admission['report'], directory, '/opt/tt-metal')
        for name, source in generated.items():
            if (directory / name).read_bytes() != source.encode():
                raise ValueError('Direct-window adapter changed during request')
