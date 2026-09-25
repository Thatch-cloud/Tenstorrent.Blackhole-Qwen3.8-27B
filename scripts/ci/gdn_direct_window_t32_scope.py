"""Request-only T32 convolution routing with native short tails and explicit hit counts."""

from contextlib import contextmanager
import importlib
from pathlib import Path
from unittest.mock import patch

from gdn_direct_window_t32_gate import qualify
from gdn_direct_window_t32_hardware_sources import payloads


@contextmanager
def scoped_direct_windows_t32(evidence, directory, runtime):
    import gdn_batched_conv
    import gdn_device_loop_state
    import ttnn

    directory = Path(directory)
    admission = qualify(evidence, directory, runtime)
    generated = payloads(directory)

    def check_sources():
        for name, source in generated.items():
            if (directory / name).read_bytes() != source.encode():
                raise ValueError('T32 window hardware adapter differs from admitted transformation: ' + name)

    check_sources()
    for name in generated:
        module = importlib.import_module(name[:-3])
        if Path(module.__file__).resolve() != (directory / name).resolve():
            raise ValueError('T32 window adapter loaded from another checkout')
    candidate = importlib.import_module('gdn_direct_window_t32_hardware_batch')
    original = gdn_batched_conv.run_batched_projected
    if (gdn_device_loop_state.run_batched_projected is not original
            or getattr(original, '_direct_window_override', False)):
        raise ValueError('Original unmodified GDN bindings required')
    audit = dict(admission=admission, rows=32, hits=0, fallbacks=0, restored=False, shapes={})

    def run(mesh, projected, *arguments, **options):
        shape = tuple(projected.shape)
        label = str(shape)
        audit['shapes'][label] = audit['shapes'].get(label, 0) + 1
        if len(shape) == 3 and shape[1] == 32 and shape != (1, 32, 8240):
            raise ValueError('Unqualified T32 convolution projection')
        if shape != (1, 32, 8240):
            audit['fallbacks'] += 1
            return original(mesh, projected, *arguments, **options)
        if any(options.get(name) is not True for name in
                ('dma_windows', 'packed_checkpoints', 'norm_batch', 'defer_conv_publication')):
            raise ValueError('Deferred packed T32 checkpoints required')
        if projected.memory_config() != ttnn.L1_MEMORY_CONFIG:
            raise ValueError('Simulator-qualified L1 projection required')
        result = candidate.run_batched_projected(mesh, projected, *arguments, **options)
        audit['hits'] += 1
        return result

    run._direct_window_override = True
    try:
        with patch.object(gdn_batched_conv, 'run_batched_projected', run), \
                patch.object(gdn_device_loop_state, 'run_batched_projected', run):
            yield audit
            if any(module.run_batched_projected is not run for module in (gdn_batched_conv, gdn_device_loop_state)):
                raise ValueError('T32 window binding changed outside its owning scope')
    finally:
        audit['restored'] = all(module.run_batched_projected is original
            for module in (gdn_batched_conv, gdn_device_loop_state))
        check_sources()
        if qualify(evidence, directory, runtime) != admission:
            raise ValueError('T32 window admission changed during request')
